"""
Multi-View Teacher: shared backbone + DP-VIC + EG-SSM + parameter-free fusion.

Implements Sec. 3.2 - 3.5 of the paper.
L_total = L_cls + alpha L_DP-VIC + beta L_rate + gamma L_smooth + delta L_sync

"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ProjectionHead, get_backbone
from .dp_vic import DPVIC, prototype_diagnostics
from .fmrd import resample_time
from .eg_ssm import (
    EGSSM,
    event_gate_regularizers,
    event_weighted_pooling,
    get_temporal_module,
)


class MeanConsensusFusion(nn.Module):
    """Sec. 3.5: parameter-free consensus fusion across cameras."""

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        
        return view_features.mean(dim=1)


class UncertaintyFusion(nn.Module):
    """Ablation only: inverse-variance weighted fusion (a learnable alternative)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        var = self.net(view_features).squeeze(-1) + 1e-6   # (B, V)
        w = (1.0 / var)
        w = w / w.sum(dim=1, keepdim=True)
        return (view_features * w.unsqueeze(-1)).sum(dim=1)


class MultiViewTeacher(nn.Module):
    """Multi-view teacher network.

    Args:
        num_classes: C.
        num_instances: M, number of unique subject-trial executions in the
            training split (size of the instance prototype bank).
        num_views: V.
        backbone: one of the names accepted by `get_backbone`.
        embedding_dim: d_z of the shared embedding space.
        temporal: temporal module name; 'egssm' is the method.
        fusion: 'mean' (paper) or 'uncertainty' (ablation).
        aux_view_logits: whether to add the per-view auxiliary classification
      
    """

    def __init__(
        self,
        num_classes: int,
        num_instances: int,
        num_views: int = 3,
        backbone: str = "r3d_18",
        embedding_dim: int = 256,
        temporal: str = "egssm",
        egssm_hidden_dim: int = 512,
        egssm_state_dim: int = 256,
        egssm_layers: int = 2,
        fusion: str = "mean",
        pretrained_backbone: bool = True,
        temperature: float = 0.07,
        class_weight: float = 1.0,
        instance_weight: float = 0.5,
        prototype_momentum: float = 0.99,
        target_rate: float = 0.3,
        aux_view_logits: bool = True,
        dropout: float = 0.1,
        target_len: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_views = num_views
        self.embedding_dim = embedding_dim
        self.target_rate = target_rate
        self.aux_view_logits = aux_view_logits
        self.target_len = target_len

        self.backbone = get_backbone(backbone, pretrained=pretrained_backbone, frame_level=True)
        self.projection = ProjectionHead(
            input_dim=self.backbone.feature_dim,
            hidden_dim=embedding_dim * 2,
            output_dim=embedding_dim,
        )

        self.temporal_name = temporal
        self.temporal = get_temporal_module(
            temporal,
            embedding_dim,
            hidden_dim=egssm_hidden_dim,
            state_dim=egssm_state_dim,
            num_layers=egssm_layers,
            dropout=dropout,
            init_rate=target_rate,
        ) if temporal in ("egssm", "ssm") else get_temporal_module(
            temporal, embedding_dim, hidden_dim=egssm_hidden_dim, dropout=dropout
        )

        self.dp_vic = DPVIC(
            num_classes=num_classes,
            num_instances=num_instances,
            feature_dim=embedding_dim,
            temperature=temperature,
            class_weight=class_weight,
            instance_weight=instance_weight,
            momentum=prototype_momentum,
        )

        self.fusion = MeanConsensusFusion() if fusion == "mean" else UncertaintyFusion(embedding_dim)

        # A single classifier, applied to both the fused and the per-view
      
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    # ------------------------------------------------------------------ utils
    def encode_view(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, target_len, d) L2-normalized frame embeddings.

        """
        f = self.backbone(x)
        if f.shape[1] != self.target_len:
            f = resample_time(f, self.target_len)
        return self.projection(f)

    @staticmethod
    def _stack_views(views) -> torch.Tensor:
        if isinstance(views, (list, tuple)):
            return torch.stack(views, dim=1)
        return views

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        views,
        class_labels: Optional[torch.Tensor] = None,
        instance_indices: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
        update_prototypes: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            views: list of V tensors (B, C, T, H, W), or a single (B, V, C, T, H, W).
        """
        x = self._stack_views(views)              # (B, V, C, T, H, W)
        B, V = x.shape[:2]

        # Shared backbone: fold views into the batch so the backbone sees one
       
        flat = x.reshape(B * V, *x.shape[2:])
        z = self.encode_view(flat)                 # (B*V, T, d)
        T, d = z.shape[1], z.shape[2]
        z = z.reshape(B, V, T, d)

        # --- Temporal branch (EG-SSM). Gate is computed per view by the shared
       
        y, gates = self.temporal(z.reshape(B * V, T, d))
        y = y.reshape(B, V, T, d)
        if gates is not None:
            gates = gates.reshape(B, V, T)

        # --- Event-weighted pooling -> per-view descriptors for classification.
        s_view = event_weighted_pooling(
            y.reshape(B * V, T, d),
            None if gates is None else gates.reshape(B * V, T),
        ).reshape(B, V, d)

        # --- Parameter-free consensus fusion.
        s_fused = self.fusion(s_view)
        logits = self.classifier(s_fused)

        out: Dict[str, torch.Tensor] = {"logits": logits, "fused_features": s_fused}

        if self.aux_view_logits:
            out["view_logits"] = self.classifier(s_view.reshape(B * V, d)).reshape(B, V, -1)

        # --- Losses -------------------------------------------------------
        if class_labels is not None:
            out["cls_loss"] = F.cross_entropy(logits, class_labels)

            if self.aux_view_logits:
                out["aux_cls_loss"] = F.cross_entropy(
                    out["view_logits"].reshape(B * V, -1),
                    class_labels.repeat_interleave(V),
                )

            if instance_indices is not None:
                # DP-VIC operates on temporally mean-pooled *projected*
                # embeddings, i.e. before EG-SSM.
                view_descriptors = z.mean(dim=2)          # (B, V, d)
                dpvic = self.dp_vic(
                    view_descriptors, class_labels, instance_indices,
                    update_prototypes=update_prototypes,
                )
                out["dp_vic_loss"] = dpvic["loss"]
                for k, v in dpvic.items():
                    if k != "loss":
                        out[k] = v

            if gates is not None:
                regs = event_gate_regularizers(gates, self.target_rate)
                out["rate_loss"] = regs["rate"]
                out["smooth_loss"] = regs["smooth"]
                out["sync_loss"] = regs["sync"]
                out["gate_mean"] = gates.mean().detach()

        # --- Intermediates for FMRD ---------------------------------------
        if return_intermediates:
            out["view_embeddings"] = z          # (B, V, T, d) - FMRD targets
            out["consensus"] = z.mean(dim=1)    # (B, T, d)    - Eq. (10)
            out["view_pooled"] = s_view
            if gates is not None:
                out["view_gates"] = gates

        return out

    # ---------------------------------------------------------------- Eq. (9)
    def total_loss(
        self,
        out: Dict[str, torch.Tensor],
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.05,
        delta: float = 0.1,
        aux_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """L_total = L_cls + a L_DP-VIC + b L_rate + g L_smooth + d L_sync.
       
        """
        device = out["logits"].device
        zero = torch.zeros((), device=device)

        total = out["cls_loss"]
        parts = {"cls": out["cls_loss"]}

        for name, weight, key in (
            ("dp_vic", alpha, "dp_vic_loss"),
            ("rate", beta, "rate_loss"),
            ("smooth", gamma, "smooth_loss"),
            ("sync", delta, "sync_loss"),
        ):
            value = out.get(key, zero)
            parts[name] = value
            total = total + weight * value

        if aux_weight > 0.0 and "aux_cls_loss" in out:
            parts["aux_cls"] = out["aux_cls_loss"]
            total = total + aux_weight * out["aux_cls_loss"]

        parts["total"] = total
        return parts

    @torch.no_grad()
    def prototype_stats(self) -> Dict[str, torch.Tensor]:
        return prototype_diagnostics(self.dp_vic)


@torch.no_grad()
def extract_teacher_targets(
    teacher: MultiViewTeacher,
    views,
    target_len: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Run a frozen teacher and return the stop-gradient FMRD targets.
   
    """
    was_training = teacher.training
    teacher.eval()
    out = teacher(views, return_intermediates=True, update_prototypes=False)
    if was_training:
        teacher.train()

    targets = {
        "view_embeddings": out["view_embeddings"],
        "logits": out["logits"],
    }
    if target_len is not None:
        from .fmrd import resample_time

        targets["view_embeddings"] = resample_time(targets["view_embeddings"], target_len)
    return targets
