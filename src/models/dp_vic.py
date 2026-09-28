"""
Dual-Prototype View-Invariant Contrastive Learning (DP-VIC).

"""

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _dist_active() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


@torch.no_grad()
def _all_gather_pairs(
    indices: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if not _dist_active():
        return indices, targets

    world = dist.get_world_size()
    n = torch.tensor([indices.numel()], device=targets.device)
    counts = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(counts, n)
    n_max = int(max(int(c.item()) for c in counts))
    if n_max == 0:
        return indices[:0], targets[:0]

    pad_i = torch.full((n_max,), -1, dtype=indices.dtype, device=targets.device)
    pad_t = torch.zeros(n_max, targets.shape[1], dtype=targets.dtype, device=targets.device)
    pad_i[: indices.numel()] = indices
    pad_t[: targets.shape[0]] = targets

    gi = [torch.zeros_like(pad_i) for _ in range(world)]
    gt = [torch.zeros_like(pad_t) for _ in range(world)]
    dist.all_gather(gi, pad_i)
    dist.all_gather(gt, pad_t)

    idx = torch.cat(gi)
    tgt = torch.cat(gt)
    keep = idx >= 0
    return idx[keep], tgt[keep]


class PrototypeMemory(nn.Module):
    """
    L2-normalized prototype bank with EMA updates (Eq. 2).

    Prototypes that have never been observed are initialized on first sight with
    the observed consensus vector rather than being EMA-blended with random
    noise. 
    """

    def __init__(self, num_prototypes: int, feature_dim: int, momentum: float = 0.99):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim
        self.momentum = momentum

        protos = F.normalize(torch.randn(num_prototypes, feature_dim), p=2, dim=1)
        self.register_buffer("prototypes", protos)
        self.register_buffer("initialized", torch.zeros(num_prototypes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, indices: torch.Tensor, targets: torch.Tensor) -> None:
        """
        EMA update.

        Args:
            indices: (N,) long tensor of prototype slots to update.
            targets: (N, D) update targets (the multi-view consensus of Eq. 1).

        """
        if indices.numel() == 0 and not _dist_active():
            return

        indices, targets = _all_gather_pairs(indices, targets)
        if indices.numel() == 0:
            return

        targets = F.normalize(targets.detach().float(), p=2, dim=1)

 
        uniq, inverse = torch.unique(indices, return_inverse=True)
        summed = torch.zeros(uniq.numel(), self.feature_dim, device=targets.device, dtype=targets.dtype)
        summed.index_add_(0, inverse, targets)
        counts = torch.zeros(uniq.numel(), device=targets.device, dtype=targets.dtype)
        counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=targets.dtype))
        mean_targets = summed / counts.unsqueeze(1).clamp_min(1.0)

        seen = self.initialized[uniq]
        current = self.prototypes[uniq]

        m = self.momentum
        updated = torch.where(
            seen.unsqueeze(1),
            m * current + (1.0 - m) * mean_targets,
            mean_targets,
        )
        self.prototypes[uniq] = F.normalize(updated, p=2, dim=1).to(self.prototypes.dtype)
        self.initialized[uniq] = True

    def forward(self) -> torch.Tensor:  # pragma: no cover - trivial
        return self.prototypes


class DPVIC(nn.Module):
    """
    Dual-Prototype View-Invariant Contrastive Learning.

    Args:
        num_classes: number of action categories C (class prototypes).
        num_instances: number of unique executions M, i.e. subject-trial pairs
            (instance prototypes).
        feature_dim: embedding dimension d_z.
        
    """

    def __init__(
        self,
        num_classes: int,
        num_instances: int,
        feature_dim: int,
        temperature: float = 0.07,
        class_weight: float = 1.0,
        instance_weight: float = 0.5,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.temperature = temperature
        self.class_weight = class_weight
        self.instance_weight = instance_weight

        self.class_memory = PrototypeMemory(num_classes, feature_dim, momentum)
        self.instance_memory = PrototypeMemory(num_instances, feature_dim, momentum)

    # ------------------------------------------------------------------ Eq. 1
    @staticmethod
    def multi_view_consensus(view_descriptors: torch.Tensor) -> torch.Tensor:
        """

        Args:
            view_descriptors: (B, V, D) temporally mean-pooled per-view descriptors.
        Returns:
            (B, D) L2-normalized consensus.
        """
        return F.normalize(view_descriptors.mean(dim=1), p=2, dim=-1)

    # ------------------------------------------------------------------ Eq. 2
    @torch.no_grad()
    def update_prototypes(
        self,
        view_descriptors: torch.Tensor,  # (B, V, D)
        class_labels: torch.Tensor,      # (B,)
        instance_indices: torch.Tensor,  # (B,)
    ) -> None:
        consensus = self.multi_view_consensus(view_descriptors)
        valid_c = class_labels >= 0
        self.class_memory.update(class_labels[valid_c], consensus[valid_c])
        
        valid_i = (instance_indices >= 0) & (instance_indices < self.num_instances)
        self.instance_memory.update(instance_indices[valid_i], consensus[valid_i])

    # ------------------------------------------------------------------ Eq. 3
    def forward(
        self,
        view_descriptors: torch.Tensor,  # (B, V, D)
        class_labels: torch.Tensor,      # (B,)
        instance_indices: torch.Tensor,  # (B,)
        update_prototypes: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute L_DP-VIC averaged over all views in the batch.

        """
        B, V, D = view_descriptors.shape
        device = view_descriptors.device

        if update_prototypes is None:
            update_prototypes = self.training
        if update_prototypes:
            self.update_prototypes(view_descriptors, class_labels, instance_indices)

        s = F.normalize(view_descriptors, p=2, dim=-1).reshape(B * V, D)


        class_protos = self.class_memory.prototypes            # (C, D)
        instance_protos = self.instance_memory.prototypes      # (M, D)
        bank = torch.cat([class_protos, instance_protos], dim=0)  # (C+M, D)

        logits = (s @ bank.t()) / self.temperature             # (B*V, C+M)

        cls_idx = class_labels.repeat_interleave(V)                                   # (B*V,)
        inst_raw = instance_indices.repeat_interleave(V)

        inst_valid = (inst_raw >= 0) & (inst_raw < self.num_instances)
        inst_idx = inst_raw.clamp_min(0) + self.num_classes                           # (B*V,)

  
        max_logit = logits.max(dim=1, keepdim=True).values.detach()
        log_denom = (logits - max_logit).exp().sum(dim=1).clamp_min(1e-12).log() + max_logit.squeeze(1)

        rows = torch.arange(s.shape[0], device=device)
        pos_c = logits[rows, cls_idx]
        pos_i = logits[rows, inst_idx]

      
        w = torch.tensor(
            [self.class_weight, self.instance_weight], device=device, dtype=logits.dtype
        ).clamp_min(1e-30).log()
        inst_term = torch.where(
            inst_valid, pos_i + w[1], torch.full_like(pos_i, float("-inf"))
        )
        log_num = torch.logsumexp(torch.stack([pos_c + w[0], inst_term], dim=1), dim=1)

        loss = (log_denom - log_num).mean()

        with torch.no_grad():
            class_logits = logits[:, : self.num_classes]
            inst_logits = logits[:, self.num_classes :]
            metrics = {
                "dpvic_class_acc": (class_logits.argmax(1) == cls_idx).float().mean(),
                "dpvic_instance_acc": (
                    (inst_logits.argmax(1) == (inst_idx - self.num_classes)) & inst_valid
                ).float().sum() / inst_valid.sum().clamp_min(1),
                "dpvic_pos_class_sim": (pos_c * self.temperature).mean(),
                "dpvic_pos_instance_sim": (pos_i * self.temperature).mean(),
            }

        return {"loss": loss, **metrics}


class ClassOnlyDPVIC(DPVIC):
    

    def __init__(self, *args, **kwargs):
        kwargs["instance_weight"] = 0.0
        super().__init__(*args, **kwargs)


class InstanceOnlyDPVIC(DPVIC):
    
    def __init__(self, *args, **kwargs):
        kwargs["class_weight"] = 0.0
        super().__init__(*args, **kwargs)


@torch.no_grad()
def prototype_diagnostics(dpvic: DPVIC) -> Dict[str, torch.Tensor]:
    """
    Returns inter-class prototype cosine distance, and mean pairwise cosine
    similarity within each prototype set (a collapse indicator).
    """
    out: Dict[str, torch.Tensor] = {}

    cp = dpvic.class_memory.prototypes
    init = dpvic.class_memory.initialized
    if init.sum() > 1:
        cp = cp[init]
        sim = cp @ cp.t()
        off = ~torch.eye(cp.shape[0], dtype=torch.bool, device=cp.device)
        out["class_inter_distance"] = (1.0 - sim[off]).mean()
        out["class_pairwise_sim"] = sim[off].mean()

    ip = dpvic.instance_memory.prototypes
    iinit = dpvic.instance_memory.initialized
    if iinit.sum() > 1:
        ip = ip[iinit]
        sim = ip @ ip.t()
        off = ~torch.eye(ip.shape[0], dtype=torch.bool, device=ip.device)
        out["instance_pairwise_sim"] = sim[off].mean()

    return out
