"""
Single-View Student.

The student sees one arbitrary camera stream at train and test time.
Everything needed for FMRD lives outside this module (see `fmrd.py`)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ProjectionHead, get_backbone
from .eg_ssm import event_weighted_pooling, get_temporal_module
from .fmrd import resample_time


class AdaptiveFusion(nn.Module):
    """Lightweight adaptive fusion with non-negative mixture weights.

    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_streams: int = 3,
                 balance_scales: bool = False):
        super().__init__()
        self.num_streams = num_streams
        self.balance_scales = balance_scales
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * num_streams, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_streams),
        )

    def forward(self, streams) -> tuple:
        """
        Args:
            streams: sequence of `num_streams` tensors, each (B, D).
        Returns:
            fused (B, D), weights (B, num_streams) summing to 1 and >= 0.

        """
        stacked = torch.stack(streams, dim=1)                 # (B, S, D)
        if self.balance_scales:
            stacked = F.normalize(stacked, dim=-1)
        w = F.softmax(self.gate(stacked.flatten(1)), dim=-1)  # (B, S), non-negative
        fused = (stacked * w.unsqueeze(-1)).sum(dim=1)
        return fused, w


class SingleViewStudent(nn.Module):
    """Compact single-view model.

    Args:
        num_classes: C.
        backbone: student backbone name. 
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "r2plus1d_18",
        embedding_dim: int = 256,
        temporal: str = "egssm",
        egssm_hidden_dim: int = 256,
        egssm_state_dim: int = 128,
        egssm_layers: int = 1,
        pretrained_backbone: bool = True,
        target_rate: float = 0.3,
        dropout: float = 0.1,
        clip_len: int = 32,
        balance_fusion: bool = False,
        use_appearance_stream: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.clip_len = clip_len
      
        self.use_appearance_stream = use_appearance_stream

        self.backbone = get_backbone(backbone, pretrained=pretrained_backbone, frame_level=True)
        backbone_dim = self.backbone.feature_dim

        self.projection = ProjectionHead(
            input_dim=backbone_dim,
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

        self.appearance_proj = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.fusion = AdaptiveFusion(embedding_dim, hidden_dim=embedding_dim,
                                     num_streams=3 if use_appearance_stream else 2,
                                     balance_scales=balance_fusion)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T, H, W) single-view clip.
        """
        f = self.backbone(x)                 
        if f.shape[1] != self.clip_len:
          
            # aligned instants.
            f = resample_time(f, self.clip_len)
        z = self.projection(f)               #normalized
        y, gates = self.temporal(z)          

        s_inv = event_weighted_pooling(z, gates)
        s_event = event_weighted_pooling(y, gates)
        
        s_app = (self.appearance_proj(f.mean(dim=1))
                 if (self.use_appearance_stream or return_intermediates) else None)

        streams = [s_inv, s_event, s_app] if self.use_appearance_stream else [s_inv, s_event]
        fused, weights = self.fusion(streams)
        logits = self.classifier(fused)

        out = {"logits": logits, "fusion_weights": weights}
        if return_intermediates:
            out.update(
                {
                    "embeddings": z,        # z^S_t: the FMRD bottleneck
                    "temporal": y,
                    "gates": gates if gates is not None else torch.zeros(0, device=x.device),
                    "s_inv": s_inv,
                    "s_event": s_event,
                    "s_app": s_app,
                    "fused": fused,
                }
            )
        return out

    @torch.no_grad()
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
