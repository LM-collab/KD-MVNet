"""
Backbones producing.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import NORMALIZATION


class _NormalizedBackbone(nn.Module):
 

    def _init_norm(self, stats: str) -> None:
        mean, std = NORMALIZATION[stats]
        self.norm_stats = stats
        self.register_buffer("_mean", torch.tensor(mean).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 3, 1, 1, 1), persistent=False)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, T, H, W) in [0, 1] -> normalized."""
        return (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)


class ProjectionHead(nn.Module):
    """Maps backbone features to the shared L2-normalized embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, T, D = x.shape
            z = self.net(x.reshape(B * T, D))
            return F.normalize(z, p=2, dim=-1).reshape(B, T, -1)
        return F.normalize(self.net(x), p=2, dim=-1)


class _FrameLevelResNet3D(_NormalizedBackbone):
    """Shared wrapper for torchvision's video ResNets (R3D / R(2+1)D / MC3)."""

    def __init__(self, base: nn.Module, feature_dim: int, stats: str = "kinetics"):
        super().__init__()
        self._init_norm(stats)
        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) in [0, 1] -> (B, T', d0)"""
        x = self.normalize(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))      # keep temporal axis
        return x.squeeze(-1).squeeze(-1).permute(0, 2, 1)    # (B, T', C)


class _FrameLevelSwin3D(_NormalizedBackbone):
    """Video Swin transformer, spatially pooled at each temporal token position.

   
    """

    def __init__(self, base: nn.Module, feature_dim: int, stats: str = "imagenet"):
        super().__init__()
        self._init_norm(stats)
        self.patch_embed = base.patch_embed
        self.pos_drop = base.pos_drop
        self.features = base.features
        self.norm = base.norm
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) in [0, 1] -> (B, T/2, d0)"""
        x = self.normalize(x)
        x = self.patch_embed(x)                 # (B, T', H', W', C)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        return x.mean(dim=(2, 3))               # (B, T', C)


class _FrameLevelS3D(_NormalizedBackbone):
  

    def __init__(self, base: nn.Module, feature_dim: int, stats: str = "kinetics"):
        super().__init__()
        self._init_norm(stats)
        self.features = base.features
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) in [0, 1] -> (B, T/8, d0)"""
        x = self.normalize(x)
        x = self.features(x)                                 # (B, C, T', H', W')
        x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))
        return x.squeeze(-1).squeeze(-1).permute(0, 2, 1)


class VideoMAEBackbone(_NormalizedBackbone):
  

    def __init__(self, model_name: str = "MCG-NJU/videomae-base", pretrained: bool = True):
        super().__init__()
        self._init_norm("imagenet")
        try:
            from transformers import VideoMAEConfig, VideoMAEModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError("VideoMAE backbone requires `pip install transformers`") from exc

        if pretrained:
            self.backbone = VideoMAEModel.from_pretrained(model_name)
        else:
            # The architecture of model_name without its weights. The library default
            # VideoMAEConfig() is not that architecture: it mean-pools instead of applying
            # the final LayerNorm, so a model built from it cannot load a checkpoint that
            # was trained from model_name.
            self.backbone = VideoMAEModel(VideoMAEConfig.from_pretrained(model_name))

        cfg = self.backbone.config
        self.feature_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.tubelet_size = getattr(cfg, "tubelet_size", 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) in [0, 1] -> (B, T/tubelet, d0)"""
        B, C, T, H, W = x.shape
        x = self.normalize(x)
        tokens = self.backbone(x.permute(0, 2, 1, 3, 4)).last_hidden_state  # (B, N, d)

        n_t = T // self.tubelet_size
        n_s = tokens.shape[1] // max(n_t, 1)
        if n_t * n_s != tokens.shape[1]:  # pragma: no cover - shape guard
            raise RuntimeError(
                f"Cannot factor {tokens.shape[1]} VideoMAE tokens into {n_t} temporal "
                f"positions; check clip length, patch size and tubelet size."
            )
        return tokens.reshape(B, n_t, n_s, self.feature_dim).mean(dim=2)


_TORCHVISION_VIDEO = {
    "r3d_18": ("r3d_18", "R3D_18_Weights", 512),
    "r2plus1d_18": ("r2plus1d_18", "R2Plus1D_18_Weights", 512),
    "mc3_18": ("mc3_18", "MC3_18_Weights", 512),
}



_TORCHVISION_MODERN = {
    "swin3d_t": ("swin3d_t", "Swin3D_T_Weights", 768, "imagenet", _FrameLevelSwin3D),
    "swin3d_s": ("swin3d_s", "Swin3D_S_Weights", 768, "imagenet", _FrameLevelSwin3D),
    "swin3d_b": ("swin3d_b", "Swin3D_B_Weights", 1024, "imagenet", _FrameLevelSwin3D),
    "s3d": ("s3d", "S3D_Weights", 1024, "kinetics", _FrameLevelS3D),
}

_UNAVAILABLE = {
    "r3d_50": (
        "R3D-50 is not in torchvision. Load a Kinetics-pretrained checkpoint "
        "(PySlowFast / mmaction2) and wrap it with _FrameLevelResNet3D."
    ),
    "r2plus1d_34": (
        "R(2+1)D-34 is not in torchvision; the IG65M/Kinetics R(2+1)D-34 weights must be "
        "loaded explicitly. Note Table 1 reports R(2+1)D-34 as a teacher while Sec. 4.0.3 "
        "says only 18- and 50-layer variants were used."
    ),
    "x3d_xs": "X3D-XS requires pytorchvideo: torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs').",
    "x3d_m": "X3D-M requires pytorchvideo: torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m').",
    "mvit_v2_s": (
        "MViTv2-S ships in torchvision but does not expose a frame-level feature. "
        "Its blocks pool the token grid as they go, so recovering f_{v,t} means "
        "tracking the (T, H, W) shape through every block and reshaping the "
        "sequence at the end -- doable, but it is a reimplementation of the "
        "forward pass rather than a wrapper. Use swin3d_t for the R1-4 "
        "comparison: same parameter scale, same Kinetics-400 pretraining, and "
        "its features come out already shaped (B, T', H', W', C)."
    ),
}


def get_backbone(name: str, pretrained: bool = True, frame_level: bool = True) -> nn.Module:
    """
    Backbone factory.

    Args:
        name: e.g. 'r3d_18', 'r2plus1d_18', 'videomae_base', 'videomae_large'.
        pretrained: load Kinetics-400 / VideoMAE pretrained weights.
        frame_level: kept for API compatibility; frame-level output is always
            produced because the method needs f_{v,t}.
    """
    name = name.lower().replace("-", "_").replace("(", "").replace(")", "").replace("+", "plus")

    aliases = {
        "r3d": "r3d_18",
        "r3d18": "r3d_18",
        "r2plus1d": "r2plus1d_18",
        "r2plus1d18": "r2plus1d_18",
        "r2plus1_d_18": "r2plus1d_18",
        "videomae": "videomae_base",
    }
    name = aliases.get(name, name)

    if name in _TORCHVISION_VIDEO:
        ctor_name, weights_name, dim = _TORCHVISION_VIDEO[name]
        from torchvision.models import video as tv_video

        ctor = getattr(tv_video, ctor_name)
        try:
            weights = getattr(tv_video, weights_name).DEFAULT if pretrained else None
            base = ctor(weights=weights)
        except AttributeError:  # older torchvision
            base = ctor(pretrained=pretrained)
        return _FrameLevelResNet3D(base, dim)

    if name in _TORCHVISION_MODERN:
        ctor_name, weights_name, dim, stats, wrapper = _TORCHVISION_MODERN[name]
        from torchvision.models import video as tv_video

        ctor = getattr(tv_video, ctor_name, None)
        if ctor is None:
            raise NotImplementedError(
                f"{name} needs a newer torchvision than {__import__('torchvision').__version__}"
            )
        try:
            weights = getattr(tv_video, weights_name).DEFAULT if pretrained else None
            base = ctor(weights=weights)
        except AttributeError:  # older torchvision
            base = ctor(pretrained=pretrained)
        return wrapper(base, dim, stats)

    if name == "videomae_base":
        return VideoMAEBackbone("MCG-NJU/videomae-base", pretrained=pretrained)
    if name == "videomae_large":
        return VideoMAEBackbone("MCG-NJU/videomae-large", pretrained=pretrained)

    if name in _UNAVAILABLE:
        raise NotImplementedError(f"{name}: {_UNAVAILABLE[name]}")

    raise ValueError(
        f"Unknown backbone '{name}'. Available: "
        f"{sorted(list(_TORCHVISION_VIDEO) + list(_TORCHVISION_MODERN))} "
        f"+ ['videomae_base', 'videomae_large']."
    )
