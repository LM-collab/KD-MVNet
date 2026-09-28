"""
Feature-Space Multi-View Reconstruction Distillation (FMRD).

"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewConditionedReconstructor(nn.Module):
    """H_theta: r_hat_{v} = diag(gamma_v) W_h z^S + beta_v.

    Args:
        embedding_dim: d, dimension of the shared teacher/student embedding space.
        num_views: V, number of synchronized teacher cameras.
        bias: whether W_h carries its own (view-shared) bias term.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_views: int,
        bias: bool = False,
        center_residuals: bool = True,
        view_bias: bool = True,
        dropout: float = 0.0,
        view_rank: int = 0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_views = num_views
        self.center_residuals = center_residuals
      
        self.view_bias = view_bias
        # Dropout inside the reconstructor only
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.W_h = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        
        self.gamma = nn.Parameter(1.0 + 0.1 * torch.randn(num_views, embedding_dim))
        self.beta = nn.Parameter(torch.zeros(num_views, embedding_dim))

        nn.init.normal_(self.W_h.weight, std=embedding_dim ** -0.5)
        if bias:
            nn.init.zeros_(self.W_h.bias)

        self.view_rank = view_rank
        if view_rank > 0:
            self.U = nn.Parameter(torch.randn(num_views, embedding_dim, view_rank)
                                  * embedding_dim ** -0.5)
            self.V = nn.Parameter(torch.zeros(num_views, view_rank, embedding_dim))

    def effective_conditioning(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The (gamma, beta) actually applied, after optional centering."""
        gamma = self.gamma
        beta = self.beta if self.view_bias else torch.zeros_like(self.beta)
        if self.center_residuals:
            # Teacher residuals satisfy sum_v r^T_{v,t} = 0.
           
            gamma = gamma - gamma.mean(dim=0, keepdim=True)
            beta = beta - beta.mean(dim=0, keepdim=True)
        return gamma, beta

    def forward(self, z_student: torch.Tensor, view_idx: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            z_student: (B, T, d) student embeddings.
            view_idx: if given, return the residual for that single view (B, T, d);
                otherwise return residuals for all views (B, V, T, d).
        """
        zd = self.dropout(z_student)
        h = self.W_h(zd)                       # (B, T, d) - the shared transform
        gamma, beta = self.effective_conditioning()

        if view_idx is not None:
            out = gamma[view_idx].view(1, 1, -1) * h + beta[view_idx].view(1, 1, -1)
            if self.view_rank > 0:
                out = out + (zd @ self.V[view_idx].t()) @ self.U[view_idx].t()
            return out

        g = gamma.view(1, self.num_views, 1, -1)   # (1, V, 1, d)
        b = beta.view(1, self.num_views, 1, -1)
        out = g * h.unsqueeze(1) + b               # (B, V, T, d)

        if self.view_rank > 0:
            # (B,T,d) x (V,r,d) -> (B,V,T,r), then (B,V,T,r) x (V,d,r) -> (B,V,T,d)
            proj = torch.einsum("btd,vrd->bvtr", zd, self.V)
            low = torch.einsum("bvtr,vdr->bvtd", proj, self.U)
            if self.center_residuals:
                          
                low = low - low.mean(dim=1, keepdim=True)
            out = out + low
        return out

    def extra_repr(self) -> str:  # pragma: no cover
        n = sum(p.numel() for p in self.parameters())
        return f"embedding_dim={self.embedding_dim}, num_views={self.num_views}, params={n}"


class UnconstrainedReconstructor(nn.Module):
    """
    Ablation: a high-capacity FiLM-MLP reconstructor.
  
    """

    def __init__(
        self,
        embedding_dim: int,
        num_views: int,
        hidden_dim: int = 512,
        view_embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_views = num_views
        self.view_embeddings = nn.Parameter(torch.randn(num_views, view_embedding_dim) * 0.1)

        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "linear": nn.Linear(hidden_dim, hidden_dim),
                        "gamma": nn.Linear(view_embedding_dim, hidden_dim),
                        "beta": nn.Linear(view_embedding_dim, hidden_dim),
                    }
                )
            )
        for blk in self.blocks:
            nn.init.zeros_(blk["gamma"].weight)
            nn.init.zeros_(blk["gamma"].bias)
            nn.init.zeros_(blk["beta"].weight)
            nn.init.zeros_(blk["beta"].bias)

        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _one_view(self, z: torch.Tensor, v: int) -> torch.Tensor:
        e = self.view_embeddings[v]
        h = self.dropout(F.gelu(self.input_proj(z)))
        for blk in self.blocks:
            residual = h
            x = blk["linear"](blk["norm"](h))
            gamma = 1.0 + blk["gamma"](e)
            beta = blk["beta"](e)
            h = self.dropout(F.gelu(gamma * x + beta)) + residual
        return self.output_proj(h)

    def forward(self, z_student: torch.Tensor, view_idx: Optional[int] = None) -> torch.Tensor:
        if view_idx is not None:
            return self._one_view(z_student, view_idx)
        return torch.stack([self._one_view(z_student, v) for v in range(self.num_views)], dim=1)


# --------------------------------------------------------------------------- #
# Teacher-side targets
# --------------------------------------------------------------------------- #
def teacher_consensus_and_residuals(
    teacher_views: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        teacher_views: (B, V, T, d) per-view teacher embeddings.
    Returns:
        consensus: (B, T, d), residuals: (B, V, T, d)
    """
    consensus = teacher_views.mean(dim=1)
    residuals = teacher_views - consensus.unsqueeze(1)
    return consensus, residuals


def resample_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Linear temporal resampling of the teacher sequence to T'.
  
    """
    if x.dim() == 2:  # (B, T)
        if x.shape[1] == target_len:
            return x
        return F.interpolate(x.unsqueeze(1), size=target_len, mode="linear",
                             align_corners=False).squeeze(1)

    if x.dim() == 3:  # (B, T, D)
        if x.shape[1] == target_len:
            return x
        return F.interpolate(x.transpose(1, 2), size=target_len, mode="linear",
                             align_corners=False).transpose(1, 2)

    if x.dim() == 4:  # (B, V, T, D)
        B, V, T, D = x.shape
        if T == target_len:
            return x
        y = x.reshape(B * V, T, D).transpose(1, 2)
        y = F.interpolate(y, size=target_len, mode="linear", align_corners=False)
        return y.transpose(1, 2).reshape(B, V, target_len, D)

    raise ValueError(f"Unsupported tensor rank for temporal resampling: {x.dim()}")


# --------------------------------------------------------------------------- #
# FMRD losses
# --------------------------------------------------------------------------- #
class FMRD(nn.Module):
    """
    The FMRD objective of Eq. (16), together with its reconstructor.

    """

    def __init__(
        self,
        embedding_dim: int,
        num_views: int,
        lambda_anchor: float = 1.0,
        lambda_rec: float = 1.0,
        reconstructor: str = "capacity_controlled",
        residual_weight: float = 0.0,
        source_view_weight: float = 1.0,
        centre_residual: bool = False,
        **reconstructor_kwargs,
    ):
        super().__init__()
        self.lambda_anchor = lambda_anchor
        self.lambda_rec = lambda_rec
        self.reconstructor_type = reconstructor
        # eta of L_rec* = L_full + eta * L_residual. 
        self.residual_weight = residual_weight
       
        self.source_view_weight = source_view_weight
        # Compare residuals after removing their batch mean. 
        self.centre_residual = centre_residual

        if reconstructor == "capacity_controlled":
            self.reconstructor = ViewConditionedReconstructor(
                embedding_dim, num_views, **reconstructor_kwargs
            )
        elif reconstructor == "unconstrained":
            self.reconstructor = UnconstrainedReconstructor(
                embedding_dim, num_views, **reconstructor_kwargs
            )
        else:
            raise ValueError(f"Unknown reconstructor type: {reconstructor}")

    # ----------------------------------------------------------------- Eq. 12
    @staticmethod
    def anchor_loss(z_student: torch.Tensor, consensus: torch.Tensor) -> torch.Tensor:
        
        cos = F.cosine_similarity(z_student, consensus.detach(), dim=-1)
        return (1.0 - cos).mean()

    # ----------------------------------------------------------- Eq. 13 and 14
    def reconstruction_loss(
        self,
        z_student: torch.Tensor,      # (B, T, d)
        consensus: torch.Tensor,      # (B, T, d)
        teacher_views: torch.Tensor,  # (B, V, T, d)
        source_view: Optional[torch.Tensor] = None,   # (B,) index the student saw
        weights: Optional[torch.Tensor] = None,       # (B,) per-sample reliability
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       
        r_hat = self.reconstructor(z_student)                       # (B, V, T, d)
        z_hat = consensus.detach().unsqueeze(1) + r_hat             # Eq. (13)
        tv = teacher_views.detach()
        cos_full = F.cosine_similarity(z_hat, tv, dim=-1)           # (B, V, T)
        per_view = 1.0 - cos_full                                   # (B, V, T)

        if self.residual_weight > 0:
            r_true = tv - consensus.detach().unsqueeze(1)
            a, b = r_hat, r_true
            if self.centre_residual:
                
                a = a - a.mean(dim=0, keepdim=True)
                b = b - b.mean(dim=0, keepdim=True)
            cos_res = F.cosine_similarity(a, b, dim=-1)
            per_view = per_view + self.residual_weight * (1.0 - cos_res)

        per_view = per_view.mean(dim=-1)                            # (B, V)

        if source_view is not None:
            keep = torch.ones_like(per_view)
            keep.scatter_(1, source_view.view(-1, 1).to(per_view.device), 0.0)
            if self.source_view_weight > 0:
                keep = keep + self.source_view_weight * (1.0 - keep)
            denom = keep.sum(dim=1).clamp_min(1e-8)
            per_sample = (per_view * keep).sum(dim=1) / denom
        else:
            per_sample = per_view.mean(dim=1)

        if weights is not None:
            w = weights.to(per_sample.device).clamp_min(0)
            loss = (per_sample * w).sum() / w.sum().clamp_min(1e-8)
        else:
            loss = per_sample.mean()
        return loss, z_hat, per_view

    # ----------------------------------------------------------------- Eq. 16
    def forward(
        self,
        z_student: torch.Tensor,      # (B, T', d)
        teacher_views: torch.Tensor,  # (B, V, T, d) - resampled internally
        source_view: Optional[torch.Tensor] = None,   # (B,) camera the student saw
        weights: Optional[torch.Tensor] = None,       # (B,) teacher reliability
    ) -> Dict[str, torch.Tensor]:
        target_len = z_student.shape[1]
        teacher_views = resample_time(teacher_views.detach(), target_len)
        consensus, _ = teacher_consensus_and_residuals(teacher_views)

        l_anchor = self.anchor_loss(z_student, consensus)
        l_rec, z_hat, _per_view = self.reconstruction_loss(
            z_student, consensus, teacher_views,
            source_view=source_view, weights=weights,
        )

        out = {
            "anchor": l_anchor,
            "rec": l_rec,
            "fmrd": self.lambda_anchor * l_anchor + self.lambda_rec * l_rec,
        }

        with torch.no_grad():
            
            r_hat = z_hat - consensus.unsqueeze(1)
            per_view = r_hat.reshape(r_hat.shape[0], r_hat.shape[1], -1)
            template_ratio = (
                per_view.mean(dim=0).norm(dim=-1).mean()
                / per_view.norm(dim=-1).mean().clamp_min(1e-8)
            )
            out["residual_template_ratio"] = template_ratio

        return out


class MultiViewPredictabilityProbe:
    """
    Held-out linear probe from student embeddings to teacher view residuals.
    
    """

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self._X: List[torch.Tensor] = []
        self._Y: List[torch.Tensor] = []   # (N, V, d) per batch

    @torch.no_grad()
    def accumulate(self, z_student: torch.Tensor, teacher_views: torch.Tensor) -> None:
        """
        Args:
            z_student: (B, T, d)
            teacher_views: (B, V, T, d), temporally resampled internally.
      
        """
        teacher_views = resample_time(teacher_views, z_student.shape[1])
        _, residuals = teacher_consensus_and_residuals(teacher_views)   # (B, V, T, d)
        B, V, T, d = residuals.shape
        self._X.append(z_student.reshape(B * T, d).detach().float().cpu())
        self._Y.append(
            residuals.permute(0, 2, 1, 3).reshape(B * T, V, d).detach().float().cpu()
        )

    @staticmethod
    def _fit_score(
        X_tr: torch.Tensor, Y_tr: torch.Tensor,
        X_te: torch.Tensor, Y_te: torch.Tensor, ridge: float,
    ) -> float:
        d = X_tr.shape[1]
        A = X_tr.t() @ X_tr + ridge * X_tr.shape[0] * torch.eye(d)
        W = torch.linalg.solve(A, X_tr.t() @ Y_tr)
        pred = X_te @ W
        ss_res = ((Y_te - pred) ** 2).sum()
        ss_tot = ((Y_te - Y_te.mean(dim=0, keepdim=True)) ** 2).sum()
        return float(1.0 - ss_res / ss_tot.clamp_min(1e-12))

    @torch.no_grad()
    def score(self, train_frac: float = 0.5, seed: int = 0) -> Dict[str, float]:
        if not self._X:
            raise RuntimeError("probe has no accumulated data")
        X = torch.cat(self._X)            # (N, d)
        Y = torch.cat(self._Y)            # (N, V, d)
        N, V, _ = Y.shape

        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=g)
        n_tr = int(train_frac * N)
        tr, te = perm[:n_tr], perm[n_tr:]
        shuffle = torch.randperm(N, generator=g)

        r2s, r2_shufs = [], []
        for v in range(V):
            Yv = Y[:, v]
            r2s.append(self._fit_score(X[tr], Yv[tr], X[te], Yv[te], self.ridge))
            Ys = Yv[shuffle]
            r2_shufs.append(self._fit_score(X[tr], Ys[tr], X[te], Ys[te], self.ridge))

        return {
            "r2": float(sum(r2s) / V),
            "r2_shuffled": float(sum(r2_shufs) / V),
            "r2_per_view": [float(r) for r in r2s],
            "n_samples": float(N),
        }

    def reset(self) -> None:
        self._X.clear()
        self._Y.clear()


@torch.no_grad()
def teacher_selfview_ceiling(
    teacher_views: torch.Tensor, ridge: float = 1e-3, seed: int = 0
) -> float:
   
    B, V, T, d = teacher_views.shape
    _, residuals = teacher_consensus_and_residuals(teacher_views)

    g = torch.Generator().manual_seed(seed)
    scores = []
    for v in range(V):
        X = teacher_views[:, v].reshape(B * T, d).float().cpu()
        Y = residuals[:, v].reshape(B * T, d).float().cpu()
        perm = torch.randperm(X.shape[0], generator=g)
        n_tr = X.shape[0] // 2
        tr, te = perm[:n_tr], perm[n_tr:]
        scores.append(
            MultiViewPredictabilityProbe._fit_score(X[tr], Y[tr], X[te], Y[te], ridge)
        )
    return float(sum(scores) / V)
