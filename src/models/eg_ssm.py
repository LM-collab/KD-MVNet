"""
Event-Gated State-Space Model (EG-SSM).
"""

from typing import Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Parallel scan for h_t = a * h_{t-1} + u_t  (constant a, real, |a| < 1)
# --------------------------------------------------------------------------- #
def gated_linear_scan(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """

    Args:
        a: (N,) decay coefficients in (0, 1), one per state channel.
        u: (B, T, N) per-timestep inputs (already gated).
    Returns:
        (B, T, N) hidden states h_t = sum_{k<=t} a^{t-k} u_k.

    """
    B, T, N = u.shape
    coeff = a.reshape(1, 1, N).expand(B, T, N).clone()
    h = u.clone()

    step = 1
    while step < T:
        h_shift = F.pad(h[:, :-step], (0, 0, step, 0))
        c_shift = F.pad(coeff[:, :-step], (0, 0, step, 0))
        h = h + coeff * h_shift
        coeff = coeff * c_shift
        step *= 2
    return h


def sequential_linear_scan(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

    B, T, N = u.shape
    h = u.new_zeros(B, N)
    out = []
    for t in range(T):
        h = a * h + u[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def linear_recurrence(a: torch.Tensor, u: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    
    if mode == "scan":
        return gated_linear_scan(a, u)
    if mode == "sequential":
        return sequential_linear_scan(a, u)
    if mode != "auto":
        raise ValueError(f"unknown scan mode: {mode}")
    return gated_linear_scan(a, u) if u.is_cuda else sequential_linear_scan(a, u)


class DiagonalSSM(nn.Module):
 

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        output_dim: Optional[int] = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        scan_mode: str = "auto",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim or input_dim
        self.scan_mode = scan_mode

        
        A_diag = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_diag))

        self.B = nn.Parameter(torch.randn(state_dim, input_dim) / math.sqrt(input_dim))
        self.C = nn.Parameter(torch.randn(self.output_dim, state_dim) / math.sqrt(state_dim))
        self.D = nn.Parameter(torch.zeros(self.output_dim, input_dim))

        log_dt = torch.rand(state_dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (A_bar: (N,), B_bar: (N, input_dim))."""
        A = -torch.exp(self.A_log)              # (N,)
        dt = torch.exp(self.log_dt)             # (N,)
        A_bar = torch.exp(A * dt)               # (N,) in (0, 1)
  
        B_bar = ((A_bar - 1.0) / A).unsqueeze(1) * self.B
        return A_bar, B_bar

    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            gate: (B, T) event gate g_{v,t}; if None the layer degenerates to a
                standard (ungated) LTI SSM, which is the "Standard SSM" ablation.
        Returns:
            y: (B, T, output_dim)
        """
        A_bar, B_bar = self.discretize()

        u = x @ B_bar.t()                       # (B, T, N)
        if gate is not None:
            u = u * gate.unsqueeze(-1)          # gated input intake, Eq. (5)

        h = linear_recurrence(A_bar, u, self.scan_mode)   # (B, T, N)
        y = h @ self.C.t() + x @ self.D.t()
        return y


class EventGate(nn.Module):


    def __init__(self, input_dim: int, init_rate: float = 0.5):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, 1)
        nn.init.normal_(self.proj.weight, std=0.01)

        init_rate = min(max(init_rate, 1e-3), 1 - 1e-3)
        nn.init.constant_(self.proj.bias, math.log(init_rate / (1 - init_rate)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T) in (0, 1)."""
        return torch.sigmoid(self.proj(self.norm(z)).squeeze(-1))


class EGSSM(nn.Module):
    """
    Event-Gated State-Space Model block stack.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        state_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gate: bool = True,
        init_rate: float = 0.5,
        scan_mode: str = "auto",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_gate = use_gate

        self.event_gate = EventGate(input_dim, init_rate=init_rate) if use_gate else None
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.ssm_layers = nn.ModuleList(
            DiagonalSSM(hidden_dim, state_dim, hidden_dim, scan_mode=scan_mode)
            for _ in range(num_layers)
        )
        self.layer_norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self, z: torch.Tensor, gate: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            z: (B, T, D) frame embeddings.
            gate: optional externally supplied gate (used when a caller wants a
                single gate shared by several branches).
        Returns:
            y: (B, T, D), gates: (B, T) or None.
        """
        if gate is None and self.event_gate is not None:
            gate = self.event_gate(z)

        x = self.input_proj(z)
        for ssm, ln in zip(self.ssm_layers, self.layer_norms):
            residual = x
            x = ssm(ln(x), gate=gate)
            x = self.dropout(x)
            x = x + residual

        return self.output_proj(x), gate


class TransformerTemporal(nn.Module):
    """Ablation baseline"""

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, z: torch.Tensor, gate: Optional[torch.Tensor] = None):
        return self.output_proj(self.encoder(self.input_proj(z))), None


class AveragePoolTemporal(nn.Module):
    

    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, z: torch.Tensor, gate: Optional[torch.Tensor] = None):
        return self.identity(z), None


def get_temporal_module(name: str, input_dim: int, **kwargs) -> nn.Module:
  
    name = name.lower()
    if name in ("egssm", "eg-ssm", "eg_ssm"):
        return EGSSM(input_dim, use_gate=True, **kwargs)
    if name in ("ssm", "standard_ssm", "s5"):
        return EGSSM(input_dim, use_gate=False, **kwargs)
    if name == "transformer":
        return TransformerTemporal(input_dim, **kwargs)
    if name in ("avgpool", "average", "mean"):
        return AveragePoolTemporal(input_dim, **kwargs)
    if name == "mamba":
        raise NotImplementedError(
            "The Mamba baseline requires a real selective-SSM implementation "
            "(pip install mamba-ssm) and must not be aliased to EG-SSM."
        )
    raise ValueError(f"Unknown temporal module: {name}")


# --------------------------------------------------------------------------- #
# Gate regularizers (Eq. 6-8)
# --------------------------------------------------------------------------- #
def gate_rate_loss(gates: torch.Tensor, target_rate: float = 0.3) -> torch.Tensor:
    """
    Args:
        gates: (B, V, T) or (B, T).
    """
    return (gates.mean() - target_rate) ** 2


def gate_smoothness_loss(gates: torch.Tensor) -> torch.Tensor:
    """
    Args:
        gates: (B, V, T) or (B, T); time is the last dimension.
    """
    if gates.shape[-1] < 2:
        return gates.sum() * 0.0
    return (gates[..., 1:] - gates[..., :-1]).abs().mean()


def gate_sync_loss(gates: torch.Tensor) -> torch.Tensor:
    """
    Args:
        gates: (B, V, T) with V >= 2.
    """
    if gates.dim() != 3 or gates.shape[1] < 2:
        return gates.sum() * 0.0

    return gates.var(dim=1, unbiased=False).mean()


def event_gate_regularizers(
    gates: torch.Tensor, target_rate: float = 0.3
) -> Dict[str, torch.Tensor]:

    return {
        "rate": gate_rate_loss(gates, target_rate),
        "smooth": gate_smoothness_loss(gates),
        "sync": gate_sync_loss(gates),
    }


def event_weighted_pooling(features: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """

    Args:
        features: (B, T, D)
        gates: (B, T) or None -> uniform mean pooling.
    Returns:
        (B, D)
    """
    if gates is None:
        return features.mean(dim=1)
    w = gates.unsqueeze(-1)
    return (features * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
