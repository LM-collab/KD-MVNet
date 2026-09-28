from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.fmrd import FMRD


def kl_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


class StudentObjective(nn.Module):
    """
    Args:
        embedding_dim: d, shared teacher/student embedding dimension.
        num_views: V of the teacher.
        lambda_kl / lambda_anchor / lambda_rec: the three distillation weights.
        kd_temperature: temperature of the logit-level term.
        label_smoothing: applied to L_cls only.
        reconstructor: 'capacity_controlled' (paper, Eq. 15) or 'unconstrained'
            
    """

    def __init__(
        self,
        embedding_dim: int,
        num_views: int,
        lambda_kl: float = 1.0,
        lambda_anchor: float = 1.0,
        lambda_rec: float = 1.0,
        kd_temperature: float = 4.0,
        label_smoothing: float = 0.0,
        reconstructor: str = "capacity_controlled",
        residual_weight: float = 0.0,
        source_view_weight: float = 1.0,
        centre_residual: bool = False,
        **reconstructor_kwargs,
    ):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_anchor = lambda_anchor
        self.lambda_rec = lambda_rec
        self.kd_temperature = kd_temperature
        self.label_smoothing = label_smoothing

        self.fmrd = FMRD(
            embedding_dim=embedding_dim,
            num_views=num_views,
            lambda_anchor=lambda_anchor,
            lambda_rec=lambda_rec,
            reconstructor=reconstructor,
            residual_weight=residual_weight,
            source_view_weight=source_view_weight,
            centre_residual=centre_residual,
            **reconstructor_kwargs,
        )

    def forward(
        self,
        student_out: Dict[str, torch.Tensor],
        teacher_targets: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        use_kl: bool = True,
        use_anchor: bool = True,
        use_rec: bool = True,
        source_view: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_out: output of `SingleViewStudent(x, return_intermediates=True)`.
            teacher_targets: dict with 'logits' (B, C) and 'view_embeddings'
                (B, V, T, d), both already detached.
            labels: (B,) ground-truth classes.
            use_kl / use_anchor / use_rec: switches for the Table 2 ablation.
        """
        device = labels.device
        zero = torch.zeros((), device=device)

        losses: Dict[str, torch.Tensor] = {}
        losses["cls"] = F.cross_entropy(
            student_out["logits"], labels, label_smoothing=self.label_smoothing
        )

        losses["kl"] = (
            kl_distillation(student_out["logits"], teacher_targets["logits"], self.kd_temperature)
            if use_kl
            else zero
        )

        if use_anchor or use_rec:
            fmrd_out = self.fmrd(
                student_out["embeddings"], teacher_targets["view_embeddings"],
                source_view=source_view,
            )
            losses["anchor"] = fmrd_out["anchor"] if use_anchor else zero
            losses["rec"] = fmrd_out["rec"] if use_rec else zero
            losses["residual_template_ratio"] = fmrd_out["residual_template_ratio"]
        else:
            losses["anchor"] = zero
            losses["rec"] = zero

        losses["total"] = (
            losses["cls"]
            + self.lambda_kl * losses["kl"]
            + self.lambda_anchor * losses["anchor"]
            + self.lambda_rec * losses["rec"]
        )
        return losses


FMRD_ABLATIONS = {
    "baseline":        dict(use_kl=False, use_anchor=False, use_rec=False),
    "logit_only":      dict(use_kl=True,  use_anchor=False, use_rec=False),
    "logit_anchor":    dict(use_kl=True,  use_anchor=True,  use_rec=False),
    "logit_rec":       dict(use_kl=True,  use_anchor=False, use_rec=True),
    "full_fmrd":       dict(use_kl=True,  use_anchor=True,  use_rec=True),
}
