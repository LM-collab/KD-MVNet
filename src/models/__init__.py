"""KD-MVNet model components."""

from .backbones import ProjectionHead, get_backbone
from .dp_vic import DPVIC, PrototypeMemory, prototype_diagnostics
from .eg_ssm import (
    EGSSM,
    DiagonalSSM,
    EventGate,
    event_gate_regularizers,
    event_weighted_pooling,
    gate_rate_loss,
    gate_smoothness_loss,
    gate_sync_loss,
    gated_linear_scan,
    get_temporal_module,
    linear_recurrence,
    sequential_linear_scan,
)
from .fmrd import (
    FMRD,
    UnconstrainedReconstructor,
    ViewConditionedReconstructor,
    resample_time,
    teacher_consensus_and_residuals,
)
from .student import AdaptiveFusion, SingleViewStudent, count_parameters
from .teacher import MultiViewTeacher, extract_teacher_targets

__all__ = [
    "ProjectionHead", "get_backbone",
    "DPVIC", "PrototypeMemory", "prototype_diagnostics",
    "EGSSM", "DiagonalSSM", "EventGate", "gated_linear_scan", "sequential_linear_scan",
    "linear_recurrence", "get_temporal_module",
    "event_gate_regularizers", "event_weighted_pooling",
    "gate_rate_loss", "gate_smoothness_loss", "gate_sync_loss",
    "FMRD", "ViewConditionedReconstructor", "UnconstrainedReconstructor",
    "resample_time", "teacher_consensus_and_residuals",
    "SingleViewStudent", "AdaptiveFusion", "count_parameters",
    "MultiViewTeacher", "extract_teacher_targets",
]
