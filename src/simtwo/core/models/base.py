from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class DelayPrediction:
    """Generic model output used by GUI at runtime (before model is loaded)"""
    path_delay_ps: float | None = None
    path_delay_ns: float | None = None
    path_delay_s: float | None = None
    distance_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # basic observer view vars
    plot_value: float | None = None
    plot_label: str = "Model output"
    model_family: str = "timing"
    target_name: str | None = None

    # Polarization / Poincare sphere vars
    stokes_vector: tuple[float, float, float] | None = None
    poincare_state: Any = None


class DelayModel(Protocol):
    
    name: str

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        
        # TODO: make this later (or not? the subclasses will handle it I think.)
        pass