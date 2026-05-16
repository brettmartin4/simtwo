from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class DelayPrediction:
    path_delay_ps: float
    path_delay_ns: float
    path_delay_s: float
    distance_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DelayModel(Protocol):
    
    name: str

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        
        # TODO: make this later (or not? the subclasses will handle it I think.)
        pass