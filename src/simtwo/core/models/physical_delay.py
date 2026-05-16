from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simtwo.core.models.base import DelayPrediction


@dataclass
class PhysicalDelayModel:
    """
    Default physics-based model for path delay.

    Expected input feature:
        {
            "temperature": <float>
        }

    You can bind GUI dataset columns like temperature_x -> temperature
    using FeatureBindings in the runtime session.
    """
    base_distance_m: float = 64_000.0
    alpha_per_c: float = 5e-7
    t0_c: float = 19.995
    light_speed_m_per_ps: float = 0.0002
    jitter_std_ps: float = 2.0
    seed: int = 42
    name: str = "default_physical_model"

    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        temp_c = float(features["temperature"])

        distance_m = self.base_distance_m * (
            1.0 + self.alpha_per_c * (temp_c - self.t0_c)
        )

        base_delay_ps = distance_m / self.light_speed_m_per_ps
        jitter_ps = float(self._rng.normal(loc=0.0, scale=self.jitter_std_ps))
        delay_ps = base_delay_ps + jitter_ps

        delay_ns = delay_ps / 1000.0
        delay_s = delay_ps * 1e-12

        return DelayPrediction(
            path_delay_ps=delay_ps,
            path_delay_ns=delay_ns,
            path_delay_s=delay_s,
            distance_m=distance_m,
            metadata={
                "temperature": temp_c,
                "base_delay_ps": base_delay_ps,
                "jitter_ps": jitter_ps,
            },
        )