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

    The model expands the configured reference distance using a linear thermal expansion term, converts that distance to path delay, and adds small Gaussian jitter.
    
    It expects a ``temperature`` feature unless the runtime binds another loaded dataset column to that feature name.
    """
    base_distance_m: float = 64_000.0
    alpha_per_c: float = 5e-7
    t0_c: float = 19.995
    light_speed_m_per_ps: float = 0.0002
    jitter_std_ps: float = 2.0
    seed: int = 42
    name: str = "default_physical_model"
    model_family: str = "timing"

    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the random generator used for timing jitter."""
        self._rng = np.random.default_rng(self.seed)

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        """Predict propagation delay for one feature row.
        
        Args:
            features: Feature mapping containing "temperature" in degrees celsius.
        
        Returns:
            DelayPrediction with path delay in picoseconds, nanoseconds, and seconds, plus metadata describing the temperature and jitter sample.
        
        Raises:
            KeyError: If "temperature" is missing from the feature mapping.
            ValueError: If the temperature cannot be converted to float.
        """
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
            plot_value=delay_s,
            plot_label="Predicted propagation delay (s)",
            model_family="timing",
            target_name="path_delay_s",
            metadata={
                "model_family": "timing",
                "temperature": temp_c,
                "base_delay_ps": base_delay_ps,
                "jitter_ps": jitter_ps,
            },
        )