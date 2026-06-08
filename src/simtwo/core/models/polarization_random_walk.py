from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simtwo.core.models.base import DelayPrediction


@dataclass
class RandomWalkPolarizationModel:
    """
    Placeholder physics-model hook for polarization drift. TODO: replace with von Mises-Fischer distribution later on

    Until actual SOP drift physical model is implemented, this keeps a unit stokes vector on the Poincare sphere and applies small random tangent-plane step at each prediction.
    The default initial condition is horizontal polarization state (per the paper): S = (1,0,0)

    This also assumes the S0 parameter is 1, which should probably be configurable later on, too
    """

    step_std: float = 0.025
    seed: int = 42
    name: str = "polarization_random_walk"

    _rng: np.random.Generator = field(init=False)
    _stokes: np.ndarray = field(init=False)
    _step_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> None:
        self._stokes = np.asarray([1.0, 0.0, 0.0], dtype=float)
        self._step_index = 0

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        step = self._rng.normal(loc=0.0, scale=float(self.step_std), size=3)

        # keep the step tangent (geodesic) to the current point on the sphere so the walk behaves like drift around the sphere rather than radial expanse/contract
        step -= np.dot(step, self._stokes) * self._stokes
        candidate = self._stokes + step
        norm = float(np.linalg.norm(candidate))
        if norm <= 0.0 or not np.isfinite(norm):
            candidate = np.asarray([1.0, 0.0, 0.0], dtype=float)
            norm = 1.0

        self._stokes = candidate / norm
        self._step_index += 1

        s1, s2, s3 = (float(v) for v in self._stokes)
        drift_step = float(np.linalg.norm(step))
        state = {
            "stokes": [s1, s2, s3],
            "label": f"step_{self._step_index}",
        }

        return DelayPrediction(
            plot_value=drift_step,
            plot_label="Polarization random-walk step",
            model_family="polarization",
            target_name="polarization_random_walk",
            stokes_vector=(s1, s2, s3),
            poincare_state=state,
            metadata={
                "model_family": "polarization",
                "model_type": "physical_random_walk_placeholder",
                "step_index": self._step_index,
                "random_walk_step_norm": drift_step,
                "S1": s1,
                "S2": s2,
                "S3": s3,
            },
        )
