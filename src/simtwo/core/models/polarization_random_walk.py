from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simtwo.core.models.base import DelayPrediction

from scipy.stats import uniform_direction, vonmises_fisher


@dataclass
class RandomWalkPolarizationModel:
    """
    Placeholder physics-model hook for polarization drift. Uses time broadening vMF distribution about the poincare sphere.

    Previously, this kept a unit stokes vector on the Poincare sphere and applied small random tangent-plane step at each prediction.

    The default initial condition is horizontal polarization state (per the paper): S = (1,0,0)

    This also assumes the S0 parameter is 1, which should probably be configurable later on, too
    """

    step_std: float = 0.025
    uniform_after_s: float = 3600.0
    min_kappa: float = 1e-6
    max_kappa: float = 1_000_000.0
    time_feature: str = "posix_time"
    seed: int = 42
    name: str = "polarization_von_mises_fisher"
    initial_kappa: float | None = None

    _rng: np.random.Generator = field(init=False)
    _stokes: np.ndarray = field(init=False)
    _first_posix_time: float | None = field(default=None, init=False)
    _step_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.initial_kappa is None:
            if float(self.step_std) <= 0.0:
                self.initial_kappa = float(self.max_kappa)
            else:
                self.initial_kappa = min(float(self.max_kappa), max(1.0, 1.0 / float(self.step_std) ** 2))
        self.reset()

    def reset(self) -> None:
        self._stokes = np.asarray([1.0, 0.0, 0.0], dtype=float)
        self._first_posix_time = None
        self._step_index = 0

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        posix_time = self._read_posix_time(features)
        elapsed_s = self._elapsed_seconds(posix_time)
        spread_factor = self._spread_factor(elapsed_s)
        kappa = self._kappa_from_spread(spread_factor)
        previous_stokes = self._stokes.copy()

        if kappa <= 0.0:
            sample = uniform_direction.rvs(dim=3, random_state=self._rng)
            kappa_used = 0.0
        elif float(self.step_std) <= 0.0 and spread_factor <= 0.0:
            sample = self._stokes
            kappa_used = kappa
        else:
            sample = vonmises_fisher.rvs(mu=self._stokes, kappa=kappa, random_state=self._rng)
            kappa_used = kappa

        self._stokes = self._normalize_stokes(sample)
        self._step_index += 1

        s1, s2, s3 = (float(v) for v in self._stokes)
        angular_step_rad = self._angular_distance(previous_stokes, self._stokes)
        state = {
            "stokes": [s1, s2, s3],
            "label": f"step_{self._step_index}",
        }

        return DelayPrediction(
            plot_value=angular_step_rad,
            plot_label="Polarization angular step",
            model_family="polarization",
            target_name="polarization_random_walk",
            stokes_vector=(s1, s2, s3),
            poincare_state=state,
            metadata={
                "model_family": "polarization",
                "model_type": "physical_von_mises_fisher_placeholder",
                "step_index": self._step_index,
                "time_feature": self.time_feature,
                "posix_time": posix_time,
                "elapsed_s": elapsed_s,
                "spread_factor": spread_factor,
                "kappa": kappa_used,
                "angular_step_rad": angular_step_rad,
                "S1": s1,
                "S2": s2,
                "S3": s3,
            },
        )

    def _read_posix_time(self, features: dict[str, float]) -> float | None:
        value = features.get(self.time_feature)
        if value is None:
            return None
        try:
            posix_time = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(posix_time):
            return None
        return posix_time

    def _elapsed_seconds(self, posix_time: float | None) -> float:
        if posix_time is None:
            return float(self._step_index)
        if self._first_posix_time is None:
            self._first_posix_time = posix_time
        return max(0.0, float(posix_time - self._first_posix_time))

    def _spread_factor(self, elapsed_s: float) -> float:
        if self.uniform_after_s <= 0.0:
            return 1.0
        return min(1.0, max(0.0, float(elapsed_s) / float(self.uniform_after_s)))

    def _kappa_from_spread(self, spread_factor: float) -> float:
        kappa = float(self.initial_kappa or 0.0) * (1.0 - float(spread_factor))
        if kappa <= float(self.min_kappa):
            return 0.0
        return kappa

    def _normalize_stokes(self, sample: Any) -> np.ndarray:
        vector = np.asarray(sample, dtype=float).reshape(-1, 3)[0]
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0 or not np.isfinite(norm):
            return np.asarray([1.0, 0.0, 0.0], dtype=float)
        return vector / norm

    def _angular_distance(self, left: np.ndarray, right: np.ndarray) -> float:
        dot = float(np.clip(np.dot(left, right), -1.0, 1.0))
        return float(np.arccos(dot))
