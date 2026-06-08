from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from simtwo.core.modeling.model import load_trained_model_bundle
from simtwo.core.models.base import DelayPrediction


@dataclass
class SklearnDelayModel:
    estimator: Any
    feature_names: list[str]
    target_name: str
    name: str = "loaded_model"
    model_family: str = "timing"
    seed: int = 42

    _rng: np.random.Generator = field(init=False)
    _stokes: np.ndarray = field(init=False)

    # https://realpython.com/python-data-classes/
    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._stokes = np.asarray([1.0, 0.0, 0.0], dtype=float)

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any], *, model_family: str | None = None) -> "SklearnDelayModel":
        saved_family = str(
            bundle.get("model_family")
            or bundle.get("metadata", {}).get("model_family")
            or ""
        ).strip().lower()
        family = str(model_family or saved_family or "timing").strip().lower()
        return cls(
            estimator=bundle["estimator"],
            feature_names=list(bundle.get("feature_names") or []),
            target_name=str(bundle.get("target_name") or "").strip(),
            name=str(bundle.get("model_name") or "loaded_model"),
            model_family=family,
        )

    @classmethod
    def from_path(cls, path: str | Path, *, model_family: str | None = None) -> "SklearnDelayModel":
        bundle = load_trained_model_bundle(path)
        return cls.from_bundle(bundle, model_family=model_family)

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        row = [float(features[name]) for name in self.feature_names]
        pred = float(self.estimator.predict([row])[0])

        if self.model_family == "polarization":
            return self._predict_polarization(pred)

        return self._predict_timing(pred)

    def _predict_timing(self, pred: float) -> DelayPrediction:
        target_key = self.target_name.strip().lower()
        path_delay_ps: float | None = None
        path_delay_ns: float | None = None
        path_delay_s: float | None = None
        plot_value = pred

        if target_key in {"path_delay", "path_delay_ns", "propagation_delay_ns", "delay_ns"}:
            path_delay_ns = pred
            path_delay_ps = path_delay_ns * 1000.0
            path_delay_s = path_delay_ns * 1e-9
            plot_value = path_delay_s
        elif target_key in {"path_delay_ps", "propagation_delay_ps", "delay_ps"}:
            path_delay_ps = pred
            path_delay_ns = path_delay_ps / 1000.0
            path_delay_s = path_delay_ps * 1e-12
            plot_value = path_delay_s
        elif target_key in {"path_delay_s", "propagation_delay_s", "delay_s"}:
            path_delay_s = pred
            path_delay_ns = path_delay_s * 1e9
            path_delay_ps = path_delay_s * 1e12
            plot_value = path_delay_s
        else:
            # toime related targets such as clock_error, time_sync_error, or timing_error are still valid line plot targets even when they are not path delay
            plot_value = pred

        return DelayPrediction(
            path_delay_ps=path_delay_ps,
            path_delay_ns=path_delay_ns,
            path_delay_s=path_delay_s,
            plot_value=plot_value,
            plot_label=f"Predicted {self.target_name}",
            model_family="timing",
            target_name=self.target_name,
            metadata={
                "model_family": "timing",
                "target_name": self.target_name,
                "feature_names": list(self.feature_names),
                "predicted_value": pred,
            },
        )

    def _predict_polarization(self, pred: float) -> DelayPrediction:
        target_key = self.target_name.strip().lower()

        if target_key in {"s1", "stokes_s1", "stokes_1"}:
            self._stokes[0] = float(np.clip(pred, -1.0, 1.0))
        elif target_key in {"s2", "stokes_s2", "stokes_2"}:
            self._stokes[1] = float(np.clip(pred, -1.0, 1.0))
        elif target_key in {"s3", "stokes_s3", "stokes_3"}:
            self._stokes[2] = float(np.clip(pred, -1.0, 1.0))
        else:
            # A singleoutput model cannot fully define a Poincare point!
            # treat the pred as a drift step magnitude and move the Stokes vector by a random geodesic step
            # this keeps the observer view usable until a multi output polarization model is added later
            # TODO: Figure this out later (or just use the metric used in the IEEE paper)
            step_scale = float(np.clip(abs(pred), 0.0, 0.25))
            direction = self._rng.normal(size=3)
            direction -= np.dot(direction, self._stokes) * self._stokes
            norm = float(np.linalg.norm(direction))
            if norm > 0.0 and np.isfinite(norm):
                self._stokes = self._stokes + (step_scale * direction / norm)

        norm = float(np.linalg.norm(self._stokes))
        if norm <= 0.0 or not np.isfinite(norm):
            self._stokes = np.asarray([1.0, 0.0, 0.0], dtype=float)
        else:
            self._stokes = self._stokes / norm

        s1, s2, s3 = (float(v) for v in self._stokes)
        state = {
            "stokes": [s1, s2, s3],
            "target_name": self.target_name,
            "predicted_value": pred,
        }

        return DelayPrediction(
            plot_value=pred,
            plot_label=f"Predicted {self.target_name}",
            model_family="polarization",
            target_name=self.target_name,
            stokes_vector=(s1, s2, s3),
            poincare_state=state,
            metadata={
                "model_family": "polarization",
                "target_name": self.target_name,
                "feature_names": list(self.feature_names),
                "predicted_value": pred,
                "S1": s1,
                "S2": s2,
                "S3": s3,
            },
        )
