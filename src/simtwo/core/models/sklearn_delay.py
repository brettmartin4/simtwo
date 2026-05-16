from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simtwo.core.modeling.model import load_trained_model_bundle
from simtwo.core.models.base import DelayPrediction


@dataclass
class SklearnDelayModel:
    estimator: Any
    feature_names: list[str]
    target_name: str
    name: str = "loaded_model"

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> "SklearnDelayModel":
        return cls(
            estimator=bundle["estimator"],
            feature_names=list(bundle.get("feature_names") or []),
            target_name=str(bundle.get("target_name") or "").strip(),
            name=str(bundle.get("model_name") or "loaded_model"),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "SklearnDelayModel":
        bundle = load_trained_model_bundle(path)
        return cls.from_bundle(bundle)

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        row = [float(features[name]) for name in self.feature_names]
        pred = float(self.estimator.predict([row])[0])

        if self.target_name in {"path_delay", "path_delay_ns"}:
            path_delay_ns = pred
            path_delay_ps = path_delay_ns * 1000.0
            path_delay_s = path_delay_ns * 1e-9
        elif self.target_name == "path_delay_ps":
            path_delay_ps = pred
            path_delay_ns = path_delay_ps / 1000.0
            path_delay_s = path_delay_ps * 1e-12
        elif self.target_name == "path_delay_s":
            path_delay_s = pred
            path_delay_ns = path_delay_s * 1e9
            path_delay_ps = path_delay_s * 1e12
        else:
            raise ValueError(
                f"Unsupported target_name '{self.target_name}'. "
                "Expected one of: path_delay, path_delay_ns, path_delay_ps, path_delay_s."
            )

        return DelayPrediction(
            path_delay_ps=path_delay_ps,
            path_delay_ns=path_delay_ns,
            path_delay_s=path_delay_s,
            metadata={
                "target_name": self.target_name,
                "feature_names": list(self.feature_names),
            },
        )