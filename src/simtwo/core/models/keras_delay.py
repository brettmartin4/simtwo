from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simtwo.core.modeling.model import load_trained_model_bundle
from simtwo.core.models.base import DelayPrediction


@dataclass
class KerasDelayModel:
    estimator: Any
    feature_names: list[str]
    target_name: str
    name: str = "loaded_model"

    def predict(self, features: dict[str, float]) -> DelayPrediction:
        row = [float(features[name]) for name in self.feature_names]
        return None