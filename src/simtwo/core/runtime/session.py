from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from simtwo.core.models.base import DelayModel


@dataclass
class LoadedDataset:
    name: str
    df: pd.DataFrame
    time_column: str = "row_index"
    timezone: str = "UTC"
    posix_unit: str = "s"

    def to_records(self) -> list[dict[str, Any]]:
        return self.df.to_dict(orient="records")
    
    def iter_records(self):
        columns = [str(col) for col in self.df.columns]
        for values in self.df.itertuples(index=False, name=None):
            yield dict(zip(columns, values))


@dataclass
class FeatureBindings:
    """
    Maps experiment-facing feature names to dataset column names.

    Example:
        mapping = {"temperature": "temperature_x"}

    Then:
        bindings.extract(row) -> {"temperature": row["temperature_x"]}
    """
    mapping: dict[str, str] = field(default_factory=dict)

    def extract(self, row: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for feature_name, dataset_column in self.mapping.items():
            out[feature_name] = float(row[dataset_column])
        return out

    def has_binding(self, feature_name: str) -> bool:
        return feature_name in self.mapping

    def dataset_column_for(self, feature_name: str) -> str | None:
        return self.mapping.get(feature_name)


@dataclass
class ExecutionControls:
    # TODO: Maybe change this later? I'm less and less a fan of this step delay feature the more I use it.
    step_delay_ms: int = 10
    stop_event: threading.Event = field(default_factory=threading.Event)
    restart_requested: bool = False
    running: bool = False


@dataclass
class RuntimeSession:
    dataset: LoadedDataset | None = None
    active_model: DelayModel | None = None
    feature_bindings: FeatureBindings = field(default_factory=FeatureBindings)

    current_model_name: str = "default"
    current_data_name: str = "none"

    current_epoch: int = 0
    status: str = ""

    results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset_results(self) -> None:
        self.results.clear()
        self.current_epoch = 0

    def set_dataset(self, dataset: LoadedDataset) -> None:
        self.dataset = dataset
        self.current_data_name = dataset.name
        self.current_epoch = 0
        self.results.clear()

    def set_model(self, model: DelayModel) -> None:
        self.active_model = model
        self.current_model_name = model.name

    def require_dataset(self) -> LoadedDataset:
        if self.dataset is None:
            raise RuntimeError("No dataset is loaded in the runtime session.")
        return self.dataset

    def require_model(self) -> DelayModel:
        if self.active_model is None:
            raise RuntimeError("No active model is loaded in the runtime session.")
        return self.active_model