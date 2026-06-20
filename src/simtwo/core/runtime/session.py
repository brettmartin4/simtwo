from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from simtwo.core.models.base import DelayModel


@dataclass
class LoadedDataset:
    """Loaded table plus metadata needed during runtime execution."""
    name: str
    df: pd.DataFrame
    time_column: str = "row_index"
    timezone: str = "UTC"
    posix_unit: str = "s"

    def to_records(self) -> list[dict[str, Any]]:
        """Convert the loaded dataframe into row dicts.
        
        Returns:
            List of dicts keyed by dataset column name. This is convenient but memory heavy for large datasets--use iter_records when streaming rows.
        """
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
        """Extract model features from one dataset row.
        
        Args:
            row: Dict representing one dataset row.
        
        Returns:
            New dict mapping model feature names to float values.
        
        Raises:
            KeyError: If a bound dataset column is missing from row.
            ValueError: If a bound value cannot be converted to a float.
        """
        out: dict[str, float] = {}
        for feature_name, dataset_column in self.mapping.items():
            out[feature_name] = float(row[dataset_column])
        return out

    def has_binding(self, feature_name: str) -> bool:
        """Return whether a model feature has an explicit dataset column binding.
        
        Args:
            feature_name (str): Model facing feature name.
        
        Returns:
            bool: True when the feature is present in the binding map.
        """
        return feature_name in self.mapping

    def dataset_column_for(self, feature_name: str) -> str | None:
        """Return the dataset column bound to a model feature.
        
        Args:
            feature_name (str): Model facing feature name.
        
        Returns:
            str: Dataset column name, or None when no binding exists.
        """
        return self.mapping.get(feature_name)


@dataclass
class ExecutionControls:
    """runtime flags used to coordinate execution and cancellation."""
    # TODO: Maybe change this later? I'm less and less a fan of this step delay feature the more I use it.
    step_delay_ms: int = 10
    stop_event: threading.Event = field(default_factory=threading.Event)
    restart_requested: bool = False
    running: bool = False


@dataclass
class RuntimeSession:
    """Shared state object for dataset, model, controls, and generated results."""
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
        """Clear generated results and rewind the epoch counter. Scheduled for deletion."""
        self.results.clear()
        self.current_epoch = 0

    def set_dataset(self, dataset: LoadedDataset) -> None:
        """Set the active dataset and reset dependent runtime state.
        
        Args:
            dataset (LoadedDataset): Loaded dataset to use for subsequent generation.
        """
        self.dataset = dataset
        self.current_data_name = dataset.name
        self.current_epoch = 0
        self.results.clear()

    def set_model(self, model: DelayModel) -> None:
        """Set the active model and update the display name.
        
        Args:
            model (DelayModel): Model implementing the DelayModel protocol.
        """
        self.active_model = model
        self.current_model_name = model.name

    def require_dataset(self) -> LoadedDataset:
        """Return the loaded dataset or raise a clear runtime error.
        
        Returns:
            LoadedDataset: Active dataset.
        
        Raises:
            RuntimeError: If no dataset has been loaded.
        """
        if self.dataset is None:
            raise RuntimeError("No dataset is loaded in the runtime session.")
        return self.dataset

    def require_model(self) -> DelayModel:
        """Return the active model or raise a clear runtime error.
        
        Returns:
            DelayModel: Active model implementing DelayModel.
        
        Raises:
            RuntimeError: If no model has been configured.
        """
        if self.active_model is None:
            raise RuntimeError("No active model is loaded in the runtime session.")
        return self.active_model