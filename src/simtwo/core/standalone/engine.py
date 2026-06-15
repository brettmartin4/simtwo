from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from simtwo.core.models.base import DelayPrediction
from simtwo.core.runtime.session import (
    ExecutionControls,
    FeatureBindings,
    LoadedDataset,
    RuntimeSession,
)

PlotCallback = Callable[[int, float], None]
ConditionsCallback = Callable[[dict[str, Any]], None]
PoincareCallback = Callable[[Any], None]


@dataclass
class StandaloneEngine:
    """
    standalone runtime.

    This keeps what I had in the previoous build:
    - stepping through one observation at a time
    - using the current active model to generate plotted value
    - ensuring GUI run controls

    The GUI calls:
    - set_dataset(...)
    - set_model(...)
    - start(...)
    - stop(...)
    - reset(...)
    """
    session: RuntimeSession
    controls: ExecutionControls

    _thread: threading.Thread | None = None
    _seed: int = 123

    def _default_dataset(self) -> LoadedDataset:
        rows = []
        for i in range(30):
            rows.append(
                {
                    "epoch": i,
                    "temperature": float(20.0 + 7.0 * np.sin(i / 6.0)),
                }
            )
        df = pd.DataFrame(rows)
        return LoadedDataset(name="default_placeholder", df=df, time_column="epoch")

    def set_dataset(self, dataset: LoadedDataset) -> None:
        self.session.set_dataset(dataset)

    def load_csv(self, path: str, dataset_name: str | None = None, time_column: str = "row_index", timezone: str = "UTC", posix_unit: str = "s"):
        df = pd.read_csv(path, encoding="utf-8-sig")
        self.set_dataset(
            LoadedDataset(
                name=dataset_name or path,
                df=df,
                time_column=time_column,
                timezone=timezone,
                posix_unit=posix_unit,
            )
        )

    def set_model(self, model):
        self.session.set_model(model)

    def set_feature_bindings(self, mapping: dict[str, str]):
        self.session.feature_bindings = FeatureBindings(mapping=mapping)

    def start(self, cb_plot: PlotCallback, cb_conditions: ConditionsCallback, cb_poincare: PoincareCallback | None = None):
        
        self.stop()

        if self.session.dataset is None:
            self.session.set_dataset(self._default_dataset())

        model = self.session.require_model()
        dataset = self.session.require_dataset()

        # If the user is still on the default timing model and no explicit binding
        # has been set, infer the temperature column.  Polarization models may
        # intentionally use no feature bindings, so only do this for timing models.
        model_family = str(getattr(model, "model_family", "timing")).strip().lower()
        if model_family == "timing" and not self.session.feature_bindings.mapping:
            cols = set(str(c) for c in dataset.df.columns)
            if "temperature" in cols:
                self.session.feature_bindings = FeatureBindings(mapping={"temperature": "temperature"})
            elif "temperature_x" in cols:
                self.session.feature_bindings = FeatureBindings(mapping={"temperature": "temperature_x"})
            elif "temp_C" in cols:
                self.session.feature_bindings = FeatureBindings(mapping={"temperature": "temp_C"})

        if model is not None and hasattr(model, "reset"):
            try:
                model.reset()
            except Exception:
                pass

        self.session.reset_results()
        self.controls.stop_event.clear()
        self.controls.running = True

        rows = dataset.to_records()
        plot_points: list[tuple[int, float]] = []
        poincare_states: list[Any] = []
        latest_conditions: dict[str, Any] = {}

        for idx, source_row in enumerate(rows):
            if self.controls.stop_event.is_set():
                break

            row = dict(source_row)
            self.session.current_epoch = idx

            features = self.session.feature_bindings.extract(row)
            pred: DelayPrediction = model.predict(features)
            plot_value = self._prediction_plot_value(pred)

            result = {
                "epoch": idx,
                **row,
                "current_model": self.session.current_model_name,
                "model_family": pred.model_family,
                "target_name": pred.target_name,
                "predicted_value": plot_value,
                "predicted_path_delay_ps": pred.path_delay_ps,
                "predicted_path_delay_ns": pred.path_delay_ns,
                "predicted_path_delay_s": pred.path_delay_s,
                "distance_m": pred.distance_m,
            }
            if pred.stokes_vector is not None:
                result["S1"] = pred.stokes_vector[0]
                result["S2"] = pred.stokes_vector[1]
                result["S3"] = pred.stokes_vector[2]
            self.session.results.append(result)

            latest_conditions = {
                "current_model": self.session.current_model_name,
                "model_family": pred.model_family,
                **features,
                "predicted_value": plot_value,
            }
            if pred.path_delay_ns is not None:
                latest_conditions["predicted_path_delay_ns"] = pred.path_delay_ns
            if pred.path_delay_s is not None:
                latest_conditions["predicted_path_delay_s"] = pred.path_delay_s
            if pred.metadata:
                latest_conditions.update(pred.metadata)

            plot_points.append((idx, plot_value))
            if pred.poincare_state is not None:
                poincare_states.append(pred.poincare_state)

        if not self.controls.stop_event.is_set():
            self.session.current_epoch = len(rows)

        if latest_conditions:
            cb_conditions(latest_conditions)
        for idx, plot_value in plot_points:
            cb_plot(idx, plot_value)
        if cb_poincare is not None:
            for state in poincare_states:
                cb_poincare(state)

        self.controls.running = False

    def stop(self):
        self.controls.stop_event.set()
        self.controls.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def reset(self):
        self.stop()
        model = self.session.active_model
        if model is not None and hasattr(model, "reset"):
            try:
                model.reset()
            except Exception:
                pass
        self.session.reset_results()
        self.controls.restart_requested = False

    def export_results(self, path: str):
        if not self.session.results:
            return

        fieldnames: list[str] = []
        for row in self.session.results:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.session.results)

    @staticmethod
    def _prediction_plot_value(pred: DelayPrediction) -> float:
        if pred.plot_value is not None and np.isfinite(pred.plot_value):
            return float(pred.plot_value)
        if pred.path_delay_s is not None and np.isfinite(pred.path_delay_s):
            return float(pred.path_delay_s)
        if pred.path_delay_ns is not None and np.isfinite(pred.path_delay_ns):
            return float(pred.path_delay_ns)
        if pred.path_delay_ps is not None and np.isfinite(pred.path_delay_ps):
            return float(pred.path_delay_ps)
        return 0.0
