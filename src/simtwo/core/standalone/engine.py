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
        if self._thread and self._thread.is_alive():
            return

        if self.session.dataset is None:
            self.session.set_dataset(self._default_dataset())

        model = self.session.require_model()
        dataset = self.session.require_dataset()

        self.controls.stop_event.clear()
        self.controls.running = True

        rows = dataset.to_records()

        def worker():
            for idx in range(self.session.current_epoch, len(rows)):
                if self.controls.stop_event.is_set():
                    break

                row = dict(rows[idx])
                self.session.current_epoch = idx

                features = self.session.feature_bindings.extract(row)
                pred: DelayPrediction = model.predict(features)

                result = {
                    "epoch": idx,
                    **row,
                    "current_model": self.session.current_model_name,
                    "predicted_path_delay_ps": pred.path_delay_ps,
                    "predicted_path_delay_ns": pred.path_delay_ns,
                    "predicted_path_delay_s": pred.path_delay_s,
                    "distance_m": pred.distance_m,
                }
                self.session.results.append(result)

                conds = {
                    "current_model": self.session.current_model_name,
                    **features,
                    "predicted_path_delay_ns": pred.path_delay_ns,
                    "predicted_path_delay_s": pred.path_delay_s,
                }

                cb_conditions(conds)
                cb_plot(idx, pred.path_delay_s)

                if cb_poincare is not None:
                    cb_poincare(None)

                time.sleep(max(0.0, self.controls.step_delay_ms / 1000.0))

            if not self.controls.stop_event.is_set():
                self.session.current_epoch = len(rows)

            self.controls.running = False

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self):

        self.controls.stop_event.set()
        self.controls.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def reset(self):
        self.stop()
        self.session.reset_results()
        self.controls.restart_requested = False

    def export_results(self, path: str):
        if not self.session.results:
            return

        fieldnames = list(self.session.results[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.session.results)