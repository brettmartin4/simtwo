from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from simtwo.core.runtime.session import ExecutionControls, RuntimeSession
from simtwo.core.sequence.plugin import SequenceExperimentContext, SequenceExperimentPlugin

PlotCallback = Callable[[int, float], None]
ConditionsCallback = Callable[[dict[str, Any]], None]
PoincareCallback = Callable[[Any], None]


@dataclass
class SequenceRunner:
    session: RuntimeSession
    controls: ExecutionControls
    plugin: SequenceExperimentPlugin
    seed: int = 42

    _thread: threading.Thread | None = None

    def start(self, cb_plot: PlotCallback, cb_conditions: ConditionsCallback, cb_poincare: PoincareCallback | None = None):
        if self._thread and self._thread.is_alive():
            return

        dataset = self.session.require_dataset()
        model = self.session.require_model()

        self.controls.stop_event.clear()
        self.controls.running = True

        rows = dataset.to_records()
        ctx = SequenceExperimentContext(
            session=self.session,
            controls=self.controls,
            model=model,
            rng=np.random.default_rng(self.seed),
        )
        self.plugin.build(ctx)

        def worker():
            for idx in range(self.session.current_epoch, len(rows)):
                if self.controls.stop_event.is_set():
                    break

                row = dict(rows[idx])
                row.setdefault("epoch", idx)
                self.session.current_epoch = idx

                result = dict(self.plugin.step(ctx, row) or {})
                result.setdefault("epoch", idx)
                result.setdefault("current_model", self.session.current_model_name)
                self.session.results.append(result)

                cb_conditions(result)

                plot_value = self._extract_plot_value(result)
                if plot_value is not None:
                    cb_plot(idx, float(plot_value))

                if cb_poincare is not None:
                    cb_poincare(result.get("poincare_state"))

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
        model = self.session.active_model
        if model is not None and hasattr(model, "reset"):
            try:
                model.reset()
            except Exception:
                pass
        self.session.reset_results()
        self.controls.restart_requested = False

    def export_results(self, path: str):
        # Can probably remove this since its only called after results are obtained? Check back here later
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
    def _extract_plot_value(result: dict[str, Any]) -> float | None:
        for key in (
            "predicted_value",
            "plot_value",
            "predicted_path_delay_s",
            "path_delay_s",
            "time_sync_error",
            "clock_error",
        ):
            value = result.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None
