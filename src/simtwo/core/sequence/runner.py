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
        self.stop()

        dataset = self.session.require_dataset()
        model = self.session.require_model()

        if model is not None and hasattr(model, "reset"):
            try:
                model.reset()
            except Exception:
                pass

        self.session.reset_results()
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

        plot_points: list[tuple[int, float]] = []
        poincare_states: list[Any] = []
        latest_result: dict[str, Any] = {}

        for idx, source_row in enumerate(rows):
            if self.controls.stop_event.is_set():
                break

            row = dict(source_row)
            row.setdefault("epoch", idx)
            self.session.current_epoch = idx

            result = dict(self.plugin.step(ctx, row) or {})
            result.setdefault("epoch", idx)
            result.setdefault("current_model", self.session.current_model_name)
            self.session.results.append(result)
            latest_result = result

            plot_value = self._extract_plot_value(result)
            if plot_value is not None:
                plot_points.append((idx, float(plot_value)))

            state = result.get("poincare_state")
            if state is not None:
                poincare_states.append(state)

        if not self.controls.stop_event.is_set():
            self.session.current_epoch = len(rows)

        if latest_result:
            cb_conditions(latest_result)
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
