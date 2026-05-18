from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from typing import Any

from simtwo.core.sequence.link_model_manager import LinkModelManager


@dataclass
class SimtwoSequenceExperiment:
    observations: list[dict[str, Any]] = field(default_factory=list)
    current_epoch: int = 0
    _run_speed_ms: int = 100
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    results_rows: list[dict[str, Any]] = field(default_factory=list)

    def set_run_speed(self, value: int):
        self._run_speed_ms = int(value)

    def cleanup_after_ids(self):
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def reset_simulation(self):
        self.cleanup_after_ids()
        self.current_epoch = 0
        self.results_rows = []

    def load_file(self, file_path: str):
        rows = []
        with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append({"epoch": i, **row})
        self.observations = rows
        self.current_epoch = 0
        self.results_rows = []

    def load_data(self, file_path: str):
        self.load_file(file_path)

    def export_results(self, file_path: str):
        if not self.results_rows:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.results_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.results_rows)

    def configure_channel_model(self, config):
        self.link_model_manager.configure(config)