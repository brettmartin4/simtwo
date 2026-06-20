"""Provide a small sequence experiment wrapper for simtwo driven demos."""

from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from typing import Any

from simtwo.core.sequence.link_model_manager import LinkModelManager


@dataclass
class SimtwoSequenceExperiment:
    """Small sequence experiment wrapper controlled by simtwo.
    
    The class loads environmental data, runs a two node demonstration scenario, and exposes the same lifecycle hooks used by the GUI backend."""
    observations: list[dict[str, Any]] = field(default_factory=list)
    current_epoch: int = 0
    _run_speed_ms: int = 100
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    results_rows: list[dict[str, Any]] = field(default_factory=list)

    def set_run_speed(self, value: int):
        """Store the requested run speed value for compatibility with playback oriented code.
        
        Args:
            value (int): Input value to coerce, normalize, or assign.
        """
        self._run_speed_ms = int(value)

    def cleanup_after_ids(self):
        """Clean up scheduled timeline events or runtime identifiers after execution."""
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def reset_simulation(self):
        """Reset the sequence experiment to its initial state."""
        self.cleanup_after_ids()
        self.current_epoch = 0
        self.results_rows = []

    def load_file(self, file_path: str):
        """Load experiment data from a file path.
        
        Args:
            file_path (str): Path to sequence experiment file.
        """
        rows = []
        with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append({"epoch": i, **row})
        self.observations = rows
        self.current_epoch = 0
        self.results_rows = []

    def load_data(self, file_path: str):
        """Load a CSV dataset and make it available to the active backend or runtime session.
        
        Args:
            file_path (str): Value used for CSV file path.
        """
        self.load_file(file_path)

    def export_results(self, file_path: str):
        """Write the currently generated results to a CSV file.
        
        Args:
            file_path (str): Value used for saving file path.
        """
        if not self.results_rows:
            return
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.results_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.results_rows)

    def configure_channel_model(self, config):
        """Apply a default, existing, or newly trained channel model config.
        
        Args:
            config: Channel model config selected in the GUI.
        """
        self.link_model_manager.configure(config)