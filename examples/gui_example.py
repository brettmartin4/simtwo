"""

This example just launches an example GUI. Will add support for CSV loading later.

Also, need to fine-tune simulation parameters for expected behavior to jive with 
my predictive models.

Adapter uses sim core components:

  - ThermalQuantumChannel (temperature-dependent distance)
  - RxNodeWithJitter (arrival time jitter, like from time tagger)

It runs one "send photon A -> B" simulation per epoch (TODO: verify terminology here), 
reports the travel time back to the GUI, and displays the current per epoch conditions.

"""

# do not get rid of this (it will cause a runtime error when the example tries to grab the sim class)
from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from sequence.components.photon import Photon
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node

from simtwo.core.GUI import SimulationGUI
from simtwo.core.JitterNode import RxNodeWithJitter
from simtwo.core.ThermalQuantumChannel import ThermalQuantumChannel


@dataclass
class SimulatorAdapter:
    """Adapter that gives the interface expected by SimulationGUI class

    The GUI expects:
      - nodes['A'], nodes['B']
      - observations (len() works)
      - current_epoch
      - run_sim(sender, receiver, update_plot, update_conditions, update_poincare_sphere)
      - set_run_speed(ms)
      - cleanup_after_ids()
      - load_file(path)
      - export_file(path)
    """

    # channel / physics knobs (integrate these with sim later)
    base_distance_m: float = 120_000.0
    alpha_per_C: float = 5e-7
    T0_C: float = 20.0
    attenuation_db_per_m: float = 0.0
    light_speed_m_per_ps: float = 0.0002
    jitter_std_ps: float = 5_000_000.0  # 5 us

    seed: int = 123

    # GUI-facing
    nodes: dict[str, Any] = field(default_factory=lambda: {"A": "A", "B": "B"})
    observations: list[dict[str, Any]] = field(default_factory=list)
    current_epoch: int = 0

    # runtime
    _run_speed_ms: int = 100
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        if not self.observations:
            # For the demo, just create a sort of back and forth temp behavior. Can replace with actual CSV vals later
            self.observations = [
                {"temp_C": float(20.0 + 7.0 * np.sin(i / 6.0)), "epoch": i}
                for i in range(30)
            ]

    # gui controols
    def set_run_speed(self, value: int):
        self._run_speed_ms = int(value)

    def cleanup_after_ids(self):
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def load_file(self, file_path: str):
        """Load observations from a CSV. Each row becomes one epochs conditions. Not working just yet"""
        obs: list[dict[str, Any]] = []
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean: dict[str, Any] = {}
                for k, v in row.items():
                    if v is None:
                        clean[k] = None
                        continue
                    s = v.strip()
                    try:
                        clean[k] = float(s)
                    except Exception:
                        clean[k] = s
                obs.append(clean)
        if obs:
            self.observations = obs
            self.current_epoch = 0

    def export_file(self, file_path: str):
        """Very small placeholder export."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("simtwo gui example export\n")
            f.write(f"epochs={len(self.observations)}\n")
            f.write(f"base_distance_m={self.base_distance_m}\n")
            f.write(f"jitter_std_ps={self.jitter_std_ps}\n")

    # sim entry
    def run_sim(
        self,
        sender: Any,
        receiver: Any,
        update_plot: Callable[[int, float], None],
        update_conditions: Callable[[dict[str, Any]], None] | None = None,
        update_poincare_sphere: Callable[[Any], None] | None = None,
    ):
        """Run epochs in a background thread and call GUI callbacks on main"""

        self._stop_evt.clear()

        def worker():
            for epoch in range(self.current_epoch, len(self.observations)):
                if self._stop_evt.is_set():
                    break

                self.current_epoch = epoch
                cond = self.observations[epoch]

                if update_conditions is not None:
                    update_conditions(cond)

                temp_C = float(cond.get("temp_C", self.T0_C))
                travel_s = self._run_one_epoch(temp_C=temp_C)
                update_plot(epoch, travel_s)

                time.sleep(max(0.0, self._run_speed_ms / 1000.0))

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _run_one_epoch(self, temp_C: float) -> float:
        """One photon A->B using the new ThermalQuantumChannel + RxNodeWithJitter classes"""

        tl = Timeline()
        alice = Node("A", tl)
        bob = RxNodeWithJitter("B", tl, rng=self._rng, jitter_std_ps=self.jitter_std_ps)

        ch = ThermalQuantumChannel(
            name="qc",
            timeline=tl,
            base_distance_m=self.base_distance_m,
            alpha_per_C=self.alpha_per_C,
            T0_C=self.T0_C,
            attenuation=self.attenuation_db_per_m,
            polarization_fidelity=1.0,
            light_speed=self.light_speed_m_per_ps,
        )
        ch.set_ends(alice, bob.name)
        ch.set_temperature(temp_C)

        photon = Photon("p0", tl)
        t_tx_ps = 0

        send_proc = Process(alice, "send_qubit", [bob.name, photon], {})
        tl.schedule(Event(t_tx_ps, send_proc))

        tl.init()
        tl.run()

        if bob.t_done_ps is None:
            return float("nan")

        flight_ps = bob.t_done_ps - t_tx_ps
        return float(flight_ps) * 1e-12


def main():
    sim = SimulatorAdapter()
    gui = SimulationGUI()
    gui.run_sim(sim)


if __name__ == "__main__":
    # TODO: Do arg parsing here? Or do I want to load args from YAML or JSON? Or should everything be configurable from GUI? Ask committee
    main()
