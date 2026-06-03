from __future__ import annotations

import csv
import threading
from dataclasses import dataclass, field
from typing import Any

from sequence.kernel.timeline import Timeline
from sequence.topology.node import QuantumRouter, BSMNode

from simtwo.core.ThermalQuantumChannel import ThermalQuantumChannel
from simtwo.core.ThermalClassicalChannel import ThermalClassicalChannel
from simtwo.core.sequence.link_model_manager import LinkModelManager
from simtwo.core.sequence.scheduler import SequenceExperimentScheduler


@dataclass
class GuiTwoNodeEntanglementDistribution:
    """
    2 node entanglement distribution experiment for Simtwo demo
    """

    attempts = 100

    # Physical baseline used by Simtwo default model
    base_distance_m = 64_000.0
    alpha_per_c = 5e-7
    t0_c = 19.995
    light_speed_m_per_ps = 0.0002

    # Channel settings
    qchannel_attenuation = 0.0002
    target_fidelity = 0.9

    memo_size = 1

    request_start_offset_ps = 1_000_000_000
    request_end_offset_ps = 3_000_000_000

    # space modeled link updates apart
    row_spacing_ps = 5_000_000_000

    #only run full entanglement request every N rows
    request_every_n_rows = 50

    # state expected by current SequenceBackend
    observations = field(default_factory=list)
    current_epoch = 0
    nodes = field(default_factory=lambda: {"A": "A", "B": "B"})

    _run_speed_ms = 100
    _stop_evt = field(default_factory=threading.Event)
    _thread = None

    results_rows = field(default_factory=list)
    auto_output_csv = "gui_two_node_entanglement_results.csv"

    def __post_init__(self):
        if not self.observations:
            self.observations = [
                {
                    "attempt": i + 1,
                    "epoch": i,
                    "temperature_x": self.t0_c,
                }
                for i in range(self.attempts)
            ]

        self.link_model_manager = LinkModelManager(
            base_distance_m=self.base_distance_m,
            alpha_per_c=self.alpha_per_c,
            t0_c=self.t0_c,
            light_speed_m_per_ps=self.light_speed_m_per_ps,
        )

    # gui hooks

    def set_run_speed(self, value: int):

        self._run_speed_ms = int(value)

    def cleanup_after_ids(self):

        self._stop_evt.set()

    def reset_simulation(self):
        self.cleanup_after_ids()
        self.current_epoch = 0
        self.results_rows = []
        self._thread = None

    def load_file(self, file_path: str):
        rows = []
        with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append({"attempt": i + 1, "epoch": i, **row})

        if rows:
            self.observations = rows
            self.current_epoch = 0
            self.results_rows = []

    def load_data(self, file_path: str):
        self.load_file(file_path)

    def export_results(self, file_path: str):
        self._write_results_csv(file_path)

    def export_file(self, file_path: str):
        self._write_results_csv(file_path)

    def configure_channel_model(self, config):
        self.link_model_manager.configure(config)

    # helper funcs

    def _write_results_csv(self, file_path: str):
        if not self.results_rows:
            return

        fieldnames = list(self.results_rows[0].keys())
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results_rows)

    def _build_topology(self):
        tl = Timeline()

        node_a = QuantumRouter("A", tl, memo_size=self.memo_size)
        node_b = QuantumRouter("B", tl, memo_size=self.memo_size)
        node_m = BSMNode("M", tl, [node_a.name, node_b.name])

        # required by SeQUeNCe reservation rule generation
        node_a.map_to_middle_node[node_b.name] = node_m.name
        node_b.map_to_middle_node[node_a.name] = node_m.name

        # quantum arms to middle BSM node
        qc_a_m = ThermalQuantumChannel(
            "qc_A_M",
            tl,
            base_distance_m=self.base_distance_m / 2.0,
            alpha_per_C=self.alpha_per_c,
            T0_C=self.t0_c,
            attenuation=self.qchannel_attenuation,
            polarization_fidelity=1.0,
            light_speed=self.light_speed_m_per_ps,
            frequency=1e12,
        )
        qc_b_m = ThermalQuantumChannel(
            "qc_B_M",
            tl,
            base_distance_m=self.base_distance_m / 2.0,
            alpha_per_C=self.alpha_per_c,
            T0_C=self.t0_c,
            attenuation=self.qchannel_attenuation,
            polarization_fidelity=1.0,
            light_speed=self.light_speed_m_per_ps,
            frequency=1e12,
        )
        qc_a_m.set_ends(node_a, node_m.name)
        qc_b_m.set_ends(node_b, node_m.name)

        # classical channels:
        # AtoB uses full distance
        cc_a_b = ThermalClassicalChannel(
            "cc_A_B",
            tl,
            base_distance_m=self.base_distance_m,
            alpha_per_C=self.alpha_per_c,
            T0_C=self.t0_c,
            light_speed=self.light_speed_m_per_ps,
        )
        cc_b_a = ThermalClassicalChannel(
            "cc_B_A",
            tl,
            base_distance_m=self.base_distance_m,
            alpha_per_C=self.alpha_per_c,
            T0_C=self.t0_c,
            light_speed=self.light_speed_m_per_ps,
        )
        cc_a_b.set_ends(node_a, node_b.name)
        cc_b_a.set_ends(node_b, node_a.name)

        # A-M and B-M use half distance
        cc_a_m = ThermalClassicalChannel(
            "cc_A_M",
            tl,
            base_distance_m=self.base_distance_m / 2.0,
            alpha_per_C=self.alpha_per_c,
            T0_C=self.t0_c,
            light_speed=self.light_speed_m_per_ps,
        )
        cc_m_a = ThermalClassicalChannel(
            "cc_M_A",
            tl,
            base_distance_m=self.base_distance_m / 2.0,
            alpha_per_C=self.alpha_per_c,
            T0_C=self.t0_c,
            light_speed=self.light_speed_m_per_ps,
        )
        cc_b_m = ThermalClassicalChannel(
            "cc_B_M",
            tl,
            base_distance_m=self.base_distance_m / 2.0,
            alpha_per_C=self.alpha_per_c,
            T0_C=self.t0_c,
            light_speed=self.light_speed_m_per_ps,
        )
        cc_m_b = ThermalClassicalChannel(
            "cc_M_B",
            tl,
            base_distance_m=self.base_distance_m / 2.0,
            alpha_per_C=self.alpha_per_c,
            T0_C=self.t0_c,
            light_speed=self.light_speed_m_per_ps,
        )

        cc_a_m.set_ends(node_a, node_m.name)
        cc_m_a.set_ends(node_m, node_a.name)
        cc_b_m.set_ends(node_b, node_m.name)
        cc_m_b.set_ends(node_m, node_b.name)

        # routing entries required by sequence routing layer
        routing_a = next(
            p for p in node_a.network_manager.protocol_stack
            if hasattr(p, "forwarding_table")
        )
        routing_b = next(
            p for p in node_b.network_manager.protocol_stack
            if hasattr(p, "forwarding_table")
        )

        routing_a.forwarding_table[node_b.name] = node_b.name
        routing_b.forwarding_table[node_a.name] = node_a.name

        return tl, node_a, node_b, node_m, qc_a_m, qc_b_m, cc_a_b, cc_b_a, cc_a_m, cc_m_a, cc_b_m, cc_m_b

    # main sim code here:

    def run_sim(
        self,
        sender: Any,
        receiver: Any,
        update_plot,
        update_conditions=None,
        update_poincare_sphere=None,
    ):
        self._stop_evt.clear()

        def worker():
            (
                tl,
                node_a,
                node_b,
                node_m,
                qc_a_m,
                qc_b_m,
                cc_a_b,
                cc_b_a,
                cc_a_m,
                cc_m_a,
                cc_b_m,
                cc_m_b,
            ) = self._build_topology()

            tl.init()

            # reg simtwo link groups once
            self.link_model_manager.reset_groups()
            self.link_model_manager.register_group(
                "A_to_M",
                [qc_a_m, cc_a_m, cc_m_a],
                delay_fraction=0.5,
                distance_fraction=0.5,
            )
            self.link_model_manager.register_group(
                "B_to_M",
                [qc_b_m, cc_b_m, cc_m_b],
                delay_fraction=0.5,
                distance_fraction=0.5,
            )
            self.link_model_manager.register_group(
                "A_to_B_classical",
                [cc_a_b, cc_b_a],
                delay_fraction=1.0,
                distance_fraction=1.0,
            )

            scheduler = SequenceExperimentScheduler(
                timeline=tl,
                rows=[dict(row) for row in self.observations],
                link_model_manager=self.link_model_manager,
                requester=node_a,
                responder=node_b,
                request_start_offset_ps=self.request_start_offset_ps,
                request_end_offset_ps=self.request_end_offset_ps,
                row_spacing_ps=self.row_spacing_ps,
                memory_size=1,
                target_fidelity=self.target_fidelity,
                request_every_n_rows=self.request_every_n_rows,
                results_rows=self.results_rows,
                cb_plot=update_plot,
                cb_conditions=update_conditions,
            )

            scheduler.schedule_all()
            tl.run()

            self.current_epoch = len(self.observations)
            self._write_results_csv(self.auto_output_csv)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()


def build_sim():
    return GuiTwoNodeEntanglementDistribution()