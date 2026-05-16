from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sequence.components.photon import Photon
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node

from simtwo.core.JitterNode import RxNodeWithJitter
from simtwo.core.ThermalQuantumChannel import ThermalQuantumChannel
from simtwo.core.sequence.plugin import BaseSequenceExperiment


class RecordingRxNode(RxNodeWithJitter):
    def __init__(self, name, timeline, rng, jitter_std_ps=0.0):
        super().__init__(name=name, timeline=timeline, rng=rng, jitter_std_ps=jitter_std_ps)
        self.arrival_log = []

    def receive_qubit(self, src: str, qubit):
        super().receive_qubit(src, qubit)
        self.arrival_log.append(
            {
                "photon_name": getattr(qubit, "name", ""),
                "src": src,
                "arrival_time_ps_raw": int(self.t_rx_ps) if self.t_rx_ps is not None else None,
                "arrival_time_ps_reported": int(self.t_done_ps) if self.t_done_ps is not None else None,
            }
        )


@dataclass
class TimeOfFlightPlugin(BaseSequenceExperiment):
    name: str = "Time of Flight (v2)"

    def build(self, ctx) -> None:
        ctx.extra["base_distance_m"] = 64_000.0
        ctx.extra["alpha_per_c"] = 5e-7
        ctx.extra["t0_c"] = 19.995
        ctx.extra["attenuation_db_per_m"] = 0.0
        ctx.extra["light_speed_m_per_ps"] = 0.0002

    def step(self, ctx, row):
        pred = ctx.predict(row)
        features = ctx.session.feature_bindings.extract(row)

        temp_value = None
        if "temperature" in features:
            temp_value = features["temperature"]
        elif "temperature_x" in features:
            temp_value = features["temperature_x"]

        tl = Timeline()
        rng = np.random.default_rng(42 + int(row.get("epoch", 0)))

        tx = Node("tx", tl)
        rx = RecordingRxNode("rx", tl, rng=rng, jitter_std_ps=2.0)

        channel = ThermalQuantumChannel(
            name="thermal_qc",
            timeline=tl,
            base_distance_m=ctx.extra["base_distance_m"],
            alpha_per_C=ctx.extra["alpha_per_c"],
            T0_C=ctx.extra["t0_c"],
            attenuation=ctx.extra["attenuation_db_per_m"],
            polarization_fidelity=1.0,
            light_speed=ctx.extra["light_speed_m_per_ps"],
        )
        channel.set_ends(tx, rx.name)

        if pred.distance_m is not None:
            channel.distance = float(pred.distance_m)
        channel.delay = int(round(pred.path_delay_ps))

        photon = Photon(f"p{row.get('epoch', 0)}", tl)
        send_time_ps = 0
        send_proc = Process(tx, "send_qubit", [rx.name, photon], {})
        tl.schedule(Event(send_time_ps, send_proc))

        tl.init()
        tl.run()

        raw_arrival_ps = rx.t_rx_ps
        reported_arrival_ps = rx.t_done_ps

        flight_time_ps_raw = None if raw_arrival_ps is None else int(raw_arrival_ps - send_time_ps)
        flight_time_ps_reported = None if reported_arrival_ps is None else int(reported_arrival_ps - send_time_ps)

        predicted_path_delay_ps = flight_time_ps_reported
        predicted_path_delay_ns = None if predicted_path_delay_ps is None else predicted_path_delay_ps / 1000.0
        predicted_path_delay_s = None if predicted_path_delay_ps is None else predicted_path_delay_ps * 1e-12

        return {
            "epoch": row.get("epoch", 0),
            "current_model": ctx.session.current_model_name,
            "input_temperature": temp_value,
            "input_t_sec": row.get("t_sec"),
            "input_path_delay_ns": row.get("path_delay", row.get("path_delay_ns")),
            "send_time_ps": send_time_ps,
            "arrival_time_ps_raw": raw_arrival_ps,
            "arrival_time_ps_reported": reported_arrival_ps,
            "flight_time_ps_raw": flight_time_ps_raw,
            "flight_time_ps_reported": flight_time_ps_reported,
            "predicted_path_delay_ps": predicted_path_delay_ps,
            "predicted_path_delay_ns": predicted_path_delay_ns,
            "predicted_path_delay_s": predicted_path_delay_s,
            "distance_m": pred.distance_m,
        }


def build_plugin():
    return TimeOfFlightPlugin()