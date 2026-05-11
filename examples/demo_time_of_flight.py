import csv
from pathlib import Path

import numpy as np

from sequence.components.photon import Photon
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node

from simtwo.core.JitterNode import RxNodeWithJitter
from simtwo.core.TempDriver import TempDriver
from simtwo.core.ThermalQuantumChannel import ThermalQuantumChannel


class RecordingRxNode(RxNodeWithJitter):
    def __init__(self, name, timeline, rng, jitter_std_ps=2.0):
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


# Vars

INPUT_CSV = "time_of_flight.csv"
OUTPUT_CSV = "time_of_flight_output.csv"

# Distance between both labs from which dataset was collected:
BASE_DISTANCE_M = 64_000.0

# Ideal conditions:
ALPHA_PER_C = 5e-7
ATTENUATION_DB_PER_M = 0.0
POLARIZATION_FIDELITY = 1.0
LIGHT_SPEED_M_PER_PS = 0.0002

# Taken from data exploration notebook:
JITTER_STD_PS = 2.0
T0_C = 19.995

SEED = 42

# t_sec val increment by 1 sec, so this is 1 second in picosecs
UPDATE_PERIOD_PS = 1_000_000_000_000

# Keep the simulation timeline starting at zero
T0_PS = 0

# Fire the photon just after each temperature update
LAUNCH_OFFSET_PS = 1


# Get csv

rows = []
with open(INPUT_CSV, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        rows.append(
            {
                "epoch": i,
                "temperature": float(row["temperature"]),
                "t_sec": float(row["t_sec"]),
                "path_delay_ns": float(row["path_delay"]),
            }
        )

first_t_sec = rows[0]["t_sec"]
temps = [row["temperature"] for row in rows]


# Setup env

timeline = Timeline()
rng = np.random.default_rng(SEED)

tx_node = Node("tx", timeline)
rx_node = RecordingRxNode("rx", timeline, rng=rng, jitter_std_ps=JITTER_STD_PS)

channel = ThermalQuantumChannel(
    name="thermal_qc",
    timeline=timeline,
    base_distance_m=BASE_DISTANCE_M,
    alpha_per_C=ALPHA_PER_C,
    T0_C=T0_C,
    attenuation=ATTENUATION_DB_PER_M,
    polarization_fidelity=POLARIZATION_FIDELITY,
    light_speed=LIGHT_SPEED_M_PER_PS,
)
channel.set_ends(tx_node, rx_node.name)

temp_driver = TempDriver(
    timeline=timeline,
    channel=channel,
    temps_C=temps,
    update_period_ps=UPDATE_PERIOD_PS,
)
temp_driver.start(t0_ps=T0_PS)


# One photon per dataframe obs

scheduled_meta = {}

for i, row in enumerate(rows):
    send_time_ps = T0_PS + i * UPDATE_PERIOD_PS + LAUNCH_OFFSET_PS
    photon_name = f"p{i}"
    photon = Photon(photon_name, timeline)

    send_proc = Process(tx_node, "send_qubit", [rx_node.name, photon], {})
    timeline.schedule(Event(send_time_ps, send_proc))

    scheduled_meta[photon_name] = {
        "epoch": row["epoch"],
        "temperature": row["temperature"],
        "t_sec": row["t_sec"],
        "path_delay_ns": row["path_delay_ns"],
        "path_delay_ps": row["path_delay_ns"] * 1000.0,
        "send_time_ps": int(send_time_ps),
    }


# Run exp
timeline.init()
timeline.run()


# Writeout results

arrivals_by_name = {row["photon_name"]: row for row in rx_node.arrival_log}

fieldnames = [
    "epoch",
    "photon_name",
    "input_t_sec",
    "input_path_delay_ns",
    "input_path_delay_ps",
    "temperature",
    "channel_distance_m",
    "send_time_ps",
    "arrival_time_ps_raw",
    "arrival_time_ps_reported",
    "flight_time_ps_raw",
    "flight_time_ps_reported",
    "arrival_t_sec_raw",
    "arrival_t_sec_reported",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for photon_name, meta in scheduled_meta.items():
        rx = arrivals_by_name.get(photon_name)
        if rx is None:
            continue

        temp_C = meta["temperature"]
        send_time_ps = meta["send_time_ps"]
        raw_arrival_ps = rx["arrival_time_ps_raw"]
        reported_arrival_ps = rx["arrival_time_ps_reported"]

        channel_distance_m = BASE_DISTANCE_M * (1.0 + ALPHA_PER_C * (temp_C - T0_C))

        arrival_t_sec_raw = None
        if raw_arrival_ps is not None:
            arrival_t_sec_raw = first_t_sec + (raw_arrival_ps * 1e-12)

        arrival_t_sec_reported = None
        if reported_arrival_ps is not None:
            arrival_t_sec_reported = first_t_sec + (reported_arrival_ps * 1e-12)

        writer.writerow(
            {
                "epoch": meta["epoch"],
                "photon_name": photon_name,
                "input_t_sec": meta["t_sec"],
                "input_path_delay_ns": meta["path_delay_ns"],
                "input_path_delay_ps": meta["path_delay_ps"],
                "temperature": temp_C,
                "channel_distance_m": channel_distance_m,
                "send_time_ps": send_time_ps,
                "arrival_time_ps_raw": raw_arrival_ps,
                "arrival_time_ps_reported": reported_arrival_ps,
                "flight_time_ps_raw": None if raw_arrival_ps is None else int(raw_arrival_ps - send_time_ps),
                "flight_time_ps_reported": None if reported_arrival_ps is None else int(reported_arrival_ps - send_time_ps),
                "arrival_t_sec_raw": arrival_t_sec_raw,
                "arrival_t_sec_reported": arrival_t_sec_reported,
            }
        )

print(f"Wrote {len(rx_node.arrival_log)} arrival records to {OUTPUT_CSV}")