import csv
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt

from sequence.kernel.timeline import Timeline
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.topology.node import Node
from sequence.components.photon import Photon

from simtwo.core.TempDriver import TempDriver
from simtwo.core.ThermalQuantumChannel import ThermalQuantumChannel


# config (replace with yaml or other file later)
CSV_PATH = "D:/simulator2/data/consolidated_weather_timesync.csv"
TEMP_COL: Union[str, int] = "temperature_x"
SKIP_HEADER_FOR_INDEX_COL = False

DISTANCE_M = 120_000
ALPHA_PER_C = 5e-7
T0_C = 20.0

# 1 Hz CSV -> 1 second per temperature update
UPDATE_PERIOD_S = 1.0

# Send one photon right at each tick
SEND_OFFSET_S = 0.0

# Jitter stdev = 2 ps
JITTER_STD_PS = 2

SEED = 123

# Downsample plot points (depends on input data--definitely change for WR data)
PLOT_EVERY_N = 1


# helper funcs
def load_temps_from_csv(csv_path: str, temp_col: Union[str, int], *, skip_header_for_index_col: bool = False, delimiter: str = ",") -> List[float]:
    temps: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        try:
            first_row = next(reader)
        except StopIteration:
            raise ValueError(f"CSV is empty: {csv_path}")

        if isinstance(temp_col, str):
            header = first_row
            if temp_col not in header:
                raise ValueError(f"Column '{temp_col}' not found in header: {header}")
            col_idx = header.index(temp_col)
        else:
            col_idx = temp_col
            if not skip_header_for_index_col:
                temps.append(float(first_row[col_idx]))

        for row in reader:
            if not row:
                continue
            try:
                temps.append(float(row[col_idx]))
            except Exception:
                continue

    if not temps:
        raise ValueError(f"No temperature values parsed from {csv_path} (col={temp_col!r}).")
    return temps


class RxNodeWithJitterLogged(Node):
    """
    Receiver that adds non-negative Gaussian (normal) jitter (ps) and logs per photon
    """
    def __init__(self, name, timeline, rng, jitter_std_ps: float):
        super().__init__(name, timeline)
        self.rng = rng
        self.jitter_std_ps = float(jitter_std_ps)
        self.rx_ps: List[int] = []
        self.done_ps: List[int] = []

    def receive_qubit(self, src: str, qubit):
        t_rx_ps = int(self.timeline.now())
        extra = self.rng.normal(loc=0.0, scale=self.jitter_std_ps)
        extra_ps = int(max(0.0, extra))
        t_done_ps = t_rx_ps + extra_ps

        self.rx_ps.append(t_rx_ps)
        self.done_ps.append(t_done_ps)


def make_timeline_with_stop_time(stop_time_ps: int) -> Timeline:
    """
    Sequence forks differ so try different runtimes?
    """
    for kw in ("stop_time", "end_time", "runtime", "time_limit"):
        try:
            return Timeline(**{kw: stop_time_ps})
        except TypeError:
            pass
    try:
        return Timeline(stop_time_ps)
    except TypeError:
        pass

    tl = Timeline()
    for attr in ("stop_time", "end_time", "runtime", "time_limit"):
        if hasattr(tl, attr):
            setattr(tl, attr, stop_time_ps)
            break
    return tl


# main test
def main():
    temps = load_temps_from_csv(CSV_PATH, TEMP_COL, skip_header_for_index_col=SKIP_HEADER_FOR_INDEX_COL)

    # One whole day of data using 86400 rows:
    # temps = temps[:86400]

    update_period_ps = int(UPDATE_PERIOD_S / 1e-12)
    send_offset_ps = int(SEND_OFFSET_S / 1e-12)

    light_speed_m_per_ps = 0.0002
    max_flight_ps = int((DISTANCE_M / light_speed_m_per_ps) * 1.1)  # ~600e6 ps for 120 km (change to 64 km tyo match WR data?)
    margin_ps = int(2e9)  # this is a 2 ms margin

    last_send_ps = (len(temps) - 1) * update_period_ps + send_offset_ps
    stop_time_ps = last_send_ps + max_flight_ps + margin_ps

    tl = make_timeline_with_stop_time(stop_time_ps)

    rng = np.random.default_rng(SEED)
    alice = Node("alice", tl)
    bob = RxNodeWithJitterLogged("bob", tl, rng=rng, jitter_std_ps=JITTER_STD_PS)

    ch = ThermalQuantumChannel(
        name="thermal_qc",
        timeline=tl,
        base_distance_m=DISTANCE_M,
        alpha_per_C=ALPHA_PER_C,
        T0_C=T0_C,
        attenuation=0.0,
        polarization_fidelity=1.0,
        light_speed=light_speed_m_per_ps,
    )
    ch.set_ends(alice, bob.name)

    driver = TempDriver(timeline=tl, channel=ch, temps_C=temps, update_period_ps=update_period_ps)
    driver.start(t0_ps=0)

    # Schedule photons at 1 Hz (same rate as temp updates)
    send_times_ps: List[int] = []
    for k in range(len(temps)):
        t_send_ps = k * update_period_ps + send_offset_ps
        send_times_ps.append(t_send_ps)
        photon = Photon(f"p{k}", tl)
        send_proc = Process(alice, "send_qubit", [bob.name, photon], {})
        tl.schedule(Event(t_send_ps, send_proc))

    print(f"Loaded {len(temps)} temps from {CSV_PATH} col={TEMP_COL!r}")
    print(f"update_period = {update_period_ps:,} ps ({UPDATE_PERIOD_S:.3e} s)")
    print(f"jitter_std    = {JITTER_STD_PS} ps")
    print(f"timeline stop = {stop_time_ps:,} ps (~{stop_time_ps*1e-12/3600:.3f} hours)\n")

    tl.init()
    tl.run()

    arrivals_ps = bob.done_ps
    n = min(len(arrivals_ps), len(send_times_ps))
    arrivals_ps = arrivals_ps[:n]
    send_times_ps = send_times_ps[:n]

    if n < 2:
        raise RuntimeError(f"Only {n} photon(s) recorded; can't form a time series.")

    # TIME SYNC
    # propagation delay per photon:
    delays_ps = np.array([t_arr - t_send for t_arr, t_send in zip(arrivals_ps, send_times_ps)], dtype=np.int64)

    # normalize
    y_ps = delays_ps - delays_ps[0]

    # x-axis as elapsed time in hours (1 sample per second)
    x_hours = (np.arange(n) * UPDATE_PERIOD_S) / 3600.0

    # downsample for plotting (change to config value later. set default to none? Shouldnt need to downsample for current test data)
    x_hours = x_hours[::PLOT_EVERY_N]
    y_ps = y_ps[::PLOT_EVERY_N]

    print("First 10 propagation delays (ps):", delays_ps[:10].tolist())
    print("First 10 normalized delays Δd (ps):", (delays_ps[:10] - delays_ps[0]).tolist())

    plt.figure(figsize=(14, 5))
    plt.plot(x_hours, y_ps)
    plt.xlabel("Elapsed time [hours]")
    plt.ylabel("Propagation delay variation Δd [ps]")
    plt.title("Time synchronization error proxy: Δ(one-way propagation delay) vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
