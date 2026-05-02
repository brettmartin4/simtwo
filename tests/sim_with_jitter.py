import numpy as np

from sequence.kernel.timeline import Timeline
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.topology.node import Node
from sequence.components.optical_channel import QuantumChannel
from sequence.components.photon import Photon
from simtwo.core.JitterNode import RxNodeWithJitter


def run_one_trial(seed: int, distance_m=120_000, jitter_std_ps=50_000):
    rng = np.random.default_rng(seed)

    tl = Timeline()
    alice = Node("alice", tl)
    bob = RxNodeWithJitter("bob", tl, rng=rng, jitter_std_ps=jitter_std_ps)

    qc = QuantumChannel(
        name="qc_alice_bob",
        timeline=tl,
        attenuation=0.0,# keep lossless so every trial arrives
        distance=distance_m,
        polarization_fidelity=1.0,
        light_speed=0.0002 # m/ps (about 2e8 m/s)
    )
    qc.set_ends(alice, bob.name)

    photon = Photon("p0", tl)

    t_tx_ps = 0
    send_proc = Process(alice, "send_qubit", [bob.name, photon], {})
    tl.schedule(Event(t_tx_ps, send_proc))

    tl.init()
    tl.run()

    # "Propagation-only" vs "propagation + jitter"
    flight_ps = bob.t_rx_ps - t_tx_ps
    reported_ps = bob.t_done_ps - t_tx_ps
    return flight_ps, reported_ps


if __name__ == "__main__":
    distance_m = 120_000

    # Pick jitter based on time sync research findings
    # 50 ns is small, 5 us is larger, 50 us is very large
    jitter_std_ps = 5_000_000  # 5 microseconds = about 5e6 ps

    print(f"Distance: {distance_m/1000:.1f} km")
    print(f"Jitter std dev: {jitter_std_ps*1e-12:.6e} s ({jitter_std_ps*1e-6:.3f} us)\n")

    for i in range(5):
        flight_ps, reported_ps = run_one_trial(seed=100 + i, distance_m=distance_m, jitter_std_ps=jitter_std_ps)

        flight_s = flight_ps * 1e-12
        reported_s = reported_ps * 1e-12

        print(f"Trial {i+1}: propagation = {flight_s*1e3:.3f} ms | reported = {reported_s*1e3:.3f} ms")
