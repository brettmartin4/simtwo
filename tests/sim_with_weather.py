import numpy as np
from sequence.kernel.timeline import Timeline
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.topology.node import Node
from sequence.components.photon import Photon
from simtwo.core.TempDriver import TempDriver
from simtwo.core.ThermalQuantumChannel import ThermalQuantumChannel


class RxNode(Node):
    def __init__(self, name, timeline):
        super().__init__(name, timeline)
        self.arrivals = []

    def receive_qubit(self, src: str, qubit):
        t = self.timeline.now()
        self.arrivals.append(t)
        print(f"[{self.name}] received photon from {src} at t={t:,} ps")


def main():
    tl = Timeline()
    alice = Node("alice", tl)
    bob = RxNode("bob", tl)

    # just a quick 5 value sequence to test
    temps = [20.0, 21.0, 19.5, 23.0, 18.0]

    # every 0.5 ms (measured in ps)
    # TODO: Check back on this later--behavior is too different from actual results from paper
    update_period_ps = int(0.5e-3 / 1e-12)

    # Custom channel with thermal expansion
    ch = ThermalQuantumChannel(
        name="thermal_qc",
        timeline=tl,
        base_distance_m=120_000,     # 120 km (loopback dist based on nodeA->nodeB)
        alpha_per_C=5e-7,            # silica order-of-mag (see citations in paper)
        T0_C=20.0,
        attenuation=0.0,             # keep lossless for demonstration
        polarization_fidelity=1.0,
        light_speed=0.0002           # m/ps ( about 2e8 m/s)
    )
    ch.set_ends(alice, bob.name)

    # Start temperature driver
    driver = TempDriver(timeline=tl, channel=ch, temps_C=temps, update_period_ps=update_period_ps)
    driver.start(t0_ps=0)

    # Send one photon shortly after each temperature update
    # (so each send sees a different distance)
    for k in range(len(temps)):
        t_send = k * update_period_ps + int(0.05e-3 / 1e-12)  # 0.05 ms after update
        photon = Photon(f"p{k}", tl)
        send_proc = Process(alice, "send_qubit", [bob.name, photon], {})
        tl.schedule(Event(t_send, send_proc))

    tl.init()
    tl.run()

    print("\nArrival times (ps):", bob.arrivals)


if __name__ == "__main__":
    main()
