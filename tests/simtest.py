
from sequence.kernel.timeline import Timeline
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.topology.node import Node
from sequence.components.optical_channel import QuantumChannel
from sequence.components.photon import Photon


class RxNode(Node):
    def __init__(self, name, timeline):
        super().__init__(name, timeline)
        self.t_rx_ps = None

    # signature per docs: receive_qubit(src: str, qubit) :contentReference[oaicite:2]{index=2}
    def receive_qubit(self, src: str, qubit):
        self.t_rx_ps = self.timeline.now()
        print(f"[{self.name}] received photon from {src} at t = {self.t_rx_ps} ps")
        self.timeline.stop()  # end sim once we got it


def main():
    tl = Timeline()

    alice = Node("alice", tl)
    bob   = RxNode("bob", tl)

    # QuantumChannel(distance in meters; light_speed in m/ps; delay derived from those) :contentReference[oaicite:3]{index=3}
    # NOTE: attenuation is in dB/m; 120 km with realistic attenuation would almost always "lose" a single photon.
    # For a pure time-of-flight test, set (attenuation=0, polarization_fidelity=1.0) to avoid loss/errors. :contentReference[oaicite:4]{index=4}
    # TODO: change distances to 64 km to match topology from past polarization/time sync study?
    qc = QuantumChannel(
        name="qc_alice_bob",
        timeline=tl,
        attenuation=0.0,           # lossless for this simple demo
        distance=120_000,          # 120 km
        polarization_fidelity=1.0, # no polarization error
        light_speed=0.0002         # default: 2e8 m/s expressed as m/ps :contentReference[oaicite:5]{index=5}
    )
    qc.set_ends(alice, bob.name)   # attaches channel to node endpoints :contentReference[oaicite:6]{index=6}

    photon = Photon("p0", tl)      # simple photon object :contentReference[oaicite:7]{index=7}

    t_tx_ps = 0
    send_proc = Process(alice, "send_qubit", [bob.name, photon], {})
    tl.schedule(Event(t_tx_ps, send_proc))

    tl.init()
    tl.run()

    if bob.t_rx_ps is None:
        print("Photon was never received (likely lost).")
    else:
        flight_ps = bob.t_rx_ps - t_tx_ps
        flight_s  = flight_ps * 1e-12
        print(f"\nOne-way flight time: {flight_ps:,} ps  =  {flight_s:.6f} s ({flight_s*1e3:.3f} ms)")


if __name__ == "__main__":
    main()
