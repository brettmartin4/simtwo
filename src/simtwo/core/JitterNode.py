import numpy as np
from sequence.topology.node import Node


class RxNodeWithJitter(Node):
    def __init__(self, name, timeline, rng, jitter_std_ps=50_000):  # 50,000 ps = 50 ns (TODO: verify correct jitter range later)
        super().__init__(name, timeline)
        self.rng = rng
        self.jitter_std_ps = float(jitter_std_ps)

        self.t_rx_ps = None # raw arrival time
        self.t_done_ps = None # arrival + jitter

    def receive_qubit(self, src: str, qubit):
        # This time is just "when the photon arrived at the node"
        # Can convert to time sync error later
        self.t_rx_ps = self.timeline.now()

        # Add non negative jitter to emulate timetag behavior
        extra = self.rng.normal(loc=0.0, scale=self.jitter_std_ps)
        extra_ps = int(max(0.0, extra))

        self.t_done_ps = self.t_rx_ps + extra_ps

        print(f"[{self.name}] arrived at {self.t_rx_ps:,} ps; + jitter {extra_ps:,} ps -> reported {self.t_done_ps:,} ps")
        #self.timeline.stop()
        