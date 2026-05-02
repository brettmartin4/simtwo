import numpy as np
from sequence.kernel.timeline import Timeline
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from simtwo.core import ThermalQuantumChannel


class TempDriver:
    """
    Schedules regular temperature updates on the Timeline.
    # TODO: Update this later to get vals from csvs from time sync research
    """
    def __init__(self, timeline: Timeline, channel: ThermalQuantumChannel,
                 temps_C, update_period_ps: int):
        self.timeline = timeline
        self.channel = channel
        # TODO: Update to numpy typecasting here
        self.temps_C = list(temps_C)
        self.update_period_ps = int(update_period_ps)
        self.i = 0

    def start(self, t0_ps=0):
        self._schedule_update(t0_ps)

    def _schedule_update(self, t_ps):
        proc = Process(self, "update", [], {})
        self.timeline.schedule(Event(t_ps, proc))

    def update(self):
        if self.i >= len(self.temps_C):
            return  # this just stops updating when list ends

        T = self.temps_C[self.i]
        self.channel.set_temperature(T)

        now = self.timeline.now()
        print(f"[TempDriver] t={now:,} ps, T={T:.2f} C, channel distance={self.channel.distance:.3f} m")

        self.i += 1
        self._schedule_update(now + self.update_period_ps)