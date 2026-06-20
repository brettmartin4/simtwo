import numpy as np
from sequence.kernel.timeline import Timeline
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from simtwo.core import ThermalQuantumChannel


class TempDriver:
    """
    Schedules regular temperature updates on the Timeline.
    """
    def __init__(self, timeline: Timeline, channel: ThermalQuantumChannel, temps_C, update_period_ps: int):
        """Create a driver for applying a sequence of channel temperatures.
        
        Args:
            timeline (Timeline): SeQUeNCe timeline on which update events are scheduled.
            channel (ThermalQuantumChannel): Thermal channel whose temperature will be updated.
            temps_C: Iterable of temperatures in degrees celsius.
            update_period_ps (int): Time between updates in picoseconds.
        """
        self.timeline = timeline
        self.channel = channel
        # TODO: Update to numpy typecasting eventually
        self.temps_C = list(temps_C)
        self.update_period_ps = int(update_period_ps)
        self.i = 0

    def start(self, t0_ps=0):
        """Schedule the first temperature update.
        
        Args:
            t0_ps: Timeline time, in picoseconds, for the first update event.
        """
        self._schedule_update(t0_ps)

    def _schedule_update(self, t_ps):
        """Schedule the next call to "update" on the timeline.
        
        Args:
            t_ps: Timeline time, in ps, at which the update should run.
        """
        proc = Process(self, "update", [], {})
        self.timeline.schedule(Event(t_ps, proc))

    def update(self):
        """Apply the next temperature value and schedule the following update.
        
        The method stops scheduling new events when all provided temperatures have
        been consumed.
        """
        if self.i >= len(self.temps_C):
            return  # this just stops updating when list ends

        T = self.temps_C[self.i]
        self.channel.set_temperature(T)

        now = self.timeline.now()
        print(f"[TempDriver] t={now:,} ps, T={T:.2f} C, channel distance={self.channel.distance:.3f} m")

        self.i += 1
        self._schedule_update(now + self.update_period_ps)