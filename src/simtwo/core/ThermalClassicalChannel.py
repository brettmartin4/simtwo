from sequence.components.optical_channel import ClassicalChannel


class ThermalClassicalChannel(ClassicalChannel):
    """
    ClassicalChannel with externally controlled effective distance/delay/loss.
    I'm fairly certain not having this was what was giving me grief with the BK protocol in the ED demo.
    """

    def __init__(self, name, timeline, base_distance_m, alpha_per_C=5e-7, T0_C=20.0, **kwargs):
        super().__init__(name=name, timeline=timeline, distance=base_distance_m, **kwargs)
        self.base_distance_m = float(base_distance_m)
        self.alpha_per_C = float(alpha_per_C)
        self.T0_C = float(T0_C)
        self.current_T_C = float(T0_C)

    def init(self) -> None:
        self.distance = self._expanded_distance(self.current_T_C)
        self._refresh_channel_params()

    def _expanded_distance(self, T_C: float) -> float:
        return self.base_distance_m * (1.0 + self.alpha_per_C * (float(T_C) - self.T0_C))

    def _refresh_channel_params(self):
        self.delay = round(self.distance / self.light_speed)

    def set_temperature(self, T_C: float):
        self.current_T_C = float(T_C)
        self.distance = self._expanded_distance(self.current_T_C)
        self._refresh_channel_params()

    def set_effective_distance(self, distance_m: float):
        self.distance = float(distance_m)
        self._refresh_channel_params()