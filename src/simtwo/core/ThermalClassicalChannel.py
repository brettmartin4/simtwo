from sequence.components.optical_channel import ClassicalChannel


class ThermalClassicalChannel(ClassicalChannel):
    """ClassicalChannel with externally controlled effective distance/delay/loss."""

    def __init__(self, name, timeline, base_distance_m, alpha_per_C=5e-7, T0_C=20.0, light_speed=None, **kwargs):
        """Initialize the ThermalClassicalChannel instance."""
        # Do NOT forward light_speed into ClassicalChannel.__init__()
        super().__init__(name=name, timeline=timeline, distance=base_distance_m)
        
        self.base_distance_m = float(base_distance_m)
        self.alpha_per_C = float(alpha_per_C)
        self.T0_C = float(T0_C)
        self.current_T_C = float(T0_C)

    def init(self) -> None:
        """Called by sequence Timeline during tl.init()--ensures the channel starts with delay corresponding to current_T_C."""
        self.distance = self._expanded_distance(self.current_T_C)
        self._refresh_channel_params()

    def _expanded_distance(self, T_C: float) -> float:
        """L(T) = L0 * (1 + alpha*(T - T0)) (TODO: Include photon ingress/egress offset)"""
        return self.base_distance_m * (1.0 + self.alpha_per_C * (float(T_C) - self.T0_C))

    def _refresh_channel_params(self):
        """Recompute cache sequence channel params that depend on distance; QuantumChannel.transmit() uses self.delay directly, not self.distance"""
        self.delay = round(self.distance / self.light_speed)
        self.loss = 1 - 10 ** (self.distance * self.attenuation / -10)

    def set_temperature(self, T_C: float):
        """Update fiber len and cache prop delay for future sends (updated to actually impact delay and loss (was not doing this previously))"""
        self.current_T_C = float(T_C)
        self.distance = self._expanded_distance(self.current_T_C)
        self._refresh_channel_params()

    def set_effective_distance(self, distance_m: float):
        """Set effective distance."""
        self.distance = float(distance_m)
        self._refresh_channel_params()