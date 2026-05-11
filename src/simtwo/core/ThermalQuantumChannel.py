import numpy as np
from sequence.components.optical_channel import QuantumChannel

class ThermalQuantumChannel(QuantumChannel):
    """
    QuantumChannel with temperature-dependent effective fiber length via thermal expansion.
    """
    def __init__(self, name, timeline, base_distance_m, alpha_per_C=5e-7, T0_C=20.0, **kwargs):
        super().__init__(name=name, timeline=timeline, distance=base_distance_m, **kwargs)
        # TODO: Update these to use numpy typecasting
        self.base_distance_m = float(base_distance_m)
        self.alpha_per_C = float(alpha_per_C)
        self.T0_C = float(T0_C)
        self.current_T_C = float(T0_C)

    def init(self) -> None:
        """
        Called by the SeQUeNCe Timeline during tl.init().
        This ensures the channel starts with the delay corresponding to current_T_C.
        """
        self.distance = self._expanded_distance(self.current_T_C)
        self._refresh_channel_params()

    def _expanded_distance(self, T_C: float) -> float:
        """
        # L(T) = L0 * (1 + alpha*(T - T0))
        """
        return self.base_distance_m * (
            1.0 + self.alpha_per_C * (float(T_C) - self.T0_C)
        )

    def _refresh_channel_params(self):
        """
        Recompute cached SeQUeNCe channel parameters that depend on distance.
        QuantumChannel.transmit() apparently uses self.delay directly, not self.distance.
        """
        self.delay = round(self.distance / self.light_speed)

        # Added for consistency
        self.loss = 1 - 10 ** (self.distance * self.attenuation / -10)

    def set_temperature(self, T_C: float):
        """
        Update fiber distance and cached propagation delay for future transmissions.
        Updated to actually impact delay and loss (was not doing this previously)
        """
        self.current_T_C = float(T_C)

        self.distance = self._expanded_distance(self.current_T_C)

        # Important:
        self._refresh_channel_params()