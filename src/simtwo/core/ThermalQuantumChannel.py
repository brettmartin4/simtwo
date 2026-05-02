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

    def set_temperature(self, T_C: float):
        self.current_T_C = float(T_C)
        # L(T) = L0 * (1 + alpha*(T - T0))
        expanded = self.base_distance_m * (1.0 + self.alpha_per_C * (self.current_T_C - self.T0_C))
        # QuantumChannel uses self.distance for propagation delay
        self.distance = expanded
        # TODO: also set delay here. reference sequence docs for exactly how this works in a simulation like BSD