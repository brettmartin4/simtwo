"""
Two-node entanglement distribution demo using simtwo dynamic thermal channels.

This mirrors SeQUeNCe's two_node_eg.ipynb as closely as possible, but replaces
SeQUeNCe's static ClassicalChannel/QuantumChannel objects with simtwo's
ThermalClassicalChannel/ThermalQuantumChannel. A physical temperature-to-distance
model updates the channel distances during the simulation.

Run from the root of your simtwo project, for example:
    python examples/simtwo_two_node_eg_dynamic.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from matplotlib import pyplot as plt

from sequence.constants import MILLISECOND
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.resource_management.rule_manager import Rule
from sequence.topology.node import BSMNode, QuantumRouter

from simtwo.core.ThermalClassicalChannel import ThermalClassicalChannel
from simtwo.core.ThermalQuantumChannel import ThermalQuantumChannel
from simtwo.core.sequence.link_model_manager import LinkModelManager


# -----------------------------------------------------------------------------
# Barrett-Kok entanglement-generation rule logic from the original notebook
# -----------------------------------------------------------------------------

def eg_rule_condition(memory_info, manager, args):
    """Use memories that are still RAW/unentangled."""
    if memory_info.state == "RAW":
        return [memory_info]
    return []


def eg_rule_action1(memories_info, args):
    """Rule action for router r1."""

    def eg_req_func(protocols, args):
        for protocol in protocols:
            if isinstance(protocol, EntanglementGenerationA):
                return protocol
        return None

    memory = memories_info[0].memory

    # In newer SeQUeNCe versions EntanglementGenerationA is abstract, so use the
    # factory method. This returns the concrete Barrett-Kok protocol.
    protocol = EntanglementGenerationA.create(
        None,
        "EGA." + memory.name,
        "m1",   # BSM node
        "r2",   # remote router
        memory,
    )
    protocol.primary = True

    return [protocol, ["r2"], [eg_req_func], [None]]


def eg_rule_action2(memories_info, args):
    """Rule action for router r2."""
    memory = memories_info[0].memory

    protocol = EntanglementGenerationA.create(
        None,
        "EGA." + memory.name,
        "m1",   # BSM node
        "r1",   # remote router
        memory,
    )

    return [protocol, [None], [None], [None]]


# -----------------------------------------------------------------------------
# Dynamic physical-model timeline driver
# -----------------------------------------------------------------------------

@dataclass
class DynamicLinkUpdater:
    """
    Timeline-scheduled driver that updates registered simtwo channels.

    Each update row needs a temperature key accepted by LinkModelManager:
        temperature_x, temperature, temp_C, or temp

    LinkModelManager converts temperature into full-path distance using:
        L(T) = L0 * (1 + alpha * (T - T0))

    Then it applies distance fractions to the registered groups:
        r1 <-> m1 arm: 0.5 of full path
        r2 <-> m1 arm: 0.5 of full path
        r1 <-> r2 classical: 1.0 of full path
    """

    timeline: Timeline
    link_model_manager: LinkModelManager
    rows: list[dict[str, Any]]
    update_period_ps: int
    verbose: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)
    i: int = 0

    def start(self, t0_ps: int = 0) -> None:
        self._schedule_update(int(t0_ps))

    def _schedule_update(self, t_ps: int) -> None:
        proc = Process(self, "update", [], {})
        self.timeline.schedule(Event(int(t_ps), proc))

    def update(self) -> None:
        if self.i >= len(self.rows):
            return

        row = self.rows[self.i]
        state = self.link_model_manager.apply_to_registered_links(row)
        state["timeline_ps"] = self.timeline.now()
        state["update_index"] = self.i
        self.log.append(state)

        if self.verbose:
            print(
                f"[DynamicLinkUpdater] t={self.timeline.now():,} ps, "
                f"T={state['input_temperature']:.3f} C, "
                f"full_dist={state['predicted_distance_m']:.6f} m, "
                f"full_delay={state['predicted_path_delay_ns']:.6f} ns"
            )

        self.i += 1
        self._schedule_update(self.timeline.now() + self.update_period_ps)


def make_temperature_rows(
    *,
    sim_time_ps: int,
    update_period_ps: int,
    t0_c: float = 20.0,
    amplitude_c: float = 5.0,
) -> list[dict[str, float]]:
    """
    Synthetic temperature profile for the demo.

    Real fiber temperature changes would normally be much slower than a 1-2 s
    entanglement demo. This sinusoid is only here to prove that the dynamic
    link update mechanism works. Replace this with rows from your weather data
    later if desired.
    """
    n = max(2, math.ceil(sim_time_ps / update_period_ps) + 1)
    rows: list[dict[str, float]] = []

    for i in range(n):
        phase = 2.0 * math.pi * i / max(1, n - 1)
        rows.append({"temperature": t0_c + amplitude_c * math.sin(phase)})

    return rows


def _force_light_speed(channel: Any, light_speed_m_per_ps: float) -> None:
    """
    simtwo's ThermalClassicalChannel accepts light_speed but currently does not
    pass it to the SeQUeNCe ClassicalChannel constructor. Set it explicitly so
    quantum/classical channels use the same propagation speed.
    """
    if hasattr(channel, "light_speed"):
        channel.light_speed = float(light_speed_m_per_ps)
    if hasattr(channel, "_refresh_channel_params"):
        channel._refresh_channel_params()


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def test_simtwo_dynamic(
    sim_time: float = 1000,
    qc_atten: float = 1e-4,
    qc_dist: float = 1.0,
    alpha_per_c: float = 5e-7,
    t0_c: float = 20.0,
    temp_amplitude_c: float = 5.0,
    update_period_ms: float = 10.0,
    light_speed_m_per_ps: float = 0.0002,
    verbose_updates: bool = False,
):
    """
    sim_time: simulation duration in ms
    qc_atten: attenuation on quantum channels in dB/m
    qc_dist: original notebook meaning: router-to-BSM arm distance in km
    alpha_per_c: thermal expansion coefficient, 1/C
    t0_c: reference temperature in C
    temp_amplitude_c: amplitude of synthetic temperature variation in C
    update_period_ms: how often the physical model updates channel distances
    light_speed_m_per_ps: approx. speed of light in fiber, m/ps
    """

    PS_PER_MS = 1e9
    M_PER_KM = 1e3

    sim_time_ps = int(sim_time * PS_PER_MS)
    update_period_ps = int(update_period_ms * PS_PER_MS)

    # The original notebook's qc_dist is one router-to-BSM arm. The full
    # r1-to-r2 physical path is therefore two arms.
    base_arm_distance_m = float(qc_dist * M_PER_KM)
    base_full_path_distance_m = 2.0 * base_arm_distance_m

    # -------------------------------------------------------------------------
    # 1. Create timeline and nodes, same as original notebook
    # -------------------------------------------------------------------------
    tl = Timeline(sim_time_ps)

    r1 = QuantumRouter("r1", tl, 50)
    r2 = QuantumRouter("r2", tl, 50)
    m1 = BSMNode("m1", tl, ["r1", "r2"])

    r1.set_seed(0)
    r2.set_seed(1)
    m1.set_seed(2)

    nodes = [r1, r2, m1]

    # -------------------------------------------------------------------------
    # 2. Create simtwo thermal classical channels instead of static channels
    # -------------------------------------------------------------------------
    classical_channels: dict[tuple[str, str], ThermalClassicalChannel] = {}

    for node1 in nodes:
        for node2 in nodes:
            if node1 == node2:
                continue

            # Physical topology: r1<->r2 is the full path; router<->m1 is an arm.
            if {node1.name, node2.name} == {"r1", "r2"}:
                base_distance_m = base_full_path_distance_m
            else:
                base_distance_m = base_arm_distance_m

            cc = ThermalClassicalChannel(
                name="_".join(["cc", node1.name, node2.name]),
                timeline=tl,
                base_distance_m=base_distance_m,
                alpha_per_C=alpha_per_c,
                T0_C=t0_c,
                light_speed=light_speed_m_per_ps,
            )
            _force_light_speed(cc, light_speed_m_per_ps)
            cc.set_ends(node1, node2.name)
            classical_channels[(node1.name, node2.name)] = cc

    # -------------------------------------------------------------------------
    # 3. Create simtwo thermal quantum channels instead of static channels
    # -------------------------------------------------------------------------
    qc1 = ThermalQuantumChannel(
        name="qc_r1_m1",
        timeline=tl,
        base_distance_m=base_arm_distance_m,
        alpha_per_C=alpha_per_c,
        T0_C=t0_c,
        attenuation=qc_atten,
        light_speed=light_speed_m_per_ps,
    )
    qc1.set_ends(r1, m1.name)

    qc2 = ThermalQuantumChannel(
        name="qc_r2_m1",
        timeline=tl,
        base_distance_m=base_arm_distance_m,
        alpha_per_C=alpha_per_c,
        T0_C=t0_c,
        attenuation=qc_atten,
        light_speed=light_speed_m_per_ps,
    )
    qc2.set_ends(r2, m1.name)

    # -------------------------------------------------------------------------
    # 4. Register physical-model link groups with simtwo's LinkModelManager
    # -------------------------------------------------------------------------
    link_model_manager = LinkModelManager(
        base_distance_m=base_full_path_distance_m,
        alpha_per_c=alpha_per_c,
        t0_c=t0_c,
        light_speed_m_per_ps=light_speed_m_per_ps,
    )

    # Quantum arms: each arm is half of the full r1-r2 path.
    link_model_manager.register_group(
        "r1_to_m1_quantum_arm",
        [qc1],
        delay_fraction=0.5,
        distance_fraction=0.5,
    )
    link_model_manager.register_group(
        "r2_to_m1_quantum_arm",
        [qc2],
        delay_fraction=0.5,
        distance_fraction=0.5,
    )

    # Classical channels: pairwise physical distances in the same topology.
    full_classical = [
        ch for (src, dst), ch in classical_channels.items()
        if {src, dst} == {"r1", "r2"}
    ]
    arm_classical = [
        ch for (src, dst), ch in classical_channels.items()
        if {src, dst} != {"r1", "r2"}
    ]

    link_model_manager.register_group(
        "r1_r2_classical_full_path",
        full_classical,
        delay_fraction=1.0,
        distance_fraction=1.0,
    )
    link_model_manager.register_group(
        "router_bsm_classical_arms",
        arm_classical,
        delay_fraction=0.5,
        distance_fraction=0.5,
    )

    # Apply initial physical state before tl.init(), then schedule timeline updates.
    initial_row = {"temperature": t0_c}
    link_model_manager.apply_to_registered_links(initial_row)

    temperature_rows = make_temperature_rows(
        sim_time_ps=sim_time_ps,
        update_period_ps=update_period_ps,
        t0_c=t0_c,
        amplitude_c=temp_amplitude_c,
    )

    dynamic_driver = DynamicLinkUpdater(
        timeline=tl,
        link_model_manager=link_model_manager,
        rows=temperature_rows,
        update_period_ps=update_period_ps,
        verbose=verbose_updates,
    )
    dynamic_driver.start(t0_ps=0)

    # -------------------------------------------------------------------------
    # 5. Initialize, load rules, and run, same as original notebook
    # -------------------------------------------------------------------------
    tl.init()

    rule1 = Rule(10, eg_rule_action1, eg_rule_condition, None, None)
    r1.resource_manager.load(rule1)

    rule2 = Rule(10, eg_rule_action2, eg_rule_condition, None, None)
    r2.resource_manager.load(rule2)

    tick = time.time()
    tl.run()
    print("execution time %.2f sec" % (time.time() - tick))

    # -------------------------------------------------------------------------
    # 6. Collect and plot entanglement completion times, same as original
    # -------------------------------------------------------------------------
    entangle_times_ms: list[float] = []

    for info in r1.resource_manager.memory_manager:
        if info.entangle_time > 0:
            entangle_times_ms.append(info.entangle_time / MILLISECOND)

    entangle_times_ms.sort()

    print(f"Number of entangled memories: {len(entangle_times_ms)}")
    print(f"Entanglement times in ms: {entangle_times_ms}")

    plt.figure()
    plt.plot(entangle_times_ms, range(1, len(entangle_times_ms) + 1), marker="o")
    plt.xlabel("Simulation Time (ms)")
    plt.ylabel("Aggregated Number of Entangled Memories")
    plt.title("Two-Node Entanglement Distribution with simtwo Dynamic Links")
    plt.grid(True)
    plt.show()

    # Optional diagnostic plot: full physical path distance over update time.
    if dynamic_driver.log:
        update_times_ms = [row["timeline_ps"] / MILLISECOND for row in dynamic_driver.log]
        full_distances_m = [row["predicted_distance_m"] for row in dynamic_driver.log]

        plt.figure()
        plt.plot(update_times_ms, full_distances_m, marker=".")
        plt.xlabel("Simulation Time (ms)")
        plt.ylabel("Predicted Full r1-r2 Distance (m)")
        plt.title("Physical-Model Distance Updates")
        plt.grid(True)
        plt.show()

    return {
        "entangle_times_ms": entangle_times_ms,
        "dynamic_link_log": dynamic_driver.log,
        "r1": r1,
        "r2": r2,
        "m1": m1,
        "quantum_channels": {"qc_r1_m1": qc1, "qc_r2_m1": qc2},
        "classical_channels": classical_channels,
    }


if __name__ == "__main__":
    test_simtwo_dynamic(
        sim_time=1000,
        qc_atten=1e-4,
        qc_dist=1.0,
        alpha_per_c=5e-7,
        t0_c=20.0,
        temp_amplitude_c=5.0,
        update_period_ms=10.0,
        verbose_updates=False,
    )
