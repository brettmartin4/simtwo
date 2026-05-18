from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sequence.kernel.timeline import Timeline
from sequence.topology.node import QuantumRouter, BSMNode
from sequence.components.optical_channel import QuantumChannel, ClassicalChannel

from simtwo.core.modeling.model import load_trained_model_bundle


@dataclass
class GuiTwoNodeEntanglementWithModeledDelay:
    """
    Two-node entanglement experiment for the current Simtwo GUI.

    The currently selected model in the modeling suite determines the
    quantum-channel delay for each attempt:
      - default model -> physical delay from temperature
      - existing model -> sklearn model prediction from temperature

    This lets you run the same experiment multiple times and compare:
      1. default physical delay model
      2. loaded sklearn delay model
    """

    attempts: int = 100

    # Physical/default model parameters
    base_distance_m: float = 64_000.0
    alpha_per_c: float = 5e-7
    t0_c: float = 19.995
    light_speed_m_per_ps: float = 0.0002
    jitter_std_ps: float = 2.0

    # Entanglement/request settings
    qchannel_attenuation: float = 0.0002
    target_fidelity: float = 0.9
    memo_size: int = 1
    request_start_time_ps: int = 1_000_000_000_000
    request_end_time_ps: int = 10_000_000_000_000

    # GUI-facing / backend-facing state
    observations: list[dict[str, Any]] = field(default_factory=list)
    current_epoch: int = 0
    nodes: dict[str, Any] = field(default_factory=lambda: {"A": "A", "B": "B"})

    _run_speed_ms: int = 100
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    # Model-selection state from GUI
    channel_model_config: Any = None
    channel_model_bundle: dict[str, Any] | None = None
    channel_model_name: str = "default_physical_model"

    results_rows: list[dict[str, Any]] = field(default_factory=list)
    auto_output_csv: str = "gui_two_node_entanglement_modeled_delay_results.csv"

    def __post_init__(self):
        if not self.observations:
            self.observations = [{"attempt": i + 1} for i in range(self.attempts)]

    # ------------------------------------------------------------------
    # GUI control hooks
    # ------------------------------------------------------------------

    def set_run_speed(self, value: int):
        self._run_speed_ms = int(value)

    def cleanup_after_ids(self):
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def reset_simulation(self):
        self.cleanup_after_ids()
        self.current_epoch = 0
        self.results_rows = []

    def load_file(self, file_path: str):
        loaded_rows = []
        with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                loaded_rows.append({"attempt": i + 1, "epoch": i, **row})

        if loaded_rows:
            self.observations = loaded_rows
            self.current_epoch = 0
            self.results_rows = []

    def load_data(self, file_path: str):
        self.load_file(file_path)

    def export_results(self, file_path: str):
        self._write_results_csv(file_path)

    def export_file(self, file_path: str):
        self._write_results_csv(file_path)

    # ------------------------------------------------------------------
    # Modeling-suite integration
    # ------------------------------------------------------------------

    def configure_channel_model(self, config):
        """
        Called by the GUI backend when the user selects Apply Default
        or Load Model in the modeling suite.
        """
        self.channel_model_config = config
        self.channel_model_bundle = None

        # Default physical model: no target required
        if config.mode == "default":
            self.channel_model_name = "default_physical_model"
            return

        # Existing sklearn model: enforce path-delay target
        if config.mode == "existing":
            if not config.model_path:
                raise RuntimeError("Choose a trained model file in the GUI before loading it.")

            bundle = load_trained_model_bundle(config.model_path)
            bundle_features = list(bundle.get("feature_names") or [])
            bundle_target = str(bundle.get("target_name") or "").strip()

            allowed_targets = {"path_delay", "path_delay_ns", "path_delay_ps", "path_delay_s"}

            if bundle_features not in (["temperature"], ["temperature_x"]):
                raise RuntimeError(
                    "Loaded model bundle must have feature_names == ['temperature'] "
                    "or ['temperature_x'] for this experiment."
                )

            if bundle_target not in allowed_targets:
                raise RuntimeError(
                    "Loaded model bundle target_name must be one of: "
                    "path_delay, path_delay_ns, path_delay_ps, path_delay_s."
                )

            self.channel_model_bundle = bundle
            self.channel_model_name = bundle.get("model_name", Path(config.model_path).stem)
            return

        raise RuntimeError(
            "This experiment only supports the default physical model "
            "or an existing trained sklearn model."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_results_csv(self, file_path: str):
        if not self.results_rows:
            return

        fieldnames = list(self.results_rows[0].keys())
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results_rows)

    def _get_temperature(self, row: dict[str, Any]) -> float:
        if "temperature_x" in row:
            return float(row["temperature_x"])
        if "temperature" in row:
            return float(row["temperature"])
        raise RuntimeError(
            "Loaded dataset must contain either 'temperature_x' or 'temperature'."
        )

    def _physical_delay_prediction(self, temperature_c: float) -> tuple[float, float, float, float]:
        """
        Returns:
            (path_delay_ps, path_delay_ns, path_delay_s, distance_m)
        """
        distance_m = self.base_distance_m * (
            1.0 + self.alpha_per_c * (temperature_c - self.t0_c)
        )
        path_delay_ps = distance_m / self.light_speed_m_per_ps
        path_delay_ns = path_delay_ps / 1000.0
        path_delay_s = path_delay_ps * 1e-12
        return path_delay_ps, path_delay_ns, path_delay_s, distance_m

    def _ml_delay_prediction(self, temperature_c: float) -> tuple[float, float, float]:
        if self.channel_model_bundle is None:
            raise RuntimeError("No trained model bundle is loaded.")

        estimator = self.channel_model_bundle["estimator"]
        feature_names = list(self.channel_model_bundle.get("feature_names") or [])
        target_name = str(self.channel_model_bundle.get("target_name") or "").strip()

        # Accept either feature name convention
        if feature_names == ["temperature"]:
            pred = float(estimator.predict([[temperature_c]])[0])
        elif feature_names == ["temperature_x"]:
            pred = float(estimator.predict([[temperature_c]])[0])
        else:
            raise RuntimeError("Unsupported loaded-model feature list for this experiment.")

        if target_name in {"path_delay", "path_delay_ns"}:
            path_delay_ns = pred
            path_delay_ps = path_delay_ns * 1000.0
            path_delay_s = path_delay_ns * 1e-9
            return path_delay_ps, path_delay_ns, path_delay_s

        if target_name == "path_delay_ps":
            path_delay_ps = pred
            path_delay_ns = path_delay_ps / 1000.0
            path_delay_s = path_delay_ps * 1e-12
            return path_delay_ps, path_delay_ns, path_delay_s

        if target_name == "path_delay_s":
            path_delay_s = pred
            path_delay_ns = path_delay_s * 1e9
            path_delay_ps = path_delay_s * 1e12
            return path_delay_ps, path_delay_ns, path_delay_s

        raise RuntimeError(f"Unsupported model target: {target_name}")

    def _predict_delay_from_active_model(self, row: dict[str, Any]) -> tuple[float, float, float, float | None]:
        temperature_c = self._get_temperature(row)

        if self.channel_model_config is None or self.channel_model_config.mode == "default":
            path_delay_ps, path_delay_ns, path_delay_s, distance_m = self._physical_delay_prediction(temperature_c)
            return path_delay_ps, path_delay_ns, path_delay_s, distance_m

        path_delay_ps, path_delay_ns, path_delay_s = self._ml_delay_prediction(temperature_c)
        return path_delay_ps, path_delay_ns, path_delay_s, None

    def _run_one_attempt(self, row: dict[str, Any]) -> tuple[bool, float, float, float, float | None]:
        """
        TODO: This experiment keeps breaking somewhere, but I'm not entirely sure at the moment. I think the BK protocol doesn't like
        what's going on with the changing distance. Will look into in next session.

        Run one entanglement attempt using the active GUI model to determine
        the channel delay.

        Returns:
            success,
            predicted_path_delay_ps,
            predicted_path_delay_ns,
            predicted_path_delay_s,
            predicted_distance_m
        """
        predicted_path_delay_ps, predicted_path_delay_ns, predicted_path_delay_s, predicted_distance_m = (
            self._predict_delay_from_active_model(row)
        )

        tl = Timeline()

        node_a = QuantumRouter("A", tl, memo_size=self.memo_size)
        node_b = QuantumRouter("B", tl, memo_size=self.memo_size)
        node_m = BSMNode("M", tl, [node_a.name, node_b.name])

        node_a.map_to_middle_node[node_b.name] = node_m.name
        node_b.map_to_middle_node[node_a.name] = node_m.name

        qc_a_m = QuantumChannel(
            "qc_A_M",
            tl,
            attenuation=self.qchannel_attenuation,
            distance=self.base_distance_m,
        )
        qc_b_m = QuantumChannel(
            "qc_B_M",
            tl,
            attenuation=self.qchannel_attenuation,
            distance=self.base_distance_m,
        )

        qc_a_m.set_ends(node_a, node_m.name)
        qc_b_m.set_ends(node_b, node_m.name)

        qc_a_m.delay = int(round(predicted_path_delay_ps))
        qc_b_m.delay = int(round(predicted_path_delay_ps))

        classical_pairs = [
            ("A", "B"),
            ("B", "A"),
            ("A", "M"),
            ("M", "A"),
            ("B", "M"),
            ("M", "B"),
        ]
        node_map = {"A": node_a, "B": node_b, "M": node_m}

        for src_name, dst_name in classical_pairs:
            cc = ClassicalChannel(
                f"cc_{src_name}_{dst_name}",
                tl,
                distance=self.base_distance_m,
            )
            cc.set_ends(node_map[src_name], dst_name)

        # ADD THIS
        routing_a = next(
            p for p in node_a.network_manager.protocol_stack
            if hasattr(p, "forwarding_table")
        )
        routing_b = next(
            p for p in node_b.network_manager.protocol_stack
            if hasattr(p, "forwarding_table")
        )

        routing_a.forwarding_table[node_b.name] = node_b.name
        routing_b.forwarding_table[node_a.name] = node_a.name

        tl.init()

        node_a.network_manager.request(
            node_b.name,
            start_time=self.request_start_time_ps,
            end_time=self.request_end_time_ps,
            memory_size=1,
            target_fidelity=self.target_fidelity,
        )

        tl.run()

        success = False
        for info in node_a.resource_manager.memory_manager:
            state = getattr(info, "state", None)
            remote_node = getattr(info, "remote_node", None)
            if state == "ENTANGLED" and remote_node == node_b.name:
                success = True
                break

        return (
            success,
            predicted_path_delay_ps,
            predicted_path_delay_ns,
            predicted_path_delay_s,
            predicted_distance_m,
        )

    # ------------------------------------------------------------------
    # Main simulation entry
    # ------------------------------------------------------------------

    def run_sim(
        self,
        sender: Any,
        receiver: Any,
        update_plot,
        update_conditions=None,
        update_poincare_sphere=None,
    ):
        self._stop_evt.clear()

        def worker():
            success_count = 0

            if self.current_epoch > 0 and self.results_rows:
                success_count = int(self.results_rows[-1]["cumulative_successes"])

            for epoch in range(self.current_epoch, len(self.observations)):
                if self._stop_evt.is_set():
                    break

                self.current_epoch = epoch
                row = dict(self.observations[epoch])

                (
                    attempt_success,
                    predicted_path_delay_ps,
                    predicted_path_delay_ns,
                    predicted_path_delay_s,
                    predicted_distance_m,
                ) = self._run_one_attempt(row)

                if attempt_success:
                    success_count += 1

                attempts_completed = epoch + 1
                success_pct = 100.0 * success_count / attempts_completed

                result_row = {
                    "attempt": attempts_completed,
                    "success": int(attempt_success),
                    "cumulative_successes": success_count,
                    "success_rate_pct": success_pct,
                    "current_model": self.channel_model_name,
                    "input_temperature": self._get_temperature(row),
                    "predicted_path_delay_ps": predicted_path_delay_ps,
                    "predicted_path_delay_ns": predicted_path_delay_ns,
                    "predicted_path_delay_s": predicted_path_delay_s,
                    "predicted_distance_m": predicted_distance_m,
                }
                self.results_rows.append(result_row)

                if update_conditions is not None:
                    update_conditions(
                        {
                            "attempt": attempts_completed,
                            "successful_entanglements": success_count,
                            "success_rate_pct": success_pct,
                            "last_attempt_success": int(attempt_success),
                            "current_model": self.channel_model_name,
                            "predicted_path_delay_ns": predicted_path_delay_ns,
                            "predicted_path_delay_s": predicted_path_delay_s,
                            "input_temperature": self._get_temperature(row),
                        }
                    )

                # Plot cumulative success percentage
                update_plot(epoch, predicted_path_delay_ns)

                time.sleep(max(0.0, self._run_speed_ms / 1000.0))

            if not self._stop_evt.is_set():
                self._write_results_csv(self.auto_output_csv)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()


def build_sim():
    return GuiTwoNodeEntanglementWithModeledDelay()