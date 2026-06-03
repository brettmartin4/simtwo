import csv
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from sequence.components.photon import Photon
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.timeline import Timeline
from sequence.topology.node import Node

from simtwo.core.SimulatorAdapter import SimulatorAdapter
from simtwo.core.JitterNode import RxNodeWithJitter
from simtwo.core.ThermalQuantumChannel import ThermalQuantumChannel
from simtwo.core.modeling.model import load_trained_model_bundle


class RecordingRxNode(RxNodeWithJitter):
    def __init__(self, name, timeline, rng, jitter_std_ps=2.0):
        super().__init__(name=name, timeline=timeline, rng=rng, jitter_std_ps=jitter_std_ps)
        self.arrival_log = []

    def receive_qubit(self, src: str, qubit):
        super().receive_qubit(src, qubit)
        self.arrival_log.append(
            {
                "photon_name": getattr(qubit, "name", ""),
                "src": src,
                "arrival_time_ps_raw": int(self.t_rx_ps) if self.t_rx_ps is not None else None,
                "arrival_time_ps_reported": int(self.t_done_ps) if self.t_done_ps is not None else None,
            }
        )


class GuiTemperatureToPathDelayExperiment(SimulatorAdapter):
    def __post_init__(self):
        super().__post_init__()

        # match physical experiment settings
        self.base_distance_m = 64_000.0
        self.alpha_per_C = 5e-7
        self.T0_C = 19.995
        self.attenuation_db_per_m = 0.0
        self.light_speed_m_per_ps = 0.0002
        self.jitter_std_ps = 2.0
        self.seed = 42

        self.channel_model_config = None
        self.channel_model_bundle = None
        self.channel_model_name = "<no model selected>"

        self.gui_dataset_loaded = False
        self.results_rows = []
        self.default_output_csv = "gui_sequence_path_delay_results.csv"

        self._rng = np.random.default_rng(self.seed)

    # hooks

    def load_file(self, file_path: str):
        super().load_file(file_path)
        self.gui_dataset_loaded = True
        self.current_epoch = 0
        self.results_rows = []

    def load_data(self, file_path: str):
        self.load_file(file_path)

    def reset_simulation(self):
        self.cleanup_after_ids()
        self.current_epoch = 0
        self.results_rows = []

    def export_results(self, file_path: str):
        self._write_results_csv(file_path)

    # modeling stuff:
    def configure_channel_model(self, config):
        self.channel_model_config = config
        self.channel_model_bundle = None

        allowed_targets = {"path_delay", "path_delay_ns", "path_delay_ps", "path_delay_s"}
        selected_features = list(config.feature_names or [])

        has_temp_feature = ("temperature_x" in selected_features) or ("temperature" in selected_features)

        if config.mode == "default":
            if not has_temp_feature:
                # TODO: change this to be dynamic later as opposed to looking for specific var names
                raise RuntimeError(
                    "This experiment requires either 'temperature_x' or 'temperature' "
                    "to be selected as a feature in the modeling suite."
                )
            self.channel_model_name = "default_physical_model"
            return

        if config.mode == "existing":
            if not has_temp_feature:
                raise RuntimeError(
                    "This experiment requires either 'temperature_x' or 'temperature' "
                    "to be selected as a feature in the modeling suite."
                )

            bundle = load_trained_model_bundle(config.model_path)
            bundle_features = list(bundle.get("feature_names") or [])
            bundle_target = str(bundle.get("target_name") or "").strip()

            if bundle_features not in (["temperature_x"], ["temperature"]):
                raise RuntimeError(
                    "Loaded model bundle must have feature_names == ['temperature_x'] "
                    "or ['temperature'] for this experiment."
                )

            self.channel_model_bundle = bundle
            self.channel_model_name = bundle.get("model_name", Path(config.model_path).stem)
            return

        raise RuntimeError(
            "Sequence backend training for 'new model' is not implemented yet. "
            "Use either the default model or load an existing trained model bundle."
        )

    # helper funcs

    def _validate_ready(self):
        if not self.gui_dataset_loaded:
            raise RuntimeError(
                "Load a dataset in the GUI and send it to the modeling suite before running this experiment."
            )

        if self.channel_model_config is None:
            raise RuntimeError(
                "Select and apply a model in the modeling suite before running this experiment."
            )

        if not self.observations:
            raise RuntimeError("No observations are currently loaded.")

        first_row = self.observations[0]
        # TODO: again, change this later:
        if "temperature_x" not in first_row and "temperature" not in first_row:
            raise RuntimeError(
                "Loaded dataset must contain either a 'temperature_x' or 'temperature' column."
            )

    # TODO: remove after fixing previous few TODOs?
    def _get_temperature_value(self, row: dict[str, Any]) -> float:
        if "temperature_x" in row:
            return float(row["temperature_x"])
        if "temperature" in row:
            return float(row["temperature"])
        raise RuntimeError("Row is missing both 'temperature_x' and 'temperature'.")

    # TODO: definitely change this
    def _target_to_seconds_and_ns(self, value: float, target_name: str) -> tuple[float, float]:
        target_name = str(target_name).strip()

        if target_name in {"path_delay", "path_delay_ns"}:
            delay_ns = float(value)
            delay_s = delay_ns * 1e-9
            return delay_s, delay_ns

        if target_name == "path_delay_ps":
            delay_ps = float(value)
            delay_ns = delay_ps / 1000.0
            delay_s = delay_ps * 1e-12
            return delay_s, delay_ns

        if target_name == "path_delay_s":
            delay_s = float(value)
            delay_ns = delay_s * 1e9
            return delay_s, delay_ns

        raise RuntimeError(f"Unsupported target name: {target_name}")

    def _predict_ml_delay(self, temperature_value: float) -> tuple[float, float]:
        if self.channel_model_bundle is None:
            raise RuntimeError("No trained model bundle is loaded.")

        estimator = self.channel_model_bundle["estimator"]
        target_name = str(self.channel_model_bundle.get("target_name") or "").strip()

        pred = float(estimator.predict([[float(temperature_value)]])[0])
        return self._target_to_seconds_and_ns(pred, target_name)

    def _run_one_physical_epoch(self, temp_c: float) -> dict[str, Any]:
        tl = Timeline()
        rng = np.random.default_rng(self.seed + int(self.current_epoch))

        tx_node = Node("tx", tl)
        rx_node = RecordingRxNode("rx", tl, rng=rng, jitter_std_ps=self.jitter_std_ps)

        channel = ThermalQuantumChannel(
            name="thermal_qc",
            timeline=tl,
            base_distance_m=self.base_distance_m,
            alpha_per_C=self.alpha_per_C,
            T0_C=self.T0_C,
            attenuation=self.attenuation_db_per_m,
            polarization_fidelity=1.0,
            light_speed=self.light_speed_m_per_ps,
        )
        channel.set_ends(tx_node, rx_node.name)
        channel.set_temperature(temp_c)

        photon = Photon(f"p{self.current_epoch}", tl)
        send_time_ps = 0

        send_proc = Process(tx_node, "send_qubit", [rx_node.name, photon], {})
        tl.schedule(Event(send_time_ps, send_proc))

        tl.init()
        tl.run()

        raw_arrival_ps = rx_node.t_rx_ps
        reported_arrival_ps = rx_node.t_done_ps

        flight_time_ps_raw = None
        if raw_arrival_ps is not None:
            flight_time_ps_raw = int(raw_arrival_ps - send_time_ps)

        flight_time_ps_reported = None
        if reported_arrival_ps is not None:
            flight_time_ps_reported = int(reported_arrival_ps - send_time_ps)

        predicted_path_delay_ps = flight_time_ps_reported
        predicted_path_delay_ns = None if predicted_path_delay_ps is None else predicted_path_delay_ps / 1000.0
        predicted_path_delay_s = None if predicted_path_delay_ps is None else predicted_path_delay_ps * 1e-12

        return {
            "send_time_ps": send_time_ps,
            "arrival_time_ps_raw": raw_arrival_ps,
            "arrival_time_ps_reported": reported_arrival_ps,
            "flight_time_ps_raw": flight_time_ps_raw,
            "flight_time_ps_reported": flight_time_ps_reported,
            "predicted_path_delay_ps": predicted_path_delay_ps,
            "predicted_path_delay_ns": predicted_path_delay_ns,
            "predicted_path_delay_s": predicted_path_delay_s,
        }

    def _write_results_csv(self, file_path: str):
        fieldnames = [
            "epoch",
            "model_mode",
            "current_model",
            "input_t_sec",
            "input_temperature",
            "input_path_delay_ns",
            "input_path_delay_ps",
            "send_time_ps",
            "arrival_time_ps_raw",
            "arrival_time_ps_reported",
            "flight_time_ps_raw",
            "flight_time_ps_reported",
            "predicted_path_delay_ps",
            "predicted_path_delay_ns",
            "predicted_path_delay_s",
            "arrival_t_sec_raw",
            "arrival_t_sec_reported",
        ]

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results_rows)

    # main sim code:
    def run_sim(self, sender: Any, receiver: Any, update_plot, update_conditions=None, update_poincare_sphere=None):
        self._validate_ready()
        self._stop_evt.clear()
        self.results_rows = []

        def worker():
            rows = [dict(row) for row in self.observations]
            for i, row in enumerate(rows):
                row.setdefault("epoch", i)

            for i in range(self.current_epoch, len(rows)):
                if self._stop_evt.is_set():
                    break

                row = rows[i]
                self.current_epoch = i

                temperature_value = self._get_temperature_value(row)

                input_path_delay_ns = row.get("path_delay", row.get("path_delay_ns"))
                input_path_delay_ps = None
                if input_path_delay_ns is not None:
                    input_path_delay_ps = float(input_path_delay_ns) * 1000.0

                if self.channel_model_config.mode == "default":
                    result = self._run_one_physical_epoch(temperature_value)
                    model_mode = "default"
                else:
                    predicted_path_delay_s, predicted_path_delay_ns = self._predict_ml_delay(temperature_value)
                    result = {
                        "send_time_ps": None,
                        "arrival_time_ps_raw": None,
                        "arrival_time_ps_reported": None,
                        "flight_time_ps_raw": None,
                        "flight_time_ps_reported": None,
                        "predicted_path_delay_ps": predicted_path_delay_ns * 1000.0,
                        "predicted_path_delay_ns": predicted_path_delay_ns,
                        "predicted_path_delay_s": predicted_path_delay_s,
                    }
                    model_mode = "existing"

                arrival_t_sec_raw = None
                if result["arrival_time_ps_raw"] is not None and row.get("t_sec") is not None:
                    arrival_t_sec_raw = float(row["t_sec"]) + (float(result["arrival_time_ps_raw"]) * 1e-12)

                arrival_t_sec_reported = None
                if model_mode == "default":
                    if result["arrival_time_ps_reported"] is not None and row.get("t_sec") is not None:
                        arrival_t_sec_reported = float(row["t_sec"]) + (float(result["arrival_time_ps_reported"]) * 1e-12)
                else:
                    if row.get("t_sec") is not None and result["predicted_path_delay_s"] is not None:
                        arrival_t_sec_reported = float(row["t_sec"]) + float(result["predicted_path_delay_s"])

                result_row = {
                    "epoch": row["epoch"],
                    "model_mode": model_mode,
                    "current_model": self.channel_model_name,
                    "input_t_sec": row.get("t_sec"),
                    "input_temperature": temperature_value,
                    "input_path_delay_ns": input_path_delay_ns,
                    "input_path_delay_ps": input_path_delay_ps,
                    "send_time_ps": result["send_time_ps"],
                    "arrival_time_ps_raw": result["arrival_time_ps_raw"],
                    "arrival_time_ps_reported": result["arrival_time_ps_reported"],
                    "flight_time_ps_raw": result["flight_time_ps_raw"],
                    "flight_time_ps_reported": result["flight_time_ps_reported"],
                    "predicted_path_delay_ps": result["predicted_path_delay_ps"],
                    "predicted_path_delay_ns": result["predicted_path_delay_ns"],
                    "predicted_path_delay_s": result["predicted_path_delay_s"],
                    "arrival_t_sec_raw": arrival_t_sec_raw,
                    "arrival_t_sec_reported": arrival_t_sec_reported,
                }
                self.results_rows.append(result_row)

                cond_row = {
                    "current_model": self.channel_model_name,
                    "temperature": temperature_value,
                    "input_t_sec": row.get("t_sec"),
                    "input_path_delay_ns": input_path_delay_ns,
                    "predicted_path_delay_ns": result["predicted_path_delay_ns"],
                    "predicted_path_delay_s": result["predicted_path_delay_s"],
                }

                if update_conditions is not None:
                    update_conditions(cond_row)

                if result["predicted_path_delay_s"] is not None:
                    update_plot(i, float(result["predicted_path_delay_s"]))

                time.sleep(max(0.0, self._run_speed_ms / 1000.0))

            self._write_results_csv(self.default_output_csv)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()


def build_sim():
    return GuiTemperatureToPathDelayExperiment()