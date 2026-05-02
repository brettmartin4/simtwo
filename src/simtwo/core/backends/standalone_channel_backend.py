from __future__ import annotations

import csv
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from simtwo.core.backends.protocol import ChannelModelConfig
from simtwo.core.modeling.model import fit_model_bundle, load_model_spec, load_trained_model_bundle, save_trained_model_bundle


@dataclass
class StandaloneBackend:
    base_distance_m: float = 120_000.0
    alpha_per_C: float = 5e-7
    T0_C: float = 20.0
    attenuation_db_per_m: float = 0.0
    light_speed_m_per_ps: float = 0.0002
    jitter_std_ps: float = 5_000_000.0
    seed: int = 123

    observations: list[dict[str, Any]] = field(default_factory=list)
    current_epoch: int = 0

    _run_speed_ms: int = 100
    _stop_evt: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _rng: np.random.Generator = field(init=False)
    _results: list[dict[str, Any]] = field(default_factory=list)
    _channel_model_config: ChannelModelConfig | None = None
    _active_model_bundle: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if not self.observations:
            self.observations = self._default_observations()

    def get_mode_name(self) -> str:
        return "Standalone Channel Workbench"

    def set_run_speed(self, ms: int) -> None:
        self._run_speed_ms = int(ms)

    def start(self, cb_plot, cb_conditions, cb_poincare) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_evt.clear()

        def worker() -> None:
            for idx in range(self.current_epoch, len(self.observations)):
                if self._stop_evt.is_set():
                    break

                obs = dict(self.observations[idx])
                self.current_epoch = idx
                series_value = self._compute_series_value(obs)
                state = self._compute_poincare_state(idx, obs)
                result = {
                    "epoch": idx,
                    "plot_value": series_value,
                    **obs,
                }
                if self._active_model_bundle is None:
                    result["travel_time_s"] = series_value
                else:
                    result["predicted_value"] = series_value
                    result["active_model"] = self._active_model_bundle.get("model_name", "trained_model")
                self._results.append(result)

                conds = dict(obs)
                if self._active_model_bundle is not None:
                    conds["active_model"] = self._active_model_bundle.get("model_name", "trained_model")
                    conds["predicted_value"] = series_value
                cb_conditions(conds)
                cb_plot(idx, series_value)
                cb_poincare(state)

                time.sleep(max(0.0, self._run_speed_ms / 1000.0))

            if not self._stop_evt.is_set():
                self.current_epoch = len(self.observations)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def reset(self) -> None:
        self.stop()
        self.current_epoch = 0
        self._results.clear()

    def load_data(self, path: str) -> None:
        observations: list[dict[str, Any]] = []
        #with open(path, "r", newline="", encoding="utf-8") as fh:
        with open(path, "r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row_idx, row in enumerate(reader):
                clean: dict[str, Any] = {"epoch": row_idx}
                for key, value in row.items():
                    clean[key] = self._coerce_value(value)
                observations.append(clean)

        if not observations:
            raise ValueError("CSV contained no observation rows.")

        self.observations = observations
        self.current_epoch = 0
        self._results.clear()

    def export_results(self, path: str) -> None:
        out_path = Path(path)
        rows = self._results if self._results else [{"epoch": idx, **obs} for idx, obs in enumerate(self.observations)]
        if not rows:
            raise ValueError("There is no data available to export.")

        fieldnames: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def configure_channel_model(self, config: ChannelModelConfig) -> None:
        self._channel_model_config = config

        if config.mode == "default":
            self._active_model_bundle = None
            return

        if config.mode == "existing" and config.model_path:
            suffix = Path(config.model_path).suffix.lower()
            if suffix in {".joblib", ".pkl", ".pickle"}:
                self._active_model_bundle = load_trained_model_bundle(config.model_path)
                return

            if suffix == ".json":
                spec = load_model_spec(config.model_path)
                params = spec.get("params", {}) if isinstance(spec, dict) else {}
                if isinstance(params, dict):
                    for attr in (
                        "base_distance_m",
                        "alpha_per_C",
                        "T0_C",
                        "attenuation_db_per_m",
                        "light_speed_m_per_ps",
                        "jitter_std_ps",
                    ):
                        if attr in params:
                            setattr(self, attr, float(params[attr]))
                self._active_model_bundle = None
                return

        # TODO: Remove later?
        if config.mode == "new":
            # Training happens w/ train_channel_model()
            return

    def train_channel_model(self, config: ChannelModelConfig) -> dict[str, Any]:
        bundle = fit_model_bundle(self.observations, config)
        self._active_model_bundle = bundle
        self._channel_model_config = config
        metadata = dict(bundle.get("metadata", {}))
        metadata["model_name"] = bundle.get("model_name", config.model_name)
        metadata["model_kind"] = bundle.get("model_kind", config.model_kind)
        metadata["target_name"] = bundle.get("target_name", config.target_name)
        return metadata

    def save_current_model(self, path: str) -> None:
        if self._active_model_bundle is None:
            raise ValueError("There is no trained model loaded to save.")
        save_trained_model_bundle(self._active_model_bundle, path)

    def _default_observations(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(60):
            out.append(
                {
                    "epoch": i,
                    "temp_C": float(20.0 + 7.0 * math.sin(i / 6.0)),
                    "humidity": float(50.0 + 20.0 * math.sin(i / 11.0 + 0.3)),
                    "wind_speed": float(2.5 + 1.0 * math.cos(i / 8.0)),
                }
            )
        return out

    def _compute_series_value(self, obs: dict[str, Any]) -> float:
        pred = self._predict_with_active_model(obs)
        if pred is not None and np.isfinite(pred):
            return float(pred)
        return self._compute_travel_time(obs)

    def _predict_with_active_model(self, obs: dict[str, Any]) -> float | None:
        bundle = self._active_model_bundle
        if not bundle:
            return None

        estimator = bundle.get("estimator")
        feature_names = list(bundle.get("feature_names") or [])
        if estimator is None or not feature_names:
            return None

        row: list[float] = []
        for feature in feature_names:
            value = obs.get(feature)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(numeric):
                return None
            row.append(numeric)

        try:
            pred = estimator.predict(np.asarray([row], dtype=float))
            return float(pred[0])
        except Exception:
            return None

    def _compute_travel_time(self, obs: dict[str, Any]) -> float:
        temp_c = self._first_numeric(obs, ["temp_C", "temperature", "temp", "temperature_C"], self.T0_C)
        humidity = self._first_numeric(obs, ["humidity", "humidity_pct", "relative_humidity"], 50.0)
        wind_speed = self._first_numeric(obs, ["wind_speed", "wind", "wind_mps"], 0.0)

        expanded_distance_m = self.base_distance_m * (1.0 + self.alpha_per_C * (temp_c - self.T0_C))
        base_ps = expanded_distance_m / self.light_speed_m_per_ps

        humidity_ps = (humidity - 50.0) * 50.0
        wind_ps = wind_speed * 100.0
        jitter_ps = float(self._rng.normal(0.0, self.jitter_std_ps))

        total_ps = max(0.0, base_ps + humidity_ps + wind_ps + jitter_ps)
        return total_ps * 1e-12

    def _compute_poincare_state(self, epoch: int, obs: dict[str, Any]) -> list[complex]:
        temp_c = self._first_numeric(obs, ["temp_C", "temperature", "temp", "temperature_C"], self.T0_C)
        theta = 0.2 + 0.02 * (temp_c - self.T0_C)
        phi = epoch * 0.1
        a = math.cos(theta / 2.0)
        b = complex(math.cos(phi), math.sin(phi)) * math.sin(theta / 2.0)
        return [a, b]

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip()
        if text == "":
            return ""
        try:
            return float(text)
        except ValueError:
            return text

    @staticmethod
    def _first_numeric(obs: dict[str, Any], keys: list[str], default: float) -> float:
        for key in keys:
            value = obs.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)
