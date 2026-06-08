from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from simtwo.core.backends.protocol import ChannelModelConfig
from simtwo.core.modeling.model import fit_model_bundle, load_trained_model_bundle, save_trained_model_bundle
from simtwo.core.models.physical_delay import PhysicalDelayModel
from simtwo.core.models.polarization_random_walk import RandomWalkPolarizationModel
from simtwo.core.models.sklearn_delay import SklearnDelayModel
from simtwo.core.runtime.session import ExecutionControls, FeatureBindings, LoadedDataset, RuntimeSession
from simtwo.core.sequence.runner import SequenceRunner
from simtwo.core.standalone.engine import StandaloneEngine


@dataclass
class GuiRuntimeBackend:
    mode_name: str
    engine: Any
    session: RuntimeSession
    controls: ExecutionControls

    _channel_model_config: ChannelModelConfig | None = None
    _active_model_bundle: dict[str, Any] | None = None

    def get_mode_name(self) -> str:
        return self.mode_name

    def set_run_speed(self, ms: int) -> None:
        self.controls.step_delay_ms = int(ms)

    def start(self, cb_plot, cb_conditions, cb_poincare) -> None:
        self.engine.start(cb_plot, cb_conditions, cb_poincare)

    def stop(self) -> None:
        self.engine.stop()

    def reset(self) -> None:
        self.engine.reset()

    def load_data(self, path: str) -> None:
        df = pd.read_csv(path, encoding="utf-8-sig")
        dataset = LoadedDataset(
            name=Path(path).stem,
            df=df,
            time_column=self._infer_time_column(df),
            timezone="UTC",
            posix_unit="s",
        )
        self.session.set_dataset(dataset)

    def export_results(self, path: str) -> None:
        self.engine.export_results(path)

    def configure_channel_model(self, config: ChannelModelConfig) -> None:
        family = self._normalize_model_family(config.model_family)
        self._channel_model_config = config
        self._active_model_bundle = None

        if config.mode == "default":
            if family == "timing":
                self.session.set_model(
                    PhysicalDelayModel(
                        base_distance_m=64_000.0,
                        alpha_per_c=5e-7,
                        t0_c=19.995,
                        light_speed_m_per_ps=0.0002,
                        jitter_std_ps=2.0,
                        seed=42,
                    )
                )
                if self.session.dataset is not None:
                    self.session.feature_bindings = self._build_default_bindings(self.session.dataset)
                return

            if family == "polarization":
                self.session.feature_bindings = FeatureBindings(mapping={})
                self.session.set_model(RandomWalkPolarizationModel(seed=42))
                return

        if self.session.dataset is None:
            raise RuntimeError("Load a dataset before configuring a trained or existing model.")

        if config.mode == "existing":
            if not config.model_path:
                raise RuntimeError("Choose a trained model file before loading it.")

            bundle = load_trained_model_bundle(config.model_path)
            saved_family = self._normalize_model_family(
                str(bundle.get("model_family") or bundle.get("metadata", {}).get("model_family") or family)
            )
            if saved_family != family:
                raise RuntimeError(
                    f"Selected model family is '{family}', but the saved model was marked as '{saved_family}'."
                )

            model = SklearnDelayModel.from_bundle(bundle, model_family=family)
            self.session.feature_bindings = self._build_bindings_for_features(
                dataset=self.session.dataset,
                feature_names=model.feature_names,
            )
            self.session.set_model(model)
            return

        if config.mode == "new":
            #  actual fit happens in train_channel_model
            return

        raise RuntimeError(f"Unsupported model mode: {config.mode}")

    def train_channel_model(self, config: ChannelModelConfig) -> dict[str, Any]:
        family = self._normalize_model_family(config.model_family)
        if self.session.dataset is None:
            raise RuntimeError("Load a dataset before training a model.")

        observations = self.session.dataset.to_records()
        bundle = fit_model_bundle(observations, config)
        self._active_model_bundle = bundle

        model = SklearnDelayModel.from_bundle(bundle, model_family=family)
        self.session.feature_bindings = self._build_bindings_for_features(
            dataset=self.session.dataset,
            feature_names=model.feature_names,
        )
        self.session.set_model(model)

        metadata = dict(bundle.get("metadata", {}))
        metadata["model_name"] = bundle.get("model_name", config.model_name)
        metadata["model_kind"] = bundle.get("model_kind", config.model_kind)
        metadata["model_family"] = family
        metadata["target_name"] = bundle.get("target_name", config.target_name)
        return metadata

    def save_current_model(self, path: str) -> None:
        if self._active_model_bundle is None:
            raise ValueError("There is no trained model loaded to save.")
        save_trained_model_bundle(self._active_model_bundle, path)

    @staticmethod
    def _normalize_model_family(value: str | None) -> str:
        family = str(value or "").strip().lower()
        if family not in {"timing", "polarization"}:
            raise RuntimeError("Select a model family first: timing or polarization.")
        return family

    @staticmethod
    def _infer_time_column(df: pd.DataFrame) -> str:
        for candidate in ("current_time", "t_sec", "posix_time", "epoch"):
            if candidate in df.columns:
                return candidate
        return "row_index"

    @staticmethod
    def _build_default_bindings(dataset: LoadedDataset) -> FeatureBindings:
        # TODO: Change this later to search for the temp substring in all columns and print a notice if multiple exist
        cols = set(str(c) for c in dataset.df.columns)
        if "temperature" in cols:
            return FeatureBindings(mapping={"temperature": "temperature"})
        if "temperature_x" in cols:
            return FeatureBindings(mapping={"temperature": "temperature_x"})
        if "temp_C" in cols:
            return FeatureBindings(mapping={"temperature": "temp_C"})
        raise RuntimeError(
            "Default physical timing model requires one of: temperature, temperature_x, or temp_C. "
            "Use the polarization family if you want the random-walk polarization placeholder."
        )

    @staticmethod
    def _build_bindings_for_features(dataset: LoadedDataset, feature_names: list[str]) -> FeatureBindings:
        cols = set(str(c) for c in dataset.df.columns)
        mapping: dict[str, str] = {}

        for feature in feature_names:
            if feature in cols:
                mapping[feature] = feature
                continue

            if feature == "temperature" and "temperature_x" in cols:
                mapping["temperature"] = "temperature_x"
                continue

            if feature == "temperature_x" and "temperature" in cols:
                mapping["temperature_x"] = "temperature"
                continue

            if feature == "temperature" and "temp_C" in cols:
                mapping["temperature"] = "temp_C"
                continue

            raise RuntimeError(
                f"Could not bind model feature '{feature}' to a dataset column."
            )

        return FeatureBindings(mapping=mapping)


def build_standalone_gui_backend() -> GuiRuntimeBackend:
    session = RuntimeSession()
    controls = ExecutionControls(step_delay_ms=10)
    engine = StandaloneEngine(session=session, controls=controls)

    backend = GuiRuntimeBackend(
        mode_name="Standalone Channel Workbench",
        engine=engine,
        session=session,
        controls=controls,
    )

    session.set_model(
        PhysicalDelayModel(
            base_distance_m=64_000.0,
            alpha_per_c=5e-7,
            t0_c=19.995,
            light_speed_m_per_ps=0.0002,
            jitter_std_ps=2.0,
            seed=42,
        )
    )
    session.feature_bindings = FeatureBindings(mapping={"temperature": "temperature"})

    return backend


def build_sequence_gui_backend(plugin) -> GuiRuntimeBackend:
    session = RuntimeSession()
    controls = ExecutionControls(step_delay_ms=10)
    engine = SequenceRunner(session=session, controls=controls, plugin=plugin)

    backend = GuiRuntimeBackend(
        mode_name="SeQUeNCe Experiment",
        engine=engine,
        session=session,
        controls=controls,
    )

    session.set_model(
        PhysicalDelayModel(
            base_distance_m=64_000.0,
            alpha_per_c=5e-7,
            t0_c=19.995,
            light_speed_m_per_ps=0.0002,
            jitter_std_ps=2.0,
            seed=42,
        )
    )

    return backend