"""Connect GUI actions to the selected runtime engine, dataset, and channel model."""

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
    """Adapter that connects the GUI to a runtime session and execution engine.
    
    The adapter owns the active channel model config, maps CSV columns onto model features, and delegates generation/export calls to the selected runtime engine. 
    It is used for both standalone and sequence-backed GUI modes."""
    mode_name: str
    engine: Any
    session: RuntimeSession
    controls: ExecutionControls

    _channel_model_config: ChannelModelConfig | None = None
    _active_model_bundle: dict[str, Any] | None = None

    def get_mode_name(self) -> str:
        """Return the display name for the active backend mode.
        
        Returns:
            The requested text value.
        """
        return self.mode_name

    def set_run_speed(self, ms: int) -> None:
        """Store the requested run speed value for compatibility with playback oriented code. Scheduled for deletion.
        
        Args:
            ms (int): Run speed delay in millis.
        """
        self.controls.step_delay_ms = int(ms)

    def start(self, cb_plot, cb_conditions, cb_poincare) -> None:
        """Run generation and emit plot, condition, and polarization callbacks.
        
        Args:
            cb_plot: Callback that receives the epoch index and plot value.
            cb_conditions: Callback that receives the current environment/condition dictionary.
            cb_poincare: Callback that receives the current polarization state, when available.
        """
        self.engine.start(cb_plot, cb_conditions, cb_poincare)

    def stop(self) -> None:
        """Stop any active execution and release runtime resources."""
        self.engine.stop()

    def reset(self) -> None:
        """Return the backend or runner to its initial state and clear generated results."""
        self.engine.reset()

    def load_data(self, path: str) -> None:
        """Load a CSV dataset and make it available to the active backend or runtime session.
        
        Args:
            path (str): File path used for loading or saving data.
        """
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
        """Write the currently generated results to a CSV file.
        
        Args:
            path (str): File path used for loading or saving data.
        """
        self.engine.export_results(path)

    def configure_channel_model(self, config: ChannelModelConfig) -> None:
        """Apply a default, existing, or newly trained channel model config.
        
        Args:
            config (ChannelModelConfig): Channel model config selected in the GUI.
        
        Raises:
            RuntimeError: If the operation cannot be completed with the current inputs or state.
        """
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
        """Train a model from the currently loaded dataset and activate the trained bundle.
        
        Args:
            config (ChannelModelConfig): Channel model config selected in the GUI.
        
        Returns:
            Metadata or result values produced by the operation.
        
        Raises:
            RuntimeError: If the operation cannot be completed with the current inputs or state.
        """
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
        """Save the currently trained model bundle to disk.
        
        Args:
            path (str): File path used for loading or saving data.
        
        Raises:
            ValueError: If the operation cannot be completed with the current inputs or state.
        """
        if self._active_model_bundle is None:
            raise ValueError("There is no trained model loaded to save.")
        save_trained_model_bundle(self._active_model_bundle, path)

    @staticmethod
    def _normalize_model_family(value: str | None) -> str:
        """Casts the model family string to all lowercase.
        
        Args:
            value (str): Model family name.
        
        Returns:
            str: stripped and lowercase model family name.

        Raises:
            RuntimeError: If the model family type is not recognized.
        """
        family = str(value or "").strip().lower()
        if family not in {"timing", "polarization"}:
            raise RuntimeError("Select a model family first: timing or polarization.")
        return family

    @staticmethod
    def _infer_time_column(df: pd.DataFrame) -> str:
        """Helper for infering time feature within a dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame to search.
        
        Returns:
            The requested time column.
        """
        for candidate in ("current_time", "t_sec", "posix_time", "epoch"):
            if candidate in df.columns:
                return candidate
        return "row_index"

    @staticmethod
    def _build_default_bindings(dataset: LoadedDataset) -> FeatureBindings:
        """Build default bindings from the current inputs.
        
        Args:
            dataset (LoadedDataset): Loaded dataset or processing dataset to operate on.
        
        Returns:
            A feature binding object that maps model features to dataset columns.
        
        Raises:
            RuntimeError: If the operation cannot be completed with the current inputs or state.
        """
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
        """Build bindings for features from the current inputs.
        
        Args:
            dataset (LoadedDataset): Loaded dataset or processing dataset to operate on.
            feature_names: Model feature names that must be bound to dataset columns.
        
        Returns:
            A feature binding object that maps model features to dataset columns.
        
        Raises:
            RuntimeError: If the operation cannot be completed with the current inputs or state.
        """
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
    """Build standalone gui backend from the current inputs.
    
    Returns:
        GuiRuntimeBackend: A configured GUI runtime backend.
    """
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
    """Build sequence gui backend from the current inputs.
    
    Args:
        plugin: Sequence plugin used by the runner.
    
    Returns:
        GuiRuntimeBackend: A configured GUI runtime backend.
    """
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