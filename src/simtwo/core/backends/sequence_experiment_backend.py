"""Wrapper for legacy sequence experiment objects behind the GUI backend interface."""

from __future__ import annotations

from typing import Any

from simtwo.core.backends.protocol import ChannelModelConfig
from simtwo.core.modeling.model import load_model_spec


class SequenceBackend:
    """Compatibility wrapper around a sequence experiment object.
    
    The wrapper adapts an experiment exposing methods run_sim, load_file, or export_file and more to the common backend interface expected by the GUI."""
    def __init__(self, sequence_sim: Any):
        """Initialize the object and store the runtime dependencies it needs.
        
        Args:
            sequence_sim: sequence experiment object being wrapped.
        """
        self.sim = sequence_sim
        self._run_speed_ms = 100
        self._channel_model_config: ChannelModelConfig | None = None

    def get_mode_name(self) -> str:
        """Return the display name for the active backend mode."""
        return "SeQUeNCe Experiment"

    def set_run_speed(self, ms: int) -> None:
        """Store the requested run speed val for compatibility with playback oriented code. Scheduled for deletion.
        
        Args:
            ms (int): Run speed delay in millis.
        """
        self._run_speed_ms = int(ms)
        if hasattr(self.sim, "set_run_speed"):
            self.sim.set_run_speed(ms)

    def start(self, cb_plot, cb_conditions, cb_poincare) -> None:
        """Run generation and emit plot, condition, and polarization callbacks.
        
        Args:
            cb_plot: Callback that receives the epoch index and plot value.
            cb_conditions: Callback that receives the current environment/condition dict.
            cb_poincare: Callback that receives the current polarization state, when available.
        """
        sender = self.sim.nodes["A"]
        receiver = self.sim.nodes["B"]
        self.sim.run_sim(sender, receiver, cb_plot, cb_conditions, cb_poincare)

    def stop(self) -> None:
        """Stop any active execution and release runtime resources."""
        try:
            self.sim.cleanup_after_ids()
        except Exception:
            pass

    def reset(self) -> None:
        """Return the backend or runner to its initial state and clear generated results."""
        self.stop()
        if hasattr(self.sim, "reset_simulation"):
            self.sim.reset_simulation()
        elif hasattr(self.sim, "current_epoch"):
            self.sim.current_epoch = 0

    def load_data(self, path: str) -> None:
        """Load a CSV dataset and make it available to the active backend or runtime session.
        
        Args:
            path (str): File path used for loading or saving data.
        
        Raises:
            ValueError: If the operation cannot be completed with the current inputs or state.
        """
        if hasattr(self.sim, "load_data"):
            self.sim.load_data(path)
        else:
            self.sim.load_file(path)

    def export_results(self, path: str) -> None:
        """Write the currently generated results to a CSV file.
        
        Args:
            path (str): File path used for loading or saving data.
        
        Raises:
            ValueError: If the operation cannot be completed with the current inputs or state.
        """
        if hasattr(self.sim, "export_results"):
            self.sim.export_results(path)
        else:
            self.sim.export_file(path)

    def configure_channel_model(self, config: ChannelModelConfig) -> None:
        """Apply a default, existing, or newly trained channel model config.
        
        Args:
            config (ChannelModelConfig): Channel model config selected in the GUI.
        """
        self._channel_model_config = config

        if hasattr(self.sim, "configure_channel_model"):
            self.sim.configure_channel_model(config)
            return

        if config.mode == "existing" and config.model_path:
            spec = load_model_spec(config.model_path)
            setattr(self.sim, "channel_model_spec", spec)

        setattr(self.sim, "channel_model_config", config)

    def train_channel_model(self, config: ChannelModelConfig) -> dict[str, Any]:
        """Train a model from the currently loaded dataset and activate the trained bundle.
        
        Args:
            config (ChannelModelConfig): Channel model config selected in the GUI.
        
        Returns:
            Metadata or result values produced by the operation.
        """
        self._channel_model_config = config
        if hasattr(self.sim, "train_channel_model"):
            return dict(self.sim.train_channel_model(config) or {})
        raise NotImplementedError(
            "Training is not implemented for the SeQUeNCe backnd yet."
            "Right now the sklearn training flow is wired up for only the standalone backend."
        )

    def save_current_model(self, path: str) -> None:
        """Save the currently trained model bundle to disk.
        
        Args:
            path (str): File path used for loading or saving data.
        
        Raises:
            ValueError: If the operation cannot be completed with the current inputs or state.
        """
        if hasattr(self.sim, "save_current_model"):
            self.sim.save_current_model(path)
            return
        raise NotImplementedError(
            "Saving trained models is not implemented yet for the SeQUeNCe backend yet."
        )
