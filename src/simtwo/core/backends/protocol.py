"""Define shared backend config objects and callback protocols used by the GUI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

PlotCallback = Callable[[int, float], None]
ConditionsCallback = Callable[[dict[str, Any]], None]
PoincareCallback = Callable[[Any], None]


@dataclass
class ChannelModelConfig:
    """Describe how a channel model should be loaded, trained, or applied.
    
    The GUI passes this config to a backend whenever a default model is selected, loads an existing model bundle, or trains a new sklearn model.
    The fields also define which model family should be used and which dataset columns are treated as features and targets."""
    # either "default", "existing", or "new":
    mode: str
    # either "timing" or "polarization"; the UI requires this before activation
    model_family: str = ""
    model_path: str = ""
    model_name: str = "my_model"
    epochs: int = 50
    learning_rate: float = 1e-3
    feature_names: list[str] | None = None
    target_name: str | None = None
    model_kind: str = "linear_regression"
    model_params: dict[str, Any] = field(default_factory=dict)
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15


class SimulationBackend(Protocol):
    """Protocol implemented by backend adapters used by the GUI.
    
    Backends hide whether predictions come from the standalone runtime, a sequence experiment, or another execution engine. 
    Implementations should expose the same lifecycle methods so the UI can load data, configure models, generate plots, export results, and save trained models."""
    # according to PEP 544, ellipsis placeholders are fine here for protocol method bodies. TODO: make standard in other siimilar instances?
    def get_mode_name(self) -> str: 
        """Return the display name for the active backend mode.
        
        Returns:
            The requested text value."""
        ...

    def set_run_speed(self, ms: int) -> None:
        """Store the requested run speed val for compatibility with playback oriented code.
        
        Args:
            ms (int): Run-speed delay in milliseconds. """
        ...

    def start(self, cb_plot: PlotCallback, cb_conditions: ConditionsCallback, cb_poincare: PoincareCallback) -> None:
        """Run generation and emit plot, condition, and polarization callbacks.
        
        Args:
            cb_plot (PlotCallback): Callback that receives the epoch index and plot value.
            cb_conditions (ConditionsCallback): Callback that receives the current environment/condition dictionary.
            cb_poincare (PoincareCallback): Callback that receives the current polarization state, when available."""
        ...

    def stop(self) -> None:
        """Stop any active execution and release runtime resources."""
        ...

    def reset(self) -> None:
        """Return the backend or runner to its initial state and clear generated results."""
        ...

    def load_data(self, path: str) -> None:
        """Load a CSV dataset and make it available to the active backend or runtime session.
        
        Args:
            path (str): File path used for loading or saving data."""
        ...

    def export_results(self, path: str) -> None:
        """Write the currently generated results to a CSV file.
        
        Args:
            path (str): File path used for loading or saving data."""
        ...

    def configure_channel_model(self, config: ChannelModelConfig) -> None:
        """Apply a default, existing, or newly trained channel model config.
        
        Args:
            config (ChannelModelConfig): Channel model config selected in the GUI."""
        ...

    def train_channel_model(self, config: ChannelModelConfig) -> dict[str, Any]:
        """Train a model from the currently loaded dataset and activate the trained bundle.
        
        Args:
            config (ChannelModelConfig): Channel model config selected in the GUI.
        
        Returns:
            Metadata or result values produced by the operation."""
        ...

    def save_current_model(self, path: str) -> None:
        """Save the currently trained model bundle to disk.
        
        Args:
            path (str): File path used for loading or saving data."""
        ...
