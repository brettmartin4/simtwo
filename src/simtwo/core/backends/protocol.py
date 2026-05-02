from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

PlotCallback = Callable[[int, float], None]
ConditionsCallback = Callable[[dict[str, Any]], None]
PoincareCallback = Callable[[Any], None]


@dataclass
class ChannelModelConfig:
    # either "default", "existing", or "new":
    mode: str
    model_path: str = ""
    model_name: str = "my_model"
    epochs: int = 50
    learning_rate: float = 1e-3
    feature_names: list[str] | None = None
    target_name: str | None = None
    model_kind: str = "linear_regression"
    model_params: dict[str, Any] = field(default_factory=dict)


class SimulationBackend(Protocol):
    # according to PEP 544, ellipsis placeholders are fine here for protocol method bodies.
    def get_mode_name(self) -> str: ...
    def set_run_speed(self, ms: int) -> None: ...
    def start(
        self,
        cb_plot: PlotCallback,
        cb_conditions: ConditionsCallback,
        cb_poincare: PoincareCallback,
    ) -> None: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
    def load_data(self, path: str) -> None: ...
    def export_results(self, path: str) -> None: ...
    def configure_channel_model(self, config: ChannelModelConfig) -> None: ...
    def train_channel_model(self, config: ChannelModelConfig) -> dict[str, Any]: ...
    def save_current_model(self, path: str) -> None: ...
