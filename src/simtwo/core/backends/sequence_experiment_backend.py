from __future__ import annotations

from typing import Any

from simtwo.core.backends.protocol import ChannelModelConfig
from simtwo.core.modeling.model import load_model_spec


class SequenceBackend:
    def __init__(self, sequence_sim: Any):
        self.sim = sequence_sim
        self._run_speed_ms = 100
        self._channel_model_config: ChannelModelConfig | None = None

    def get_mode_name(self) -> str:
        return "SeQUeNCe Experiment"

    def set_run_speed(self, ms: int) -> None:
        self._run_speed_ms = int(ms)
        if hasattr(self.sim, "set_run_speed"):
            self.sim.set_run_speed(ms)

    def start(self, cb_plot, cb_conditions, cb_poincare) -> None:
        sender = self.sim.nodes["A"]
        receiver = self.sim.nodes["B"]
        self.sim.run_sim(sender, receiver, cb_plot, cb_conditions, cb_poincare)

    def stop(self) -> None:
        try:
            self.sim.cleanup_after_ids()
        except Exception:
            pass

    def reset(self) -> None:
        self.stop()
        if hasattr(self.sim, "reset_simulation"):
            self.sim.reset_simulation()
        elif hasattr(self.sim, "current_epoch"):
            self.sim.current_epoch = 0

    def load_data(self, path: str) -> None:
        if hasattr(self.sim, "load_data"):
            self.sim.load_data(path)
        else:
            self.sim.load_file(path)

    def export_results(self, path: str) -> None:
        if hasattr(self.sim, "export_results"):
            self.sim.export_results(path)
        else:
            self.sim.export_file(path)

    def configure_channel_model(self, config: ChannelModelConfig) -> None:
        self._channel_model_config = config

        if hasattr(self.sim, "configure_channel_model"):
            self.sim.configure_channel_model(config)
            return

        if config.mode == "existing" and config.model_path:
            spec = load_model_spec(config.model_path)
            setattr(self.sim, "channel_model_spec", spec)

        setattr(self.sim, "channel_model_config", config)

    def train_channel_model(self, config: ChannelModelConfig) -> dict[str, Any]:
        self._channel_model_config = config
        if hasattr(self.sim, "train_channel_model"):
            return dict(self.sim.train_channel_model(config) or {})
        raise NotImplementedError(
            "Training is not implemented for the SeQUeNCe backnd yet."
            "Right now the sklearn training flow is wired up for only the standalone backend."
        )

    def save_current_model(self, path: str) -> None:
        if hasattr(self.sim, "save_current_model"):
            self.sim.save_current_model(path)
            return
        raise NotImplementedError(
            "Saving trained models is not implemented yet for the SeQUeNCe backend yet."
        )
