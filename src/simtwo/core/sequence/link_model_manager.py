"""Manage simtwo backed link groups inside sequence network experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simtwo.core.modeling.model import load_trained_model_bundle


@dataclass
class LinkGroup:
    """Describe a set of sequence links that share the same simtwo channel model.
    
    Each group records one or more channels and optional nodes that should be updated together when a new pred is generated."""
    name: str
    channels: list[Any]
    delay_fraction: float = 1.0
    distance_fraction: float = 1.0


class LinkModelManager:
    """Owns current active GUI model and applies pred fullpath delay/distance to sequence channels."""

    def __init__(self, *, base_distance_m: float, alpha_per_c: float, t0_c: float, light_speed_m_per_ps: float):
        """Initialize the object and store the runtime dependencies it needs.
        
        Args:
            base_distance_m (float): Value used for base distance m.
            alpha_per_c (float): Value used for alpha per c.
            t0_c (float): Value used for t0 c.
            light_speed_m_per_ps (float): Value used for light speed m per ps.
        """
        self.base_distance_m = float(base_distance_m)
        self.alpha_per_c = float(alpha_per_c)
        self.t0_c = float(t0_c)
        self.light_speed_m_per_ps = float(light_speed_m_per_ps)

        self.model_config = None
        self.model_bundle: dict[str, Any] | None = None
        self.model_name = "default_physical_model"

        self.link_groups: list[LinkGroup] = []

    def reset_groups(self):
        """Clear all registered link groups."""
        self.link_groups = []

    def register_group(self, name: str, channels: list[Any], *,  delay_fraction: float = 1.0, distance_fraction: float = 1.0):
        """Register a link group that should receive simtwo model updates.
        
        Args:
            name (str): Name assigned to the object or dataset.
            channels: List containing channels.
            delay_fraction (float): Value used for delay fraction.
            distance_fraction (float): Value used for distance fraction.
        """
        self.link_groups.append(
            LinkGroup(
                name=name,
                channels=list(channels),
                delay_fraction=float(delay_fraction),
                distance_fraction=float(distance_fraction),
            )
        )

    def configure(self, config):
        """Configure a link group or manager from the current runtime context.
        
        Args:
            config: Channel model config selected in the GUI.
        
        Raises:
            RuntimeError: If the operation cannot be completed with the current inputs or state.
        """
        self.model_config = config
        self.model_bundle = None
        self.model_name = "default_physical_model"

        if config.mode == "default":
            self.model_name = "default_physical_model"
            return

        if config.mode == "existing":
            if not config.model_path:
                raise RuntimeError("Choose a trained model file before loading it.")

            self.model_bundle = load_trained_model_bundle(config.model_path)
            self.model_name = self.model_bundle.get("model_name", Path(config.model_path).stem)
            return

        raise RuntimeError(
            "Sequence mode training for new models not implemented yet "
            "Use default physical model or existing traind model."
        )

    def _get_temperature(self, row: dict[str, Any]) -> float:
        """Return temp for internal use.
        
        Args:
            row: Input observation row.
        
        Returns:
            The computed temp value.
        
        Raises:
            RuntimeError: If the operation cannot be completed with the current inputs or state.
        """
        for key in ("temperature_x", "temperature", "temp_C", "temp"):
            if key in row:
                return float(row[key])
        raise RuntimeError(
            "Dataset row must contain one of: temperature_x, temperature, temp_C, temp."#change later to be dynamic?
        )

    def _predict_full_path(self, row: dict[str, Any]) -> tuple[float, float, float, float | None]:
        """
        Returns:
            full_path_delay_ps, full_path_delay_ns, full_path_delay_s, full_path_distance_m_or_None
        """
        temp_c = self._get_temperature(row)

        if self.model_config is None or self.model_config.mode == "default":
            distance_m = self.base_distance_m * (
                1.0 + self.alpha_per_c * (temp_c - self.t0_c)
            )
            delay_ps = distance_m / self.light_speed_m_per_ps
            delay_ns = delay_ps / 1000.0
            delay_s = delay_ps * 1e-12
            return delay_ps, delay_ns, delay_s, distance_m

        estimator = self.model_bundle["estimator"]
        feature_names = list(self.model_bundle.get("feature_names") or [])
        target_name = str(self.model_bundle.get("target_name") or "").strip()

        feature_row = []
        for name in feature_names:
            if name in row:
                feature_row.append(float(row[name]))
            elif name == "temperature_x" and "temperature" in row:
                feature_row.append(float(row["temperature"]))
            elif name == "temperature" and "temperature_x" in row:
                feature_row.append(float(row["temperature_x"]))
            else:
                raise RuntimeError(f"Missing model feature '{name}' in current row.")

        pred = float(estimator.predict([feature_row])[0])

        if target_name in {"path_delay", "path_delay_ns"}:
            delay_ns = pred
            delay_ps = delay_ns * 1000.0
            delay_s = delay_ns * 1e-9
            return delay_ps, delay_ns, delay_s, None

        if target_name == "path_delay_ps":
            delay_ps = pred
            delay_ns = delay_ps / 1000.0
            delay_s = delay_ps * 1e-12
            return delay_ps, delay_ns, delay_s, None

        if target_name == "path_delay_s":
            delay_s = pred
            delay_ns = delay_s * 1e9
            delay_ps = delay_s * 1e12
            return delay_ps, delay_ns, delay_s, None

        raise RuntimeError(
            f"Unsupported model target '{target_name}'. "
            "Expected path_delay/path_delay_ns/path_delay_ps/path_delay_s."
        )

    def apply_to_registered_links(self, row: dict[str, Any]) -> dict[str, Any]:
        """Apply to registered links to the current state.
        
        Args:
            row: Input observation row.
        
        Returns:
            Metadata or result values produced by the operation.
        """
        full_delay_ps, full_delay_ns, full_delay_s, full_distance_m = self._predict_full_path(row)

        if full_distance_m is None:
            full_distance_m = full_delay_ps * self.light_speed_m_per_ps

        for group in self.link_groups:
            group_delay_ps = full_delay_ps * group.delay_fraction
            group_distance_m = full_distance_m * group.distance_fraction

            for ch in group.channels:
                if hasattr(ch, "set_effective_distance"):
                    ch.set_effective_distance(group_distance_m)
                else:
                    if hasattr(ch, "distance"):
                        ch.distance = float(group_distance_m)
                    if hasattr(ch, "delay"):
                        ch.delay = int(round(group_delay_ps))

                if hasattr(ch, "loss") and hasattr(ch, "attenuation"):
                    ch.loss = 1 - 10 ** (ch.distance * ch.attenuation / -10)

        return {
            "current_model": self.model_name,
            "predicted_path_delay_ps": full_delay_ps,
            "predicted_path_delay_ns": full_delay_ns,
            "predicted_path_delay_s": full_delay_s,
            "predicted_distance_m": full_distance_m,
            "input_temperature": self._get_temperature(row),
        }