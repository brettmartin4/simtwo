"""Defines the plugin interface used to integrate sequence experiments with simtwo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from simtwo.core.models.base import DelayModel, DelayPrediction
from simtwo.core.runtime.session import ExecutionControls, RuntimeSession


@dataclass
class SequenceExperimentContext:
    """Bundle runtime state that a sequence plugin needs while running.
    
    The context exposes the Simtwo runtime session, execution controls, channel model, feature bindings, and loaded dataset so plugins can request preds and update their sequence objects consistently."""
    session: RuntimeSession
    controls: ExecutionControls
    model: DelayModel
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    timeline: Any = None
    nodes: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def predict(self, row: dict[str, Any]) -> DelayPrediction:
        """Evaluate the active simtwo channel model for one observation row.
        
        Args:
            row: Input observation row.
        
        Returns:
            DelayPrediction: The computed pred value.
        """
        features = self.session.feature_bindings.extract(row)
        return self.model.predict(features)


class SequenceExperimentPlugin(Protocol):
    """Base interface for simtwo-aware sequence experiment plugins.
    
    Plugins implement setup and per-row update behavior while the runner handles dataset iteration, pred callbacks, and result export."""
    
    name: str

    def build(self, ctx: SequenceExperimentContext):
        """Called once before stepping starts.

        Create the timeline, nodes, channels, and any experiment state here.

        Args:
            ctx (SequenceExperimentContext)
        """
        return

    def step(self, ctx: SequenceExperimentContext, row: dict[str, Any]) -> dict[str, Any]:
        """TODO: Not implemented and scheduled for deletion pending tests to determine if removal will break the program."""
        return


class BaseSequenceExperiment:
    # Convenience base class for subclassing over protocol-only style in case a user wants to use this method instead (not tested until after first full build)
    name = "Unnamed SeQUeNCe Experiment"

    # TODO: Use not implemented error raising for all of these?
    def build(self, ctx: SequenceExperimentContext):
        return

    def step(self, ctx: SequenceExperimentContext, row: dict[str, Any]) -> dict[str, Any]:
        return