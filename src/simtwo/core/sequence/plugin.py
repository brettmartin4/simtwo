from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from simtwo.core.models.base import DelayModel, DelayPrediction
from simtwo.core.runtime.session import ExecutionControls, RuntimeSession


@dataclass
class SequenceExperimentContext:
    session: RuntimeSession
    controls: ExecutionControls
    model: DelayModel
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    timeline: Any = None
    nodes: dict[str, Any] = field(default_factory=dict)
    channels: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def predict(self, row: dict[str, Any]) -> DelayPrediction:
        features = self.session.feature_bindings.extract(row)
        return self.model.predict(features)


class SequenceExperimentPlugin(Protocol):
    
    name: str

    def build(self, ctx: SequenceExperimentContext):
        """
        Called once before stepping starts.
        Create the timeline, nodes, channels, and any experiment state here.
        """
        return

    def step(self, ctx: SequenceExperimentContext, row: dict[str, Any]) -> dict[str, Any]:
        """
        Called once per dataset row.
        # TODO: Implement

        Must return a dict containing at least one plottable field--  something lke:
            {
                "path_delay_s": ...,
                "clockerror": ...,
                ...
            }
        """
        return


class BaseSequenceExperiment:
    """
    Convenience base class for subclassing over protocol-only style in case a user wants to use this method instead (not tested until after first full build)
    # TODO: Test this later
    """
    name = "Unnamed SeQUeNCe Experiment"

    # TODO: Use not implemented error raising for all of these?
    def build(self, ctx: SequenceExperimentContext):
        return

    def step(self, ctx: SequenceExperimentContext, row: dict[str, Any]) -> dict[str, Any]:
        return