from __future__ import annotations

import math

import pytest

from simtwo.core.models.physical_delay import PhysicalDelayModel


def test_physical_delay_returns_consistent_units():

    model = PhysicalDelayModel(jitter_std_ps=0.0)
    pred = model.predict({"temperature": 20.0})

    assert pred.model_family == "timing"
    assert pred.target_name == "path_delay_s"
    assert pred.path_delay_ps is not None
    assert pred.path_delay_ns == pytest.approx(pred.path_delay_ps / 1000.0)
    assert pred.path_delay_s == pytest.approx(pred.path_delay_ps * 1e-12)
    assert pred.plot_value == pytest.approx(pred.path_delay_s)
    assert pred.distance_m is not None
    assert math.isfinite(pred.distance_m)


def test_physical_delay_distance_increases_with_temperature_when_jitter_disabled():

    model = PhysicalDelayModel(jitter_std_ps=0.0)

    cold = model.predict({"temperature": 10.0})
    warm = model.predict({"temperature": 30.0})

    assert warm.distance_m > cold.distance_m
    assert warm.path_delay_ps > cold.path_delay_ps


def test_physical_delay_requires_temperature_feature():
    
    model = PhysicalDelayModel(jitter_std_ps=0.0)

    with pytest.raises(KeyError):
        model.predict({})
