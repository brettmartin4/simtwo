from __future__ import annotations

import numpy as np
import pytest

from simtwo.core.models.polarization_random_walk import RandomWalkPolarizationModel


def test_random_walk_prediction_is_unit_stokes_vector():

    model = RandomWalkPolarizationModel(step_std=0.05, seed=1)
    pred = model.predict({})

    assert pred.model_family == "polarization"
    assert pred.target_name == "polarization_random_walk"
    assert pred.stokes_vector is not None
    assert pred.poincare_state is not None
    assert np.linalg.norm(pred.stokes_vector) == pytest.approx(1.0)
    assert pred.metadata["S1"] == pytest.approx(pred.stokes_vector[0])
    assert pred.metadata["S2"] == pytest.approx(pred.stokes_vector[1])
    assert pred.metadata["S3"] == pytest.approx(pred.stokes_vector[2])


def test_random_walk_reset_restores_starting_stokes_state():

    model = RandomWalkPolarizationModel(step_std=0.05, seed=1)
    model.predict({})
    model.predict({})
    model.reset()

    assert tuple(model._stokes) == pytest.approx((1.0, 0.0, 0.0))
    assert model._step_index == 0


def test_random_walk_zero_step_stays_horizontal():
    
    model = RandomWalkPolarizationModel(step_std=0.0, seed=1)
    pred = model.predict({})

    assert pred.stokes_vector == pytest.approx((1.0, 0.0, 0.0))
    assert pred.plot_value == pytest.approx(0.0)
