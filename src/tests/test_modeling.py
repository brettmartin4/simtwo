from __future__ import annotations

import numpy as np
import pytest



from simtwo.core.backends.protocol import ChannelModelConfig
from simtwo.core.modeling.model import (
    build_training_arrays,
    fit_model_bundle,
    load_trained_model_bundle,
    save_trained_model_bundle,
)
from simtwo.core.models.sklearn_delay import SklearnDelayModel



def test_build_training_arrays_skips_non_numeric_rows():

    observations = [
        {"temperature": 20.0, "target": 1.0},
        {"temperature": "bad", "target": 2.0},
        {"temperature": 22.0, "target": "bad"},
        {"temperature": 23.0, "target": 4.0},
    ]

    x_arr, y_arr, skipped = build_training_arrays(observations, ["temperature"], "target")

    assert x_arr.tolist() == [[20.0], [23.0]]
    assert y_arr.tolist() == [1.0, 4.0]
    assert skipped == 2


def test_fit_save_load_bundle_and_predict_timing_model(tmp_path):

    observations = [
        {"temperature": float(i), "path_delay_ns": float(2 * i + 1)}
        for i in range(12)
    ]
    config = ChannelModelConfig(
        mode="new",
        model_family="timing",
        model_name="test_model",
        feature_names=["temperature"],
        target_name="path_delay_ns",
        model_kind="linear_regression",
    )

    bundle = fit_model_bundle(observations, config)
    path = tmp_path / "model.joblib"
    save_trained_model_bundle(bundle, path)
    loaded = load_trained_model_bundle(path)
    model = SklearnDelayModel.from_bundle(loaded)
    pred = model.predict({"temperature": 20.0})

    assert loaded["model_family"] == "timing"
    assert loaded["metadata"]["train_count"] > 0
    assert loaded["metadata"]["validation_count"] > 0
    assert loaded["metadata"]["test_count"] > 0
    assert pred.model_family == "timing"
    assert pred.path_delay_ns == pytest.approx(41.0)
    assert pred.path_delay_ps == pytest.approx(41000.0)
    assert pred.path_delay_s == pytest.approx(41.0e-9)


def test_fit_model_bundle_requires_enough_usable_rows():

    config = ChannelModelConfig(
        mode="new",
        model_family="timing",
        feature_names=["temperature"],
        target_name="target",
    )

    with pytest.raises(ValueError):
        fit_model_bundle([{"temperature": 20.0, "target": 1.0}], config)


def test_sklearn_delay_model_polarization_output_is_unit_stokes_vector():
    
    class ConstantEstimator:
        def predict(self, rows):
            return np.asarray([0.5 for _ in rows], dtype=float)

    model = SklearnDelayModel(
        estimator=ConstantEstimator(),
        feature_names=["temperature"],
        target_name="S2",
        model_family="polarization",
    )

    pred = model.predict({"temperature": 20.0})

    assert pred.model_family == "polarization"
    assert pred.stokes_vector is not None
    assert np.linalg.norm(pred.stokes_vector) == pytest.approx(1.0)
    assert pred.poincare_state is not None
