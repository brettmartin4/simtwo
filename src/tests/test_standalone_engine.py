from __future__ import annotations

import csv

import pandas as pd




from simtwo.core.models.physical_delay import PhysicalDelayModel
from simtwo.core.models.polarization_random_walk import RandomWalkPolarizationModel
from simtwo.core.runtime.session import ExecutionControls, FeatureBindings, LoadedDataset, RuntimeSession
from simtwo.core.standalone.engine import StandaloneEngine


def test_standalone_engine_generates_timing_results_and_callbacks(tmp_path):

    session = RuntimeSession()
    controls = ExecutionControls()
    engine = StandaloneEngine(session=session, controls=controls)
    dataset = LoadedDataset(
        name="timing_demo",
        df=pd.DataFrame({"temperature": [20.0, 21.0, 22.0], "posix_time": [1.0, 2.0, 3.0]}),
        time_column="posix_time",
    )

    session.set_dataset(dataset)
    session.set_model(PhysicalDelayModel(jitter_std_ps=0.0))
    session.feature_bindings = FeatureBindings(mapping={"temperature": "temperature"})

    plot_points = []
    conditions = []

    engine.start(lambda epoch, value: plot_points.append((epoch, value)), conditions.append)

    assert controls.running is False
    assert session.current_epoch == 3
    assert len(session.results) == 3
    assert len(plot_points) == 3
    assert len(conditions) == 1
    assert session.results[0]["predicted_path_delay_s"] is not None
    assert session.results[0]["model_family"] == "timing"

    out_path = tmp_path / "results.csv"
    engine.export_results(str(out_path))

    with out_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert "predicted_path_delay_s" in rows[0]


def test_standalone_engine_generates_polarization_poincare_callbacks():

    session = RuntimeSession()
    controls = ExecutionControls()
    engine = StandaloneEngine(session=session, controls=controls)
    dataset = LoadedDataset(name="polarization_demo", df=pd.DataFrame({"epoch": [0, 1, 2, 3]}))

    session.set_dataset(dataset)
    session.set_model(RandomWalkPolarizationModel(step_std=0.0))
    session.feature_bindings = FeatureBindings(mapping={})

    plot_points = []
    conditions = []
    poincare_states = []

    engine.start(
        lambda epoch, value: plot_points.append((epoch, value)),
        conditions.append,
        poincare_states.append,
    )

    assert len(session.results) == 4
    assert len(plot_points) == 4
    assert len(conditions) == 1
    assert len(poincare_states) == 4
    assert session.results[0]["model_family"] == "polarization"
    assert {"S1", "S2", "S3"}.issubset(session.results[0])
