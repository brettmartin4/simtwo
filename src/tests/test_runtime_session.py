from __future__ import annotations

import pandas as pd
import pytest


from simtwo.core.runtime.session import FeatureBindings, LoadedDataset, RuntimeSession
from simtwo.core.models.physical_delay import PhysicalDelayModel


def test_loaded_dataset_iter_records_uses_in_memory_dataframe_rows():

    dataset = LoadedDataset(
        name="demo",
        df=pd.DataFrame({"temperature": [20.0, 21.0], "posix_time": [100.0, 101.0]}),
    )

    rows = list(dataset.iter_records())

    assert rows == [
        {"temperature": 20.0, "posix_time": 100.0},
        {"temperature": 21.0, "posix_time": 101.0},
    ]


def test_feature_bindings_extracts_mapped_numeric_values():

    bindings = FeatureBindings(mapping={"temperature": "temperature_x"})

    features = bindings.extract({"temperature_x": "22.5", "unused": 10})

    assert features == {"temperature": 22.5}
    assert bindings.has_binding("temperature")
    assert bindings.dataset_column_for("temperature") == "temperature_x"


def test_runtime_session_requires_dataset_and_model():
    
    session = RuntimeSession()

    with pytest.raises(RuntimeError):
        session.require_dataset()
    with pytest.raises(RuntimeError):
        session.require_model()

    dataset = LoadedDataset(name="demo", df=pd.DataFrame({"temperature": [20.0]}))
    model = PhysicalDelayModel()
    session.set_dataset(dataset)
    session.set_model(model)

    assert session.require_dataset() is dataset
    assert session.require_model() is model
    assert session.current_data_name == "demo"
    assert session.current_model_name == model.name
