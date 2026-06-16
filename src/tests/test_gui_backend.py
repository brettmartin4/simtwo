from __future__ import annotations

import pandas as pd
import pytest



from simtwo.core.backends.gui_backend import build_standalone_gui_backend
from simtwo.core.backends.protocol import ChannelModelConfig
from simtwo.core.models.polarization_random_walk import RandomWalkPolarizationModel
from simtwo.core.models.physical_delay import PhysicalDelayModel


def test_backend_loads_csv_and_infers_time_column(tmp_path):

    path = tmp_path / "data.csv"
    pd.DataFrame({"temperature_x": [20.0, 21.0], "posix_time": [100.0, 101.0]}).to_csv(path, index=False)

    backend = build_standalone_gui_backend()
    backend.load_data(str(path))

    assert backend.session.dataset is not None
    assert backend.session.dataset.time_column == "posix_time"
    assert backend.session.current_data_name == "data"


def test_backend_configures_default_timing_with_temperature_x_binding(tmp_path):

    path = tmp_path / "data.csv"
    pd.DataFrame({"temperature_x": [20.0, 21.0], "posix_time": [100.0, 101.0]}).to_csv(path, index=False)

    backend = build_standalone_gui_backend()
    backend.load_data(str(path))
    backend.configure_channel_model(ChannelModelConfig(mode="default", model_family="timing"))

    assert isinstance(backend.session.active_model, PhysicalDelayModel)
    assert backend.session.feature_bindings.mapping == {"temperature": "temperature_x"}


def test_backend_configures_default_polarization_random_walk(tmp_path):

    path = tmp_path / "data.csv"
    pd.DataFrame({"epoch": [0, 1, 2]}).to_csv(path, index=False)

    backend = build_standalone_gui_backend()
    backend.load_data(str(path))
    backend.configure_channel_model(ChannelModelConfig(mode="default", model_family="polarization"))

    assert isinstance(backend.session.active_model, RandomWalkPolarizationModel)
    assert backend.session.feature_bindings.mapping == {}


def test_backend_requires_model_family_for_configuration():

    backend = build_standalone_gui_backend()

    with pytest.raises(RuntimeError):
        backend.configure_channel_model(ChannelModelConfig(mode="default", model_family=""))
