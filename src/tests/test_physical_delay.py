from simtwo.core.models.physical_delay import PhysicalDelayModel

def test_physical_delay_returns_prediction():
    model = PhysicalDelayModel()
    pred = model.predict({"temperature": 20.0})

    assert pred is not None
    assert pred.path_delay_ps is not None
    assert pred.path_delay_ns is not None
    assert pred.path_delay_s is not None
    assert pred.distance_m is not None
    assert pred.plot_value is not None
    assert pred.model_family == "timing"