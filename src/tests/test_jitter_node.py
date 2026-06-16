from __future__ import annotations

import numpy as np
import pytest

# This should be unnecessary since sequence is part of the prereqs
pytest.importorskip("sequence")

from sequence.kernel.timeline import Timeline
from simtwo.core.JitterNode import RxNodeWithJitter


def test_default_jitter_samples_are_non_negative():

    rng = np.random.default_rng(56)
    rx = RxNodeWithJitter("B", Timeline(), rng=rng, jitter_std_ps=50_000, jitter_method="default")

    values = [rx._sample_jitter_ps() for _ in range(100)]

    assert len(values) == 100
    assert all(isinstance(value, int) for value in values)
    assert all(value >= 0 for value in values)


def test_perlin_jitter_samples_are_non_negative_and_bounded_by_amplitude():

    rng = np.random.default_rng(56)
    rx = RxNodeWithJitter(
        "B",
        Timeline(),
        rng=rng,
        jitter_std_ps=50_000,
        jitter_method="perlin",
        perlin_step=0.05,
        perlin_amplitude_ps=1000,
    )

    values = [rx._sample_jitter_ps() for _ in range(200)]

    assert len(values) == 200
    assert all(value >= 0 for value in values)
    assert all(value <= 1000 for value in values)
    assert len(set(values)) > 1


def test_jitter_method_aliases_and_invalid_method():

    assert RxNodeWithJitter._normalize_jitter_method("gaussian") == "default"
    assert RxNodeWithJitter._normalize_jitter_method("perlin1d") == "perlin"

    with pytest.raises(ValueError):
        RxNodeWithJitter._normalize_jitter_method("unsupported")
