"""Tests for disturbance signal generation (port of indpensim_run.m §dist)."""
import numpy as np

from bioperator_env.plant.disturbances import generate_disturbances


def test_generate_disturbances_shape():
    d = generate_disturbances(T=230.0, h=0.2, seed=42)
    n = int(round(230.0 / 0.2)) + 1
    expected_keys = {"distMuP", "distMuX", "distcs", "distcoil", "distabc",
                     "distPAA", "distTcin", "distO_2in"}
    assert set(d.keys()) == expected_keys
    for k, sig in d.items():
        assert sig.shape == (n,), f"{k}: expected ({n},), got {sig.shape}"


def test_generate_disturbances_deterministic():
    d1 = generate_disturbances(T=10.0, h=0.2, seed=7)
    d2 = generate_disturbances(T=10.0, h=0.2, seed=7)
    for k in d1:
        assert np.array_equal(d1[k], d2[k])


def test_generate_disturbances_different_seeds_differ():
    d1 = generate_disturbances(T=10.0, h=0.2, seed=1)
    d2 = generate_disturbances(T=10.0, h=0.2, seed=2)
    assert not np.array_equal(d1["distMuP"], d2["distMuP"])


def test_generate_disturbances_low_freq():
    """Low-pass filter (b=0.005, a=[1, -0.995]) means high-freq energy is small."""
    d = generate_disturbances(T=200.0, h=0.2, seed=0)
    sig = d["distMuP"]
    diffs = np.diff(sig)
    # adjacent-sample jumps should be much smaller than overall spread
    assert np.max(np.abs(diffs)) < 5 * np.std(sig)
