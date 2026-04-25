"""Process disturbance signal generators.

Ports indpensim_run.m lines 143-183. Eight disturbance channels are generated
as IIR low-pass filtered Gaussian noise (b=0.005, a=[1, -0.995]) so disturbances
are slow-varying, like real plant drift. Each channel has its own gain.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter


_FILTER_B = np.array([1 - 0.995])      # b1
_FILTER_A = np.array([1.0, -0.995])    # a1


# Channel name -> standard deviation gain applied to unit Gaussian noise
# Values come directly from indpensim_run.m lines 148-183.
_GAINS: dict[str, float] = {
    "distMuP":   0.03,
    "distMuX":   0.25,
    "distcs":    5.0 * 300.0,    # SD ~+/- 15 g/L spread shrunk by filter
    "distcoil":  300.0,
    "distabc":   0.2,
    "distPAA":   300000.0,
    "distTcin":  100.0,
    "distO_2in": 0.02,
}


def generate_disturbances(T: float, h: float, seed: int) -> dict[str, np.ndarray]:
    """Return dict of 8 disturbance time-series, length N = T/h + 1.

    Parameters
    ----------
    T : float
        Total simulated time in hours (e.g., 230).
    h : float
        Sampling period in hours (e.g., 0.2).
    seed : int
        Random seed; same seed -> same disturbances.
    """
    n = int(round(T / h)) + 1
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name, gain in _GAINS.items():
        v = rng.standard_normal(n)
        out[name] = lfilter(_FILTER_B, _FILTER_A, gain * v).astype(np.float64)
    return out
