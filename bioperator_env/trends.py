"""Qualitative trend labels for the plant-console observation.

Real plant operators read trend arrows on the console, not raw delta values.
We mirror that by computing a linear-regression slope over the trailing N
samples and bucketing into one of five labels.
"""
from __future__ import annotations
import numpy as np


_LABELS = ("falling_fast", "falling", "stable", "rising", "rising_fast")


def trend_label(history: list[float], window: int = 5,
                fast_threshold: float = 0.05) -> str:
    """Compute qualitative trend over last `window` samples.

    Parameters
    ----------
    history : list of float
        Recent values, most recent last.
    window : int
        Trailing window size.
    fast_threshold : float
        Slope normalized by mean. Above this -> "fast"; under stable_eps -> "stable".
    """
    if len(history) < 2:
        return "stable"
    h = np.asarray(history[-window:], dtype=np.float64)
    n = len(h)
    if n < 2:
        return "stable"
    x = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(x, h, 1)[0])
    mean_abs = max(float(np.mean(np.abs(h))), 1e-9)
    norm = slope / mean_abs
    if norm > fast_threshold:
        return "rising_fast"
    if norm > fast_threshold * 0.2:
        return "rising"
    if norm < -fast_threshold:
        return "falling_fast"
    if norm < -fast_threshold * 0.2:
        return "falling"
    return "stable"


def all_labels() -> tuple[str, ...]:
    return _LABELS
