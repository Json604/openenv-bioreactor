"""Sequential Batch Control (SBC) recipes - port of fctrl_indpensim.m §SBC.

Piecewise-constant setpoint schedules indexed by simulation step k. Used by:
  - the fixed-recipe baseline agent (the plant's default behavior),
  - the env's reset/step to populate non-agent-controlled MVs.

Lookup convention: returns the SP whose k_max is the smallest one >= the
queried k. If k exceeds the largest k_max, returns the last SP. This matches
the MATLAB `for SQ = 1:length(Recipe_*); if k<=Recipe_*(SQ); ... break;` flow.
"""
from __future__ import annotations


_SCHEDULES: dict[str, list[tuple[int, float]]] = {
    "Fs": [
        (15, 8.0), (60, 15.0), (80, 30.0), (100, 75.0), (120, 150.0),
        (140, 30.0), (160, 37.0), (180, 43.0), (200, 47.0), (220, 51.0),
        (240, 57.0), (260, 61.0), (280, 65.0), (300, 72.0), (320, 76.0),
        (340, 80.0), (360, 84.0), (380, 90.0), (400, 116.0),
        (800, 90.0), (1750, 80.0),
    ],
    "Foil": [
        (20, 22.0), (80, 30.0), (280, 35.0), (300, 34.0), (320, 33.0),
        (340, 32.0), (360, 31.0), (380, 30.0), (400, 29.0), (1750, 23.0),
    ],
    "Fg": [
        (40, 30.0), (100, 42.0), (200, 55.0), (450, 60.0),
        (1000, 75.0), (1250, 65.0), (1750, 60.0),
    ],
    "pressure": [
        (62, 0.6), (125, 0.7), (150, 0.8), (200, 0.9),
        (500, 1.1), (750, 1.0), (1000, 0.9), (1750, 0.9),
    ],
    "F_discharge": [
        (500, 0.0), (510, -4000.0), (650, 0.0), (660, -4000.0),
        (750, 0.0), (760, -4000.0), (850, 0.0), (860, -4000.0),
        (950, 0.0), (960, -4000.0), (1050, 0.0), (1060, -4000.0),
        (1150, 0.0), (1160, -4000.0), (1250, 0.0), (1260, -4000.0),
        (1350, 0.0), (1360, -4000.0), (1750, 0.0),
    ],
    "Fw": [
        (250, 0.0), (375, 500.0), (750, 100.0), (800, 0.0), (850, 400.0),
        (1000, 150.0), (1250, 250.0), (1350, 0.0), (1750, 100.0),
    ],
    "Fpaa": [
        (25, 5.0), (200, 0.0), (1000, 10.0), (1500, 4.0), (1750, 0.0),
    ],
}


def sbc_lookup(name: str, k: int) -> float:
    """Return the SBC setpoint for `name` at simulation step `k`."""
    if name not in _SCHEDULES:
        raise KeyError(f"Unknown SBC variable: {name}")
    for k_max, sp in _SCHEDULES[name]:
        if k <= k_max:
            return sp
    return _SCHEDULES[name][-1][1]


def available_schedules() -> list[str]:
    """List of variable names with SBC schedules."""
    return list(_SCHEDULES.keys())
