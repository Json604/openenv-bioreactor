"""Alarm rules for the plant-console observation.

Single string output (or None) so the LLM gets a clear signal. Priority order
matters: critical safety alarms should be reported first; if multiple
conditions trigger simultaneously, the highest-severity one wins.
"""
from __future__ import annotations
from typing import Optional


def evaluate_alarm(measurements: dict, setpoints: dict) -> Optional[str]:
    """Return the highest-priority active alarm, or None.

    Parameters
    ----------
    measurements : dict
        Current process variables; expected keys include
        `dissolved_oxygen_pct`, `substrate_g_L`, `temperature_C`, `pH`.
    setpoints : dict
        Target / limit values; expected keys include
        `DO_min_safe_pct`, `substrate_max_g_L`, `temperature_target_C`,
        `pH_target`.
    """
    DO = measurements.get("dissolved_oxygen_pct")
    DO_min = setpoints.get("DO_min_safe_pct", 20.0)
    if DO is not None and DO < DO_min * 0.5:
        return "DO_CRITICAL"
    if DO is not None and DO < DO_min:
        return "DO_NEAR_LOW_LIMIT"

    S = measurements.get("substrate_g_L")
    S_max = setpoints.get("substrate_max_g_L", 0.30)
    if S is not None and S > S_max * 1.5:
        return "S_OVERSHOOT"

    T = measurements.get("temperature_C")
    T_target = setpoints.get("temperature_target_C", 25.0)
    if T is not None and abs(T - T_target) > 1.5:
        return "TEMP_DRIFT"

    pH = measurements.get("pH")
    pH_target = setpoints.get("pH_target", 6.5)
    if pH is not None and abs(pH - pH_target) > 0.3:
        return "PH_DRIFT"

    return None
