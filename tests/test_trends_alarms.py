"""Tests for trend labelling and alarm rules."""
from bioperator_env.trends import trend_label
from bioperator_env.alarms import evaluate_alarm


def test_stable_history_is_stable():
    assert trend_label([10.0, 10.05, 9.95, 10.02, 10.0]) == "stable"


def test_monotone_rising_history_is_rising():
    label = trend_label([10.0, 10.5, 11.0, 11.5, 12.0])
    assert label in ("rising", "rising_fast")


def test_steep_rising_is_fast():
    label = trend_label([10.0, 12.0, 14.0, 16.0, 18.0])
    assert label == "rising_fast"


def test_steep_falling_is_fast():
    label = trend_label([20.0, 18.0, 16.0, 14.0, 12.0])
    assert label == "falling_fast"


def test_short_history_returns_stable():
    assert trend_label([10.0]) == "stable"


def test_alarm_do_critical_priority():
    a = evaluate_alarm(
        measurements={"dissolved_oxygen_pct": 5.0, "substrate_g_L": 1.0,
                       "temperature_C": 25.0, "pH": 6.5},
        setpoints={"DO_min_safe_pct": 20.0, "substrate_max_g_L": 0.30,
                   "temperature_target_C": 25.0, "pH_target": 6.5},
    )
    assert a == "DO_CRITICAL"


def test_alarm_do_near_low():
    a = evaluate_alarm(
        measurements={"dissolved_oxygen_pct": 18.0, "substrate_g_L": 0.1,
                       "temperature_C": 25.0, "pH": 6.5},
        setpoints={"DO_min_safe_pct": 20.0, "substrate_max_g_L": 0.30,
                   "temperature_target_C": 25.0, "pH_target": 6.5},
    )
    assert a == "DO_NEAR_LOW_LIMIT"


def test_alarm_substrate_overshoot():
    a = evaluate_alarm(
        measurements={"dissolved_oxygen_pct": 25.0, "substrate_g_L": 1.0,
                       "temperature_C": 25.0, "pH": 6.5},
        setpoints={"DO_min_safe_pct": 20.0, "substrate_max_g_L": 0.30,
                   "temperature_target_C": 25.0, "pH_target": 6.5},
    )
    assert a == "S_OVERSHOOT"


def test_alarm_none_when_clean():
    a = evaluate_alarm(
        measurements={"dissolved_oxygen_pct": 25.0, "substrate_g_L": 0.1,
                       "temperature_C": 25.0, "pH": 6.5},
        setpoints={"DO_min_safe_pct": 20.0, "substrate_max_g_L": 0.30,
                   "temperature_target_C": 25.0, "pH_target": 6.5},
    )
    assert a is None
