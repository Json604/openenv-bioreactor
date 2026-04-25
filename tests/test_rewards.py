"""Tests for the 7 reward components + composer."""
import pytest

from bioperator_env.rewards import (
    DEFAULT_WEIGHTS,
    RewardContext,
    compose_reward,
    score_control_effort,
    score_do_safety,
    score_format_validity,
    score_productivity,
    score_stability,
    score_substrate_control,
    score_terminal_yield_bonus,
)


# ---- per-component ----

def test_format_validity_binary():
    assert score_format_validity(True) == 1.0
    assert score_format_validity(False) == 0.0


@pytest.mark.parametrize("do,expected", [
    (30.0, 1.0),
    (25.0, 1.0),
    (22.0, 0.3),
    (18.0, -0.5),
    (10.0, -1.0),
])
def test_do_safety_piecewise(do, expected):
    assert score_do_safety(do, do_min_safe=20.0) == expected


def test_productivity_zero_when_no_growth():
    assert score_productivity(-0.01) == 0.0
    assert score_productivity(0.0) == 0.0


def test_productivity_clamps_to_one():
    assert score_productivity(10.0) == 1.0


def test_substrate_in_band_is_one():
    assert score_substrate_control(0.15, 0.05, 0.30) == 1.0


def test_substrate_below_band_is_negative():
    assert score_substrate_control(0.0, 0.05, 0.30) == -1.0


def test_substrate_far_above_band_is_negative():
    assert score_substrate_control(1.0, 0.05, 0.30) == -1.0


def test_stability_perfect_tracking_is_one():
    s = score_stability(25.0, 6.5)
    assert abs(s - 1.0) < 1e-9


def test_stability_drops_with_drift():
    s = score_stability(28.0, 6.5)
    assert 0.0 < s < 1.0


def test_control_effort_zero_when_idle():
    assert score_control_effort({"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0,
                                  "agitation_delta_rpm": 0}) == 0.0


def test_control_effort_negative_under_action():
    r = score_control_effort({"feed_delta_L_h": 5, "aeration_delta_vvm": 0.10,
                              "agitation_delta_rpm": 5})
    assert r < 0


def test_terminal_yield_only_at_end():
    assert score_terminal_yield_bonus(15.0, is_terminal=False) == 0.0


def test_terminal_yield_at_end_clamped():
    assert score_terminal_yield_bonus(40.0, reference_yield_g_L=20.0, is_terminal=True) == 1.0
    assert score_terminal_yield_bonus(10.0, reference_yield_g_L=20.0, is_terminal=True) == 0.5


# ---- composer ----

def test_compose_reward_returns_total_and_components():
    ctx = RewardContext(
        action_was_valid=True,
        action={"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0},
        do_pct=22.0, do_min_safe=20.0,
        d_penicillin=0.025,
        s_g_L=0.15, s_min=0.05, s_max=0.30,
        temperature_C=25.0, pH=6.5,
        T_target=25.0, pH_target=6.5,
        final_penicillin_g_L=10.0, is_terminal=False,
    )
    total, comps = compose_reward(ctx)
    assert -1.0 <= total <= 1.0
    assert comps.format_validity == 1.0
    assert comps.do_safety == 0.3
    assert comps.substrate_control == 1.0
    assert comps.stability > 0.99


def test_compose_reward_invalid_format_drops_total():
    ctx_valid = RewardContext(
        action_was_valid=True,
        action={"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0},
        do_pct=25.0, do_min_safe=20.0,
        d_penicillin=0.0, s_g_L=0.15, s_min=0.05, s_max=0.30,
        temperature_C=25.0, pH=6.5, T_target=25.0, pH_target=6.5,
        final_penicillin_g_L=0.0, is_terminal=False,
    )
    ctx_invalid = RewardContext(**{**ctx_valid.__dict__, "action_was_valid": False})
    total_v, _ = compose_reward(ctx_valid)
    total_i, _ = compose_reward(ctx_invalid)
    assert total_v > total_i


def test_compose_reward_clamps_to_unit_interval():
    """Even a worst-case state should not produce reward outside [-1, 1]."""
    ctx = RewardContext(
        action_was_valid=False,
        action={"feed_delta_L_h": 5, "aeration_delta_vvm": 0.10, "agitation_delta_rpm": 5},
        do_pct=0.0, do_min_safe=20.0,
        d_penicillin=-1.0,
        s_g_L=2.0, s_min=0.05, s_max=0.30,
        temperature_C=40.0, pH=4.0,
        T_target=25.0, pH_target=6.5,
        final_penicillin_g_L=0.0, is_terminal=False,
    )
    total, _ = compose_reward(ctx)
    assert -1.0 <= total <= 1.0
