"""Tests for pydantic Action / Observation / State models."""
import pytest
from pydantic import ValidationError

from bioperator_env.models import (
    BioOperatorAction,
    BioOperatorObservation,
    BioOperatorState,
    RewardComponents,
    StepInfo,
)


# ---- Action ----

def test_action_accepts_canonical_values():
    a = BioOperatorAction(
        feed_delta_L_h=5,
        aeration_delta_vvm=0.10,
        agitation_delta_rpm=-5,
        reason="DO falling, cut feed",
    )
    assert a.feed_delta_L_h == 5
    assert a.aeration_delta_vvm == 0.10
    assert a.agitation_delta_rpm == -5
    assert a.reason == "DO falling, cut feed"


def test_action_reason_is_optional():
    a = BioOperatorAction(feed_delta_L_h=0, aeration_delta_vvm=0.0, agitation_delta_rpm=0)
    assert a.reason is None


def test_action_rejects_out_of_set_feed():
    with pytest.raises(ValidationError):
        BioOperatorAction(feed_delta_L_h=3, aeration_delta_vvm=0.0, agitation_delta_rpm=0)


def test_action_rejects_out_of_set_aeration():
    with pytest.raises(ValidationError):
        BioOperatorAction(feed_delta_L_h=0, aeration_delta_vvm=0.05, agitation_delta_rpm=0)


def test_action_rejects_out_of_set_rpm():
    with pytest.raises(ValidationError):
        BioOperatorAction(feed_delta_L_h=0, aeration_delta_vvm=0.0, agitation_delta_rpm=10)


def test_action_truncates_long_reason():
    long_reason = "x" * 500
    a = BioOperatorAction(
        feed_delta_L_h=0, aeration_delta_vvm=0.0, agitation_delta_rpm=0, reason=long_reason
    )
    assert a.reason is not None
    assert len(a.reason) <= 200


def test_action_27_arms_are_all_constructible():
    """Verify the full discrete 3x3x3 product is constructible."""
    feeds = [-5, 0, 5]
    aers = [-0.10, 0.0, 0.10]
    rpms = [-5, 0, 5]
    n = 0
    for f in feeds:
        for a in aers:
            for r in rpms:
                BioOperatorAction(feed_delta_L_h=f, aeration_delta_vvm=a, agitation_delta_rpm=r)
                n += 1
    assert n == 27


# ---- Observation ----

def _minimal_obs_kwargs() -> dict:
    return dict(
        time_h=42.0,
        batch_phase="production",
        measurements={
            "temperature_C": 25.0,
            "pH": 6.5,
            "dissolved_oxygen_pct": 22.0,
            "substrate_g_L": 0.15,
            "volume_L": 80000.0,
            "OUR": 0.5,
            "CER": 0.4,
            "CO2_outgas_pct": 4.0,
            "O2_outgas_pct": 19.5,
        },
        setpoints_or_limits={
            "temperature_target_C": 25.0,
            "pH_target": 6.5,
            "DO_min_safe_pct": 20.0,
            "substrate_max_g_L": 0.30,
            "substrate_min_g_L": 0.05,
        },
        current_controls={
            "feed_rate_L_h": 80.0,
            "aeration_rate_vvm": 0.85,
            "agitation_rpm": 100.0,
            "cooling_valve_pct": 45.0,
            "pressure_bar": 0.9,
        },
        recent_trends={
            "DO": "stable",
            "pH": "stable",
            "temperature": "stable",
            "substrate": "stable",
        },
        alarm=None,
        previous_action=None,
        offline_lab=None,
        instruction="...",
    )


def test_observation_minimal():
    obs = BioOperatorObservation(**_minimal_obs_kwargs())
    assert obs.time_h == 42.0
    assert obs.batch_phase == "production"


def test_observation_phase_literal_enforced():
    kwargs = _minimal_obs_kwargs()
    kwargs["batch_phase"] = "invalid_phase"
    with pytest.raises(ValidationError):
        BioOperatorObservation(**kwargs)


def test_observation_serializable_to_json():
    obs = BioOperatorObservation(**_minimal_obs_kwargs())
    s = obs.model_dump_json()
    assert "production" in s
    assert "dissolved_oxygen_pct" in s


# ---- RewardComponents / StepInfo ----

def test_reward_components_are_floats():
    rc = RewardComponents(
        format_validity=1.0,
        do_safety=0.5,
        productivity=0.3,
        substrate_control=0.7,
        stability=0.9,
        control_effort=-0.1,
        terminal_yield_bonus=0.0,
    )
    assert rc.format_validity == 1.0
    assert rc.terminal_yield_bonus == 0.0


def test_step_info_carries_reward_breakdown():
    info = StepInfo(
        reward_total=0.4,
        reward_components=RewardComponents(
            format_validity=1.0, do_safety=0.5, productivity=0.3,
            substrate_control=0.7, stability=0.9, control_effort=-0.1,
            terminal_yield_bonus=0.0,
        ),
        safety_violation=False,
        success=False,
        done_reason="",
    )
    assert info.reward_total == 0.4
    assert info.safety_violation is False


# ---- State ----

def test_state_default_construction():
    s = BioOperatorState(
        task_id="do-recovery-medium",
        seed=42,
        step_count=0,
        time_h=40.0,
        ode_state=[0.0] * 33,
    )
    assert s.task_id == "do-recovery-medium"
    assert s.cumulative_reward == 0.0
    assert len(s.ode_state) == 33
