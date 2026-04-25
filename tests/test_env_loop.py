"""Integration tests for BioOperatorEnv reset/step loop."""
import numpy as np
import pytest

from bioperator_env.env import BioOperatorEnv
from bioperator_env.models import BioOperatorAction, BioOperatorObservation


def test_reset_returns_observation():
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    obs = env.reset()
    assert isinstance(obs, BioOperatorObservation)
    assert "dissolved_oxygen_pct" in obs.measurements
    assert "feed_rate_L_h" in obs.current_controls
    assert obs.batch_phase in ("inoculation", "growth", "production", "stationary")


def test_step_with_dict_action():
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    env.reset()
    obs, reward, done, info = env.step({
        "feed_delta_L_h": 0,
        "aeration_delta_vvm": 0.0,
        "agitation_delta_rpm": 0,
    })
    assert isinstance(obs, BioOperatorObservation)
    assert -1.0 <= reward <= 1.0
    assert done is False
    assert "reward_components" in info
    assert info["format_valid"] is True


def test_step_with_pydantic_action():
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    env.reset()
    a = BioOperatorAction(feed_delta_L_h=5, aeration_delta_vvm=0.10, agitation_delta_rpm=0)
    obs, reward, done, info = env.step(a)
    assert info["format_valid"] is True


def test_step_with_invalid_dict_uses_default_and_flags_format():
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    env.reset()
    obs, reward, done, info = env.step({"feed_delta_L_h": 99,  # invalid literal
                                         "aeration_delta_vvm": 0.0,
                                         "agitation_delta_rpm": 0})
    assert info["format_valid"] is False
    # default no-op was applied, env still advanced
    assert env.step_count == 1


def test_step_with_invalid_json_string():
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    env.reset()
    obs, reward, done, info = env.step("not valid json {{{")
    assert info["format_valid"] is False


def test_50_steps_terminates_on_timeout():
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 100:
        _, _, done, info = env.step({
            "feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0,
        })
        steps += 1
    assert done is True
    assert info["done_reason"] in ("timeout", "safety_violation", "integrator_failure")


def test_state_endpoint_is_complete():
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    env.reset()
    s = env.state()
    assert s.task_id == "do-recovery-medium"
    assert s.seed == 42
    assert len(s.ode_state) == 33


def test_observation_does_not_leak_full_ode_state():
    """Anti-cheat: agent must not see the 33-vector ODE state."""
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    obs = env.reset()
    obs_dict = obs.model_dump()
    # The full state list should NEVER appear in any observation field
    flat = str(obs_dict)
    # 11 morphological / vacuole vars should not be exposed
    for forbidden in ("a0", "a1", "a3", "a4", "n0", "phi0", "vacuole"):
        assert forbidden not in flat.lower(), f"observation leaks hidden state: {forbidden}"
