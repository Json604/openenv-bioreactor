"""Tests for scenario specs and prompt builder."""
import json
import pytest

from bioperator_env.scenarios import get_task, list_tasks, TaskSpec
from bioperator_env.prompt import build_prompt, format_observation, SYSTEM_PROMPT
from bioperator_env.models import BioOperatorObservation


def test_list_tasks_has_four_scenarios():
    tasks = list_tasks()
    assert "do-recovery-easy" in tasks
    assert "do-recovery-medium" in tasks
    assert "aeration-limit-hard" in tasks
    assert "normal-baseline" in tasks
    assert len(tasks) == 4


def test_get_task_returns_spec():
    spec = get_task("do-recovery-medium")
    assert isinstance(spec, TaskSpec)
    assert spec.fault_code == 3
    assert spec.difficulty == "medium"


def test_get_task_unknown_raises():
    with pytest.raises(KeyError):
        get_task("frobnicate")


def _obs() -> BioOperatorObservation:
    return BioOperatorObservation(
        time_h=42.0,
        batch_phase="production",
        measurements={
            "temperature_C": 25.0, "pH": 6.5,
            "dissolved_oxygen_pct": 22.0,
            "substrate_g_L": 0.15, "volume_L": 80000.0,
            "OUR": 0.5, "CER": 0.4,
            "CO2_outgas_pct": 4.0, "O2_outgas_pct": 19.5,
        },
        setpoints_or_limits={
            "temperature_target_C": 25.0, "pH_target": 6.5,
            "DO_min_safe_pct": 20.0,
            "substrate_max_g_L": 0.30, "substrate_min_g_L": 0.05,
        },
        current_controls={
            "feed_rate_L_h": 80.0, "aeration_rate_vvm": 0.85,
            "agitation_rpm": 100.0, "cooling_valve_pct": 45.0,
            "pressure_bar": 0.9,
        },
        recent_trends={"DO": "stable", "pH": "stable",
                       "temperature": "stable", "substrate": "stable"},
        alarm=None,
        previous_action=None,
        offline_lab=None,
        instruction=SYSTEM_PROMPT,
    )


def test_format_observation_is_valid_json():
    s = format_observation(_obs())
    parsed = json.loads(s)
    assert parsed["time_h"] == 42.0
    assert parsed["batch_phase"] == "production"


def test_build_prompt_contains_all_sections():
    p = build_prompt(_obs())
    assert "<system>" in p
    assert "<observation>" in p
    assert "<action>" in p
    assert "DO_min_safe_pct" in p or "dissolved_oxygen_pct" in p
