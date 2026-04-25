"""Tests for the training pipeline pieces that don't need a GPU."""
import json

from training.reward_fn import _parse_action, format_only_reward_fn, reward_fn
from training.rollout import build_dataset, env_to_snapshot
from bioperator_env.env import BioOperatorEnv


def test_parse_action_extracts_valid_json():
    text = (
        'Sure, here is the action:\n'
        '{"feed_delta_L_h": -5, "aeration_delta_vvm": 0.10, '
        '"agitation_delta_rpm": 0, "reason": "DO low"}\n'
        'Hope that helps.'
    )
    action, valid = _parse_action(text)
    assert valid is True
    assert action["feed_delta_L_h"] == -5
    assert action["aeration_delta_vvm"] == 0.10


def test_parse_action_handles_no_json():
    text = "I think we should reduce the feed."
    _, valid = _parse_action(text)
    assert valid is False


def test_parse_action_rejects_bad_values():
    text = '{"feed_delta_L_h": 999, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0}'
    _, valid = _parse_action(text)
    assert valid is False


def test_env_to_snapshot_roundtrip():
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    env.reset()
    env.step({"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0})
    snap = env_to_snapshot(env)
    assert snap.task_id == "do-recovery-medium"
    assert len(snap.plant_state) == 33
    # serializable
    s = json.dumps(snap.__dict__)
    assert isinstance(s, str)


def test_build_dataset_smoke():
    rows = build_dataset(num_samples=4, task_ids=["do-recovery-easy"], seed=0)
    assert len(rows) == 4
    for r in rows:
        assert "prompt" in r
        assert "snapshot_json" in r
        assert "task_id" in r
        snap = json.loads(r["snapshot_json"])
        assert "plant_state" in snap
        assert len(snap["plant_state"]) == 33


def test_format_only_reward_zero_for_garbage():
    rewards = format_only_reward_fn(["I think...", "..."])
    assert rewards == [0.0, 0.0]


def test_format_only_reward_one_for_valid():
    valid = ('Action: {"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, '
             '"agitation_delta_rpm": 0}')
    rewards = format_only_reward_fn([valid])
    assert rewards == [1.0]


def test_reward_fn_runs_on_synthetic_dataset():
    rows = build_dataset(num_samples=2, task_ids=["do-recovery-easy"], seed=0)
    completions = [
        '{"feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0}',
        '{"feed_delta_L_h": -5, "aeration_delta_vvm": 0.10, "agitation_delta_rpm": 0}',
    ]
    rewards = reward_fn(
        completions,
        snapshot_json=[r["snapshot_json"] for r in rows],
    )
    assert len(rewards) == 2
    for r in rewards:
        assert -1.0 <= r <= 1.0
