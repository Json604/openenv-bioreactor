"""Smoke tests for the lightweight (no-LLM) baselines."""
from baselines.fixed_recipe_agent import FixedRecipeAgent
from baselines.random_agent import RandomAgent
from baselines.rule_based_agent import RuleBasedAgent
from bioperator_env.env import BioOperatorEnv
from bioperator_env.models import BioOperatorAction


def _act_validates(action_dict: dict) -> bool:
    try:
        BioOperatorAction(**action_dict)
        return True
    except Exception:
        return False


def test_random_emits_valid_actions():
    agent = RandomAgent(seed=42)
    env = BioOperatorEnv(task_id="do-recovery-easy", seed=42)
    obs = env.reset()
    for _ in range(20):
        a = agent.act(obs)
        assert _act_validates(a)
        obs, _, done, _ = env.step(a)
        if done:
            break


def test_fixed_recipe_emits_zero_deltas():
    agent = FixedRecipeAgent()
    env = BioOperatorEnv(task_id="normal-baseline", seed=42)
    obs = env.reset()
    a = agent.act(obs)
    assert _act_validates(a)
    assert a["feed_delta_L_h"] == 0
    assert a["aeration_delta_vvm"] == 0.0
    assert a["agitation_delta_rpm"] == 0


def test_rule_based_responds_to_low_DO():
    agent = RuleBasedAgent()
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    obs = env.reset()
    # Force a low-DO observation
    obs.measurements["dissolved_oxygen_pct"] = 18.0
    obs.recent_trends["DO"] = "falling_fast"
    a = agent.act(obs)
    assert _act_validates(a)
    assert a["aeration_delta_vvm"] >= 0.10  # bumps aeration
    assert a["feed_delta_L_h"] <= 0          # cuts or holds feed


def test_full_episode_with_rule_based():
    agent = RuleBasedAgent()
    env = BioOperatorEnv(task_id="do-recovery-medium", seed=42)
    obs = env.reset()
    done = False
    total = 0.0
    while not done:
        a = agent.act(obs)
        obs, r, done, info = env.step(a)
        total += r
    assert env.step_count > 0
    assert -1.0 * env.spec.max_steps <= total <= 1.0 * env.spec.max_steps
