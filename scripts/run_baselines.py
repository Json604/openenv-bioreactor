"""Run a set of baseline agents on a set of scenarios, dump per-episode CSVs.

Usage:
    python scripts/run_baselines.py \
        --agents random,fixed_recipe,rule_based \
        --tasks do-recovery-easy,do-recovery-medium,aeration-limit-hard \
        --seeds 0,1,2,3,4 \
        --out results/

Heavy LLM baselines (untrained_qwen, claude_zero_shot, trained_qwen) are
opt-in — they only run when listed in --agents and require GPU / API key.

Each episode produces one row in `baseline_<agent>.csv`:
    seed, task_id, total_reward, steps, success, safety_violations,
    final_DO_pct, final_S_g_L, final_P_g_L, final_X_g_L
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Type

from baselines.fixed_recipe_agent import FixedRecipeAgent
from baselines.random_agent import RandomAgent
from baselines.rule_based_agent import RuleBasedAgent
from bioperator_env.env import BioOperatorEnv


_AGENT_FACTORIES: dict[str, callable] = {
    "random":       lambda: RandomAgent(seed=0),
    "fixed_recipe": lambda: FixedRecipeAgent(),
    "rule_based":   lambda: RuleBasedAgent(),
}


def _maybe_register_heavy_agents() -> None:
    """Lazily register LLM-backed agents only when requested."""
    try:
        from baselines.untrained_qwen_agent import UntrainedQwenAgent
        _AGENT_FACTORIES["untrained_qwen"] = lambda: UntrainedQwenAgent()
    except Exception:
        pass
    try:
        from baselines.claude_zero_shot_agent import ClaudeZeroShotAgent
        _AGENT_FACTORIES["claude_zero_shot"] = lambda: ClaudeZeroShotAgent()
    except Exception:
        pass
    try:
        from baselines.trained_qwen_agent import TrainedQwenAgent
        _AGENT_FACTORIES["trained_qwen"] = lambda: TrainedQwenAgent()
    except Exception:
        pass


def run_one_episode(agent, task_id: str, seed: int) -> dict:
    env = BioOperatorEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    do_min_safe = obs.setpoints_or_limits.get("DO_min_safe_pct", 20.0)
    s_min = obs.setpoints_or_limits.get("substrate_min_g_L", 0.05)
    s_max = obs.setpoints_or_limits.get("substrate_max_g_L", 0.30)
    done = False
    total = 0.0
    do_steps_above_floor = 0
    sub_steps_in_band = 0
    n_steps = 0
    format_valid_count = 0
    while not done:
        action = agent.act(obs)
        obs, r, done, info = env.step(action)
        total += r
        n_steps += 1
        if obs.measurements["dissolved_oxygen_pct"] >= do_min_safe:
            do_steps_above_floor += 1
        if s_min <= obs.measurements["substrate_g_L"] <= s_max:
            sub_steps_in_band += 1
        if info.get("format_valid", False):
            format_valid_count += 1
    state = env.state()
    return {
        "agent": agent.name,
        "seed": seed,
        "task_id": task_id,
        "total_reward": total,
        "steps": state.step_count,
        "success": int(info.get("success", False)),
        "safety_violations": state.safety_violations,
        "do_above_floor_pct": (do_steps_above_floor / n_steps * 100.0) if n_steps else 0.0,
        "substrate_in_band_pct": (sub_steps_in_band / n_steps * 100.0) if n_steps else 0.0,
        "format_valid_pct": (format_valid_count / n_steps * 100.0) if n_steps else 0.0,
        "final_DO_pct": obs.measurements["dissolved_oxygen_pct"],
        "final_S_g_L": obs.measurements["substrate_g_L"],
        "final_P_g_L": float(state.ode_state[3]),
        "final_X_g_L": float(sum(state.ode_state[11:15])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default="random,fixed_recipe,rule_based")
    parser.add_argument("--tasks", default="do-recovery-medium")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out", default="results/")
    args = parser.parse_args()

    _maybe_register_heavy_agents()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for agent_name in agents:
        if agent_name not in _AGENT_FACTORIES:
            print(f"[skip] unknown agent: {agent_name}")
            continue
        agent = _AGENT_FACTORIES[agent_name]()
        rows = []
        for task in tasks:
            for seed in seeds:
                print(f"  [{agent_name}] {task} seed={seed} ...", flush=True)
                rows.append(run_one_episode(agent, task, seed))
        out_path = out_dir / f"baseline_{agent_name}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  -> wrote {out_path} ({len(rows)} episodes)")


if __name__ == "__main__":
    main()
