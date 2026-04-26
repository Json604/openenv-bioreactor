"""Env snapshot/restore + GRPO prompt dataset builder.

Why this exists: TRL's GRPOTrainer wants a (prompt, reward_fn) pair, not a
custom rollout loop. We bridge the env to GRPO by:
  1) Pre-building a dataset of (prompt, env_snapshot) rows ahead of training.
  2) The reward function (training/reward_fn.py) restores a Plant from the
     snapshot, applies each parsed action, and scores it.

Each row corresponds to one decision point (one env step). We sample many
decision points across many seeds + scenarios to produce a diverse dataset.
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass
from typing import Iterable, Optional

from bioperator_env.env import BioOperatorEnv
from bioperator_env.prompt import build_prompt
from bioperator_env.scenarios import list_tasks


@dataclass
class EnvSnapshot:
    """Minimal serializable env state. Sufficient to recompute reward
    components for any candidate action."""
    task_id: str
    seed: int
    step_count: int
    plant_t_h: float
    plant_k: int
    plant_state: list[float]   # 33-vector
    plant_u_prev: dict
    Fs: float
    Fg_m3min: float
    RPM: float
    last_P: float
    do_min_safe: float
    s_min: float
    s_max: float
    T_target: float
    pH_target: float


def env_to_snapshot(env: BioOperatorEnv) -> EnvSnapshot:
    return EnvSnapshot(
        task_id=env.task_id,
        seed=env.seed,
        step_count=env.step_count,
        plant_t_h=float(env.plant.t_h),
        plant_k=int(env.plant.k),
        plant_state=[float(x) for x in env.plant.state],
        plant_u_prev=dict(env.plant._u_prev),
        Fs=float(env._Fs),
        Fg_m3min=float(env._Fg_m3min),
        RPM=float(env._RPM),
        last_P=float(env._last_P),
        do_min_safe=float(env.spec.setpoints["DO_min_safe_pct"]),
        s_min=float(env.spec.setpoints.get("substrate_min_g_L", 0.05)),
        s_max=float(env.spec.setpoints.get("substrate_max_g_L", 0.30)),
        T_target=float(env.spec.setpoints["temperature_target_C"]),
        pH_target=float(env.spec.setpoints["pH_target"]),
    )


def _is_critical(obs) -> bool:
    """A snapshot is 'critical' (action choice matters) when:
       - an alarm is active, OR
       - DO is below the safe floor + 5% buffer, OR
       - substrate is outside the healthy band, OR
       - DO trend is falling_fast.
    Calm middle-of-batch states get filtered out so GRPO doesn't waste
    rollouts on prompts where any action gives ~the same reward.
    """
    if obs.alarm:
        return True
    DO = obs.measurements.get("dissolved_oxygen_pct", 100.0)
    DO_min = obs.setpoints_or_limits.get("DO_min_safe_pct", 20.0)
    if DO < DO_min + 5.0:
        return True
    S = obs.measurements.get("substrate_g_L", 0.15)
    S_min = obs.setpoints_or_limits.get("substrate_min_g_L", 0.05)
    S_max = obs.setpoints_or_limits.get("substrate_max_g_L", 0.30)
    if S < S_min or S > S_max:
        return True
    if obs.recent_trends.get("DO") in ("falling_fast", "falling"):
        return True
    return False


def build_dataset(num_samples: int = 256,
                  task_ids: Optional[Iterable[str]] = None,
                  steps_per_episode_skip: int = 5,
                  seed: int = 0,
                  critical_only: bool = True,
                  fallback_calm_ratio: float = 0.2) -> list[dict]:
    """Generate `num_samples` (prompt, snapshot) rows for GRPO training.

    `critical_only=True` (default) filters to snapshots where an alarm is
    active, DO is near the floor, substrate is outside the band, or DO is
    falling. This is essential for GRPO learning: at calm states all
    actions produce ~identical rewards, so within-group variance collapses
    to zero and gradients vanish.

    A small `fallback_calm_ratio` of calm states is still allowed to keep
    the dataset diverse (so the model also learns to emit no-op when
    nothing's happening).

    Each row: {
        "prompt": str,
        "snapshot_json": str,    # JSON-serialized EnvSnapshot
        "task_id": str,
        "is_critical": bool,
    }
    """
    rng = random.Random(seed)
    tasks = list(task_ids) if task_ids else [t for t in list_tasks() if t != "normal-baseline"]
    rows = []
    calm_collected = 0
    target_calm = int(num_samples * fallback_calm_ratio) if critical_only else num_samples

    attempts = 0
    max_attempts = num_samples * 50   # bound search to avoid infinite loops
    while len(rows) < num_samples and attempts < max_attempts:
        attempts += 1
        task = rng.choice(tasks)
        ep_seed = rng.randint(0, 1_000_000)
        env = BioOperatorEnv(task_id=task, seed=ep_seed)
        obs = env.reset()
        steps_to_take = rng.randint(0, env.spec.max_steps - 1)
        for _ in range(steps_to_take):
            obs, _, done, _ = env.step({"feed_delta_L_h": 0,
                                         "aeration_delta_vvm": 0.0,
                                         "agitation_delta_rpm": 0})
            if done:
                break
        if env.step_count >= env.spec.max_steps:
            continue

        critical = _is_critical(obs)
        if critical_only:
            if not critical:
                if calm_collected >= target_calm:
                    continue
                calm_collected += 1

        rows.append({
            "prompt": build_prompt(obs),
            "snapshot_json": json.dumps(env_to_snapshot(env).__dict__),
            "task_id": task,
            "is_critical": critical,
        })

    if attempts >= max_attempts and len(rows) < num_samples:
        # Hit the search ceiling. Fall back to fully unfiltered to top up.
        while len(rows) < num_samples:
            task = rng.choice(tasks)
            ep_seed = rng.randint(0, 1_000_000)
            env = BioOperatorEnv(task_id=task, seed=ep_seed)
            obs = env.reset()
            steps_to_take = rng.randint(0, env.spec.max_steps - 1)
            for _ in range(steps_to_take):
                obs, _, done, _ = env.step({"feed_delta_L_h": 0,
                                             "aeration_delta_vvm": 0.0,
                                             "agitation_delta_rpm": 0})
                if done:
                    break
            if env.step_count >= env.spec.max_steps:
                continue
            rows.append({
                "prompt": build_prompt(obs),
                "snapshot_json": json.dumps(env_to_snapshot(env).__dict__),
                "task_id": task,
                "is_critical": _is_critical(obs),
            })
    return rows
