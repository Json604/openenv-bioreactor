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


def build_dataset(num_samples: int = 256,
                  task_ids: Optional[Iterable[str]] = None,
                  steps_per_episode_skip: int = 5,
                  seed: int = 0) -> list[dict]:
    """Generate `num_samples` (prompt, snapshot) rows for GRPO training.

    Each row: {
        "prompt": str,
        "snapshot_json": str,    # JSON-serialized EnvSnapshot
        "task_id": str,
    }
    """
    rng = random.Random(seed)
    tasks = list(task_ids) if task_ids else [t for t in list_tasks() if t != "normal-baseline"]
    rows = []
    while len(rows) < num_samples:
        task = rng.choice(tasks)
        ep_seed = rng.randint(0, 1_000_000)
        env = BioOperatorEnv(task_id=task, seed=ep_seed)
        obs = env.reset()
        # Sample a step uniformly within the episode
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
        })
    return rows
