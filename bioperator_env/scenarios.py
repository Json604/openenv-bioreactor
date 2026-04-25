"""Task definitions for BioOperatorEnv.

Each task is a configurable starting point + disturbance schedule + reward
weighting. The MVP set focuses on DO/feed disturbance recovery during the
production phase (~hour 40 of the batch).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    """One reset()-able scenario."""
    task_id: str
    description: str
    difficulty: str             # "easy" / "medium" / "hard" / "dev"
    fault_code: int             # passed through to PlantConfig
    start_t_h: float            # episode begins this far into the batch
    max_steps: int = 50
    setpoints: dict = field(default_factory=lambda: {
        "temperature_target_C": 25.0,
        "pH_target": 6.5,
        "DO_min_safe_pct": 20.0,
        "substrate_max_g_L": 0.30,
        "substrate_min_g_L": 0.05,
    })


_TASKS: dict[str, TaskSpec] = {
    "do-recovery-easy": TaskSpec(
        task_id="do-recovery-easy",
        description=("DO/feed disturbance recovery, mild. Substrate fault "
                     "(Faults=3) triggers a gentle DO drop in production "
                     "phase. Agent must rebalance feed and aeration."),
        difficulty="easy",
        fault_code=3,
        start_t_h=40.0,
        max_steps=50,
    ),
    "do-recovery-medium": TaskSpec(
        task_id="do-recovery-medium",
        description=("DO/feed disturbance recovery with productivity goal. "
                     "Substrate fault overlaid with normal disturbances; "
                     "agent must keep DO safe AND keep penicillin growing."),
        difficulty="medium",
        fault_code=3,
        start_t_h=60.0,
        max_steps=50,
    ),
    "aeration-limit-hard": TaskSpec(
        task_id="aeration-limit-hard",
        description=("High-density fermentation with aeration fault. "
                     "Faults=1 forces aeration to drop; agent must use "
                     "feed cut and RPM to compensate."),
        difficulty="hard",
        fault_code=1,
        start_t_h=80.0,
        max_steps=50,
    ),
    "normal-baseline": TaskSpec(
        task_id="normal-baseline",
        description=("No-fault dev scenario. Used for sanity checks and "
                     "fixed-recipe baseline comparisons."),
        difficulty="dev",
        fault_code=0,
        start_t_h=40.0,
        max_steps=50,
    ),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in _TASKS:
        raise KeyError(f"Unknown task_id: {task_id}. Available: {list(_TASKS)}")
    return _TASKS[task_id]


def list_tasks() -> list[str]:
    return list(_TASKS.keys())


def all_specs() -> dict[str, TaskSpec]:
    return dict(_TASKS)
