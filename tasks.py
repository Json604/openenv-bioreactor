"""Task definitions for the bioreactor OpenEnv environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Disturbance:
    oxygen_delta: float = 0.0
    mixing_delta: float = 0.0
    nutrient_delta: float = 0.0


@dataclass(frozen=True)
class BioreactorTask:
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int
    initial_oxygen: float
    initial_mixing: float
    initial_nutrient: float
    initial_stirrer: float
    initial_oxygen_input: float
    target_oxygen: float = 0.70
    target_mixing: float = 0.80
    target_nutrient: float = 0.60
    oxygen_load: float = 1.0
    viscosity: float = 1.0
    nutrient_use: float = 1.0
    actuator_penalty: float = 0.02
    foam_penalty_threshold: float = 1.65
    noise: float = 0.0
    seed: int = 7
    disturbances: dict[int, Disturbance] = field(default_factory=dict)


TASKS: dict[str, BioreactorTask] = {
    "batch-startup-easy": BioreactorTask(
        task_id="batch-startup-easy",
        name="Batch Startup Stabilization",
        difficulty="easy",
        description=(
            "Warm-started batch reactor. The agent should hold oxygen near 0.70 "
            "and mixing near 0.80 while nutrients drift toward 0.60."
        ),
        max_steps=50,
        initial_oxygen=0.62,
        initial_mixing=0.68,
        initial_nutrient=0.72,
        initial_stirrer=0.55,
        initial_oxygen_input=0.55,
        oxygen_load=0.85,
        viscosity=0.80,
        nutrient_use=0.80,
        actuator_penalty=0.010,
        noise=0.002,
        seed=11,
    ),
    "fed-batch-shift-medium": BioreactorTask(
        task_id="fed-batch-shift-medium",
        name="Fed-Batch Disturbance Recovery",
        difficulty="medium",
        description=(
            "A fed-batch run starts oxygen-poor and under-mixed. Feed events and "
            "viscosity shifts require active recovery without over-driving oxygen."
        ),
        max_steps=50,
        initial_oxygen=0.48,
        initial_mixing=0.45,
        initial_nutrient=0.86,
        initial_stirrer=0.45,
        initial_oxygen_input=0.45,
        oxygen_load=1.12,
        viscosity=1.10,
        nutrient_use=1.05,
        actuator_penalty=0.020,
        noise=0.004,
        seed=23,
        disturbances={
            16: Disturbance(oxygen_delta=-0.08, nutrient_delta=0.08),
            32: Disturbance(mixing_delta=-0.10, nutrient_delta=0.05),
        },
    ),
    "high-density-hard": BioreactorTask(
        task_id="high-density-hard",
        name="High-Density Fermentation",
        difficulty="hard",
        description=(
            "High cell density increases oxygen demand and viscosity. Aggressive "
            "actuation can create foam penalties, so the agent must balance both loops."
        ),
        max_steps=50,
        initial_oxygen=0.42,
        initial_mixing=0.36,
        initial_nutrient=0.92,
        initial_stirrer=0.40,
        initial_oxygen_input=0.40,
        oxygen_load=1.35,
        viscosity=1.30,
        nutrient_use=1.20,
        actuator_penalty=0.035,
        foam_penalty_threshold=1.45,
        noise=0.006,
        seed=41,
        disturbances={
            12: Disturbance(oxygen_delta=-0.10),
            24: Disturbance(mixing_delta=-0.12, nutrient_delta=0.04),
            38: Disturbance(oxygen_delta=-0.06, mixing_delta=-0.08),
        },
    ),
}


DEFAULT_TASK_ID = "batch-startup-easy"


def get_task(task_id: str | None) -> BioreactorTask:
    if task_id is None:
        return TASKS[DEFAULT_TASK_ID]
    if task_id not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"unknown task_id={task_id!r}; valid tasks: {valid}")
    return TASKS[task_id]
