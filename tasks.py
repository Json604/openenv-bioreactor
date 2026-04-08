"""Task definitions for the bioreactor OpenEnv environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Disturbance:
    oxygen_delta: float = 0.0
    mixing_delta: float = 0.0
    nutrient_delta: float = 0.0
    biomass_delta: float = 0.0
    byproduct_delta: float = 0.0
    feed_rate_delta: float = 0.0


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
    initial_biomass: float
    initial_byproduct: float
    initial_stirrer: float
    initial_oxygen_input: float
    initial_feed_rate: float
    target_oxygen: float = 0.68
    target_mixing: float = 0.78
    target_nutrient: float = 0.28
    target_biomass: float = 0.72
    max_safe_byproduct: float = 0.30
    terminal_biomass_target: float = 0.72
    terminal_byproduct_limit: float = 0.30
    terminal_nutrient_range: tuple[float, float] = (0.08, 0.30)
    terminal_oxygen_floor: float = 0.45
    oxygen_load: float = 1.0
    oxygen_transfer: float = 1.0
    viscosity_base: float = 1.0
    nutrient_use: float = 1.0
    growth_rate: float = 1.0
    actuator_penalty: float = 0.015
    foam_penalty_threshold: float = 0.65
    shear_threshold: float = 0.72
    noise: float = 0.0
    seed: int = 7
    feed_schedule: dict[int, float] = field(default_factory=dict)
    disturbances: dict[int, Disturbance] = field(default_factory=dict)


TASKS: dict[str, BioreactorTask] = {
    "startup-stabilization-easy": BioreactorTask(
        task_id="startup-stabilization-easy",
        name="Startup Stabilization",
        difficulty="easy",
        description=(
            "Recover a fresh batch after inoculation. Hold dissolved oxygen and "
            "mixing in the productive band while growing biomass without creating "
            "excess byproduct."
        ),
        max_steps=50,
        initial_oxygen=0.58,
        initial_mixing=0.62,
        initial_nutrient=0.82,
        initial_biomass=0.22,
        initial_byproduct=0.08,
        initial_stirrer=0.52,
        initial_oxygen_input=0.50,
        initial_feed_rate=0.18,
        oxygen_load=0.85,
        oxygen_transfer=0.95,
        viscosity_base=0.78,
        nutrient_use=0.82,
        growth_rate=1.12,
        actuator_penalty=0.010,
        foam_penalty_threshold=0.74,
        shear_threshold=0.78,
        noise=0.0015,
        seed=11,
        target_biomass=0.54,
        max_safe_byproduct=0.22,
        terminal_biomass_target=0.56,
        terminal_byproduct_limit=0.18,
        terminal_nutrient_range=(0.10, 0.32),
        terminal_oxygen_floor=0.50,
        feed_schedule={18: 0.16, 34: 0.10},
    ),
    "fed-batch-optimization-medium": BioreactorTask(
        task_id="fed-batch-optimization-medium",
        name="Fed-Batch Productivity Optimization",
        difficulty="medium",
        description=(
            "A fed-batch campaign enters aggressive growth. Feed pulses increase "
            "oxygen demand and viscosity, so the controller must protect growth "
            "while keeping byproduct and foam under control."
        ),
        max_steps=50,
        initial_oxygen=0.46,
        initial_mixing=0.48,
        initial_nutrient=0.52,
        initial_biomass=0.34,
        initial_byproduct=0.16,
        initial_stirrer=0.48,
        initial_oxygen_input=0.46,
        initial_feed_rate=0.18,
        oxygen_load=1.12,
        oxygen_transfer=1.00,
        viscosity_base=0.98,
        nutrient_use=1.05,
        growth_rate=1.00,
        actuator_penalty=0.018,
        foam_penalty_threshold=0.66,
        shear_threshold=0.74,
        noise=0.003,
        seed=23,
        target_biomass=0.78,
        max_safe_byproduct=0.30,
        terminal_biomass_target=0.80,
        terminal_byproduct_limit=0.24,
        terminal_nutrient_range=(0.08, 0.22),
        terminal_oxygen_floor=0.44,
        feed_schedule={10: 0.22, 20: 0.28, 33: 0.24, 42: 0.18},
        disturbances={
            16: Disturbance(oxygen_delta=-0.06, nutrient_delta=0.07),
            31: Disturbance(mixing_delta=-0.08, byproduct_delta=0.05),
        },
    ),
    "oxygen-limited-recovery-hard": BioreactorTask(
        task_id="oxygen-limited-recovery-hard",
        name="Oxygen-Limited High-Density Recovery",
        difficulty="hard",
        description=(
            "Late-stage high-cell-density fermentation runs near oxygen limitation. "
            "The agent must recover from disturbances, avoid shear damage, and finish "
            "with high biomass and acceptable metabolite quality."
        ),
        max_steps=50,
        initial_oxygen=0.38,
        initial_mixing=0.40,
        initial_nutrient=0.44,
        initial_biomass=0.56,
        initial_byproduct=0.26,
        initial_stirrer=0.44,
        initial_oxygen_input=0.42,
        initial_feed_rate=0.20,
        oxygen_load=1.34,
        oxygen_transfer=1.03,
        viscosity_base=1.18,
        nutrient_use=1.18,
        growth_rate=1.05,
        actuator_penalty=0.028,
        foam_penalty_threshold=0.61,
        shear_threshold=0.70,
        noise=0.0045,
        seed=41,
        target_biomass=0.90,
        max_safe_byproduct=0.34,
        terminal_biomass_target=0.88,
        terminal_byproduct_limit=0.28,
        terminal_nutrient_range=(0.06, 0.18),
        terminal_oxygen_floor=0.40,
        feed_schedule={8: 0.26, 18: 0.30, 29: 0.24, 40: 0.16},
        disturbances={
            12: Disturbance(oxygen_delta=-0.10, byproduct_delta=0.03),
            24: Disturbance(mixing_delta=-0.10, feed_rate_delta=0.05),
            37: Disturbance(oxygen_delta=-0.06, nutrient_delta=0.05, byproduct_delta=0.04),
        },
    ),
}


DEFAULT_TASK_ID = "startup-stabilization-easy"


def get_task(task_id: str | None) -> BioreactorTask:
    if task_id is None:
        return TASKS[DEFAULT_TASK_ID]
    if task_id not in TASKS:
        valid = ", ".join(sorted(TASKS))
        raise ValueError(f"unknown task_id={task_id!r}; valid tasks: {valid}")
    return TASKS[task_id]
