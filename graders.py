"""Deterministic graders for bioreactor control trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from tasks import BioreactorTask


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    oxygen: float
    mixing: float
    nutrient: float
    biomass: float
    byproduct: float
    foam_risk: float
    shear_damage: float
    reward: float
    collapsed: bool


def classify_phase(step: int, max_steps: int) -> str:
    if step <= max_steps // 3:
        return "startup"
    if step <= (2 * max_steps) // 3:
        return "growth"
    return "stress"


def score_signal(value: float, target: float, tolerance: float) -> float:
    return clamp(1.0 - abs(value - target) / tolerance)


def grade_step(
    *,
    oxygen: float,
    mixing: float,
    nutrient: float,
    biomass: float,
    byproduct: float,
    biomass_growth: float,
    stirrer_speed: float,
    oxygen_input: float,
    feed_rate: float,
    foam_risk: float,
    shear_damage: float,
    task: BioreactorTask,
) -> tuple[float, dict[str, float]]:
    oxygen_score = score_signal(oxygen, task.target_oxygen, 0.30)
    mixing_score = score_signal(mixing, task.target_mixing, 0.30)
    nutrient_score = score_signal(nutrient, task.target_nutrient, 0.35)
    production_score = clamp(biomass / max(task.target_biomass, 1e-6))
    purity_score = clamp(1.0 - byproduct / max(task.max_safe_byproduct, 1e-6))
    safety_score = clamp(1.0 - 0.60 * foam_risk - 0.85 * shear_damage)
    growth_bonus = clamp(biomass_growth * 12.0)
    actuator_penalty = task.actuator_penalty * (
        0.40 * stirrer_speed + 0.34 * oxygen_input + 0.26 * feed_rate
    )
    foam_penalty = clamp(max(0.0, foam_risk - task.foam_penalty_threshold) * 0.6, 0.0, 0.4)

    reward = (
        0.19 * oxygen_score
        + 0.16 * mixing_score
        + 0.10 * nutrient_score
        + 0.25 * production_score
        + 0.12 * purity_score
        + 0.10 * safety_score
        + 0.08 * growth_bonus
        - actuator_penalty
        - foam_penalty
    )
    reward = clamp(reward)
    return reward, {
        "oxygen_score": oxygen_score,
        "mixing_score": mixing_score,
        "nutrient_score": nutrient_score,
        "production_score": production_score,
        "purity_score": purity_score,
        "safety_score": safety_score,
        "growth_bonus": growth_bonus,
        "penalty": actuator_penalty + foam_penalty,
    }


def grade_trajectory(points: list[TrajectoryPoint], task: BioreactorTask) -> float:
    if not points:
        return 0.0

    average_reward = sum(point.reward for point in points) / len(points)
    final_biomass = clamp(points[-1].biomass / max(task.target_biomass, 1e-6))
    terminal_biomass = clamp(points[-1].biomass / max(task.terminal_biomass_target, 1e-6))
    terminal_byproduct = clamp(1.0 - points[-1].byproduct / max(task.terminal_byproduct_limit, 1e-6))
    terminal_nutrient = clamp(
        1.0
        - abs(
            points[-1].nutrient
            - 0.5 * (task.terminal_nutrient_range[0] + task.terminal_nutrient_range[1])
        )
        / max(0.5 * (task.terminal_nutrient_range[1] - task.terminal_nutrient_range[0]), 1e-6)
    )
    terminal_oxygen = clamp(points[-1].oxygen / max(task.terminal_oxygen_floor, 1e-6))
    terminal_score = clamp(
        0.40 * terminal_biomass
        + 0.25 * terminal_byproduct
        + 0.20 * terminal_nutrient
        + 0.15 * terminal_oxygen
    )
    safe_region = [
        point
        for point in points
        if abs(point.oxygen - task.target_oxygen) <= 0.10
        and abs(point.mixing - task.target_mixing) <= 0.12
        and point.byproduct <= task.max_safe_byproduct
        and point.foam_risk <= task.foam_penalty_threshold
    ]
    stability = len(safe_region) / len(points)
    efficiency = 1.0 - (
        0.5 * sum(point.byproduct for point in points) / len(points)
        + 0.5 * sum(point.shear_damage for point in points) / len(points)
    )
    survival = len(points) / task.max_steps
    collapse_penalty = 0.25 if any(point.collapsed for point in points) else 0.0
    score = (
        0.28 * average_reward
        + 0.18 * final_biomass
        + 0.17 * terminal_score
        + 0.17 * stability
        + 0.10 * clamp(efficiency)
        + 0.10 * survival
        - collapse_penalty
    )
    return clamp(score)


def phase_summary(points: list[TrajectoryPoint], task: BioreactorTask) -> dict[str, dict[str, float]]:
    if not points:
        return {
            phase: {"reward": 0.0, "biomass": 0.0, "safety": 0.0}
            for phase in ("startup", "growth", "stress")
        }

    grouped: dict[str, list[TrajectoryPoint]] = {"startup": [], "growth": [], "stress": []}
    for point in points:
        grouped[classify_phase(point.step, task.max_steps)].append(point)

    summary: dict[str, dict[str, float]] = {}
    for phase, phase_points in grouped.items():
        if not phase_points:
            summary[phase] = {"reward": 0.0, "biomass": 0.0, "safety": 0.0}
            continue
        avg_reward = sum(point.reward for point in phase_points) / len(phase_points)
        avg_biomass = sum(point.biomass for point in phase_points) / len(phase_points)
        avg_safety = 1.0 - (
            0.6 * sum(point.foam_risk for point in phase_points) / len(phase_points)
            + 0.8 * sum(point.shear_damage for point in phase_points) / len(phase_points)
        )
        summary[phase] = {
            "reward": round(clamp(avg_reward), 4),
            "biomass": round(clamp(avg_biomass), 4),
            "safety": round(clamp(avg_safety), 4),
        }
    return summary
