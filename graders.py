"""Deterministic graders for bioreactor control trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from tasks import BioreactorTask


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class TrajectoryPoint:
    oxygen: float
    mixing: float
    nutrient: float
    reward: float
    collapsed: bool


def score_signal(value: float, target: float, tolerance: float) -> float:
    return clamp(1.0 - abs(value - target) / tolerance)


def grade_step(
    *,
    oxygen: float,
    mixing: float,
    nutrient: float,
    stirrer_speed: float,
    oxygen_input: float,
    task: BioreactorTask,
) -> tuple[float, dict[str, float]]:
    oxygen_score = score_signal(oxygen, task.target_oxygen, 0.70)
    mixing_score = score_signal(mixing, task.target_mixing, 0.80)
    nutrient_score = score_signal(nutrient, task.target_nutrient, 0.60)
    stable = (
        abs(oxygen - task.target_oxygen) <= 0.10
        and abs(mixing - task.target_mixing) <= 0.12
        and nutrient >= 0.15
    )
    stability_bonus = 0.05 if stable else 0.0
    actuator_penalty = task.actuator_penalty * (stirrer_speed + oxygen_input)
    foam_penalty = 0.0
    if stirrer_speed + oxygen_input > task.foam_penalty_threshold:
        foam_penalty = 0.08 * (stirrer_speed + oxygen_input - task.foam_penalty_threshold)

    reward = (
        0.42 * oxygen_score
        + 0.38 * mixing_score
        + 0.15 * nutrient_score
        + stability_bonus
        - actuator_penalty
        - foam_penalty
    )
    reward = clamp(reward)
    return reward, {
        "oxygen_score": oxygen_score,
        "mixing_score": mixing_score,
        "nutrient_score": nutrient_score,
        "stability_bonus": stability_bonus,
        "penalty": actuator_penalty + foam_penalty,
    }


def grade_trajectory(points: list[TrajectoryPoint], task: BioreactorTask) -> float:
    if not points:
        return 0.0
    average_reward = sum(point.reward for point in points) / len(points)
    survival = len(points) / task.max_steps
    stable_steps = [
        point
        for point in points
        if abs(point.oxygen - task.target_oxygen) <= 0.10
        and abs(point.mixing - task.target_mixing) <= 0.12
        and point.nutrient >= 0.15
    ]
    stability = len(stable_steps) / len(points)
    collapse_penalty = 0.25 if any(point.collapsed for point in points) else 0.0
    return clamp(0.65 * average_reward + 0.25 * stability + 0.10 * survival - collapse_penalty)
