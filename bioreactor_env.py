"""OpenEnv-compatible bioreactor control environment."""

from __future__ import annotations

import random
from uuid import uuid4

from graders import TrajectoryPoint, clamp, grade_step, grade_trajectory, phase_summary, score_signal
from models import BioreactorAction, BioreactorObservation, BioreactorState
from tasks import DEFAULT_TASK_ID, get_task


class BioreactorEnv:
    """Step-based RL environment for a lightweight fermentation control benchmark."""

    def __init__(self, task_id: str = DEFAULT_TASK_ID, seed: int | None = None):
        self.task = get_task(task_id)
        self.seed_override = seed
        self.rng = random.Random(seed if seed is not None else self.task.seed)
        self.episode_id = str(uuid4())
        self.step_count = 0
        self.done = False
        self.last_error: str | None = None
        self.cumulative_reward = 0.0
        self.score = 0.0
        self.phase_scores = phase_summary([], self.task)
        self.trajectory: list[TrajectoryPoint] = []
        self._load_task_defaults()

    def reset(self, task_id: str | None = None, seed: int | None = None) -> BioreactorObservation:
        if task_id is not None:
            self.task = get_task(task_id)
        self.seed_override = seed
        self.rng = random.Random(seed if seed is not None else self.task.seed)
        self.episode_id = str(uuid4())
        self.step_count = 0
        self.done = False
        self.last_error = None
        self.cumulative_reward = 0.0
        self.score = 0.0
        self.phase_scores = phase_summary([], self.task)
        self.trajectory = []
        self._load_task_defaults()
        return self._observation(reward=0.0, message="reset")

    def step(self, action: BioreactorAction | int) -> BioreactorObservation:
        if self.done:
            self.last_error = "episode already complete; call reset"
            return self._observation(reward=0.0, message=self.last_error)

        action_id = action if isinstance(action, int) else action.action
        if action_id not in (0, 1, 2, 3, 4, 5, 6):
            action_id = 4
            self.last_error = "invalid action defaulted to 4"
        else:
            self.last_error = None

        previous_biomass = self.biomass_concentration
        self.step_count += 1
        self._apply_schedule_and_disturbances()
        self._apply_action(action_id)
        self._apply_dynamics()
        biomass_growth = max(0.0, self.biomass_concentration - previous_biomass)

        collapsed = self.oxygen_level < 0.08
        reward, _breakdown = grade_step(
            oxygen=self.oxygen_level,
            mixing=self.mixing_uniformity,
            nutrient=self.nutrient_concentration,
            biomass=self.biomass_concentration,
            byproduct=self.byproduct_load,
            biomass_growth=biomass_growth,
            stirrer_speed=self.stirrer_speed,
            oxygen_input=self.oxygen_input,
            feed_rate=self.feed_rate,
            foam_risk=self.foam_risk,
            shear_damage=self.shear_damage,
            task=self.task,
        )
        if collapsed:
            reward = 0.0

        self.done = self.step_count >= self.task.max_steps or collapsed
        self.cumulative_reward += reward
        self.trajectory.append(
            TrajectoryPoint(
                step=self.step_count,
                oxygen=self.oxygen_level,
                mixing=self.mixing_uniformity,
                nutrient=self.nutrient_concentration,
                biomass=self.biomass_concentration,
                byproduct=self.byproduct_load,
                foam_risk=self.foam_risk,
                shear_damage=self.shear_damage,
                reward=reward,
                collapsed=collapsed,
            )
        )
        self.score = grade_trajectory(self.trajectory, self.task)
        self.phase_scores = phase_summary(self.trajectory, self.task)
        message = self._status_message(collapsed)
        return self._observation(reward=reward, message=message)

    @property
    def state(self) -> BioreactorState:
        return BioreactorState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            done=self.done,
            cumulative_reward=self.cumulative_reward,
            score=self.score,
            oxygen_level=self.oxygen_level,
            mixing_uniformity=self.mixing_uniformity,
            nutrient_concentration=self.nutrient_concentration,
            biomass_concentration=self.biomass_concentration,
            byproduct_load=self.byproduct_load,
            stirrer_speed=self.stirrer_speed,
            oxygen_input=self.oxygen_input,
            feed_rate=self.feed_rate,
            foam_risk=self.foam_risk,
            shear_damage=self.shear_damage,
            phase_scores=self.phase_scores,
            last_error=self.last_error,
        )

    def close(self) -> None:
        pass

    def _load_task_defaults(self) -> None:
        self.oxygen_level = self.task.initial_oxygen
        self.mixing_uniformity = self.task.initial_mixing
        self.nutrient_concentration = self.task.initial_nutrient
        self.biomass_concentration = self.task.initial_biomass
        self.byproduct_load = self.task.initial_byproduct
        self.stirrer_speed = self.task.initial_stirrer
        self.oxygen_input = self.task.initial_oxygen_input
        self.feed_rate = self.task.initial_feed_rate
        self.foam_risk = clamp(0.22 * self.oxygen_input + 0.16 * self.feed_rate + 0.10 * self.byproduct_load)
        self.shear_damage = clamp(max(0.0, self.stirrer_speed - self.task.shear_threshold) * 0.2)

    def _apply_schedule_and_disturbances(self) -> None:
        if self.step_count in self.task.feed_schedule:
            self.feed_rate = clamp(self.task.feed_schedule[self.step_count])

        disturbance = self.task.disturbances.get(self.step_count)
        if disturbance is None:
            return

        self.oxygen_level = clamp(self.oxygen_level + disturbance.oxygen_delta)
        self.mixing_uniformity = clamp(self.mixing_uniformity + disturbance.mixing_delta)
        self.nutrient_concentration = clamp(self.nutrient_concentration + disturbance.nutrient_delta)
        self.biomass_concentration = clamp(self.biomass_concentration + disturbance.biomass_delta)
        self.byproduct_load = clamp(self.byproduct_load + disturbance.byproduct_delta)
        self.feed_rate = clamp(self.feed_rate + disturbance.feed_rate_delta)

    def _apply_action(self, action_id: int) -> None:
        if action_id == 0:
            self.stirrer_speed = clamp(self.stirrer_speed + 0.10)
        elif action_id == 1:
            self.stirrer_speed = clamp(self.stirrer_speed - 0.10)
        elif action_id == 2:
            self.oxygen_input = clamp(self.oxygen_input + 0.10)
        elif action_id == 3:
            self.oxygen_input = clamp(self.oxygen_input - 0.10)
        elif action_id == 5:
            self.feed_rate = clamp(self.feed_rate + 0.06)
        elif action_id == 6:
            self.feed_rate = clamp(self.feed_rate - 0.06)

    def _apply_dynamics(self) -> None:
        viscosity = self.task.viscosity_base + 0.35 * self.biomass_concentration
        oxygen_growth_factor = score_signal(self.oxygen_level, self.task.target_oxygen, 0.35)
        mixing_growth_factor = score_signal(self.mixing_uniformity, self.task.target_mixing, 0.35)
        nutrient_growth_factor = clamp(self.nutrient_concentration / max(self.task.target_nutrient + 0.15, 1e-6))

        oxygen_transfer = self.task.oxygen_transfer * self.oxygen_input * (0.55 + 0.45 * self.mixing_uniformity)
        oxygen_demand = self.task.oxygen_load * self.biomass_concentration * (0.35 + 0.65 * self.feed_rate)
        mixing_decay = viscosity * (1.0 - self.stirrer_speed)

        biomass_growth = (
            0.05
            * self.task.growth_rate
            * self.biomass_concentration
            * oxygen_growth_factor
            * mixing_growth_factor
            * nutrient_growth_factor
            * (1.0 - 0.70 * self.shear_damage)
        )
        nutrient_consumption = (
            0.03 * self.task.nutrient_use * self.biomass_concentration * (0.40 + 0.60 * mixing_growth_factor)
        )
        byproduct_formation = (
            0.03 * max(0.0, self.feed_rate - self.oxygen_level)
            + 0.02 * max(0.0, 0.45 - self.oxygen_level)
            + 0.01 * max(0.0, self.nutrient_concentration - 0.55)
        )

        self.oxygen_level += 0.12 * oxygen_transfer - 0.09 * oxygen_demand
        self.mixing_uniformity += 0.10 * self.stirrer_speed - 0.05 * mixing_decay - 0.015 * self.foam_risk
        self.nutrient_concentration += 0.08 * self.feed_rate - nutrient_consumption
        self.biomass_concentration += biomass_growth
        self.byproduct_load += byproduct_formation - 0.015 * self.mixing_uniformity

        self.foam_risk += 0.04 * self.oxygen_input + 0.025 * self.feed_rate + 0.015 * self.byproduct_load
        self.foam_risk -= 0.025 * self.mixing_uniformity

        self.shear_damage += max(0.0, self.stirrer_speed - self.task.shear_threshold) * 0.045
        self.shear_damage -= 0.012 * (1.0 - self.stirrer_speed)

        if self.task.noise:
            self.oxygen_level += self.rng.uniform(-self.task.noise, self.task.noise)
            self.mixing_uniformity += self.rng.uniform(-self.task.noise, self.task.noise)
            self.nutrient_concentration += self.rng.uniform(-self.task.noise, self.task.noise)
            self.byproduct_load += self.rng.uniform(-self.task.noise, self.task.noise)

        self.oxygen_level = clamp(self.oxygen_level)
        self.mixing_uniformity = clamp(self.mixing_uniformity)
        self.nutrient_concentration = clamp(self.nutrient_concentration)
        self.biomass_concentration = clamp(self.biomass_concentration)
        self.byproduct_load = clamp(self.byproduct_load)
        self.foam_risk = clamp(self.foam_risk)
        self.shear_damage = clamp(self.shear_damage)

    def _status_message(self, collapsed: bool) -> str:
        if collapsed:
            return "oxygen collapse"
        warnings: list[str] = []
        if self.foam_risk > self.task.foam_penalty_threshold:
            warnings.append("foam_risk")
        if self.shear_damage > 0.35:
            warnings.append("shear_damage")
        if self.byproduct_load > self.task.max_safe_byproduct:
            warnings.append("byproduct_high")
        if not warnings:
            return "ok"
        return ",".join(warnings)

    def _observation(self, reward: float, message: str) -> BioreactorObservation:
        return BioreactorObservation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            instruction=self.task.description,
            step=self.step_count,
            max_steps=self.task.max_steps,
            oxygen_level=self.oxygen_level,
            mixing_uniformity=self.mixing_uniformity,
            nutrient_concentration=self.nutrient_concentration,
            biomass_concentration=self.biomass_concentration,
            byproduct_load=self.byproduct_load,
            feed_rate=self.feed_rate,
            target_oxygen=self.task.target_oxygen,
            target_mixing=self.task.target_mixing,
            target_nutrient=self.task.target_nutrient,
            target_biomass=self.task.target_biomass,
            max_safe_byproduct=self.task.max_safe_byproduct,
            terminal_biomass_target=self.task.terminal_biomass_target,
            terminal_byproduct_limit=self.task.terminal_byproduct_limit,
            terminal_nutrient_low=self.task.terminal_nutrient_range[0],
            terminal_nutrient_high=self.task.terminal_nutrient_range[1],
            terminal_oxygen_floor=self.task.terminal_oxygen_floor,
            reward=clamp(reward),
            score=self.score,
            done=self.done,
            message=message,
        )
