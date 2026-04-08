"""OpenEnv-compatible bioreactor control environment."""

from __future__ import annotations

import random
from uuid import uuid4

from graders import TrajectoryPoint, clamp, grade_step, grade_trajectory
from models import BioreactorAction, BioreactorObservation, BioreactorState
from tasks import DEFAULT_TASK_ID, BioreactorTask, get_task


class BioreactorEnv:
    """Step-based RL environment for simplified bioreactor process control."""

    def __init__(self, task_id: str = DEFAULT_TASK_ID, seed: int | None = None):
        self.task = get_task(task_id)
        self.seed_override = seed
        self.rng = random.Random(seed if seed is not None else self.task.seed)
        self.episode_id = str(uuid4())
        self.step_count = 0
        self.done = False
        self.last_error: str | None = None
        self.oxygen_level = self.task.initial_oxygen
        self.mixing_uniformity = self.task.initial_mixing
        self.nutrient_concentration = self.task.initial_nutrient
        self.stirrer_speed = self.task.initial_stirrer
        self.oxygen_input = self.task.initial_oxygen_input
        self.cumulative_reward = 0.0
        self.score = 0.0
        self.trajectory: list[TrajectoryPoint] = []

    def reset(self, task_id: str | None = None, seed: int | None = None) -> BioreactorObservation:
        if task_id is not None:
            self.task = get_task(task_id)
        self.seed_override = seed
        self.rng = random.Random(seed if seed is not None else self.task.seed)
        self.episode_id = str(uuid4())
        self.step_count = 0
        self.done = False
        self.last_error = None
        self.oxygen_level = self.task.initial_oxygen
        self.mixing_uniformity = self.task.initial_mixing
        self.nutrient_concentration = self.task.initial_nutrient
        self.stirrer_speed = self.task.initial_stirrer
        self.oxygen_input = self.task.initial_oxygen_input
        self.cumulative_reward = 0.0
        self.score = 0.0
        self.trajectory = []
        return self._observation(reward=0.0, message="reset")

    def step(self, action: BioreactorAction | int) -> BioreactorObservation:
        if self.done:
            self.last_error = "episode already complete; call reset"
            return self._observation(reward=0.0, message=self.last_error)

        action_id = action if isinstance(action, int) else action.action
        if action_id not in (0, 1, 2, 3, 4):
            action_id = 4
            self.last_error = "invalid action defaulted to 4"
        else:
            self.last_error = None

        self.step_count += 1
        self._apply_action(action_id)
        self._apply_dynamics()

        collapsed = self.oxygen_level < 0.10
        reward, _breakdown = grade_step(
            oxygen=self.oxygen_level,
            mixing=self.mixing_uniformity,
            nutrient=self.nutrient_concentration,
            stirrer_speed=self.stirrer_speed,
            oxygen_input=self.oxygen_input,
            task=self.task,
        )
        if collapsed:
            reward = 0.0

        self.done = self.step_count >= self.task.max_steps or collapsed
        self.cumulative_reward += reward
        self.trajectory.append(
            TrajectoryPoint(
                oxygen=self.oxygen_level,
                mixing=self.mixing_uniformity,
                nutrient=self.nutrient_concentration,
                reward=reward,
                collapsed=collapsed,
            )
        )
        self.score = grade_trajectory(self.trajectory, self.task)
        message = "oxygen collapse" if collapsed else "ok"
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
            stirrer_speed=self.stirrer_speed,
            oxygen_input=self.oxygen_input,
            last_error=self.last_error,
        )

    def close(self) -> None:
        pass

    def _apply_action(self, action_id: int) -> None:
        if action_id == 0:
            self.stirrer_speed = clamp(self.stirrer_speed + 0.10)
        elif action_id == 1:
            self.stirrer_speed = clamp(self.stirrer_speed - 0.10)
        elif action_id == 2:
            self.oxygen_input = clamp(self.oxygen_input + 0.10)
        elif action_id == 3:
            self.oxygen_input = clamp(self.oxygen_input - 0.10)

    def _apply_dynamics(self) -> None:
        disturbance = self.task.disturbances.get(self.step_count)
        if disturbance is not None:
            self.oxygen_level = clamp(self.oxygen_level + disturbance.oxygen_delta)
            self.mixing_uniformity = clamp(self.mixing_uniformity + disturbance.mixing_delta)
            self.nutrient_concentration = clamp(self.nutrient_concentration + disturbance.nutrient_delta)

        consumption = self.task.oxygen_load * (1.0 - self.nutrient_concentration)
        decay = self.task.viscosity * (1.0 - self.stirrer_speed)

        self.oxygen_level += 0.1 * self.oxygen_input - 0.05 * consumption
        self.mixing_uniformity += 0.1 * self.stirrer_speed - 0.05 * decay
        self.nutrient_concentration -= 0.03 * self.task.nutrient_use * self.mixing_uniformity

        if self.task.noise:
            self.oxygen_level += self.rng.uniform(-self.task.noise, self.task.noise)
            self.mixing_uniformity += self.rng.uniform(-self.task.noise, self.task.noise)
            self.nutrient_concentration += self.rng.uniform(-self.task.noise, self.task.noise)

        self.oxygen_level = clamp(self.oxygen_level)
        self.mixing_uniformity = clamp(self.mixing_uniformity)
        self.nutrient_concentration = clamp(self.nutrient_concentration)

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
            target_oxygen=self.task.target_oxygen,
            target_mixing=self.task.target_mixing,
            target_nutrient=self.task.target_nutrient,
            reward=clamp(reward),
            score=self.score,
            done=self.done,
            message=message,
        )
