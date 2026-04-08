"""Typed Pydantic models for the OpenEnv bioreactor environment."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except Exception:
    Action = BaseModel
    Observation = BaseModel
    State = BaseModel


class BioreactorAction(Action):
    """Discrete actuator command for the bioreactor."""

    action: int = Field(
        default=4,
        ge=0,
        le=6,
        description=(
            "0=increase stirrer, 1=decrease stirrer, 2=increase oxygen, "
            "3=decrease oxygen, 4=do nothing, 5=increase feed, 6=decrease feed"
        ),
    )


class BioreactorObservation(Observation):
    """Observation returned by reset() and step()."""

    task_id: str = Field(..., description="Current task identifier")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Task difficulty")
    instruction: str = Field(..., description="Task objective")
    step: int = Field(..., ge=0, description="Current episode step")
    max_steps: int = Field(..., ge=1, description="Maximum episode length")
    oxygen_level: float = Field(..., ge=0.0, le=1.0)
    mixing_uniformity: float = Field(..., ge=0.0, le=1.0)
    nutrient_concentration: float = Field(..., ge=0.0, le=1.0)
    biomass_concentration: float = Field(..., ge=0.0, le=1.0)
    byproduct_load: float = Field(..., ge=0.0, le=1.0)
    feed_rate: float = Field(..., ge=0.0, le=1.0)
    target_oxygen: float = Field(..., ge=0.0, le=1.0)
    target_mixing: float = Field(..., ge=0.0, le=1.0)
    target_nutrient: float = Field(..., ge=0.0, le=1.0)
    target_biomass: float = Field(..., ge=0.0, le=1.0)
    max_safe_byproduct: float = Field(..., ge=0.0, le=1.0)
    terminal_biomass_target: float = Field(..., ge=0.0, le=1.0)
    terminal_byproduct_limit: float = Field(..., ge=0.0, le=1.0)
    terminal_nutrient_low: float = Field(..., ge=0.0, le=1.0)
    terminal_nutrient_high: float = Field(..., ge=0.0, le=1.0)
    terminal_oxygen_floor: float = Field(..., ge=0.0, le=1.0)
    reward: float = Field(..., ge=0.0, le=1.0, description="Per-step reward in [0, 1]")
    score: float = Field(..., ge=0.0, le=1.0, description="Trajectory grader score in [0, 1]")
    done: bool = Field(..., description="Whether the episode has ended")
    valid_actions: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    message: str = Field(default="", description="Status or failure message")

    def vector(self) -> list[float]:
        return [
            self.oxygen_level,
            self.mixing_uniformity,
            self.nutrient_concentration,
            self.biomass_concentration,
            self.byproduct_load,
            self.feed_rate,
        ]


class BioreactorReward(BaseModel):
    """Reward breakdown for transparent grading."""

    reward: float = Field(..., ge=0.0, le=1.0)
    oxygen_score: float = Field(..., ge=0.0, le=1.0)
    mixing_score: float = Field(..., ge=0.0, le=1.0)
    nutrient_score: float = Field(..., ge=0.0, le=1.0)
    production_score: float = Field(..., ge=0.0, le=1.0)
    purity_score: float = Field(..., ge=0.0, le=1.0)
    safety_score: float = Field(..., ge=0.0, le=1.0)
    growth_bonus: float = Field(..., ge=0.0, le=1.0)
    penalty: float = Field(..., ge=0.0)


class BioreactorState(State):
    """Server-side episode state."""

    episode_id: str = Field(..., description="Unique episode id")
    step_count: int = Field(default=0, ge=0)
    task_id: str = Field(..., description="Current task id")
    difficulty: str = Field(..., description="Current task difficulty")
    done: bool = Field(default=False)
    cumulative_reward: float = Field(default=0.0)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    oxygen_level: float = Field(..., ge=0.0, le=1.0)
    mixing_uniformity: float = Field(..., ge=0.0, le=1.0)
    nutrient_concentration: float = Field(..., ge=0.0, le=1.0)
    biomass_concentration: float = Field(..., ge=0.0, le=1.0)
    byproduct_load: float = Field(..., ge=0.0, le=1.0)
    stirrer_speed: float = Field(..., ge=0.0, le=1.0)
    oxygen_input: float = Field(..., ge=0.0, le=1.0)
    feed_rate: float = Field(..., ge=0.0, le=1.0)
    foam_risk: float = Field(..., ge=0.0, le=1.0)
    shear_damage: float = Field(..., ge=0.0, le=1.0)
    phase_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    last_error: str | None = None


class ResetRequest(BaseModel):
    task_id: str | None = Field(default=None, description="Optional task id to run")
    seed: int | None = Field(default=None, description="Optional deterministic seed override")


class StepResponse(BaseModel):
    observation: BioreactorObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
