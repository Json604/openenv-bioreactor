"""Pydantic data models for BioOperatorEnv (Action / Observation / State).

These define the contract at every boundary the agent touches:
- BioOperatorAction: what the LLM emits (validated, clipped)
- BioOperatorObservation: what the LLM sees (SCADA-style plant console)
- BioOperatorState: server-side full debug state (NOT shown to the agent)
- RewardComponents / StepInfo: reward + done bookkeeping returned by env.step
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class BioOperatorAction(BaseModel):
    """Operator action: discrete deltas on three control levers + optional reason.

    The action space is intentionally small (27 arms = 3 x 3 x 3) so GRPO
    can explore it densely. PID controllers for pH and temperature stay
    behind the agent's reach; this is correct operator behavior AND the
    main anti-cheat surface.
    """

    feed_delta_L_h: Literal[-5, 0, 5]
    aeration_delta_vvm: Literal[-0.10, 0.0, 0.10]
    agitation_delta_rpm: Literal[-5, 0, 5]
    reason: Optional[str] = Field(default=None)

    @field_validator("reason", mode="before")
    @classmethod
    def _truncate_reason(cls, v):
        if isinstance(v, str) and len(v) > 200:
            return v[:200]
        return v


class BioOperatorObservation(BaseModel):
    """SCADA-style plant-console snapshot the agent sees each step."""

    time_h: float
    batch_phase: Literal["inoculation", "growth", "production", "stationary"]
    measurements: dict
    setpoints_or_limits: dict
    current_controls: dict
    recent_trends: dict
    alarm: Optional[str]
    previous_action: Optional[dict]
    offline_lab: Optional[dict]
    instruction: str


class RewardComponents(BaseModel):
    """Per-step reward breakdown. Logged independently for diagnosability."""
    format_validity: float
    do_safety: float
    productivity: float
    substrate_control: float
    stability: float
    control_effort: float
    terminal_yield_bonus: float


class StepInfo(BaseModel):
    """Auxiliary info returned alongside (obs, reward, done)."""
    reward_total: float
    reward_components: RewardComponents
    safety_violation: bool
    success: bool
    done_reason: str


class BioOperatorState(BaseModel):
    """Server-side full debug state. NOT shown to the agent (anti-cheat)."""
    task_id: str
    seed: int
    step_count: int
    time_h: float
    ode_state: list[float]                # 33-vector
    last_action: Optional[dict] = None
    cumulative_reward: float = 0.0
    component_history: list[dict] = []    # one RewardComponents.dict() per step
    safety_violations: int = 0

    model_config = {"arbitrary_types_allowed": True}
