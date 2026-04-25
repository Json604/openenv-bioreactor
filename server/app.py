"""FastAPI server exposing the OpenEnv-compliant interface for BioOperatorEnv.

Endpoints:
    GET  /              -> service metadata
    GET  /health        -> liveness check
    GET  /tasks         -> list of available scenarios
    POST /reset         -> {task_id, seed} -> Observation
    POST /step          -> {action} -> {observation, reward, done, info}
    GET  /state         -> full server-side state (debugging only)
"""
from __future__ import annotations
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from bioperator_env import __version__
from bioperator_env.env import BioOperatorEnv
from bioperator_env.scenarios import all_specs, list_tasks


app = FastAPI(title="BioOperatorEnv", version=__version__)


# Global env instance (single-tenant by design — judges hit one Space at a time)
_env: BioOperatorEnv = BioOperatorEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Any = Field(..., description="JSON action object or string")


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


@app.get("/")
def root() -> dict:
    return {
        "name": "BioOperatorEnv",
        "version": __version__,
        "tagline": "Flight simulator for autonomous bioreactor operators",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict:
    specs = all_specs()
    return {
        "tasks": [
            {
                "task_id": s.task_id,
                "description": s.description,
                "difficulty": s.difficulty,
                "fault_code": s.fault_code,
                "max_steps": s.max_steps,
            }
            for s in specs.values()
        ],
        "default_task_id": list_tasks()[0],
    }


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    global _env
    _env = BioOperatorEnv(
        task_id=req.task_id or _env.task_id,
        seed=req.seed if req.seed is not None else _env.seed,
    )
    obs = _env.reset()
    return {"observation": obs.model_dump()}


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    obs, reward, done, info = _env.step(req.action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def state() -> dict:
    return _env.state().model_dump()
