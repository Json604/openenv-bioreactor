"""FastAPI server exposing the OpenEnv reset/step/state API."""

from __future__ import annotations

from fastapi import Body, FastAPI, HTTPException, Query

from bioreactor_env import BioreactorEnv
from models import BioreactorAction, ResetRequest, StepResponse
from tasks import TASKS


ENV_NAME = "openenv-bioreactor"

app = FastAPI(
    title="Bioreactor Control OpenEnv",
    version="0.1.0",
    description="RL-compatible bioreactor oxygen and mixing control environment.",
)

env = BioreactorEnv()
env.reset()


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "env": ENV_NAME}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": ENV_NAME}


@app.get("/tasks")
def tasks() -> dict[str, list[dict[str, object]]]:
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "name": task.name,
                "difficulty": task.difficulty,
                "description": task.description,
                "max_steps": task.max_steps,
            }
            for task in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(
    payload: ResetRequest | None = Body(default=None),
    task_id: str | None = Query(default=None),
) -> dict[str, object]:
    request = payload or ResetRequest()
    selected_task_id = task_id or request.task_id
    try:
        observation = env.reset(task_id=selected_task_id, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _model_dump(observation)


@app.post("/step")
def step(action: BioreactorAction) -> dict[str, object]:
    observation = env.step(action)
    response = StepResponse(
        observation=observation,
        reward=observation.reward,
        done=observation.done,
        info={
            "score": observation.score,
            "task_id": observation.task_id,
            "message": observation.message,
            "phase_scores": env.phase_scores,
        },
    )
    return _model_dump(response)


@app.get("/state")
def state() -> dict[str, object]:
    return _model_dump(env.state)


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
