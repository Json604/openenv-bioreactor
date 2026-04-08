"""Small HTTP client for the bioreactor OpenEnv service."""

from __future__ import annotations

import requests

from models import BioreactorAction, BioreactorObservation, BioreactorState


class BioreactorClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str | None = None) -> BioreactorObservation:
        payload = {} if task_id is None else {"task_id": task_id}
        response = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        response.raise_for_status()
        return BioreactorObservation(**response.json())

    def step(self, action: int | BioreactorAction) -> BioreactorObservation:
        action_model = action if isinstance(action, BioreactorAction) else BioreactorAction(action=action)
        response = requests.post(
            f"{self.base_url}/step",
            json=_model_dump(action_model),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return BioreactorObservation(**data["observation"])

    def state(self) -> BioreactorState:
        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return BioreactorState(**response.json())

    def close(self) -> None:
        pass


def _model_dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
