"""Tests for the FastAPI server."""
from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    j = r.json()
    assert j["name"] == "BioOperatorEnv"
    assert "/reset" in j["endpoints"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_tasks():
    r = client.get("/tasks")
    assert r.status_code == 200
    j = r.json()
    ids = [t["task_id"] for t in j["tasks"]]
    assert "do-recovery-medium" in ids


def test_reset_then_step_roundtrip():
    r = client.post("/reset", json={"task_id": "do-recovery-medium", "seed": 42})
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert "measurements" in obs

    r = client.post("/step", json={"action": {
        "feed_delta_L_h": 0, "aeration_delta_vvm": 0.0, "agitation_delta_rpm": 0,
    }})
    assert r.status_code == 200
    j = r.json()
    assert "observation" in j
    assert -1.0 <= j["reward"] <= 1.0
    assert j["done"] in (False, True)
    assert "reward_components" in j["info"]


def test_state_endpoint():
    client.post("/reset", json={"task_id": "do-recovery-medium", "seed": 42})
    r = client.get("/state")
    assert r.status_code == 200
    j = r.json()
    assert j["task_id"] == "do-recovery-medium"
    assert len(j["ode_state"]) == 33


def test_step_with_invalid_action_string():
    client.post("/reset", json={"task_id": "do-recovery-medium", "seed": 42})
    r = client.post("/step", json={"action": "not valid json at all"})
    assert r.status_code == 200
    assert r.json()["info"]["format_valid"] is False
