import pytest
from fastapi.testclient import TestClient

import agent.main as agent_main
import api.main as inference
from agent.orchestrator import Agent
from tests.fakes import stub_registry


def test_chat_endpoint_returns_structured_evidence(monkeypatch):
    monkeypatch.setattr(agent_main, "_agent", Agent(stub_registry()))
    client = TestClient(agent_main.app)
    r = client.post("/chat", json={"message": "Which dataset version was used for experiment 25?"}, headers={"X-Request-ID": "abc-123"})
    body = r.json()
    assert r.status_code == 200 and body["request_id"] == "abc-123"
    assert body["tool_calls"][0]["tool"] == "dvc_experiment_dataset" and "v2.1" in body["answer"]


def test_chat_endpoint_validates_input(monkeypatch):
    monkeypatch.setattr(agent_main, "_agent", Agent(stub_registry()))
    client = TestClient(agent_main.app)
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_confirm_flag_enables_state_change(monkeypatch):
    monkeypatch.setattr(agent_main, "_agent", Agent(stub_registry()))
    client = TestClient(agent_main.app)
    assert client.post("/chat", json={"message": "restart the deployment"}).json()["tool_calls"][0]["status"] == "confirmation_required"
    assert client.post("/chat", json={"message": "restart the deployment", "confirm": True}).json()["tool_calls"][0]["status"] == "ok"


def test_tool_catalogue_lists_permissions(monkeypatch):
    monkeypatch.setattr(agent_main, "_agent", Agent(stub_registry()))
    perms = {t["name"]: t["permission"] for t in TestClient(agent_main.app).get("/tools").json()}
    assert perms["k8s_restart_deployment"] == "write" and perms["mlflow_best_run"] == "read"


class FakeModel:
    n_features_in_ = 3

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


@pytest.fixture
def inference_client(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.joblib")
    with TestClient(inference.app) as c:
        yield c
    inference.state["model"] = None


def test_inference_not_ready_without_model_but_alive(inference_client):
    assert inference_client.get("/health").status_code == 200       # liveness: process is fine
    assert inference_client.get("/ready").status_code == 503        # readiness: no model -> no traffic
    assert inference_client.post("/predict", json={"features": [1, 2, 3]}).status_code == 503


def test_inference_predicts_when_model_loaded(inference_client):
    inference.state["model"] = FakeModel()
    assert inference_client.get("/ready").status_code == 200
    r = inference_client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    assert r.status_code == 200 and r.json()["prediction"] == 1
    assert inference_client.post("/predict", json={"features": [0.1]}).status_code == 422
