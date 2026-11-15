import httpx
import pytest

import agent.tools.airflow_tools as af
from agent.registry import REGISTRY, ToolContext

RUN = "manual__2026-11-14T10:00:00+00:00"


def handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/auth/token":
        return httpx.Response(201, json={"access_token": "jwt-token"})
    assert request.headers["authorization"] == "Bearer jwt-token"  # authenticated requests
    if path == "/api/v2/dags/training_pipeline/dagRuns" and request.method == "GET":
        return httpx.Response(200, json={"dag_runs": [{"dag_run_id": RUN, "state": "failed", "start_date": "2026-11-14"}]})
    if path.endswith("/taskInstances"):
        return httpx.Response(200, json={"task_instances": [
            {"task_id": "validate_dataset", "state": "success", "try_number": 1, "max_tries": 2},
            {"task_id": "quality_gate", "state": "failed", "try_number": 1, "max_tries": 2},
            {"task_id": "register_model", "state": "upstream_failed", "try_number": 0, "max_tries": 2}]})
    if path.endswith("/dagRuns/" + RUN):
        return httpx.Response(200, json={"dag_run_id": RUN, "state": "failed"})
    if path.endswith("/logs/1"):
        return httpx.Response(200, json={"content": [{"event": "Quality gate FAILED: f1=0.70 < 0.85"}]})
    if path == "/api/v2/dags":
        return httpx.Response(200, json={"dags": [{"dag_id": "training_pipeline", "is_paused": False}]})
    return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture
def client(monkeypatch):
    c = af.AirflowClient("http://airflow.test", "admin", "pw", transport=httpx.MockTransport(handler))
    monkeypatch.setattr(af, "get_client", lambda: c)
    return c


def call(name, args=None, confirmed=False):
    return REGISTRY.execute(name, args or {}, ToolContext("t", confirmed=confirmed))


def test_list_dags_marks_approved(client):
    d = call("airflow_list_dags").data
    assert d["dags"][0]["approved_for_trigger"] is True


def test_failed_tasks_are_identified(client):
    d = call("airflow_failed_tasks", {"dag_id": "training_pipeline"}).data
    assert d["run_state"] == "failed" and [t["task_id"] for t in d["failed_tasks"]] == ["quality_gate", "register_model"]


def test_task_logs_are_returned(client):
    d = call("airflow_task_logs", {"dag_id": "training_pipeline", "dag_run_id": RUN, "task_id": "quality_gate"}).data
    assert "Quality gate FAILED" in d["log_tail"][0]


def test_trigger_requires_confirmation(client):
    assert call("airflow_trigger_dag").status == "confirmation_required"


def test_trigger_of_unapproved_dag_is_rejected(client):
    res = call("airflow_trigger_dag", {"dag_id": "some_other_dag"}, confirmed=True)
    assert res.status == "error" and "not approved" in res.error


def test_airflow_unreachable_is_reported(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused")

    c = af.AirflowClient("http://airflow.test", "admin", "pw", transport=httpx.MockTransport(refuse))
    monkeypatch.setattr(af, "get_client", lambda: c)
    res = call("airflow_dag_runs")
    assert res.status == "error" and "unreachable" in res.error


def test_bad_credentials_are_reported_without_leaking_password(monkeypatch):
    c = af.AirflowClient("http://airflow.test", "admin", "SUPERSECRET",
                         transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "bad"})))
    monkeypatch.setattr(af, "get_client", lambda: c)
    res = call("airflow_dag_runs")
    assert res.status == "error" and "authentication failed" in res.error and "SUPERSECRET" not in res.error
