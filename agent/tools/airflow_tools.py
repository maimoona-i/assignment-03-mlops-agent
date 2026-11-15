"""Apache Airflow 3 workflow tools (Task 3). Uses the Airflow 3 REST API (/api/v2) with JWT authentication."""
from __future__ import annotations

import time
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.errors import ToolError
from agent.logging_utils import redact_text
from agent.registry import REGISTRY
from agent.schemas import NoArgs

API = "/api/v2"


class AirflowClient:
    """Minimal authenticated Airflow 3 client (token is cached in memory, never logged)."""

    def __init__(self, base_url: str, username: str, password: str, transport: httpx.BaseTransport | None = None,
                 timeout: float = 10.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._username, self._password = username, password
        self._token: str | None = None

    def _auth_header(self) -> dict:
        if self._token is None:
            try:
                r = self.http.post("/auth/token", json={"username": self._username, "password": self._password})
            except httpx.HTTPError as exc:
                raise ToolError(f"Airflow is unreachable: {type(exc).__name__}") from exc
            if r.status_code >= 400:
                raise ToolError(f"Airflow authentication failed (HTTP {r.status_code})")
            self._token = r.json()["access_token"]
        return {"Authorization": f"Bearer {self._token}"}

    def request(self, method: str, path: str, **kwargs):
        extra_headers = kwargs.pop("headers", {})
        for attempt in (1, 2):
            try:
                r = self.http.request(method, path, headers={**self._auth_header(), **extra_headers}, **kwargs)
            except httpx.HTTPError as exc:
                raise ToolError(f"Airflow request failed: {type(exc).__name__}") from exc
            if r.status_code == 401 and attempt == 1:  # token expired -> re-authenticate once
                self._token = None
                continue
            break
        if r.status_code >= 400:
            raise ToolError(f"Airflow API {r.status_code} for {path}: {redact_text(r.text[:300])}")
        if "json" in r.headers.get("content-type", ""):
            return r.json()
        return {"_text": r.text}


def get_client() -> AirflowClient:
    s = get_settings()
    return AirflowClient(s.airflow_url, s.airflow_username, s.airflow_password)


def _run_path(dag_id: str, run_id: str) -> str:
    return f"{API}/dags/{quote(dag_id, safe='')}/dagRuns/{quote(run_id, safe='')}"


def _require_approved(dag_id: str) -> None:
    approved = get_settings().approved_dags
    if dag_id not in approved:
        raise ToolError(f"DAG '{dag_id}' is not approved for this agent (approved: {', '.join(approved)})")


def _run_info(r: dict) -> dict:
    return {k: r.get(k) for k in ("dag_run_id", "state", "run_type", "logical_date", "start_date", "end_date")}


def _latest_run_id(client: AirflowClient, dag_id: str) -> str:
    runs = client.request("GET", f"{API}/dags/{quote(dag_id, safe='')}/dagRuns",
                          params={"limit": 1, "order_by": "-start_date"}).get("dag_runs", [])
    if not runs:
        raise ToolError(f"DAG '{dag_id}' has no runs yet")
    return runs[0]["dag_run_id"]


class DagArgs(BaseModel):
    dag_id: str = Field(default="training_pipeline", pattern=r"^[\w.\-]+$")


# ---------------------------------------------------------------- tools
@REGISTRY.tool(name="airflow_list_dags", module="airflow", args=NoArgs,
               description="List the DAGs available in Airflow (paused flag, and whether this agent is approved to trigger them).")
def airflow_list_dags(_: NoArgs) -> dict:
    approved = get_settings().approved_dags
    dags = get_client().request("GET", f"{API}/dags", params={"limit": 100}).get("dags", [])
    return {"dags": [{"dag_id": d["dag_id"], "is_paused": d.get("is_paused"),
                      "approved_for_trigger": d["dag_id"] in approved} for d in dags], "count": len(dags)}


class RunsArgs(DagArgs):
    limit: int = Field(default=5, ge=1, le=25)


@REGISTRY.tool(name="airflow_dag_runs", module="airflow", args=RunsArgs,
               description="Get the most recent DAG runs and their states (success/failed/running/queued).")
def airflow_dag_runs(args: RunsArgs) -> dict:
    data = get_client().request("GET", f"{API}/dags/{quote(args.dag_id, safe='')}/dagRuns",
                                params={"limit": args.limit, "order_by": "-start_date"})
    return {"dag_id": args.dag_id, "runs": [_run_info(r) for r in data.get("dag_runs", [])]}


class TriggerArgs(DagArgs):
    conf: dict = Field(default_factory=dict, description="Optional run configuration, e.g. {'n_estimators': 200}")
    wait_seconds: int = Field(default=0, ge=0, le=120, description="Poll up to this many seconds for completion")


@REGISTRY.tool(name="airflow_trigger_dag", module="airflow", args=TriggerArgs, mutating=True, timeout_s=150,
               description="Trigger an APPROVED training DAG and optionally wait for the result. Never call without an explicit user request.")
def airflow_trigger_dag(args: TriggerArgs) -> dict:
    _require_approved(args.dag_id)
    client = get_client()
    dag = client.request("GET", f"{API}/dags/{quote(args.dag_id, safe='')}")
    if dag.get("is_paused"):
        raise ToolError(f"DAG '{args.dag_id}' is paused; unpause it in Airflow before triggering")
    run = client.request("POST", f"{API}/dags/{quote(args.dag_id, safe='')}/dagRuns",
                         json={"logical_date": None, "conf": args.conf})
    run_id, state = run["dag_run_id"], run.get("state")
    deadline = time.monotonic() + args.wait_seconds
    while args.wait_seconds and state in ("queued", "running") and time.monotonic() < deadline:
        time.sleep(3)
        state = client.request("GET", _run_path(args.dag_id, run_id)).get("state")
    return {"dag_id": args.dag_id, "dag_run_id": run_id, "state": state,
            "finished": state in ("success", "failed"),
            "note": None if state in ("success", "failed") else "Run has not finished yet; check status again later."}


class RunRefArgs(DagArgs):
    dag_run_id: str | None = Field(default=None, description="Run id (default: latest run)")


@REGISTRY.tool(name="airflow_task_instances", module="airflow", args=RunRefArgs,
               description="Get all task instances of a DAG run: state, try number, retries, start/end times.")
def airflow_task_instances(args: RunRefArgs) -> dict:
    client = get_client()
    run_id = args.dag_run_id or _latest_run_id(client, args.dag_id)
    run = client.request("GET", _run_path(args.dag_id, run_id))
    tis = client.request("GET", _run_path(args.dag_id, run_id) + "/taskInstances").get("task_instances", [])
    return {"dag_id": args.dag_id, "dag_run_id": run_id, "run_state": run.get("state"),
            "tasks": [{"task_id": t["task_id"], "state": t.get("state"), "try_number": t.get("try_number"),
                       "max_tries": t.get("max_tries"), "start_date": t.get("start_date"),
                       "end_date": t.get("end_date"), "duration": t.get("duration")} for t in tis]}


@REGISTRY.tool(name="airflow_failed_tasks", module="airflow", args=RunRefArgs,
               description="Identify failed (or upstream-failed) tasks in a DAG run (default: latest run).")
def airflow_failed_tasks(args: RunRefArgs) -> dict:
    info = airflow_task_instances(args)
    bad = [t for t in info["tasks"] if t["state"] in ("failed", "upstream_failed")]
    return {"dag_id": info["dag_id"], "dag_run_id": info["dag_run_id"], "run_state": info["run_state"],
            "failed_tasks": bad, "count": len(bad)}


class LogArgs(RunRefArgs):
    task_id: str = Field(pattern=r"^[\w.\-]+$")
    try_number: int = Field(default=1, ge=1, le=20)
    tail_lines: int = Field(default=40, ge=5, le=200)


@REGISTRY.tool(name="airflow_task_logs", module="airflow", args=LogArgs,
               description="Retrieve the log tail of one task attempt for failure diagnostics.")
def airflow_task_logs(args: LogArgs) -> dict:
    client = get_client()
    run_id = args.dag_run_id or _latest_run_id(client, args.dag_id)
    path = _run_path(args.dag_id, run_id) + f"/taskInstances/{quote(args.task_id, safe='')}/logs/{args.try_number}"
    raw = client.request("GET", path, headers={"Accept": "application/json"})
    content = raw.get("content", raw.get("_text", ""))
    if isinstance(content, list):
        lines = [c.get("event", str(c)) if isinstance(c, dict) else str(c) for c in content]
    else:
        lines = str(content).splitlines()
    return {"dag_id": args.dag_id, "dag_run_id": run_id, "task_id": args.task_id, "try_number": args.try_number,
            "log_tail": [redact_text(line) for line in lines[-args.tail_lines:]]}
