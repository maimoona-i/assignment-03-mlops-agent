"""MLflow experiment-intelligence tools (Task 2). Uses MlflowClient (never scrapes the web UI)."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.errors import ToolError
from agent.registry import REGISTRY
from agent.schemas import NoArgs

PROVENANCE_TAGS = ("dataset.version", "dvc.md5", "dvc.path", "git.commit", "quality_gate")


def get_client():
    """Create an MlflowClient (patched in tests). Short retries so an unreachable server fails fast."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=get_settings().mlflow_uri)


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat() if ms else None


def run_summary(run) -> dict:
    tags = run.data.tags
    return {
        "run_id": run.info.run_id, "run_name": tags.get("mlflow.runName"),
        "experiment_id": run.info.experiment_id, "status": run.info.status,
        "start_time": _iso(run.info.start_time), "metrics": dict(run.data.metrics),
        "params": dict(run.data.params), "provenance": {k: tags.get(k) for k in PROVENANCE_TAGS},
    }


def _resolve_experiments(client, experiment: str | None) -> list[str]:
    if experiment:
        exp = client.get_experiment_by_name(experiment)
        if exp is None and experiment.isdigit():
            try:
                exp = client.get_experiment(experiment)
            except Exception:
                exp = None
        if exp is None:
            raise ToolError(f"Experiment '{experiment}' not found")
        return [exp.experiment_id]
    exp = client.get_experiment_by_name(get_settings().mlflow_experiment)
    if exp is not None:
        return [exp.experiment_id]
    ids = [e.experiment_id for e in client.search_experiments()]
    if not ids:
        raise ToolError("No experiments exist in the MLflow tracking server")
    return ids


def _recent_runs(client, ids: list[str], limit: int, finished_only: bool = True):
    return client.search_runs(
        ids, filter_string="attributes.status = 'FINISHED'" if finished_only else "",
        order_by=["attributes.start_time DESC"], max_results=limit)


def resolve_run(identifier: str):
    """Resolve a run id, run name, or experiment id/name into one MLflow run."""
    client = get_client()
    if re.fullmatch(r"[0-9a-f]{32}", identifier):
        try:
            return client.get_run(identifier)
        except Exception as exc:
            raise ToolError(f"Run '{identifier}' not found") from exc
    exp = client.get_experiment_by_name(identifier)
    if exp is None and identifier.isdigit():
        try:
            exp = client.get_experiment(identifier)
        except Exception:
            exp = None
    if exp is not None:
        runs = _recent_runs(client, [exp.experiment_id], 1)
        if not runs:
            raise ToolError(f"Experiment '{identifier}' has no finished runs")
        return runs[0]
    safe = identifier.replace("'", "")
    runs = client.search_runs(_resolve_experiments(client, None), filter_string=f"tags.mlflow.runName = '{safe}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise ToolError(f"No run, run name, or experiment matches '{identifier}'")
    return runs[0]


# ---------------------------------------------------------------- tools
@REGISTRY.tool(name="mlflow_list_experiments", module="mlflow", args=NoArgs,
               description="List MLflow experiments with their ids and lifecycle stage.")
def mlflow_list_experiments(_: NoArgs) -> dict:
    exps = get_client().search_experiments()
    return {"experiments": [{"experiment_id": e.experiment_id, "name": e.name,
                             "lifecycle_stage": e.lifecycle_stage} for e in exps], "count": len(exps)}


class ListRunsArgs(BaseModel):
    experiment: str | None = Field(default=None, description="Experiment name or id (default: the project experiment)")
    limit: int = Field(default=10, ge=1, le=50)


@REGISTRY.tool(name="mlflow_list_runs", module="mlflow", args=ListRunsArgs,
               description="List the most recent training runs (any status) with metrics, parameters and dataset provenance tags.")
def mlflow_list_runs(args: ListRunsArgs) -> dict:
    client = get_client()
    runs = _recent_runs(client, _resolve_experiments(client, args.experiment), args.limit, finished_only=False)
    return {"runs": [run_summary(r) for r in runs], "count": len(runs)}


class RunArgs(BaseModel):
    run_id: str = Field(description="MLflow run id (32 hex chars) or run name")


@REGISTRY.tool(name="mlflow_get_run", module="mlflow", args=RunArgs,
               description="Get one run in detail: metrics, parameters, tags and artifact listing.")
def mlflow_get_run(args: RunArgs) -> dict:
    run = resolve_run(args.run_id)
    client = get_client()
    try:
        artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    except Exception:
        artifacts = None
    data = run_summary(run)
    data["artifacts"] = artifacts
    data["artifacts_note"] = None if artifacts is not None else "artifact listing unavailable"
    return data


class BestRunArgs(BaseModel):
    metric: str = Field(default="f1", description="Metric name to rank by")
    mode: str = Field(default="max", pattern="^(max|min)$")
    last_n: int = Field(default=10, ge=1, le=50, description="Consider only the N most recent finished runs")
    experiment: str | None = None


@REGISTRY.tool(name="mlflow_best_run", module="mlflow", args=BestRunArgs,
               description="Identify the best of the recent finished runs by a metric, with readiness checks and caveats (a high score alone does not mean production-ready).")
def mlflow_best_run(args: BestRunArgs) -> dict:
    client = get_client()
    runs = _recent_runs(client, _resolve_experiments(client, args.experiment), args.last_n)
    scored = [r for r in runs if args.metric in r.data.metrics]
    if not scored:
        raise ToolError(f"No recent finished runs recorded the metric '{args.metric}'")
    scored.sort(key=lambda r: r.data.metrics[args.metric], reverse=(args.mode == "max"))
    best = run_summary(scored[0])
    ranking = [{"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName"),
                args.metric: r.data.metrics[args.metric]} for r in scored]
    checks, caveats = _readiness_checks(best, ranking, args.metric)
    return {"metric": args.metric, "mode": args.mode, "runs_considered": len(runs), "best": best,
            "ranking": ranking, "checks": checks, "caveats": caveats,
            "production_ready": "not determined - needs the checks above plus deployment/serving validation"}


def _readiness_checks(best: dict, ranking: list[dict], metric: str) -> tuple[list[dict], list[str]]:
    s, checks, caveats = get_settings(), [], []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add("run_finished", "pass" if best["status"] == "FINISHED" else "fail", f"status={best['status']}")
    if metric == "f1":
        val = best["metrics"].get("f1")
        add("meets_f1_policy_threshold", "pass" if val is not None and val >= s.min_f1 else "fail",
            f"f1={val} vs policy MIN_F1={s.min_f1}")
    gate = best["provenance"].get("quality_gate")
    add("quality_gate_tag", {"passed": "pass", "failed": "fail"}.get(gate, "unknown"), f"quality_gate={gate}")
    has_prov = bool(best["provenance"].get("dvc.md5") or best["provenance"].get("dataset.version"))
    add("dataset_provenance_recorded", "pass" if has_prov else "fail", "dvc/dataset tags present" if has_prov else "missing")
    if len(ranking) > 1:
        margin = abs(ranking[0][metric] - ranking[1][metric])
        add("margin_over_runner_up", "pass" if margin >= 0.005 else "unknown", f"margin={margin:.4f}")
        if margin < 0.005:
            caveats.append("Top candidates differ by less than 0.005 - the ranking may be within noise.")
    m = best["metrics"]
    if "precision" in m and "recall" in m and abs(m["precision"] - m["recall"]) > 0.10:
        caveats.append("Precision and recall differ by more than 0.10; check whether that trade-off suits the use case.")
    caveats.append("Single hold-out evaluation; no confidence interval or drift/latency validation was recorded.")
    return checks, caveats


class CompareArgs(BaseModel):
    run_ids: list[str] | None = Field(default=None, description="Run ids/names to compare (default: last_n recent runs)")
    last_n: int = Field(default=5, ge=2, le=20)
    metric: str = "f1"
    experiment: str | None = None


@REGISTRY.tool(name="mlflow_compare_runs", module="mlflow", args=CompareArgs,
               description="Compare several runs (e.g. the last five) side by side: metrics, differing parameters, dataset provenance.")
def mlflow_compare_runs(args: CompareArgs) -> dict:
    client = get_client()
    runs = ([resolve_run(r) for r in args.run_ids] if args.run_ids
            else _recent_runs(client, _resolve_experiments(client, args.experiment), args.last_n))
    if len(runs) < 2:
        raise ToolError("Need at least two runs to compare")
    rows = [run_summary(r) for r in runs]
    all_params = {k for r in rows for k in r["params"]}
    differing = {k: {r["run_id"][:8]: r["params"].get(k) for r in rows}
                 for k in sorted(all_params) if len({r["params"].get(k) for r in rows}) > 1}
    with_metric = [r for r in rows if args.metric in r["metrics"]]
    best = max(with_metric, key=lambda r: r["metrics"][args.metric]) if with_metric else None
    return {"metric": args.metric, "runs": rows, "differing_params": differing,
            "best_by_metric": best["run_id"] if best else None,
            "note": "Highest metric is a ranking, not a production-readiness decision."}


class RegistryArgs(BaseModel):
    model_name: str | None = Field(default=None, pattern=r"^[\w.\-]+$")


@REGISTRY.tool(name="mlflow_registry_info", module="mlflow", args=RegistryArgs,
               description="List registered models and their versions (with run ids and aliases) from the MLflow model registry.")
def mlflow_registry_info(args: RegistryArgs) -> dict:
    client = get_client()
    try:
        models = client.search_registered_models()
    except Exception as exc:
        raise ToolError(f"Model registry unavailable: {exc}") from exc
    out = []
    for m in models:
        if args.model_name and m.name != args.model_name:
            continue
        versions = client.search_model_versions(f"name='{m.name}'")
        out.append({"name": m.name, "aliases": dict(getattr(m, "aliases", {}) or {}),
                    "versions": [{"version": v.version, "run_id": v.run_id, "status": v.status,
                                  "created": _iso(v.creation_timestamp)} for v in versions]})
    return {"registered_models": out, "count": len(out),
            "note": None if out else "No registered models (a model is registered only after the quality gate passes)."}
