import agent.tools.mlflow_tools as mt
from agent.registry import REGISTRY, ToolContext
from tests.fakes import FakeMlflow, mlflow_run


def call(name, args=None):
    return REGISTRY.execute(name, args or {}, ToolContext("t"))


def runs():
    return [mlflow_run("1" * 32, "gb-v2", {"f1": 0.91, "precision": 0.90, "recall": 0.92}),
            mlflow_run("2" * 32, "rf-v2", {"f1": 0.94, "precision": 0.93, "recall": 0.95}),
            mlflow_run("3" * 32, "lr-v2", {"f1": 0.80, "precision": 0.8, "recall": 0.8}),
            mlflow_run("4" * 32, "failed", {"f1": 0.99}, status="FAILED")]


def test_best_run_picks_highest_metric_and_has_readiness_checks(monkeypatch):
    monkeypatch.setattr(mt, "get_client", lambda: FakeMlflow(runs()[:3]))
    res = call("mlflow_best_run", {"metric": "f1"})
    d = res.data
    assert res.status == "ok" and d["best"]["run_name"] == "rf-v2" and d["ranking"][0]["f1"] == 0.94
    assert {c["name"] for c in d["checks"]} >= {"run_finished", "meets_f1_policy_threshold", "dataset_provenance_recorded"}
    assert "not determined" in d["production_ready"] and d["caveats"]


def test_high_score_below_policy_is_not_silently_accepted(monkeypatch):
    monkeypatch.setenv("MIN_F1", "0.97")
    monkeypatch.setattr(mt, "get_client", lambda: FakeMlflow(runs()[:3]))
    checks = {c["name"]: c for c in call("mlflow_best_run").data["checks"]}
    assert checks["meets_f1_policy_threshold"]["status"] == "fail"


def test_failed_quality_gate_tag_is_flagged(monkeypatch):
    r = [mlflow_run("5" * 32, "x", {"f1": 0.95}, tags={"quality_gate": "failed"})]
    monkeypatch.setattr(mt, "get_client", lambda: FakeMlflow(r))
    checks = {c["name"]: c for c in call("mlflow_best_run").data["checks"]}
    assert checks["quality_gate_tag"]["status"] == "fail"


def test_unknown_metric_is_an_error(monkeypatch):
    monkeypatch.setattr(mt, "get_client", lambda: FakeMlflow(runs()[:3]))
    res = call("mlflow_best_run", {"metric": "auc"})
    assert res.status == "error" and "auc" in res.error


def test_compare_runs_shows_metrics_and_differing_params(monkeypatch):
    r = runs()[:3]
    r[0].data.params["n_estimators"] = "300"
    monkeypatch.setattr(mt, "get_client", lambda: FakeMlflow(r))
    d = call("mlflow_compare_runs", {"last_n": 3, "metric": "f1"}).data
    assert len(d["runs"]) == 3 and d["best_by_metric"] == "2" * 32 and "n_estimators" in d["differing_params"]


def test_mlflow_server_down_is_reported(monkeypatch):
    def boom():
        raise ConnectionError("connection refused")

    monkeypatch.setattr(mt, "get_client", boom)
    res = call("mlflow_best_run")
    assert res.status == "error" and "ConnectionError" in res.error
