import json

import pytest

import agent.tools.dvc_tools as dvc
import agent.tools.mlflow_tools as mt
from agent.errors import ToolError
from agent.registry import REGISTRY, ToolContext
from tests.fakes import FakeMlflow, mlflow_run

DVC_FILE = "outs:\n- md5: {md5}\n  size: {size}\n  path: dataset.csv\n"


def call(name, args=None):
    return REGISTRY.execute(name, args or {}, ToolContext("t"))


@pytest.mark.parametrize("cmd", [["rm", "-rf", "/"], ["git", "push"], ["dvc", "push"], ["git", "checkout", "main"]])
def test_unapproved_commands_are_blocked(cmd):
    with pytest.raises(ToolError, match="not approved"):
        dvc.run_approved(cmd)


def test_unsafe_arguments_are_blocked():
    with pytest.raises(ToolError, match="unsafe"):
        dvc.run_approved(["git", "show", "HEAD;rm -rf /"])


def test_invalid_revision_rejected_by_compare_tool():
    res = call("dvc_compare_versions", {"rev_a": "--output=/tmp/x", "rev_b": "HEAD"})
    assert res.status == "error" and "Invalid git revision" in res.error


def test_path_traversal_in_dataset_is_invalid():
    assert call("dvc_status", {"dataset": "../../etc/passwd"}).status == "invalid"


def test_list_tracked_datasets(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dataset.csv.dvc").write_text(DVC_FILE.format(md5="abc", size=10))
    res = call("dvc_list_tracked_datasets")
    assert res.status == "ok" and res.data["datasets"][0]["path"] == "data/dataset.csv" and res.data["datasets"][0]["md5"] == "abc"


def test_compare_versions_detects_change(monkeypatch):
    files = {"v1:data/dataset.csv.dvc": DVC_FILE.format(md5="aaa", size=100),
             "v2:data/dataset.csv.dvc": DVC_FILE.format(md5="bbb", size=150)}

    def fake(cmd, timeout=20):
        if cmd[:2] == ["git", "show"]:
            if cmd[2] in files:
                return files[cmd[2]]
            raise ToolError("missing")
        if cmd[:2] == ["git", "rev-parse"]:
            return cmd[2] + "-commit\n"
        raise ToolError("unexpected")

    monkeypatch.setattr(dvc, "run_approved", fake)
    res = call("dvc_compare_versions", {"rev_a": "v1", "rev_b": "v2"})
    assert res.status == "ok" and res.data["identical"] is False and res.data["size_delta_bytes"] == 50


def test_status_reports_changed_data(monkeypatch):
    out = json.dumps({"data/dataset.csv.dvc": [{"changed outs": {"data/dataset.csv": "modified"}}]})
    monkeypatch.setattr(dvc, "run_approved", lambda cmd, timeout=20: out)
    assert call("dvc_status").data["changed"] is True
    monkeypatch.setattr(dvc, "run_approved", lambda cmd, timeout=20: "{}")
    assert call("dvc_status").data["changed"] is False


def test_missing_dvc_binary_is_reported(monkeypatch):
    monkeypatch.setattr(dvc.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    res = call("dvc_status")
    assert res.status == "error" and "not found" in res.error


def test_experiment_dataset_provenance_is_read_and_verified(monkeypatch):
    run_id = "b" * 32
    monkeypatch.setattr(mt, "get_client", lambda: FakeMlflow([mlflow_run(run_id, "rf-v2", {"f1": 0.9})]))
    monkeypatch.setattr(dvc, "run_approved", lambda cmd, timeout=20: DVC_FILE.format(md5="abc123", size=1))
    res = call("dvc_experiment_dataset", {"identifier": run_id})
    d = res.data
    assert res.status == "ok" and d["dataset_version"] == "v2.1" and d["git_commit"] == "deadbeef"
    assert d["provenance_verified_against_git"] is True


def test_explain_reproduce_returns_commands_without_executing(monkeypatch):
    res = call("dvc_explain_reproduce", {"git_rev": "v1"})
    assert res.status == "ok" and res.data["executed"] is False
    assert any(s.startswith("git checkout v1 -- data/dataset.csv.dvc") for s in res.data["steps"])
