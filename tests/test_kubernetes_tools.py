import pytest

import agent.tools.kubernetes_tools as k8s
from agent.registry import REGISTRY, ToolContext
from tests.fakes import FakeApps, FakeCore, cstatus, deployment, pod, warning_event


@pytest.fixture
def cluster(monkeypatch):
    core, apps = FakeCore(pods=[pod("p1"), pod("p2")]), FakeApps()
    monkeypatch.setattr(k8s, "get_apis", lambda: (core, apps))
    return core, apps


def run(name, args=None, confirmed=False):
    return REGISTRY.execute(name, args or {}, ToolContext("t", confirmed=confirmed))


def test_unauthorized_namespace_modification_is_rejected(cluster):
    _, apps = cluster
    res = run("k8s_restart_deployment", {"namespace": "kube-system"}, confirmed=True)
    assert res.status == "denied" and apps.patches == []


def test_restart_requires_explicit_confirmation(cluster):
    _, apps = cluster
    assert run("k8s_restart_deployment").status == "confirmation_required" and apps.patches == []
    res = run("k8s_restart_deployment", confirmed=True)
    assert res.status == "ok" and apps.patches[0][1] == "mlops-agent"
    assert "restartedAt" in str(apps.patches[0][2])


def test_scale_is_bounded_and_confirmed(cluster):
    _, apps = cluster
    assert run("k8s_scale_deployment", {"replicas": 99}, confirmed=True).status == "invalid"
    assert run("k8s_scale_deployment", {"replicas": 3}).status == "confirmation_required"
    assert run("k8s_scale_deployment", {"replicas": 3}, confirmed=True).status == "ok"
    assert apps.scales[0][2] == {"spec": {"replicas": 3}}


def test_diagnose_healthy_deployment(cluster):
    res = run("k8s_diagnose")
    assert res.status == "ok" and res.data["healthy"] is True and res.data["findings"] == []


def test_diagnose_detects_image_pull_failure(monkeypatch):
    bad = pod("p1", phase="Pending", containers=[cstatus(ready=False, waiting="ImagePullBackOff")])
    core = FakeCore(pods=[bad, pod("p2")], events=[warning_event("Failed", "Failed to pull image mlops-inference:nope")])
    monkeypatch.setattr(k8s, "get_apis", lambda: (core, FakeApps([deployment(desired=2, ready=1)])))
    d = run("k8s_diagnose").data
    assert d["healthy"] is False and "ImagePullBackOff" in d["likely_cause"] and d["unhealthy_pods"] == ["p1"]


def test_diagnose_detects_readiness_failure_and_no_endpoints(monkeypatch):
    unready = pod("p1", containers=[cstatus(ready=False)])
    core = FakeCore(pods=[unready], endpoints=0,
                    events=[warning_event("Unhealthy", "Readiness probe failed: HTTP probe failed with statuscode: 503")],
                    logs="ERROR MODEL LOAD FAILED")
    monkeypatch.setattr(k8s, "get_apis", lambda: (core, FakeApps([deployment(2, 0)])))
    d = run("k8s_diagnose").data
    text = " ".join(d["findings"])
    assert "Readiness" in text and "0 ready endpoints" in text
    assert d["log_tail_of_first_unhealthy_pod"] == ["ERROR MODEL LOAD FAILED"]


def test_diagnose_detects_oomkilled(monkeypatch):
    p = pod("p1", containers=[cstatus(ready=False, restarts=4, waiting="CrashLoopBackOff", terminated_reason="OOMKilled")])
    monkeypatch.setattr(k8s, "get_apis", lambda: (FakeCore(pods=[p]), FakeApps([deployment(2, 0)])))
    d = run("k8s_diagnose").data
    assert any("OOMKilled" in f for f in d["findings"]) and any("restarted 4" in f for f in d["findings"])


def test_cluster_unreachable_reports_error(monkeypatch):
    def boom():
        raise k8s.ToolError("No Kubernetes configuration found")

    monkeypatch.setattr(k8s, "get_apis", boom)
    assert run("k8s_list_workloads").status == "error"


def test_app_version_reports_image_and_config(cluster):
    d = run("k8s_app_version").data
    assert d["image_tags"] == ["1.0.0"] and d["config"]["MODEL_RUN_ID"] == "a" * 32
