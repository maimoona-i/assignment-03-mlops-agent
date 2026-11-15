"""Kubernetes operations tools (Task 4).

Safety: every tool is restricted to ONE approved namespace and to the configured application
(label selector / deployment name). State-changing tools additionally require explicit user confirmation
(enforced centrally in the registry).
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agent.config import get_settings
from agent.errors import ToolError
from agent.logging_utils import redact_text
from agent.registry import REGISTRY

_T = {"_request_timeout": 10}


def get_apis():
    """Return (CoreV1Api, AppsV1Api). Patched in tests. In-cluster config first, then local kubeconfig."""
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception as exc:
            raise ToolError("No Kubernetes configuration found (is a cluster running and kubectl configured?)") from exc
    return client.CoreV1Api(), client.AppsV1Api()


def _g(obj, path: str, default=None):
    """Safe nested getattr: _g(pod, 'status.phase')."""
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj


def _ns() -> str:
    return get_settings().k8s_namespace


def _selector() -> str:
    return f"app={get_settings().k8s_app_label}"


def _age_seconds(ts) -> int | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int((datetime.now(timezone.utc) - ts).total_seconds())


def _container_info(c) -> dict:
    state = "unknown"
    if _g(c, "state.running"):
        state = "running"
    elif _g(c, "state.waiting"):
        state = f"waiting:{_g(c, 'state.waiting.reason', '')}"
    elif _g(c, "state.terminated"):
        state = f"terminated:{_g(c, 'state.terminated.reason', '')}"
    return {"name": c.name, "ready": bool(c.ready), "restarts": c.restart_count or 0, "state": state,
            "last_terminated_reason": _g(c, "last_state.terminated.reason"),
            "last_exit_code": _g(c, "last_state.terminated.exit_code"), "image": getattr(c, "image", None)}


def _pod_summary(pod) -> dict:
    containers = [_container_info(c) for c in (_g(pod, "status.container_statuses", []) or [])]
    return {"name": pod.metadata.name, "phase": _g(pod, "status.phase"),
            "ready": bool(containers) and all(c["ready"] for c in containers),
            "restarts": sum(c["restarts"] for c in containers), "containers": containers,
            "node": _g(pod, "spec.node_name"), "age_seconds": _age_seconds(_g(pod, "metadata.creation_timestamp"))}


def _is_unhealthy(p: dict) -> bool:
    return p["phase"] not in ("Running", "Succeeded") or not p["ready"]


def _deployment_summary(d) -> dict:
    spec_c = _g(d, "spec.template.spec.containers", []) or []
    return {"name": d.metadata.name, "desired": _g(d, "spec.replicas", 0),
            "ready": _g(d, "status.ready_replicas", 0) or 0, "available": _g(d, "status.available_replicas", 0) or 0,
            "updated": _g(d, "status.updated_replicas", 0) or 0, "images": [c.image for c in spec_c],
            "conditions": [{"type": c.type, "status": c.status, "reason": c.reason}
                           for c in (_g(d, "status.conditions", []) or [])]}


def _list_pods(core):
    return core.list_namespaced_pod(_ns(), label_selector=_selector(), **_T).items


class NsArgs(BaseModel):
    namespace: str | None = Field(default=None, description="Optional; must equal the approved namespace")


# ---------------------------------------------------------------- read tools
@REGISTRY.tool(name="k8s_list_workloads", module="kubernetes", args=NsArgs,
               description="List the application's Deployments and Pods with replica counts, readiness and restart counts.")
def k8s_list_workloads(_: NsArgs) -> dict:
    core, apps = get_apis()
    deps = apps.list_namespaced_deployment(_ns(), label_selector=_selector(), **_T).items
    pods = [_pod_summary(p) for p in _list_pods(core)]
    return {"namespace": _ns(), "deployments": [_deployment_summary(d) for d in deps], "pods": pods,
            "unhealthy_pods": [p["name"] for p in pods if _is_unhealthy(p)]}


class LogsArgs(NsArgs):
    pod: str | None = Field(default=None, pattern=r"^[a-z0-9.\-]+$", description="Pod name (default: an unhealthy pod, else the first)")
    tail_lines: int = Field(default=50, ge=5, le=300)
    previous: bool = Field(default=False, description="Logs of the previous (crashed) container instance")


@REGISTRY.tool(name="k8s_pod_logs", module="kubernetes", args=LogsArgs,
               description="Get status and recent logs of an application pod.")
def k8s_pod_logs(args: LogsArgs) -> dict:
    core, _ = get_apis()
    pods = [_pod_summary(p) for p in _list_pods(core)]
    if not pods:
        raise ToolError(f"No pods with label {_selector()} in namespace {_ns()}")
    if args.pod:
        chosen = next((p for p in pods if p["name"] == args.pod), None)
        if chosen is None:
            raise ToolError(f"Pod '{args.pod}' is not an application pod in {_ns()}")
    else:
        chosen = next((p for p in pods if _is_unhealthy(p)), pods[0])
    try:
        logs = core.read_namespaced_pod_log(chosen["name"], _ns(), tail_lines=args.tail_lines,
                                            previous=args.previous, **_T)
    except Exception as exc:
        logs, chosen["log_error"] = "", redact_text(str(exc))[:200]
    return {"pod": chosen, "previous": args.previous, "log_tail": redact_text(logs or "").splitlines()[-args.tail_lines:]}


@REGISTRY.tool(name="k8s_list_services", module="kubernetes", args=NsArgs,
               description="Inspect the application's Services: type, ClusterIP, ports and number of ready endpoints.")
def k8s_list_services(_: NsArgs) -> dict:
    core, _apps = get_apis()
    out = []
    for svc in core.list_namespaced_service(_ns(), **_T).items:
        ready = 0
        try:
            ep = core.read_namespaced_endpoints(svc.metadata.name, _ns(), **_T)
            ready = sum(len(s.addresses or []) for s in (ep.subsets or []))
        except Exception:
            ready = -1  # unknown
        out.append({"name": svc.metadata.name, "type": _g(svc, "spec.type"), "cluster_ip": _g(svc, "spec.cluster_ip"),
                    "selector": _g(svc, "spec.selector"),
                    "ports": [{"port": p.port, "target_port": str(p.target_port)} for p in (_g(svc, "spec.ports", []) or [])],
                    "ready_endpoints": ready})
    return {"namespace": _ns(), "services": out}


_WAITING_HINTS = {
    "ImagePullBackOff": "the container image cannot be pulled (wrong image name/tag or registry access)",
    "ErrImagePull": "the container image cannot be pulled (wrong image name/tag or registry access)",
    "CrashLoopBackOff": "the container keeps crashing at startup (see logs)",
    "CreateContainerConfigError": "a referenced ConfigMap/Secret key is missing or invalid",
}


@REGISTRY.tool(name="k8s_diagnose", module="kubernetes", args=NsArgs,
               description="Diagnose why the prediction API may be unavailable: pod status, restarts, probe failures, warning events, service endpoints and logs. Returns evidence-based findings.")
def k8s_diagnose(_: NsArgs) -> dict:
    core, apps = get_apis()
    deps = [_deployment_summary(d) for d in apps.list_namespaced_deployment(_ns(), label_selector=_selector(), **_T).items]
    pods = [_pod_summary(p) for p in _list_pods(core)]
    findings: list[str] = []
    likely: str | None = None

    for d in deps:
        if d["ready"] < d["desired"]:
            findings.append(f"Deployment {d['name']}: only {d['ready']}/{d['desired']} replicas ready.")
    if not deps:
        likely = "No Deployment found for the application"
        findings.append(likely + f" (selector {_selector()} in {_ns()}).")

    unhealthy = [p for p in pods if _is_unhealthy(p)]
    for p in unhealthy:
        for c in p["containers"]:
            reason = c["state"].split(":", 1)[1] if ":" in c["state"] else ""
            if reason in _WAITING_HINTS:
                findings.append(f"Pod {p['name']}: {reason} - {_WAITING_HINTS[reason]}.")
                likely = likely or f"{reason}: {_WAITING_HINTS[reason]}"
            if c["last_terminated_reason"] == "OOMKilled":
                findings.append(f"Pod {p['name']}: previously OOMKilled (exit {c['last_exit_code']}) - memory limit too low.")
                likely = likely or "OOMKilled: memory limit too low"
            if c["restarts"]:
                findings.append(f"Pod {p['name']}: container restarted {c['restarts']} time(s).")
            if c["state"] == "running" and not c["ready"]:
                findings.append(f"Pod {p['name']}: running but NOT ready (readiness probe failing).")
        if p["phase"] == "Pending" and not p["containers"]:
            findings.append(f"Pod {p['name']}: Pending (not scheduled / not started).")

    events = []
    try:
        evs = core.list_namespaced_event(_ns(), **_T).items
        warn = [e for e in evs if getattr(e, "type", None) == "Warning"]
        warn.sort(key=lambda e: str(getattr(e, "last_timestamp", "") or getattr(e, "event_time", "") or ""), reverse=True)
        for e in warn[:12]:
            events.append({"reason": e.reason, "object": _g(e, "involved_object.name"),
                           "message": redact_text(str(e.message))[:250], "count": getattr(e, "count", None)})
        if any(ev["reason"] == "Unhealthy" and "Readiness" in ev["message"] for ev in events):
            findings.append("Events show Readiness probe failures.")
            likely = likely or "Readiness probe failing (application not reporting ready)"
        if any(ev["reason"] == "Unhealthy" and "Liveness" in ev["message"] for ev in events):
            findings.append("Events show Liveness probe failures (kubelet restarts the container).")
    except Exception as exc:
        findings.append(f"Could not read events: {type(exc).__name__}")

    log_tail: list[str] = []
    if unhealthy:
        try:
            raw = core.read_namespaced_pod_log(unhealthy[0]["name"], _ns(), tail_lines=25, **_T)
            log_tail = redact_text(raw or "").splitlines()[-25:]
        except Exception:
            log_tail = []

    endpoints = []
    try:
        for svc in core.list_namespaced_service(_ns(), **_T).items:
            ep = core.read_namespaced_endpoints(svc.metadata.name, _ns(), **_T)
            n = sum(len(s.addresses or []) for s in (ep.subsets or []))
            endpoints.append({"service": svc.metadata.name, "ready_endpoints": n})
            if n == 0:
                findings.append(f"Service {svc.metadata.name} has 0 ready endpoints -> requests cannot be served.")
                likely = likely or "Service has no ready endpoints"
    except Exception:
        pass

    healthy = bool(deps) and not findings
    return {"namespace": _ns(), "healthy": healthy, "deployments": deps, "pods": pods,
            "unhealthy_pods": [p["name"] for p in unhealthy], "warning_events": events, "service_endpoints": endpoints,
            "log_tail_of_first_unhealthy_pod": log_tail, "findings": findings,
            "likely_cause": likely if likely else ("none found" if healthy else "see findings")}


@REGISTRY.tool(name="k8s_app_version", module="kubernetes", args=NsArgs,
               description="Report the currently deployed application version: image tag, version label, and ConfigMap values (APP_VERSION, MODEL_RUN_ID, DATASET_VERSION).")
def k8s_app_version(_: NsArgs) -> dict:
    core, apps = get_apis()
    s = get_settings()
    try:
        dep = apps.read_namespaced_deployment(s.k8s_deployment, _ns(), **_T)
    except Exception as exc:
        raise ToolError(f"Deployment '{s.k8s_deployment}' not found in {_ns()}: {type(exc).__name__}") from exc
    images = [c.image for c in dep.spec.template.spec.containers]
    config = {}
    try:
        cm = core.read_namespaced_config_map(s.k8s_configmap, _ns(), **_T)
        config = {k: v for k, v in (cm.data or {}).items() if k in ("APP_VERSION", "MODEL_RUN_ID", "DATASET_VERSION", "MODEL_VERSION")}
    except Exception:
        config = {"error": "ConfigMap could not be read"}
    return {"deployment": s.k8s_deployment, "images": images,
            "image_tags": [i.rsplit(":", 1)[1] if ":" in i.split("/")[-1] else "latest" for i in images],
            "version_label": (_g(dep, "metadata.labels", {}) or {}).get("app.kubernetes.io/version"),
            "config": config, "ready_replicas": _g(dep, "status.ready_replicas", 0) or 0,
            "desired_replicas": _g(dep, "spec.replicas", 0)}


# ---------------------------------------------------------------- state-changing tools (need confirmation)
class RestartArgs(NsArgs):
    pass


@REGISTRY.tool(name="k8s_restart_deployment", module="kubernetes", args=RestartArgs, mutating=True,
               description="Rolling-restart the application Deployment (approved namespace only).")
def k8s_restart_deployment(args: RestartArgs) -> dict:
    _, apps = get_apis()
    name = get_settings().k8s_deployment
    stamp = datetime.now(timezone.utc).isoformat()
    body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": stamp}}}}}
    apps.patch_namespaced_deployment(name, _ns(), body, **_T)
    return {"deployment": name, "namespace": _ns(), "action": "rolling restart requested", "restarted_at": stamp}


class ScaleArgs(NsArgs):
    replicas: int = Field(ge=1, le=5)


@REGISTRY.tool(name="k8s_scale_deployment", module="kubernetes", args=ScaleArgs, mutating=True,
               description="Scale the application Deployment to 1-5 replicas (approved namespace only).")
def k8s_scale_deployment(args: ScaleArgs) -> dict:
    _, apps = get_apis()
    name = get_settings().k8s_deployment
    apps.patch_namespaced_deployment_scale(name, _ns(), {"spec": {"replicas": args.replicas}}, **_T)
    return {"deployment": name, "namespace": _ns(), "action": f"scaled to {args.replicas} replicas"}
