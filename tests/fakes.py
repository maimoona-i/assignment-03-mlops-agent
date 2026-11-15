"""Hand-written fakes so tests never need MLflow / Kubernetes / Airflow / an LLM."""
from __future__ import annotations

from types import SimpleNamespace as NS

from pydantic import BaseModel, ConfigDict

from agent.registry import Tool, ToolRegistry

# ---------------------------------------------------------------- Kubernetes fakes


def cstatus(ready=True, restarts=0, waiting=None, terminated_reason=None, running=True):
    state = NS(running=NS() if (running and not waiting) else None,
               waiting=NS(reason=waiting) if waiting else None, terminated=None)
    last = NS(terminated=NS(reason=terminated_reason, exit_code=137)) if terminated_reason else None
    return NS(name="api", ready=ready, restart_count=restarts, state=state, last_state=last, image="mlops-inference:1.0.0")


def pod(name, phase="Running", containers=None):
    return NS(metadata=NS(name=name, creation_timestamp=None), spec=NS(node_name="node-1"),
              status=NS(phase=phase, container_statuses=containers if containers is not None else [cstatus()]))


def deployment(desired=2, ready=2, image="mlops-inference:1.0.0"):
    return NS(metadata=NS(name="ml-inference-api", labels={"app.kubernetes.io/version": "1.0.0"}),
              spec=NS(replicas=desired, template=NS(spec=NS(containers=[NS(image=image)]))),
              status=NS(ready_replicas=ready, available_replicas=ready, updated_replicas=ready, conditions=[]))


def warning_event(reason, message, obj="pod-1"):
    return NS(type="Warning", reason=reason, message=message, involved_object=NS(name=obj), count=3,
              last_timestamp="2026-11-15T10:00:00", event_time=None)


class FakeCore:
    def __init__(self, pods=None, events=None, endpoints=2, logs="line1\nline2"):
        self.pods, self.events, self.endpoints, self.logs = pods or [], events or [], endpoints, logs

    def list_namespaced_pod(self, ns, **kw):
        return NS(items=self.pods)

    def list_namespaced_event(self, ns, **kw):
        return NS(items=self.events)

    def read_namespaced_pod_log(self, name, ns, **kw):
        return self.logs

    def list_namespaced_service(self, ns, **kw):
        return NS(items=[NS(metadata=NS(name="ml-inference-svc"),
                            spec=NS(type="ClusterIP", cluster_ip="10.0.0.1", selector={"app": "ml-inference"},
                                    ports=[NS(port=80, target_port=8000)]))])

    def read_namespaced_endpoints(self, name, ns, **kw):
        return NS(subsets=[NS(addresses=list(range(self.endpoints)))] if self.endpoints else None)

    def read_namespaced_config_map(self, name, ns, **kw):
        return NS(data={"APP_VERSION": "1.0.0", "MODEL_RUN_ID": "a" * 32, "DATASET_VERSION": "v1"})


class FakeApps:
    def __init__(self, deployments=None):
        self.deployments, self.patches, self.scales = deployments or [deployment()], [], []

    def list_namespaced_deployment(self, ns, **kw):
        return NS(items=self.deployments)

    def read_namespaced_deployment(self, name, ns, **kw):
        return self.deployments[0]

    def patch_namespaced_deployment(self, name, ns, body, **kw):
        self.patches.append((name, ns, body))

    def patch_namespaced_deployment_scale(self, name, ns, body, **kw):
        self.scales.append((name, ns, body))


# ---------------------------------------------------------------- MLflow fakes
def mlflow_run(run_id, name, metrics, status="FINISHED", tags=None, start=1_762_000_000_000):
    t = {"mlflow.runName": name, "quality_gate": "passed", "dvc.md5": "abc123", "git.commit": "deadbeef",
         "dataset.version": "v2.1", "dvc.path": "data/dataset.csv", **(tags or {})}
    return NS(info=NS(run_id=run_id, experiment_id="1", status=status, start_time=start),
              data=NS(metrics=metrics, params={"n_estimators": "100"}, tags=t))


class FakeMlflow:
    def __init__(self, runs):
        self.runs = runs

    def get_experiment_by_name(self, name):
        return NS(experiment_id="1", name=name)

    def search_runs(self, ids, filter_string="", order_by=None, max_results=100):
        return self.runs[:max_results]

    def get_run(self, run_id):
        for r in self.runs:
            if r.info.run_id == run_id:
                return r
        raise KeyError(run_id)

    def list_artifacts(self, run_id):
        return [NS(path="model")]


# ---------------------------------------------------------------- stub registry for agent tests
class Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


RUN_ID = "a" * 32
BEST = {
    "metric": "f1", "mode": "max", "runs_considered": 5,
    "best": {"run_id": RUN_ID, "run_name": "rf-v2", "experiment_id": "1", "status": "FINISHED",
             "start_time": "2026-11-10T10:00:00+00:00", "metrics": {"f1": 0.9312, "accuracy": 0.94},
             "params": {}, "provenance": {"quality_gate": "passed"}},
    "ranking": [{"run_id": RUN_ID, "run_name": "rf-v2", "f1": 0.9312}],
    "checks": [{"name": "run_finished", "status": "pass", "detail": "status=FINISHED"}],
    "caveats": ["Single hold-out evaluation; no confidence interval or drift/latency validation was recorded."],
    "production_ready": "not determined",
}
HEALTHY = {"namespace": "mlops-agent", "healthy": True, "pods": [{"name": "p1"}, {"name": "p2"}], "unhealthy_pods": [],
           "deployments": [{"name": "ml-inference-api", "desired": 2, "ready": 2, "available": 2, "updated": 2,
                            "images": ["mlops-inference:1.0.0"], "conditions": []}],
           "warning_events": [], "service_endpoints": [], "log_tail_of_first_unhealthy_pod": [], "findings": [],
           "likely_cause": "none found"}
DEFAULTS = {
    "dvc_get_dataset_version": ("dvc", False, {"dataset": "data/dataset.csv", "dataset_version": "v2.1", "md5": "abc123",
                                               "git_commit_of_dvc_file": "a4c92e1", "git_tags": ["v2"], "git_head": "f00dbabe1234"}),
    "dvc_status": ("dvc", False, {"dataset": "data/dataset.csv", "changed": False, "details": {}}),
    "dvc_list_tracked_datasets": ("dvc", False, {"datasets": [{"path": "data/dataset.csv"}], "count": 1}),
    "dvc_compare_versions": ("dvc", False, {}),
    "dvc_explain_reproduce": ("dvc", False, {"steps": ["git checkout a4c92e1 -- data/dataset.csv.dvc", "dvc checkout x", "dvc status"]}),
    "dvc_experiment_dataset": ("dvc", False, {"run_id": RUN_ID, "run_name": "rf-v2", "dataset_version": "v2.1", "dvc_md5": "abc123",
                                              "git_commit": "a4c92e1", "provenance_recorded": True,
                                              "provenance_verified_against_git": True,
                                              "dvc_remote": {"configured": True, "name": "storage"}}),
    "mlflow_best_run": ("mlflow", False, BEST),
    "mlflow_compare_runs": ("mlflow", False, {}),
    "mlflow_registry_info": ("mlflow", False, {"registered_models": [], "count": 0, "note": "none"}),
    "airflow_dag_runs": ("airflow", False, {"dag_id": "training_pipeline",
                                            "runs": [{"dag_run_id": "manual__1", "state": "success", "start_date": "2026-11-12"}]}),
    "airflow_task_instances": ("airflow", False, {"dag_run_id": "manual__1", "run_state": "success",
                                                  "tasks": [{"task_id": "validate_dataset", "state": "success"}]}),
    "airflow_failed_tasks": ("airflow", False, {"dag_run_id": "manual__1", "run_state": "success", "failed_tasks": [], "count": 0}),
    "airflow_task_logs": ("airflow", False, {"log_tail": ["boom"]}),
    "airflow_trigger_dag": ("airflow", True, {"dag_id": "training_pipeline", "dag_run_id": "manual__9", "state": "success", "finished": True}),
    "k8s_diagnose": ("kubernetes", False, HEALTHY),
    "k8s_app_version": ("kubernetes", False, {"image_tags": ["1.0.0"], "version_label": "1.0.0",
                                              "config": {"MODEL_RUN_ID": RUN_ID, "APP_VERSION": "1.0.0"}}),
    "k8s_list_services": ("kubernetes", False, {"services": []}),
    "k8s_pod_logs": ("kubernetes", False, {"pod": {"name": "p1", "phase": "Running"}, "log_tail": ["ok"]}),
    "k8s_restart_deployment": ("kubernetes", True, {"deployment": "ml-inference-api", "action": "rolling restart requested",
                                                    "restarted_at": "2026-11-15T10:00:00Z"}),
    "k8s_scale_deployment": ("kubernetes", True, {"deployment": "ml-inference-api", "action": "scaled to 3 replicas"}),
}


def stub_registry(**overrides) -> ToolRegistry:
    """A registry with stub tools under the real tool names. An override can be a dict, a callable or an Exception."""
    reg = ToolRegistry()
    for name, (module, mutating, default) in DEFAULTS.items():
        impl = overrides.get(name, default)

        def make(impl):
            def fn(args):
                if isinstance(impl, Exception):
                    raise impl
                return impl(args) if callable(impl) else impl
            return fn

        reg.register(Tool(name, "stub", Loose, make(impl), module, mutating))
    return reg
