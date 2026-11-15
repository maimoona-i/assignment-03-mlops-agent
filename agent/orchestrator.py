"""The MLOps agent: decides which tools to call, collects evidence, and writes an evidence-based answer.

Two planners share the same tool registry and the same safety layer:
  * LLM mode   - the LLM selects tools via structured tool calling (needs ANTHROPIC_API_KEY).
  * Rules mode - a deterministic keyword planner (works offline; used by tests and as a fallback).

In both modes operational claims come ONLY from tool results. If a tool fails, the failure is reported as
an uncertainty - the agent never fills the gap with guesses.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone

from agent.config import get_settings
from agent.logging_utils import log_event
from agent.registry import ToolContext, ToolRegistry
from agent.schemas import ChatResponse, ToolCallRecord, ToolResult

SYSTEM_PROMPT = """You are an MLOps assistant with tools for DVC (dataset versions), MLflow (experiments), Airflow (pipelines) and Kubernetes (deployment).
Rules:
1. Operational claims (metrics, versions, statuses, logs) MUST come from tool results in this conversation. Never use memory or guesses for them.
2. If a tool fails, times out or is denied, say so explicitly and do not invent the missing information.
3. Never assume the highest metric means a model is production-ready; mention the readiness checks and caveats returned by the tools.
4. State-changing tools (restart/scale/trigger) are only executed when the user has explicitly confirmed; if a tool returns 'confirmation_required', tell the user what confirmation is needed. Never trigger retraining on your own.
5. Recommend retraining ONLY with evidence: a failed quality gate, a metric below the stated policy threshold, an explicit freshness policy breach, or user-supplied degradation results.
6. For multi-part questions, call all relevant tools, correlate the evidence, and answer in three sections: 'Known facts', 'Uncertainties', 'Recommended actions'."""

_ADVISORY = ("whether", "should", "need", "necessary", "is it", "do we", "do i", "explain")
_DIAG_WORDS = ("why", "unavailable", "down", "failing", "unhealthy", "diagnos", "broken", "not working", "error", "crash")
_KEYWORDS = {
    "dvc": ("dataset", "dvc", "data version", "data change", "provenance", "reproduce"),
    "mlflow": ("mlflow", "best model", "best run", "experiment", "f1", "metric", "accuracy", "registry", "compare",
               "candidate", "recent model", "runs"),
    "airflow": ("airflow", "dag", "pipeline", "workflow", "trigger", "retrain"),
    "k8s": ("kubernetes", "k8s", "pod", "deployment", "deployed", "service", "unavailable", "replica", "restart",
            "rollout", "prediction api", "inference", "scale"),
}


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def _failure_text(res: ToolResult) -> str:
    return {
        "timeout": "timed out - result unknown.",
        "denied": f"request denied: {res.error}",
        "confirmation_required": f"NOT executed: {res.error}",
        "invalid": f"invalid request: {res.error}",
    }.get(res.status, f"tool unavailable or failed: {res.error}")


class _Run:
    """Collects tool calls and evidence for one user request."""

    def __init__(self, registry: ToolRegistry, ctx: ToolContext):
        self.registry, self.ctx = registry, ctx
        self.calls: list[ToolCallRecord] = []
        self.facts: list[str] = []
        self.unc: list[str] = []
        self.recs: list[str] = []

    def call(self, name: str, **args) -> ToolResult:
        res = self.registry.execute(name, args, self.ctx)
        self.calls.append(ToolCallRecord(tool=name, args=args, status=res.status, duration_ms=res.duration_ms))
        if res.status != "ok":
            self.unc.append(f"[{name}] {_failure_text(res)}")
        if res.status == "confirmation_required":
            self.recs.append("To proceed, confirm explicitly (set confirm=true / use --confirm) and resend the request.")
        return res


def render(facts: list[str], unc: list[str], recs: list[str]) -> str:
    parts = []
    parts.append("Known facts (from tool output):\n" + "\n".join(f"- {f}" for f in facts) if facts
                 else "No operational facts could be verified.")
    if unc:
        parts.append("Uncertainties / not verified:\n" + "\n".join(f"- {u}" for u in unc))
    if recs:
        parts.append("Recommended actions:\n" + "\n".join(f"- {r}" for r in recs))
    return "\n\n".join(parts)


class Agent:
    def __init__(self, registry: ToolRegistry | None = None, llm=None, max_steps: int = 8):
        if registry is None:
            import agent.tools  # noqa: F401  (registers all tools)
            from agent.registry import REGISTRY

            registry = REGISTRY
        self.registry, self.llm, self.max_steps = registry, llm, max_steps

    # ------------------------------------------------------------------ entry point
    def handle(self, message: str, confirm: bool = False, request_id: str | None = None,
               degradation_report: str | None = None) -> ChatResponse:
        rid = request_id or uuid.uuid4().hex[:12]
        ctx = ToolContext(request_id=rid, confirmed=confirm)
        log_event("request_received", request_id=rid, mode="llm" if self.llm else "rules", confirm=confirm)
        if self.llm is not None:
            try:
                return self._handle_llm(message, ctx)
            except Exception as exc:  # LLM outage must not take the agent down
                log_event("llm_failure_fallback_to_rules", request_id=rid, error=type(exc).__name__)
                resp = self._handle_rules(message, ctx, degradation_report)
                resp.uncertainties.insert(0, f"LLM planner unavailable ({type(exc).__name__}); used rule-based planner.")
                resp.answer = render(resp.facts, resp.uncertainties, resp.recommendations)
                return resp
        return self._handle_rules(message, ctx, degradation_report)

    # ------------------------------------------------------------------ LLM planner
    def _handle_llm(self, message: str, ctx: ToolContext) -> ChatResponse:
        messages: list[dict] = [{"role": "user", "content": message}]
        records: list[ToolCallRecord] = []
        failures: list[str] = []
        answer = ""
        for _ in range(self.max_steps):
            step = self.llm.step(SYSTEM_PROMPT, messages, self.registry.specs())
            messages.append({"role": "assistant", "content": step.content or step.text})
            if not step.tool_calls:
                answer = step.text
                break
            results = []
            for tc in step.tool_calls:
                res = self.registry.execute(tc.name, tc.input, ctx)
                records.append(ToolCallRecord(tool=tc.name, args=tc.input, status=res.status, duration_ms=res.duration_ms))
                if res.status != "ok":
                    failures.append(f"{tc.name}: {_failure_text(res)}")
                results.append({"type": "tool_result", "tool_use_id": tc.id, "is_error": res.status != "ok",
                                "content": json.dumps(res.model_dump(), default=str)[:12000]})
            messages.append({"role": "user", "content": results})
        else:
            answer = "I could not finish within the allowed number of tool steps."
        if failures:  # deterministic safety net, independent of what the LLM wrote
            answer += "\n\nTool problems during this request (information from these tools is NOT verified):\n" + \
                      "\n".join(f"- {f}" for f in failures)
        return ChatResponse(request_id=ctx.request_id, mode="llm", answer=answer, tool_calls=records,
                            uncertainties=failures)

    # ------------------------------------------------------------------ rule-based planner
    def _handle_rules(self, message: str, ctx: ToolContext, degradation_report: str | None = None) -> ChatResponse:
        run = _Run(self.registry, ctx)
        l = message.lower()
        domains = {d for d, words in _KEYWORDS.items() if any(w in l for w in words)}
        explicit_trigger = bool(re.search(r"\b(trigger|start|launch|kick ?off|execute)\b", l))
        retrain_advisory = "retrain" in l and not explicit_trigger and any(w in l for w in _ADVISORY)
        if domains == {"dvc", "mlflow"} and not any(w in l for w in ("best", "compare", "candidate", "f1")):
            domains = {"dvc"}

        if len(domains) >= 3 or retrain_advisory:
            self._multi(run, message, degradation_report)
        elif not domains:
            run.unc.append("I did not recognise an MLOps task. I can help with DVC datasets, MLflow experiments, "
                           "Airflow pipelines and the Kubernetes deployment.")
        else:
            if "dvc" in domains:
                self._dataset(run, message)
            if "mlflow" in domains:
                self._models(run, message)
            if "airflow" in domains:
                self._pipeline(run, message, trigger=explicit_trigger or ("retrain" in l and not retrain_advisory))
            if "k8s" in domains:
                self._deployment(run, message)
        return ChatResponse(request_id=ctx.request_id, mode="rules", answer=render(run.facts, run.unc, run.recs),
                            facts=run.facts, uncertainties=run.unc, recommendations=run.recs, tool_calls=run.calls)

    # ---- DVC ----
    def _dataset(self, run: _Run, message: str) -> None:
        l = message.lower()
        m = (re.search(r"\bexperiment\s+#?(\d+)", l) or re.search(r"\brun\s+([0-9a-f]{32})", l)
             or re.search(r"\b([0-9a-f]{32})\b", l))
        if m:
            r = run.call("dvc_experiment_dataset", identifier=m.group(1))
            if r.status == "ok":
                self._fact_provenance(run, r.data)
                if any(w in l for w in ("reproduce", "recreate", "restore", "checkout")):
                    rr = run.call("dvc_explain_reproduce", identifier=m.group(1))
                    if rr.status == "ok":
                        run.facts.append("[dvc_explain_reproduce] Reproduce with: " + " | ".join(
                            s.split("#")[0].strip() for s in rr.data["steps"][:3]))
            return
        v = run.call("dvc_get_dataset_version")
        if v.status == "ok":
            d = v.data
            run.facts.append(f"[dvc_get_dataset_version] {d['dataset']}: version label={d['dataset_version']}, "
                             f"md5={d['md5']}, last changed in commit {d['git_commit_of_dvc_file']} "
                             f"(tags: {', '.join(d['git_tags']) or 'none'}); HEAD={d['git_head'][:8]}.")
        s = run.call("dvc_status")
        if s.status == "ok":
            run.facts.append("[dvc_status] Tracked data has CHANGED vs its recorded version." if s.data["changed"]
                             else "[dvc_status] Tracked data is unchanged (matches the recorded DVC version).")
        t = run.call("dvc_list_tracked_datasets")
        if t.status == "ok":
            run.facts.append("[dvc_list_tracked_datasets] Tracked: " + ", ".join(x["path"] for x in t.data["datasets"]))
        if any(w in l for w in ("compare", "diff", "difference", "between")):
            c = run.call("dvc_compare_versions")
            if c.status == "ok":
                run.facts.append(f"[dvc_compare_versions] {c.data['a']['rev']} vs {c.data['b']['rev']}: "
                                 + ("identical data." if c.data["identical"] else
                                    f"DIFFERENT (md5 {c.data['a']['md5'][:8]} -> {c.data['b']['md5'][:8]}, "
                                    f"size delta {c.data['size_delta_bytes']} bytes)."))
        if any(w in l for w in ("reproduce", "recreate", "restore")) and v.status == "ok":
            rr = run.call("dvc_explain_reproduce", git_rev=v.data["git_commit_of_dvc_file"] or "HEAD")
            if rr.status == "ok":
                run.facts.append("[dvc_explain_reproduce] " + " | ".join(x.split("#")[0].strip() for x in rr.data["steps"][:3]))

    @staticmethod
    def _fact_provenance(run: _Run, d: dict) -> None:
        if not d["provenance_recorded"]:
            run.unc.append(f"[dvc_experiment_dataset] Run {d['run_id'][:8]} has NO recorded dataset provenance.")
            return
        ver = {True: "verified against Git", False: "MISMATCH with Git record", None: "could not be verified locally"}[
            d["provenance_verified_against_git"]]
        remote = d["dvc_remote"]
        run.facts.append(
            f"[dvc_experiment_dataset] Run {d['run_id'][:8]} ({d['run_name']}) used dataset version "
            f"{d['dataset_version']} (md5 {d['dvc_md5']}), Git commit {d['git_commit']} - provenance {ver}. "
            + (f"DVC remote: {remote['name']}." if remote.get("configured") else "No DVC remote configured."))

    # ---- MLflow ----
    def _models(self, run: _Run, message: str) -> None:
        l = message.lower()
        metric = next((m for m in ("f1", "accuracy", "precision", "recall") if m in l), "f1")
        n = re.search(r"(?:last|latest|recent)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)", l)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        last_n = int(words.get(n.group(1), n.group(1))) if n else 10
        if "compare" in l:
            r = run.call("mlflow_compare_runs", last_n=max(2, min(last_n, 20)), metric=metric)
            if r.status == "ok":
                for row in r.data["runs"]:
                    run.facts.append(f"[mlflow_compare_runs] {row['run_id'][:8]} ({row['run_name']}): "
                                     + ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(row["metrics"].items())))
                if r.data["differing_params"]:
                    run.facts.append("[mlflow_compare_runs] Differing parameters: " + json.dumps(r.data["differing_params"]))
                run.facts.append(f"[mlflow_compare_runs] Highest {metric}: run {str(r.data['best_by_metric'])[:8]}.")
                run.recs.append(r.data["note"])
        b = run.call("mlflow_best_run", metric=metric, last_n=max(1, min(last_n, 50)))
        if b.status == "ok":
            self._fact_best(run, b.data, metric)
        if any(w in l for w in ("registry", "registered", "model version")):
            reg = run.call("mlflow_registry_info")
            if reg.status == "ok":
                run.facts.append(f"[mlflow_registry_info] {reg.data['count']} registered model(s): " + "; ".join(
                    f"{m['name']} versions {[v['version'] for v in m['versions']]}" for m in reg.data["registered_models"])
                    if reg.data["count"] else f"[mlflow_registry_info] {reg.data['note']}")

    @staticmethod
    def _fact_best(run: _Run, d: dict, metric: str) -> None:
        best = d["best"]
        run.facts.append(f"[mlflow_best_run] Best of {d['runs_considered']} recent finished runs by {metric}: "
                         f"{best['run_id'][:8]} ({best['run_name']}), " +
                         ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(best["metrics"].items())) +
                         f"; started {best['start_time']}.")
        for c in d["checks"]:
            line = f"[mlflow_best_run] check {c['name']}: {c['status']} ({c['detail']})"
            (run.facts if c["status"] == "pass" else run.unc).append(line)
        run.unc.extend(f"[mlflow_best_run] Caveat: {c}" for c in d["caveats"])
        run.recs.append(f"Treat {metric} as a ranking only: a high score does not make the model production-ready - "
                        "review the checks/caveats above and validate before promotion.")

    # ---- Airflow ----
    def _pipeline(self, run: _Run, message: str, trigger: bool) -> None:
        dag = get_settings().airflow_training_dag
        if trigger:
            r = run.call("airflow_trigger_dag", dag_id=dag, wait_seconds=60)
            if r.status == "ok":
                d = r.data
                run.facts.append(f"[airflow_trigger_dag] Triggered {d['dag_id']} run {d['dag_run_id']}: state={d['state']}"
                                 + ("" if d["finished"] else " (not finished yet)") + ".")
                if d["state"] == "failed":
                    self._failed_details(run, dag, d["dag_run_id"])
            return
        r = run.call("airflow_dag_runs", dag_id=dag, limit=5)
        if r.status != "ok":
            return
        runs = r.data["runs"]
        if not runs:
            run.unc.append(f"[airflow_dag_runs] DAG {dag} has no runs yet.")
            return
        for x in runs:
            run.facts.append(f"[airflow_dag_runs] {x['dag_run_id']}: {x['state']} (start {x['start_date']})")
        t = run.call("airflow_task_instances", dag_id=dag, dag_run_id=runs[0]["dag_run_id"])
        if t.status == "ok":
            run.facts.append("[airflow_task_instances] Latest run tasks: " + ", ".join(f"{x['task_id']}={x['state']}" for x in t.data["tasks"]))
        if runs[0]["state"] == "failed":
            self._failed_details(run, dag, runs[0]["dag_run_id"])

    @staticmethod
    def _failed_details(run: _Run, dag: str, run_id: str) -> list[dict]:
        f = run.call("airflow_failed_tasks", dag_id=dag, dag_run_id=run_id)
        if f.status != "ok":
            return []
        failed = [t for t in f.data["failed_tasks"] if t["state"] == "failed"]
        for t in f.data["failed_tasks"]:
            run.facts.append(f"[airflow_failed_tasks] Task {t['task_id']} -> {t['state']} (try {t['try_number']}/{t['max_tries']})")
        if failed:
            lg = run.call("airflow_task_logs", dag_id=dag, dag_run_id=run_id, task_id=failed[0]["task_id"],
                          try_number=max(1, failed[0]["try_number"] or 1), tail_lines=8)
            if lg.status == "ok":
                run.facts.append(f"[airflow_task_logs] {failed[0]['task_id']} log tail: " + " | ".join(lg.data["log_tail"][-5:]))
        return f.data["failed_tasks"]

    # ---- Kubernetes ----
    def _deployment(self, run: _Run, message: str) -> None:
        l = message.lower()
        scale = re.search(r"scale\s+(?:\w+\s+)?(?:to\s+)?(\d+)", l)
        if "restart" in l or "rollout" in l:
            r = run.call("k8s_restart_deployment")
            if r.status == "ok":
                run.facts.append(f"[k8s_restart_deployment] {r.data['action']} for {r.data['deployment']} at {r.data['restarted_at']}.")
            return
        if scale:
            r = run.call("k8s_scale_deployment", replicas=int(scale.group(1)))
            if r.status == "ok":
                run.facts.append(f"[k8s_scale_deployment] {r.data['deployment']}: {r.data['action']}.")
            return
        if "log" in l:
            r = run.call("k8s_pod_logs", tail_lines=30)
            if r.status == "ok":
                run.facts.append(f"[k8s_pod_logs] Pod {r.data['pod']['name']} (phase {r.data['pod']['phase']}) log tail: "
                                 + " | ".join(r.data["log_tail"][-8:]))
            return
        diag = run.call("k8s_diagnose")
        ver = run.call("k8s_app_version")
        self._fact_k8s(run, diag, ver)
        if "service" in l:
            s = run.call("k8s_list_services")
            if s.status == "ok":
                for x in s.data["services"]:
                    run.facts.append(f"[k8s_list_services] {x['name']} ({x['type']}, {x['cluster_ip']}): ready endpoints={x['ready_endpoints']}")

    @staticmethod
    def _fact_k8s(run: _Run, diag: ToolResult, ver: ToolResult) -> None:
        if diag.status == "ok":
            d = diag.data
            for dep in d["deployments"]:
                run.facts.append(f"[k8s_diagnose] Deployment {dep['name']}: {dep['ready']}/{dep['desired']} replicas ready, images={dep['images']}")
            run.facts.append(f"[k8s_diagnose] {len(d['pods'])} pod(s); unhealthy: {', '.join(d['unhealthy_pods']) or 'none'}.")
            run.facts.extend(f"[k8s_diagnose] {f}" for f in d["findings"])
            if d["log_tail_of_first_unhealthy_pod"]:
                run.facts.append("[k8s_diagnose] Log tail: " + " | ".join(d["log_tail_of_first_unhealthy_pod"][-5:]))
            run.facts.append(f"[k8s_diagnose] Likely cause: {d['likely_cause']}.")
            if not d["healthy"]:
                run.recs.append("Fix the deployment issue above (infrastructure/config problem; not by itself a reason to retrain). "
                                "A restart can be done on request with explicit confirmation.")
        if ver.status == "ok":
            v = ver.data
            run.facts.append(f"[k8s_app_version] Deployed image tag(s) {v['image_tags']}, version label={v['version_label']}, config={v['config']}.")

    # ---- Multi-tool scenario ----
    def _multi(self, run: _Run, message: str, degradation_report: str | None) -> None:
        s = get_settings()
        best_res = run.call("mlflow_best_run", metric="f1", last_n=10)
        best = best_res.data["best"] if best_res.status == "ok" else None
        if best_res.status == "ok":
            self._fact_best(run, best_res.data, "f1")
        if best:
            ds = run.call("dvc_experiment_dataset", identifier=best["run_id"])
            if ds.status == "ok":
                self._fact_provenance(run, ds.data)
        else:
            run.unc.append("Dataset version for the best model was not checked because no best run could be identified.")

        diag, ver = run.call("k8s_diagnose"), run.call("k8s_app_version")
        self._fact_k8s(run, diag, ver)
        deployed_run = (ver.data["config"].get("MODEL_RUN_ID") if ver.status == "ok" else None)
        if best and ver.status == "ok":
            if not deployed_run:
                run.unc.append("Deployed model run id is not recorded in the ConfigMap, so deployed-vs-best cannot be compared.")
            elif deployed_run == best["run_id"]:
                run.facts.append("[k8s_app_version] The deployed model IS the best recent run.")
            else:
                run.facts.append(f"[k8s_app_version] Deployed model run {deployed_run[:8]} differs from the best recent run {best['run_id'][:8]}.")
                run.recs.append("Consider promoting the best run after review/validation (deployment change, not retraining).")

        dag = s.airflow_training_dag
        runs = run.call("airflow_dag_runs", dag_id=dag, limit=5)
        gate_failed = False
        if runs.status == "ok":
            for x in runs.data["runs"]:
                run.facts.append(f"[airflow_dag_runs] {x['dag_run_id']}: {x['state']}")
            if not runs.data["runs"]:
                run.unc.append(f"[airflow_dag_runs] No runs recorded for {dag}.")
            elif runs.data["runs"][0]["state"] == "failed":
                failed = self._failed_details(run, dag, runs.data["runs"][0]["dag_run_id"])
                gate_failed = any(t["task_id"] == "quality_gate" and t["state"] == "failed" for t in failed)

        # --- correlate: retraining only with explicit evidence, and NEVER triggered from here ---
        reasons = []
        if gate_failed:
            reasons.append("the latest training run failed the model quality gate")
        if best and best["metrics"].get("f1") is not None and best["metrics"]["f1"] < s.min_f1:
            reasons.append(f"the best recorded f1 ({_fmt(best['metrics']['f1'])}) is below the policy threshold {s.min_f1}")
        if s.freshness_days is None:
            run.unc.append("No freshness policy (FRESHNESS_DAYS) is configured, so model age was not used as a retraining criterion.")
        elif best and best["start_time"]:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(best["start_time"])).days
            if age > s.freshness_days:
                reasons.append(f"the best model is {age} days old, exceeding the freshness policy of {s.freshness_days} days")
        if degradation_report:
            reasons.append(f"a degradation report was supplied: {degradation_report[:200]}")
        if reasons:
            run.recs.append("Retraining is justified because " + "; and ".join(reasons) + ". It was NOT started automatically - "
                            "confirm explicitly if you want the training pipeline triggered.")
        else:
            run.recs.append("No evidence-based reason to retrain was found (no failed quality gate, no metric below policy, "
                            "no freshness-policy breach, no supplied degradation report). Retraining is not recommended.")
