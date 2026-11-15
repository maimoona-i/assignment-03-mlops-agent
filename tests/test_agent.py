from agent.llm import LLMStep, ToolUse
from agent.orchestrator import Agent
from agent.errors import ToolError
from tests.fakes import RUN_ID, stub_registry


def names(resp):
    return [c.tool for c in resp.tool_calls]


def test_infrastructure_question_selects_only_kubernetes_tools():
    resp = Agent(stub_registry()).handle("Why is my prediction API unavailable?")
    assert names(resp) and all(n.startswith("k8s_") for n in names(resp))


def test_dataset_question_uses_recorded_provenance():
    resp = Agent(stub_registry()).handle("Which dataset version was used for experiment 25?")
    assert names(resp) == ["dvc_experiment_dataset"]
    assert resp.tool_calls[0].args == {"identifier": "25"}
    assert "v2.1" in resp.answer and "a4c92e1" in resp.answer


def test_model_comparison_uses_mlflow_and_hedges_readiness():
    resp = Agent(stub_registry(mlflow_compare_runs={"runs": [{"run_id": RUN_ID, "run_name": "rf", "metrics": {"f1": 0.9}}],
                                                    "differing_params": {}, "best_by_metric": RUN_ID,
                                                    "note": "Highest metric is a ranking, not a production-readiness decision."})
                ).handle("Compare the last five training runs and recommend the best candidate based on F1 score.")
    assert {"mlflow_compare_runs", "mlflow_best_run"} <= set(names(resp))
    assert "production-ready" in resp.answer or "production-readiness" in resp.answer


def test_multi_tool_scenario_uses_all_four_systems_and_never_triggers_retraining():
    msg = ("Find the best recent model, identify the dataset version used to train it, check the deployment status, "
           "and explain whether retraining is necessary.")
    resp = Agent(stub_registry()).handle(msg)
    used = names(resp)
    assert any(n.startswith("mlflow_") for n in used) and any(n.startswith("dvc_") for n in used)
    assert any(n.startswith("k8s_") for n in used) and any(n.startswith("airflow_") for n in used)
    assert "airflow_trigger_dag" not in used
    assert any("No evidence-based reason to retrain" in r for r in resp.recommendations)
    assert resp.facts and resp.uncertainties  # facts and uncertainties are separated


def test_retraining_recommended_only_with_evidence_and_still_not_triggered():
    failing_runs = {"dag_id": "training_pipeline", "runs": [{"dag_run_id": "manual__2", "state": "failed", "start_date": "x"}]}
    failed = {"dag_run_id": "manual__2", "run_state": "failed", "count": 1,
              "failed_tasks": [{"task_id": "quality_gate", "state": "failed", "try_number": 1, "max_tries": 2}]}
    reg = stub_registry(airflow_dag_runs=failing_runs, airflow_failed_tasks=failed)
    resp = Agent(reg).handle("Should we retrain? Check the best model, dataset and deployment too.")
    assert any("justified" in r and "quality gate" in r for r in resp.recommendations)
    assert "airflow_trigger_dag" not in names(resp)


def test_degradation_report_is_accepted_as_retraining_evidence():
    resp = Agent(stub_registry()).handle(
        "Find the best recent model, its dataset version, the deployment status and whether retraining is necessary.",
        degradation_report="Live F1 dropped from 0.93 to 0.78 over two weeks")
    assert any("justified" in r and "degradation" in r for r in resp.recommendations)


def test_agent_does_not_invent_results_when_a_tool_is_unavailable():
    resp = Agent(stub_registry(mlflow_best_run=ToolError("connection refused"))).handle(
        "Which model is best by f1 in mlflow?")
    assert resp.facts == []
    assert "No operational facts could be verified" in resp.answer
    assert "connection refused" in resp.answer
    assert not any(ch.isdigit() for ch in resp.answer.split("Uncertainties")[0])  # no invented numbers


def test_partial_outage_is_reported_as_uncertainty():
    msg = "Find the best recent model, identify the dataset version, check the deployment and explain if retraining is necessary."
    resp = Agent(stub_registry(k8s_diagnose=ToolError("cluster unreachable"))).handle(msg)
    assert any("k8s_diagnose" in u and "cluster unreachable" in u for u in resp.uncertainties)
    assert any("v2.1" in f for f in resp.facts)  # other systems still reported


def test_restart_needs_confirmation_through_the_agent():
    agent = Agent(stub_registry())
    first = agent.handle("Please restart the deployment")
    assert first.tool_calls[0].status == "confirmation_required" and "NOT executed" in first.answer
    second = agent.handle("Please restart the deployment", confirm=True)
    assert second.tool_calls[0].status == "ok" and "rolling restart" in second.answer


def test_unrecognised_request_calls_no_tools():
    resp = Agent(stub_registry()).handle("What is the weather like?")
    assert resp.tool_calls == [] and "did not recognise" in resp.answer


# ------------------------------------------------------------------ LLM mode
class FakeLLM:
    def __init__(self, steps):
        self.steps, self.histories = list(steps), []

    def step(self, system, messages, tools):
        self.histories.append(list(messages))
        return self.steps.pop(0)


def test_llm_loop_executes_selected_tool_and_returns_final_text():
    call = ToolUse("tu_1", "k8s_diagnose", {})
    llm = FakeLLM([LLMStep(tool_calls=[call], content=[{"type": "tool_use", "id": "tu_1", "name": "k8s_diagnose", "input": {}}]),
                   LLMStep(text="Deployment is healthy: 2/2 replicas ready.", content=[{"type": "text", "text": "ok"}])])
    resp = Agent(stub_registry(), llm=llm).handle("how is the deployment?")
    assert resp.mode == "llm" and names(resp) == ["k8s_diagnose"] and "healthy" in resp.answer
    tool_result = llm.histories[1][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and "ml-inference-api" in tool_result["content"]


def test_llm_cannot_bypass_confirmation_for_state_changes():
    call = ToolUse("tu_1", "k8s_restart_deployment", {})
    llm = FakeLLM([LLMStep(tool_calls=[call], content=[{"type": "tool_use", "id": "tu_1", "name": "k8s_restart_deployment", "input": {}}]),
                   LLMStep(text="I restarted it.", content=[{"type": "text", "text": "x"}])])
    resp = Agent(stub_registry(), llm=llm).handle("restart it", confirm=False)
    assert resp.tool_calls[0].status == "confirmation_required"
    assert "NOT verified" in resp.answer  # safety net appended even if the LLM over-claims


def test_llm_outage_falls_back_to_rules_with_note():
    class Broken:
        def step(self, *a, **k):
            raise RuntimeError("api down")

    resp = Agent(stub_registry(), llm=Broken()).handle("Why is my prediction API unavailable?")
    assert resp.mode == "rules" and any("LLM planner unavailable" in u for u in resp.uncertainties)
