# Technical Report – MLOps AI Agent (Assignment 03)

**Student:** Maimoona I.  **Course:** MLOps & Production AI Engineering, Fall 2026
**Repository:** https://github.com/maimoona-i/assignment-03-mlops-agent

---

## 1. System architecture

The agent answers questions about a real ML system through one interface. It never answers from memory: every operational fact comes from a tool response.

```
User → FastAPI / CLI → Agent (rule-based planner, or LLM tool-calling if an API key is set)
                         │
                         ▼
                 Tool Registry  (argument validation · approved namespace · confirmation
                                 for state changes · timeouts · structured JSON logs)
        ┌───────────┬───────────┴───────────┬──────────────┐
       DVC        MLflow                 Airflow 3      Kubernetes
   (git/dvc CLI)  (MlflowClient)       (REST API + JWT) (python client)
        └───────────┴───────────┬───────────┴──────────────┘
                                ▼
          Answer = Known facts · Uncertainties · Recommended actions
```

- **DVC:** dataset v1 and v2 tracked with `dvc add`, remote `localstorage`, Git tags `v1`, `v2`.
- **MLflow:** every training run logs params, metrics and provenance tags (`dataset.version`, `dvc.md5`, `git.commit`, `quality_gate`).
- **Airflow 3:** DAG `training_pipeline`: validate → preprocess → train → evaluate/log → quality gate → register model (retries=2; validation and gate failures are not retried and block promotion).
- **Kubernetes (minikube):** FastAPI inference service, Deployment with 2 replicas, ClusterIP Service, readiness (`/ready`) and liveness (`/health`) probes, CPU/memory requests and limits, ConfigMap for non-sensitive settings.

## 2. Tool definitions and permissions

| Module | Tools | Permission |
|---|---|---|
| DVC | `dvc_list_tracked_datasets`, `dvc_get_dataset_version`, `dvc_status`, `dvc_compare_versions`, `dvc_experiment_dataset`, `dvc_explain_reproduce` | read-only (allow-listed git/dvc commands, no shell) |
| MLflow | `mlflow_list_experiments`, `mlflow_list_runs`, `mlflow_get_run`, `mlflow_best_run`, `mlflow_compare_runs`, `mlflow_registry_info` | read-only |
| Airflow | `airflow_list_dags`, `airflow_dag_runs`, `airflow_task_instances`, `airflow_failed_tasks`, `airflow_task_logs` | read-only |
| Airflow | `airflow_trigger_dag` | **write**: approved DAG only, needs explicit confirmation |
| Kubernetes | `k8s_list_workloads`, `k8s_pod_logs`, `k8s_list_services`, `k8s_diagnose`, `k8s_app_version` | read-only, namespace `mlops-agent` only |
| Kubernetes | `k8s_restart_deployment`, `k8s_scale_deployment` | **write**: needs explicit confirmation, namespace `mlops-agent` only |

Safety rules: confirmation comes only from the user's request (`--confirm`), never from the LLM; other namespaces are rejected; secrets are redacted in logs; retraining is never started automatically.

## 3. Evidence screenshots

All images are in the `screenshots/` folder.

### DVC
![dvc status and remote](../screenshots/dvc_setup.png)
The DVC remote is `localstorage` (`/home/maimoona/dvc-remote`, default). Plain `dvc status` also lists the optional `train` pipeline stage as changed because `dvc repro` was never run; the dataset itself is tracked by `data/dataset.csv.dvc`, and the agent checks that file specifically (see Failure B, where it reports "unchanged" after the restore).

![reproduce dataset v1](../screenshots/s1_reproduce.png)
Reproducible checkout: md5 `36682424…` (v2) became `f11e36df…` (v1), and `dvc status` said "Data and pipelines are up to date".

### MLflow
![mlflow first runs](../screenshots/mlflow_runs_first.png)
![mlflow all runs](../screenshots/mlflow_runs.png)
Experiment `mlops-agent-demo` with 10 finished runs (random forest, gradient boosting, logistic regression, on dataset v1 and v2). Every run passed the gate and was registered as a new version of `churn-classifier` (v1 to v10 in the screenshot; Airflow adds more).

### Airflow
![airflow health](../screenshots/airflow_home.png)
Airflow 3: MetaDatabase, Scheduler, Triggerer and Dag Processor are all healthy.

![airflow successful run](../screenshots/airflow_success.png)
The agent-triggered run succeeded: validate_dataset, preprocess, train_model, evaluate_and_log, quality_gate and register_model all `success` in 26 seconds.

### Kubernetes and inference API
![pods starting](../screenshots/k8s_pods_start.png)
![deployment and service](../screenshots/k8s_healthy.png)
Deployment `ml-inference-api` 2/2 ready, both pods `1/1 Running`, Service `ml-inference-svc` of type ClusterIP (`10.111.187.219`, port 80).

![api checks](../screenshots/api_curl.png)
`/health` → alive, `/ready` → ready, `/version` → app 1.0.0, model run `149c9a5c…`, dataset v2, `/predict` → prediction 1 (probability 0.594).

### Tests and CI
![pytest and ruff](../screenshots/pytest_pass.png)
62 tests passed and `ruff` reports "All checks passed!".

![github actions](../screenshots/github_ci_green.png)
GitHub Actions: all three CI runs (Initial project + dataset v1, Pin SQLAlchemy<2.1, dataset v2) are green.

## 4. Five tested agent conversations (actual tool evidence)

**1. Dataset provenance** (tools: `dvc_experiment_dataset`, `dvc_status`, `dvc_get_dataset_version`)
The agent reported that the best run `16661c42` (gradient_boosting-v2) used dataset **v2**, md5 `36682424f93ae6c02c2425bf5b7941d4`, Git commit `abbe5ee2…`, "provenance verified against Git", DVC remote `localstorage`. Reproducibility was shown with `scripts/reproduce_dataset.sh v1`: the md5 changed from `36682424…` (v2) to `f11e36df…` (v1), `params.yaml` became `dataset_version: v1`, and `dvc status` said "Data and pipelines are up to date". Returning to v2 restored the original md5.

**2. Model selection** (tools: `mlflow_compare_runs`, `mlflow_best_run`)

![model selection](../screenshots/s2_models.png)

For "Compare the last five training runs and recommend the best candidate based on F1", the agent returned these F1 scores: 0.9246 (rf-v2), 0.9287 (gb-v1), **0.9385 (rf-v1)**, 0.9246 (rf-v2), 0.8750 (logistic regression, v2). It listed the differing parameters (model, dataset version, train rows, depth) and named `0e02a0c5` as the highest. It then ran readiness checks (finished, above the MIN_F1=0.85 policy, quality gate passed, provenance recorded, margin 0.0098) and stated that a high F1 does not by itself make a model production-ready, with a caveat about the single hold-out evaluation.

**3. Pipeline execution** (tool: `airflow_trigger_dag`)

![confirmation required](../screenshots/s3_confirm.png)
![confirmed run](../screenshots/s3_run.png)

Without `--confirm` the agent refused: "NOT executed… explicit user confirmation is required" and verified nothing. With `--confirm` it triggered a real Airflow run, which finished with `state=success`. The Airflow UI shows all six tasks succeeded (validate_dataset, preprocess, train_model, evaluate_and_log, quality_gate, register_model) in 26 seconds. A follow-up status question returned the same task states.

**4. Deployment diagnosis** (tools: `k8s_diagnose`, `k8s_app_version`)
Healthy state: 2/2 replicas ready, no unhealthy pods, "Likely cause: none found", image tag 1.0.0, config `DATASET_VERSION=v2`. During Failure A the diagnosis changed to the one in section 5.

**5. Multi-tool reasoning** (tools: MLflow, DVC, Kubernetes, Airflow)

![multi-tool answer](../screenshots/s5_multi_healthy.png)

With a healthy system the agent separated the answer into three parts. *Facts:* best run `16661c42` f1=0.9456, dataset v2 verified, 2/2 replicas ready, last Airflow run successful. It also found that the deployed model (run `149c9a5c`) differs from the best recent run. *Uncertainties:* single hold-out evaluation only, and no freshness policy configured. *Recommendation:* consider promoting the better run (a deployment change, not retraining), and "no evidence-based reason to retrain… retraining is not recommended". Retraining was never triggered.

## 5. Two failure investigations

### Failure A – Kubernetes: image cannot be pulled

![failure A diagnosis](../screenshots/failA_agent.png)
![failure A fixed](../screenshots/failA_fixed.png)
- **Injected:** `scripts/inject_failure.sh k8s-bad-image` set the image to `mlops-inference:does-not-exist`.
- **Agent diagnosis** (`k8s_diagnose`, `k8s_app_version`): Deployment only **1/2** replicas ready; 3 pods, 2 unhealthy; both showed **ImagePullBackOff** ("image cannot be pulled, wrong image name/tag or registry access"); the deployed image tag was `does-not-exist`. Likely cause: ImagePullBackOff.
- **Root cause:** non-existent image tag in the Deployment.
- **Resolution:** `scripts/inject_failure.sh k8s-restore` re-applied the correct manifests. The rolling update brought up two new pods that reached 1/1 Running; the old pod terminated. The agent then reported a healthy deployment.

### Failure B – Data/pipeline: quality gate failure

![failure B pipeline diagnosis](../screenshots/failB_agent.png)
![failure B multi-tool answer](../screenshots/failB_multi.png)
![failure B fixed](../screenshots/failB_fixed.png)
- **Injected:** `scripts/inject_failure.sh data-hard` replaced the dataset with a noisy one (label noise 35%). DVC showed the tracked data as modified.
- **Agent diagnosis** (`airflow_dag_runs`, `airflow_task_instances`, `airflow_failed_tasks`, `airflow_task_logs`): the latest run `…18:42:05` **failed**. validate_dataset, preprocess, train_model and evaluate_and_log succeeded; **quality_gate failed (try 1/2)** and register_model was **upstream_failed**, so the bad model was never promoted. The task log said: `Quality gate FAILED: f1=0.7727 < 0.85`.
- **Root cause:** training data of poor quality; the pipeline's quality gate correctly blocked promotion. The multi-tool question also said retraining was "justified because the latest run failed the quality gate" but **did not start it**.
- **Resolution:** `scripts/inject_failure.sh data-restore` restored the committed dataset through DVC; `dvc status` then reported the data unchanged (v2, md5 `36682424…`).
- **Note:** here the right fix was restoring the data, not simply retraining on the same bad data. The agent's retraining advice is a policy trigger and needs human judgement.

## 6. Testing, reliability and safety
- 62 automated pytest tests with mocked services: tool selection, argument validation, timeouts, service failures, redacted structured logs.
- A test proves the agent does **not invent results** when a tool is unavailable (it says "No operational facts could be verified").
- A test proves an **unauthorised Kubernetes change is rejected** (other namespace → `denied`; no confirmation → `confirmation_required`).
- Logs are JSON lines with tool name, status, duration and request id (visible in the CLI output); secrets are redacted.
- GitHub Actions runs `ruff` and `pytest` on every pull request; external services are mocked in CI.

## 7. Known limitations and improvements
- The rule-based planner understands limited phrasing; LLM mode (API key) is more flexible.
- Tool timeouts stop waiting but cannot kill the underlying thread.
- Airflow runs in standalone dev mode with a dev password; production needs a real auth manager.
- The model is baked into the Docker image; the deployed model (`149c9a5c`) was not the best recent run, which the agent flagged. Better: load the model from the MLflow registry and automate promotion.
- Readiness checks are heuristic: no drift monitoring or latency testing.
- Retraining advice is policy based (gate, F1 threshold, freshness, degradation report) and should be reviewed by a person.
