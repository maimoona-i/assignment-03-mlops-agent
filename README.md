# Assignment 03 – Production-Ready MLOps AI Agent

A conversational agent that talks to **real** MLOps infrastructure — **DVC** (dataset versions), **MLflow** (experiments / registry),
**Apache Airflow 3** (training pipeline) and **Kubernetes** (FastAPI inference service) — and answers only from tool evidence.

```
User ─► FastAPI (agent/main.py) ─► Agent (orchestrator.py: LLM tool-calling OR rule-based planner)
                                      │  every call passes through the Tool Registry
                                      │  (validation · approved namespace · confirmation · timeout · structured logs)
            ┌─────────────┬───────────┴───────────┬──────────────────┐
           DVC          MLflow                 Airflow 3         Kubernetes
        dvc_tools    mlflow_tools           airflow_tools     kubernetes_tools
```
See `architecture.md` for details and `report/REPORT.md` for the technical report template.

## Repository layout
| Path | Purpose |
|---|---|
| `agent/` | Agent: `orchestrator.py`, `registry.py` (safety layer), `tools/*` (4 modules), `llm.py`, `main.py` (API), `cli.py` |
| `api/main.py` | FastAPI **inference service** deployed on Kubernetes |
| `dags/training_pipeline.py` | Airflow 3 DAG: validate → preprocess → train → evaluate/log → quality gate → register |
| `training/` | Pipeline stage code, dataset generator, provenance helpers |
| `kubernetes/` | Namespace, ConfigMap, Deployment (2 replicas, probes, limits), ClusterIP Service, RBAC |
| `docker/`, `docker-compose.yml` | Local MLflow server + Airflow 3 |
| `tests/` | pytest suite (all external systems mocked) |
| `scripts/` | Failure injection + dataset reproduction demos |

## 1. Prerequisites
Python 3.11+, Git, Docker (+ Docker Compose), `kubectl`, **minikube** or **kind**. (Windows users: use WSL2.)

## 2. Install
```bash
python -m venv .venv && source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                     # then edit .env (never commit it)
```

## 3. Environment configuration (`.env`)
| Variable | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | Optional. Enables LLM tool-calling mode. Without it the deterministic rule-based planner is used. |
| `MLFLOW_TRACKING_URI` | default `http://localhost:5000` |
| `AIRFLOW_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Airflow 3 API credentials (dev: `admin`/`admin`) |
| `AIRFLOW_APPROVED_DAGS` | Only these DAGs may be triggered (default `training_pipeline`) |
| `K8S_NAMESPACE` | The **only** namespace the agent may touch (default `mlops-agent`) |
| `MIN_F1`, `FRESHNESS_DAYS` | Retraining policy inputs (freshness unset = no freshness policy) |

## 4. Launch each service
**DVC + data** (one-time):
```bash
git init && dvc init
dvc remote add -d localstorage ../dvc-remote            # any folder outside the repo works as a remote
python training/make_dataset.py --version v1 --rows 3000 --seed 1
dvc add data/dataset.csv
git add . && git commit -m "dataset v1" && git tag v1
dvc push
```
**MLflow + Airflow:**
```bash
docker compose up -d --build        # MLflow http://localhost:5000 · Airflow http://localhost:8080 (admin/admin)
```
**Train a few runs** (each logs params, metrics and provenance tags to MLflow):
```bash
python training/train.py
python training/train.py --model gradient_boosting --n-estimators 150 --max-depth 4
python training/train.py --model logistic_regression
```
**Build + deploy the inference service:**
```bash
docker build -t mlops-inference:1.0.0 .
minikube image load mlops-inference:1.0.0                # kind: kind load docker-image mlops-inference:1.0.0
# put the MLflow run id of the model in kubernetes/configmap.yaml (MODEL_RUN_ID), then:
kubectl apply -f kubernetes/namespace.yaml -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n mlops-agent get pods
```
**Start the agent API:**
```bash
uvicorn agent.main:app --port 8000        # docs at http://localhost:8000/docs
```

## 5. Talk to the agent
```bash
python -m agent.cli "Which dataset version was used for experiment 25?"
python -m agent.cli "Compare the last five training runs and recommend the best candidate based on F1 score."
python -m agent.cli "Trigger the model training pipeline and report whether execution succeeded."            # asks for confirmation
python -m agent.cli --confirm "Trigger the model training pipeline and report whether execution succeeded." # executes
python -m agent.cli "Why is my prediction API unavailable?"
python -m agent.cli "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
```
or `curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"..."}'`
(add `"confirm": true` to authorise state-changing actions, `"degradation_report": "..."` to supply degradation evidence).

## 6. Run the tests
```bash
pytest            # unit tests, everything external is mocked
ruff check .      # lint (same as CI)
```

## 7. Failure injection (for the demo and the report)
```bash
scripts/inject_failure.sh k8s-bad-image   # ImagePullBackOff        → ask: "Why is my prediction API unavailable?"
scripts/inject_failure.sh k8s-bad-config  # readiness never passes  → model file not found in logs
scripts/inject_failure.sh k8s-oom         # OOMKilled
scripts/inject_failure.sh k8s-restore
scripts/inject_failure.sh data-corrupt    # Airflow validation fails; `dvc status` shows the data changed
scripts/inject_failure.sh data-hard       # noisy data → quality gate fails (f1 < 0.85)
scripts/inject_failure.sh data-restore
scripts/reproduce_dataset.sh v1           # reproducible dataset checkout
```

## Safety model (summary)
* Operational claims only from tool output; tool failures are reported as uncertainties.
* Only allow-listed read-only `git`/`dvc` commands (no shell). Airflow: only approved DAGs. Kubernetes: one namespace, one app.
* State-changing tools (`k8s_restart_deployment`, `k8s_scale_deployment`, `airflow_trigger_dag`) run only with explicit user confirmation — set by the *request*, never by the LLM.
* Retraining is never started automatically; it is recommended only with explicit evidence.
* Structured JSON logs (tool, status, duration, request id) with secret redaction. No secrets in the repo.
