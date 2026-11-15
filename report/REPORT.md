# Technical Report – MLOps AI Agent (Assignment 03)

**Student:** _name / roll no._  **Repository:** _GitHub URL_

## 1. Complete system architecture
_Paste the diagram from `architecture.md` (render the Mermaid in GitHub, screenshot it) and describe each component in 1–2 paragraphs:
FastAPI agent API, orchestrator (LLM + rules), tool registry/safety layer, the four tool modules, Airflow DAG, MLflow, DVC, Kubernetes app._

## 2. Tool definitions and permissions
_Copy the table from `architecture.md` (or `curl localhost:8000/tools`). Explain: read vs write tools, confirmation, approved namespace, approved DAG list, allow-listed commands._

## 3. Evidence screenshots (save in `screenshots/`)
| System | What to show | File |
|---|---|---|
| DVC | `dvc status`, `data/dataset.csv.dvc`, `git tag`, `dvc remote list` | `screenshots/dvc_*.png` |
| MLflow | experiment list with ≥5 runs, one run page with provenance tags, registry | `screenshots/mlflow_*.png` |
| Airflow | DAG graph, successful run, a failed run with the failed task | `screenshots/airflow_*.png` |
| Kubernetes | `kubectl -n mlops-agent get pods,svc,deploy`, `describe pod` | `screenshots/k8s_*.png` |

## 4. Five tested agent conversations (with actual tool evidence)
For each: the question, the full agent answer, and the `tool_calls` list (use `python -m agent.cli --json "..."`).
1. Dataset provenance – "Which dataset version was used for experiment N?" (+ reproduce the dataset with `scripts/reproduce_dataset.sh`)
2. Model selection – "Compare the last five training runs and recommend the best candidate based on F1 score."
3. Pipeline execution – "Trigger the model training pipeline and report whether execution succeeded." (first without, then with `--confirm`)
4. Deployment diagnosis – "Why is my prediction API unavailable?"
5. Multi-tool reasoning – the mandatory scenario (best model → dataset → deployment → retraining advice)

## 5. Two failure investigations
For each: injected failure → symptom → how the agent diagnosed it (tools used, evidence) → root cause → resolution.
* **Failure A (Kubernetes):** e.g. `k8s-bad-image` (ImagePullBackOff) or `k8s-bad-config` (readiness fails: model path wrong) → fixed with `k8s-restore`.
* **Failure B (Data/Pipeline):** e.g. `data-corrupt` (validation task fails, `dvc status` shows change) or `data-hard` (quality gate fails) → fixed with `data-restore`.

## 6. Testing, reliability and safety
_Summarise the pytest suite (`pytest` output screenshot): tool selection, timeouts, invalid requests, service failures, no-invention test,
unauthorised Kubernetes change rejected, redacted structured logs. Show a sample log line._

## 7. Known limitations and proposed improvements
_See `architecture.md` – expand with what you observed._ Ideas: pull the model from the MLflow registry at startup, drift monitoring,
OAuth for Airflow, richer NL understanding, async tool execution with cancellation, canary rollouts.
