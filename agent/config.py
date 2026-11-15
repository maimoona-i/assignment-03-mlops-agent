"""Central configuration. Everything comes from environment variables (optionally from a local .env).

Secrets (API keys, passwords) are read from the environment at the point of use and are never logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Tiny .env loader (does not override variables that are already set)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _opt_int(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    mlflow_uri: str
    mlflow_experiment: str
    airflow_url: str
    airflow_username: str
    airflow_password: str
    airflow_training_dag: str
    approved_dags: tuple[str, ...]
    k8s_namespace: str
    k8s_app_label: str
    k8s_deployment: str
    k8s_configmap: str
    tool_timeout_s: float
    min_f1: float
    freshness_days: int | None
    anthropic_model: str
    llm_enabled: bool


def get_settings() -> Settings:
    """Read settings from the environment on every call (keeps tests simple)."""
    approved = os.environ.get("AIRFLOW_APPROVED_DAGS", "training_pipeline")
    return Settings(
        repo_root=Path(os.environ.get("REPO_ROOT", REPO_ROOT)),
        mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment=os.environ.get("MLFLOW_EXPERIMENT", "mlops-agent-demo"),
        airflow_url=os.environ.get("AIRFLOW_URL", "http://localhost:8080"),
        airflow_username=os.environ.get("AIRFLOW_USERNAME", "admin"),
        airflow_password=os.environ.get("AIRFLOW_PASSWORD", ""),
        airflow_training_dag=os.environ.get("AIRFLOW_TRAINING_DAG", "training_pipeline"),
        approved_dags=tuple(d.strip() for d in approved.split(",") if d.strip()),
        k8s_namespace=os.environ.get("K8S_NAMESPACE", "mlops-agent"),
        k8s_app_label=os.environ.get("K8S_APP_LABEL", "ml-inference"),
        k8s_deployment=os.environ.get("K8S_DEPLOYMENT", "ml-inference-api"),
        k8s_configmap=os.environ.get("K8S_CONFIGMAP", "ml-inference-config"),
        tool_timeout_s=float(os.environ.get("TOOL_TIMEOUT_SECONDS", "30")),
        min_f1=float(os.environ.get("MIN_F1", "0.85")),
        freshness_days=_opt_int("FRESHNESS_DAYS"),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        llm_enabled=bool(os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get("AGENT_MODE") != "rules",
    )
