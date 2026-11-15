import pytest


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("K8S_NAMESPACE", "mlops-agent")
    monkeypatch.setenv("K8S_DEPLOYMENT", "ml-inference-api")
    monkeypatch.setenv("AIRFLOW_URL", "http://airflow.test")
    monkeypatch.setenv("AIRFLOW_PASSWORD", "unit-test-password")
    monkeypatch.setenv("MIN_F1", "0.85")
    monkeypatch.delenv("FRESHNESS_DAYS", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
