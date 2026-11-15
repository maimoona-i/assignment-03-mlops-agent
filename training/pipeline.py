"""Training pipeline stages. Used by BOTH the CLI (training/train.py) and the Airflow DAG (dags/training_pipeline.py).

validate_dataset -> preprocess -> train_model -> evaluate_and_log -> quality_gate -> register_model
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from training.provenance import ROOT, load_params, provenance

log = logging.getLogger("training")
WORK = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))


class DataValidationError(Exception):
    """Dataset failed validation (not retryable)."""


class QualityGateError(Exception):
    """Model did not meet the quality gate (not retryable)."""


def _mlflow(params: dict):
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", params["mlflow"]["experiment"]))
    return mlflow


# 1 -----------------------------------------------------------------------------
def validate_dataset(overrides: dict | None = None) -> dict:
    import pandas as pd

    p = load_params(overrides)
    path = ROOT / p["data"]["path"]
    if not path.exists():
        raise DataValidationError(f"Dataset not found: {path} (run `dvc pull` / `dvc checkout`)")
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise DataValidationError("Column 'target' is missing")
    if len(df) < p["data"]["min_rows"]:
        raise DataValidationError(f"Too few rows: {len(df)} < {p['data']['min_rows']}")
    nulls = int(df.isna().sum().sum())
    if nulls:
        raise DataValidationError(f"Dataset contains {nulls} missing values")
    ratio = float(df["target"].value_counts(normalize=True).min())
    if ratio < p["data"]["min_class_ratio"]:
        raise DataValidationError(f"Minority class ratio {ratio:.3f} below {p['data']['min_class_ratio']}")
    report = {"rows": int(len(df)), "columns": int(df.shape[1]), "minority_class_ratio": round(ratio, 4)}
    log.info("Dataset validation passed: %s", report)
    return report


# 2 -----------------------------------------------------------------------------
def preprocess(overrides: dict | None = None) -> dict:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    p = load_params(overrides)
    df = pd.read_csv(ROOT / p["data"]["path"])
    train, test = train_test_split(df, test_size=p["train"]["test_size"], random_state=p["train"]["seed"],
                                   stratify=df["target"])
    WORK.mkdir(parents=True, exist_ok=True)
    train.to_csv(WORK / "train.csv", index=False)
    test.to_csv(WORK / "test.csv", index=False)
    info = {"train_rows": len(train), "test_rows": len(test)}
    log.info("Preprocessing done: %s", info)
    return info


# 3 -----------------------------------------------------------------------------
def train_model(overrides: dict | None = None) -> dict:
    import joblib
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    p = load_params(overrides)["train"]
    df = pd.read_csv(WORK / "train.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    if p["model"] == "random_forest":
        model = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=p["seed"])
    elif p["model"] == "gradient_boosting":
        model = GradientBoostingClassifier(n_estimators=p["n_estimators"], max_depth=min(p["max_depth"], 5), random_state=p["seed"])
    elif p["model"] == "logistic_regression":
        model = LogisticRegression(max_iter=500)
    else:
        raise ValueError(f"Unknown model type {p['model']}")
    model.fit(X, y)
    joblib.dump(model, WORK / "model.joblib")
    log.info("Trained %s", p["model"])
    return {"model": p["model"]}


# 4 -----------------------------------------------------------------------------
def evaluate_and_log(overrides: dict | None = None) -> dict:
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    params = load_params(overrides)
    df = pd.read_csv(WORK / "test.csv")
    X, y = df.drop(columns=["target"]), df["target"]
    model = joblib.load(WORK / "model.joblib")
    pred = model.predict(X)
    metrics = {"f1": float(f1_score(y, pred)), "accuracy": float(accuracy_score(y, pred)),
               "precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred))}
    mlflow = _mlflow(params)
    from mlflow import sklearn as mlflow_sklearn

    tags = provenance(params)
    with mlflow.start_run(run_name=f"{params['train']['model']}-{params['dataset_version']}") as run:
        mlflow.log_params({k: v for k, v in params["train"].items()})
        mlflow.log_params({"dataset_version": params["dataset_version"], "train_rows": len(pd.read_csv(WORK / "train.csv"))})
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow_sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id
    out = {"run_id": run_id, "metrics": metrics}
    (WORK / "last_run.json").write_text(json.dumps(out))
    Path(ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Logged run %s to MLflow: %s", run_id, metrics)
    return out


# 5 -----------------------------------------------------------------------------
def quality_gate(overrides: dict | None = None) -> dict:
    params = load_params(overrides)
    last = json.loads((WORK / "last_run.json").read_text())
    f1, threshold = last["metrics"]["f1"], params["quality_gate"]["min_f1"]
    passed = f1 >= threshold
    from mlflow.tracking import MlflowClient

    MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")).set_tag(
        last["run_id"], "quality_gate", "passed" if passed else "failed")
    if not passed:
        raise QualityGateError(f"Quality gate FAILED: f1={f1:.4f} < {threshold}")
    log.info("Quality gate passed: f1=%.4f >= %s", f1, threshold)
    return {"run_id": last["run_id"], "f1": f1, "threshold": threshold, "passed": True}


# 6 (promotion; only reachable if every previous stage succeeded) ----------------
def register_model(overrides: dict | None = None) -> dict:
    params = load_params(overrides)
    last = json.loads((WORK / "last_run.json").read_text())
    mlflow = _mlflow(params)
    mv = mlflow.register_model(f"runs:/{last['run_id']}/model", params["mlflow"]["registered_model"])
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    shutil.copy(WORK / "model.joblib", models_dir / "model.joblib")  # picked up by `docker build` of the API image
    (models_dir / "model_info.json").write_text(json.dumps({"run_id": last["run_id"], "version": mv.version}))
    log.info("Registered model version %s (run %s)", mv.version, last["run_id"])
    return {"run_id": last["run_id"], "model_version": mv.version}
