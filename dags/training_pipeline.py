"""Airflow 3 training DAG:  validate -> preprocess -> train -> evaluate & log to MLflow -> quality gate -> register.

* retries=2 with a delay for transient failures (NOT for data/quality problems: those raise AirflowFailException).
* A failed validation or quality gate stops everything downstream; `register_model` (promotion) only runs
  when the quality gate succeeded.
* Optional run config (dag_run.conf): {"model": "...", "n_estimators": 200, "max_depth": 6}
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta

import pendulum
from airflow.sdk import dag, get_current_context, task

try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # older 3.0.x layouts
    from airflow.exceptions import AirflowFailException

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

log = logging.getLogger("airflow.task")


def _overrides() -> dict:
    conf = get_current_context()["dag_run"].conf or {}
    return {k: conf[k] for k in ("model", "n_estimators", "max_depth") if k in conf}


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args={"owner": "mlops", "retries": 2, "retry_delay": timedelta(seconds=20)},
    tags=["mlops", "assignment-03"],
    doc_md=__doc__,
)
def training_pipeline():
    @task
    def validate_dataset() -> dict:
        from training import pipeline

        try:
            return pipeline.validate_dataset(_overrides())
        except pipeline.DataValidationError as exc:
            log.error("Dataset validation failed: %s", exc)
            raise AirflowFailException(str(exc)) from exc  # no retry: bad data will not fix itself

    @task
    def preprocess(validation: dict) -> dict:
        from training import pipeline

        return pipeline.preprocess(_overrides())

    @task
    def train_model(prep: dict) -> dict:
        from training import pipeline

        return pipeline.train_model(_overrides())

    @task
    def evaluate_and_log(trained: dict) -> dict:
        from training import pipeline

        return pipeline.evaluate_and_log(_overrides())

    @task
    def quality_gate(evaluation: dict) -> dict:
        from training import pipeline

        try:
            return pipeline.quality_gate(_overrides())
        except pipeline.QualityGateError as exc:
            log.error("%s", exc)
            raise AirflowFailException(str(exc)) from exc  # no retry; blocks promotion

    @task
    def register_model(gate: dict) -> dict:
        from training import pipeline

        return pipeline.register_model(_overrides())

    register_model(quality_gate(evaluate_and_log(train_model(preprocess(validate_dataset())))))


training_pipeline()
