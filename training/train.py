"""Run the whole pipeline locally (no Airflow):  python training/train.py [--n-estimators 200 --max-depth 6 --model gradient_boosting]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["random_forest", "gradient_boosting", "logistic_regression"])
    ap.add_argument("--n-estimators", type=int)
    ap.add_argument("--max-depth", type=int)
    ap.add_argument("--no-register", action="store_true", help="skip model registration/promotion")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ov = {k: v for k, v in {"model": a.model, "n_estimators": a.n_estimators, "max_depth": a.max_depth}.items() if v is not None}
    pipeline.validate_dataset(ov)
    pipeline.preprocess(ov)
    pipeline.train_model(ov)
    res = pipeline.evaluate_and_log(ov)
    print("metrics:", res["metrics"], "run_id:", res["run_id"])
    try:
        pipeline.quality_gate(ov)
    except pipeline.QualityGateError as exc:
        print(str(exc))
        return 2
    if not a.no_register:
        print("registered:", pipeline.register_model(ov))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
