"""Generate a synthetic binary-classification dataset and set its version label in params.yaml.

  python training/make_dataset.py --version v1 --rows 3000 --seed 1
  python training/make_dataset.py --version v2 --rows 4500 --seed 2     # a "new" dataset version
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.03, help="label noise (higher = harder problem)")
    a = ap.parse_args()
    X, y = make_classification(n_samples=a.rows, n_features=20, n_informative=10, n_redundant=4,
                               flip_y=a.noise, class_sep=1.2, weights=[0.6, 0.4], random_state=a.seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
    df["target"] = y
    out = ROOT / "data" / "dataset.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    params_file = ROOT / "params.yaml"
    params = yaml.safe_load(params_file.read_text())
    params["dataset_version"] = a.version
    params_file.write_text(yaml.safe_dump(params, sort_keys=False))
    print(f"Wrote {out} ({len(df)} rows) and set dataset_version={a.version} in params.yaml")


if __name__ == "__main__":
    main()
