"""Dataset/Git provenance helpers (pure Python: no git/dvc binaries needed, so they also work inside Airflow)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parent.parent))


def load_params(overrides: dict | None = None) -> dict:
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    for key, value in (overrides or {}).items():
        if key in params["train"]:
            params["train"][key] = value
    return params


def git_commit() -> str:
    """Read HEAD commit directly from .git (works without the git binary)."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        head = (ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1]
            ref_file = ROOT / ".git" / ref
            if ref_file.exists():
                return ref_file.read_text().strip()
            for line in (ROOT / ".git" / "packed-refs").read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split(" ")[0]
            return "unknown"
        return head
    except Exception:
        return "unknown"


def dvc_info(dataset_path: str) -> dict:
    """md5/size of the dataset as recorded in its .dvc pointer file (the committed version)."""
    f = ROOT / (dataset_path + ".dvc")
    if not f.exists():
        return {"md5": None, "size": None}
    out = (yaml.safe_load(f.read_text()) or {}).get("outs", [{}])[0]
    return {"md5": out.get("md5"), "size": out.get("size")}


def provenance(params: dict) -> dict:
    path = params["data"]["path"]
    info = dvc_info(path)
    return {"dataset.version": str(params["dataset_version"]), "dvc.md5": info["md5"] or "untracked",
            "dvc.path": path, "git.commit": git_commit()}
