"""DVC data-management tools (Task 1).

Safety: only an allow-list of read-only git/dvc commands may run, with shell=False and validated arguments.
"""
from __future__ import annotations

import configparser
import json
import re
import subprocess

import yaml
from pydantic import BaseModel, Field, field_validator

from agent.config import get_settings
from agent.errors import ToolError
from agent.registry import REGISTRY
from agent.schemas import NoArgs

# ---------------------------------------------------------------- safe command runner
ALLOWED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("dvc", "status"),
    ("dvc", "data", "status"),
    ("dvc", "diff"),
    ("git", "rev-parse"),
    ("git", "show"),
    ("git", "log"),
    ("git", "tag"),
    ("git", "status"),
    ("git", "cat-file"),
)
_SAFE_ARG = re.compile(r"^[\w./:@~^=,%+\-]+$")
_REV = re.compile(r"^[A-Za-z0-9._/~^\-]{1,100}$")


def run_approved(cmd: list[str], timeout: int = 20) -> str:
    """Run an allow-listed command and return stdout. Anything else is rejected."""
    if not any(tuple(cmd[: len(p)]) == p for p in ALLOWED_COMMANDS):
        raise ToolError(f"Command not approved: {' '.join(cmd[:2])}")
    for arg in cmd:
        if not _SAFE_ARG.match(arg):
            raise ToolError("Command contains unsafe characters")
    try:
        proc = subprocess.run(cmd, cwd=get_settings().repo_root, capture_output=True, text=True,
                              timeout=timeout, check=False, shell=False)
    except FileNotFoundError as exc:
        raise ToolError(f"'{cmd[0]}' executable not found on this machine") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"'{cmd[0]}' timed out") from exc
    if proc.returncode != 0:
        raise ToolError(f"'{' '.join(cmd[:3])}' failed: {(proc.stderr or proc.stdout).strip()[:300]}")
    return proc.stdout


# ---------------------------------------------------------------- helpers
class DatasetArgs(BaseModel):
    dataset: str = Field(default="data/dataset.csv", description="Dataset path tracked by DVC (relative to repo root)")

    @field_validator("dataset")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        if v.startswith("/") or ".." in v or not re.match(r"^[\w./\-]+$", v):
            raise ValueError("dataset must be a safe relative path")
        return v


def _dvc_rel(dataset: str) -> str:
    return dataset if dataset.endswith(".dvc") else dataset + ".dvc"


def _parse_dvc_text(text: str) -> dict:
    doc = yaml.safe_load(text) or {}
    outs = doc.get("outs") or []
    if not outs:
        raise ToolError("No 'outs' entry found in .dvc file")
    return outs[0]


def _read_dvc_file(dataset: str) -> dict:
    path = get_settings().repo_root / _dvc_rel(dataset)
    if not path.exists():
        raise ToolError(f"{_dvc_rel(dataset)} not found - dataset is not tracked by DVC")
    return _parse_dvc_text(path.read_text())


def _check_rev(rev: str) -> str:
    if rev.startswith("-") or not _REV.match(rev):
        raise ToolError(f"Invalid git revision: {rev!r}")
    return rev


def _dataset_version_at(rev: str | None) -> str | None:
    """dataset_version recorded in params.yaml at a git revision (or working tree)."""
    try:
        if rev:
            text = run_approved(["git", "show", f"{_check_rev(rev)}:params.yaml"])
        else:
            text = (get_settings().repo_root / "params.yaml").read_text()
        return (yaml.safe_load(text) or {}).get("dataset_version")
    except Exception:
        return None


def _dvc_remote() -> dict:
    cfg = get_settings().repo_root / ".dvc" / "config"
    if not cfg.exists():
        return {"configured": False}
    parser = configparser.ConfigParser(strict=False)
    parser.read(cfg)
    default = parser.get("core", "remote", fallback=None)
    for section in parser.sections():
        if section.startswith("'remote") or section.startswith('"remote') or section.startswith("remote"):
            name = section.split('"')[1] if '"' in section else section.split("'")[1] if "'" in section else section
            if default is None or name == default:
                url = parser.get(section, "url", fallback="")
                url = re.sub(r"//[^/@]+@", "//", url)  # strip embedded credentials
                return {"configured": True, "name": name, "url": url}
    return {"configured": False}


# ---------------------------------------------------------------- tools
@REGISTRY.tool(name="dvc_list_tracked_datasets", module="dvc", args=NoArgs,
               description="List datasets and pipeline outputs tracked by DVC in this repository.")
def dvc_list_tracked_datasets(_: NoArgs) -> dict:
    root = get_settings().repo_root
    skip = {".git", ".venv", "venv", "node_modules", ".dvc", "artifacts"}
    datasets = []
    for p in sorted(root.rglob("*.dvc")):
        if skip & set(p.relative_to(root).parts):
            continue
        try:
            out = _parse_dvc_text(p.read_text())
        except ToolError:
            continue
        datasets.append({
            "dvc_file": str(p.relative_to(root)),
            "path": str((p.parent / out.get("path", "")).relative_to(root)),
            "md5": out.get("md5"), "size_bytes": out.get("size"), "nfiles": out.get("nfiles"),
            "kind": "data",
        })
    stages = []
    dvc_yaml = root / "dvc.yaml"
    if dvc_yaml.exists():
        for name, stage in ((yaml.safe_load(dvc_yaml.read_text()) or {}).get("stages") or {}).items():
            outs = [o if isinstance(o, str) else next(iter(o)) for o in stage.get("outs", [])]
            stages.append({"stage": name, "outs": outs, "deps": stage.get("deps", [])})
    return {"datasets": datasets, "pipeline_stages": stages, "count": len(datasets)}


@REGISTRY.tool(name="dvc_get_dataset_version", module="dvc", args=DatasetArgs,
               description="Get the current DVC dataset version (content hash, version label) and the Git commit that last changed it.")
def dvc_get_dataset_version(args: DatasetArgs) -> dict:
    out = _read_dvc_file(args.dataset)
    rel = _dvc_rel(args.dataset)
    head = run_approved(["git", "rev-parse", "HEAD"]).strip()
    last = run_approved(["git", "log", "-1", "--pretty=format:%H,%cI", "--", rel]).strip()
    last_commit, _, committed_at = last.partition(",")
    tags = run_approved(["git", "tag", "--points-at", last_commit]).split() if last_commit else []
    dirty = bool(run_approved(["git", "status", "--porcelain", "--", rel]).strip())
    return {
        "dataset": args.dataset, "dvc_file": rel, "md5": out.get("md5"),
        "size_bytes": out.get("size"), "dataset_version": _dataset_version_at(None),
        "git_head": head, "git_commit_of_dvc_file": last_commit or None,
        "committed_at": committed_at or None, "git_tags": tags,
        "dvc_file_has_uncommitted_changes": dirty,
    }


class StatusArgs(DatasetArgs):
    pass


@REGISTRY.tool(name="dvc_status", module="dvc", args=StatusArgs,
               description="Detect whether DVC-tracked data in the workspace differs from its recorded version (data changed / missing).")
def dvc_status(args: StatusArgs) -> dict:
    raw = run_approved(["dvc", "status", "--json", _dvc_rel(args.dataset)]).strip() or "{}"
    try:
        details = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError("Could not parse 'dvc status' output") from exc
    return {"dataset": args.dataset, "changed": bool(details), "details": details}


class CompareArgs(DatasetArgs):
    rev_a: str = Field(default="HEAD~1", description="Older git revision / tag")
    rev_b: str = Field(default="HEAD", description="Newer git revision / tag")


@REGISTRY.tool(name="dvc_compare_versions", module="dvc", args=CompareArgs,
               description="Compare the dataset between two git revisions or tags (content hash, size, version label).")
def dvc_compare_versions(args: CompareArgs) -> dict:
    rel = _dvc_rel(args.dataset)
    versions = {}
    for label, rev in (("a", args.rev_a), ("b", args.rev_b)):
        _check_rev(rev)
        try:
            text = run_approved(["git", "show", f"{rev}:{rel}"])
        except ToolError as exc:
            raise ToolError(f"{rel} does not exist at revision '{rev}': {exc}") from exc
        out = _parse_dvc_text(text)
        commit = run_approved(["git", "rev-parse", rev]).strip()
        versions[label] = {"rev": rev, "commit": commit, "md5": out.get("md5"),
                           "size_bytes": out.get("size"), "dataset_version": _dataset_version_at(rev)}
    same = versions["a"]["md5"] == versions["b"]["md5"]
    return {"dataset": args.dataset, "identical": same, "a": versions["a"], "b": versions["b"],
            "size_delta_bytes": (versions["b"]["size_bytes"] or 0) - (versions["a"]["size_bytes"] or 0)}


class ExperimentArgs(BaseModel):
    identifier: str = Field(description="MLflow run id, run name, or experiment id (latest finished run is used)")


@REGISTRY.tool(name="dvc_experiment_dataset", module="dvc", args=ExperimentArgs,
               description="Find which dataset version (DVC hash, version label, Git commit) was used to train an MLflow run/experiment, from recorded provenance tags.")
def dvc_experiment_dataset(args: ExperimentArgs) -> dict:
    from agent.tools.mlflow_tools import resolve_run  # local import: keeps modules decoupled

    run = resolve_run(args.identifier)
    tags = run.data.tags
    md5, commit, path = tags.get("dvc.md5"), tags.get("git.commit"), tags.get("dvc.path", "data/dataset.csv")
    verified = None
    if md5 and commit and commit != "unknown":
        try:
            recorded = _parse_dvc_text(run_approved(["git", "show", f"{_check_rev(commit)}:{_dvc_rel(path)}"]))
            verified = recorded.get("md5") == md5
        except ToolError:
            verified = None  # commit not available locally -> cannot verify
    return {
        "run_id": run.info.run_id, "run_name": tags.get("mlflow.runName"),
        "experiment_id": run.info.experiment_id, "dataset_path": path,
        "dataset_version": tags.get("dataset.version"), "dvc_md5": md5, "git_commit": commit,
        "provenance_recorded": bool(md5 or tags.get("dataset.version")),
        "provenance_verified_against_git": verified, "dvc_remote": _dvc_remote(),
    }


class ReproduceArgs(DatasetArgs):
    identifier: str | None = Field(default=None, description="MLflow run id/name/experiment id to reproduce the dataset of")
    git_rev: str | None = Field(default=None, description="Git commit/tag to reproduce (alternative to identifier)")


@REGISTRY.tool(name="dvc_explain_reproduce", module="dvc", args=ReproduceArgs,
               description="Explain (without executing) the exact commands to reproduce a previous dataset version, based on recorded provenance.")
def dvc_explain_reproduce(args: ReproduceArgs) -> dict:
    rev, source = args.git_rev, "git_rev argument"
    if args.identifier:
        info = dvc_experiment_dataset(ExperimentArgs(identifier=args.identifier))
        rev, source = info["git_commit"], f"MLflow run {info['run_id']}"
        if not rev or rev == "unknown":
            raise ToolError("The run has no recorded Git commit, so the dataset cannot be reproduced reliably")
    if not rev:
        raise ToolError("Provide identifier or git_rev")
    _check_rev(rev)
    rel = _dvc_rel(args.dataset)
    return {
        "source": source, "git_rev": rev, "executed": False,
        "steps": [
            f"git checkout {rev} -- {rel}        # restore the pointer file for that version",
            f"dvc checkout {rel}                  # restore from local cache (or: dvc pull {rel} to fetch from the remote)",
            f"dvc status {rel}                    # should report: Data and pipelines are up to date",
            f"git checkout HEAD -- {rel} && dvc checkout {rel}   # return to the latest version afterwards",
        ],
        "note": "Commands are shown for the user to run; the agent never modifies the working tree.",
    }
