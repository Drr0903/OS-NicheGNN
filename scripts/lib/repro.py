"""Reproducibility helpers shared by every project018 stage.

Every analysis stage does, in order:
    cfg = load_config()                      # config/run.yaml
    set_global_seed(cfg["seed"])
    ... work ...
    write_provenance(stage, inputs, outputs, cfg, extra={...})

A stage without results/<stage>/PROVENANCE.json is considered not run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "run.yaml"
RESULTS = ROOT / "results"
TRACK = os.environ.get("PROJECT018_TRACK", "")  # "" = set in advance frozen model; "invariant" =  second track


def track_suffix() -> str:
    return f"_{TRACK}" if TRACK else ""


def frozen_name() -> str:
    """Name of the frozen-model directory (data/processed/<frozen_name>) and of the frozen embeddings folder for the current track."""
    return f"frozen{track_suffix()}"


def results_path(rel: str) -> Path:
    """results/<stage>/<file> with the track suffix on the stage directory, e.g. results_path("08_niches/spot_niches.tsv")."""
    stage, _, rest = rel.partition("/")
    return RESULTS / f"{stage}{track_suffix()}" / rest
FIGURES = ROOT / "figures"
DATA = ROOT / "data"


# --------------------------------------------------------------------------- config
def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = sha256_file(path)
    return cfg


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """cfg_get(cfg, 'model.hidden_dim')"""
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# --------------------------------------------------------------------------- seeds
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --------------------------------------------------------------------------- hashing
def sha256_file(path: Path | str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path | str, block: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_many(paths: Iterable[Path | str]) -> dict:
    out = {}
    for p in paths:
        p = Path(p)
        if p.is_file():
            out[str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)] = {
                "md5": md5_file(p),
                "bytes": p.stat().st_size,
            }
        elif p.is_dir():
            out[str(p)] = {"dir": True, "n_files": sum(1 for _ in p.rglob("*") if _.is_file())}
        else:
            out[str(p)] = {"missing": True}
    return out


# --------------------------------------------------------------------------- git / env
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:
        return True


def env_snapshot() -> dict:
    snap = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", Path(sys.executable).parents[1].name),
    }
    for mod in ("numpy", "pandas", "scipy", "scanpy", "anndata", "sklearn", "torch", "torch_geometric"):
        try:
            m = __import__(mod)
            snap[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            snap["cuda"] = torch.version.cuda
            snap["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return snap


# --------------------------------------------------------------------------- provenance
def stage_dir(stage: str) -> Path:
    d = RESULTS / f"{stage}{track_suffix()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_provenance(
    stage: str,
    inputs: Iterable[Path | str],
    outputs: Iterable[Path | str],
    cfg: dict,
    extra: dict | None = None,
    t_start: float | None = None,
) -> Path:
    prov = {
        "stage": stage,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": None if t_start is None else round(time.time() - t_start, 1),
        "seed": cfg.get("seed"),
        "config_path": cfg.get("_config_path"),
        "config_sha256": cfg.get("_config_sha256"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "inputs": _hash_many(inputs),
        "outputs": _hash_many(outputs),
        "environment": env_snapshot(),
        "extra": extra or {},
    }
    out = stage_dir(stage) / "PROVENANCE.json"
    with open(out, "w") as fh:
        json.dump(prov, fh, indent=2, default=str)
    return out


def write_table(df, path: Path | str, index: bool = False) -> Path:
    """Tidy TSV writer used by every stage so the figure scripts read one format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)
    return path


def read_table(path: Path | str, **kw):
    import pandas as pd

    return pd.read_csv(path, sep="\t", **kw)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
