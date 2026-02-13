from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def artifacts_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["artifacts_dir"])


def cache_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["cache_dir"])


def corpus_dir(cfg: Dict[str, Any], corpus: str) -> Path:
    return artifacts_root(cfg) / "corpora" / corpus


def run_dir(cfg: Dict[str, Any], corpus: str) -> Path:
    return artifacts_root(cfg) / "runs" / str(cfg.get("run_name", "run")) / corpus

