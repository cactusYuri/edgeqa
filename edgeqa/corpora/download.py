from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from edgeqa.logging_utils import get_logger


def _run(cmd: list[str], *, cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_git_repo(*, url: str, dest_dir: str | Path, revision: Optional[str] = None) -> str:
    log = get_logger("edgeqa.corpora.download")
    dest = Path(dest_dir)
    if (dest / ".git").exists():
        log.info("Updating repo: %s", dest)
        _run(["git", "fetch", "--all", "--tags"], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Cloning repo: %s -> %s", url, dest)
        _run(["git", "clone", url, str(dest)])

    if revision:
        _run(["git", "checkout", revision], cwd=dest)
    # Return current commit SHA for provenance.
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(dest)).decode("utf-8").strip()
    return sha

