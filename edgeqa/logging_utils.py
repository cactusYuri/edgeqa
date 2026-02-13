from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(level: str | int = "INFO") -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    fmt = os.getenv("EDGEQA_LOG_FORMAT", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    formatter = logging.Formatter(fmt)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)

    log_file = (os.getenv("EDGEQA_LOG_FILE") or "").strip()
    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            has_file = any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(path) for h in root.handlers)
            if not has_file:
                fh = logging.FileHandler(str(path), mode="a", encoding="utf-8")
                fh.setFormatter(formatter)
                root.addHandler(fh)
        except Exception:
            # Best-effort: keep stdout logging even if file logging fails.
            pass


def get_logger(name: str, *, level: Optional[str | int] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    return logger
