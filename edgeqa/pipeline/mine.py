from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from edgeqa.jsonl import read_jsonl, write_jsonl
from edgeqa.logging_utils import get_logger
from edgeqa.pipeline.paths import corpus_dir
from edgeqa.text_utils import normalize_for_match


_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_CAP_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NUM_RE = re.compile(r"\d")

_DEF_CUES = [
    "definition",
    "theorem",
    "lemma",
    "corollary",
    "proof",
    "we define",
    "is defined",
    "means that",
    "if and only if",
]


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def score_passage(text: str, *, freq: Counter[str]) -> float:
    t_norm = normalize_for_match(text)
    cue = sum(1 for c in _DEF_CUES if c in t_norm)
    caps = len(_CAP_RE.findall(text or ""))
    nums = len(_NUM_RE.findall(text or ""))
    toks = _tokenize(text)
    if toks:
        rarity = sum(1.0 / math.sqrt(max(1, freq[w])) for w in toks) / len(toks)
    else:
        rarity = 0.0
    # Simple linear mix (tunable).
    return 1.5 * cue + 1.0 * rarity + 0.05 * caps + 0.02 * nums


def mine_corpus_passages(cfg: Dict[str, Any], corpus: str) -> Path:
    log = get_logger("edgeqa.mine")
    cdir = corpus_dir(cfg, corpus)
    passages_path = cdir / "passages.jsonl"
    passages = list(read_jsonl(passages_path))
    if not passages:
        raise ValueError(f"No passages found at {passages_path}")

    freq: Counter[str] = Counter()
    for p in passages:
        freq.update(_tokenize(p.get("text", "")))

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for p in passages:
        s = score_passage(p.get("text", ""), freq=freq)
        scored.append((s, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    k = int(cfg.get("edgeqa", {}).get("candidate_passages", 200))
    mined = scored[:k]

    # Candidate passages depend on k, so include it in the filename to avoid stale re-use
    # when configs change (e.g., pilot k=200 vs paper k=20000).
    out = cdir / f"mined_passages_k{k}.jsonl"
    write_jsonl(out, [{"score": s, **p} for s, p in mined])
    log.info("Mined %d/%d passages (k=%d) -> %s", len(mined), len(passages), k, out)
    return out
