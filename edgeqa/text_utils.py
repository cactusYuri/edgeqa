from __future__ import annotations

import re
import string
from typing import Iterable, List


_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def normalize_for_match(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.translate(_PUNCT_TABLE)
    t = normalize_ws(t)
    return t


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?。！？])\s+")


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    denom = len(sa | sb)
    if denom == 0:
        return 0.0
    return len(sa & sb) / denom

