from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

from edgeqa.llm.json_parse import parse_json_loose
from edgeqa.llm.client import LLMClient
from edgeqa.pipeline.prompts import equiv_prompt
from edgeqa.text_utils import normalize_for_match


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _extract_number(text: str) -> Optional[float]:
    m = _NUM_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def quick_equiv(a: str, b: str) -> bool:
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if na and na == nb:
        return True
    if na and nb:
        short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
        # Allow small amounts of extra fluff (e.g., full-sentence answers).
        if short and short in long:
            tok_n = len(short.split())
            if tok_n >= 2 or len(short) >= 8:
                return True
    xa, xb = _extract_number(na), _extract_number(nb)
    if xa is not None and xb is not None:
        if math.isfinite(xa) and math.isfinite(xb) and abs(xa - xb) <= 1e-9:
            return True
    return False


async def llm_equiv(
    client: LLMClient,
    a: str,
    b: str,
    *,
    model: str,
    max_tokens: int = 64,
) -> bool:
    resp = await client.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": equiv_prompt(a, b)},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        cache_namespace="equiv",
    )
    content = (resp.get("response") or {}).get("content") or ""
    data = parse_json_loose(content)
    if isinstance(data, dict) and "equivalent" in data:
        return bool(data["equivalent"])
    raise ValueError("equiv judge returned non-JSON or missing field")


async def answer_equiv(
    client: LLMClient,
    a: str,
    b: str,
    *,
    model: str,
    use_llm: bool = True,
) -> bool:
    if quick_equiv(a, b):
        return True
    if not use_llm:
        return False
    return await llm_equiv(client, a, b, model=model)
