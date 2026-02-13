from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", (text or "").strip())


def parse_json_loose(text: str) -> Any:
    """
    Best-effort JSON extractor for LLM outputs.
    Supports:
      - raw JSON
      - ```json ... ```
      - leading/trailing prose
    """
    t = _strip_fences(text)
    if not t:
        raise ValueError("empty response")

    # Try full parse first.
    try:
        return json.loads(t)
    except Exception:
        pass

    # Try extracting array.
    first = t.find("[")
    last = t.rfind("]")
    if 0 <= first < last:
        frag = t[first : last + 1]
        try:
            return json.loads(frag)
        except Exception:
            pass

    # Try extracting object.
    first = t.find("{")
    last = t.rfind("}")
    if 0 <= first < last:
        frag = t[first : last + 1]
        try:
            return json.loads(frag)
        except Exception:
            pass

    raise ValueError("failed to parse JSON from response")

