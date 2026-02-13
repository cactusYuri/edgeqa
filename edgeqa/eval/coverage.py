from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set

from edgeqa.jsonl import read_jsonl


def compute_edgeqa_coverage(
    *,
    passages_path: str | Path,
    units_path: str | Path,
    edgeqa_path: str | Path,
) -> Dict[str, Any]:
    passages = list(read_jsonl(passages_path))
    units = list(read_jsonl(units_path))
    data = list(read_jsonl(edgeqa_path))

    total_passages = len(passages)
    total_units = len(units)

    passage_to_unit: Dict[str, Optional[str]] = {p["passage_id"]: p.get("unit_id") for p in passages}

    covered_passages: Set[str] = set()
    covered_units: Set[str] = set()
    reasons: Set[str] = set()
    unknown = 0

    for ex in data:
        for pid in ex.get("evidence", []) or []:
            covered_passages.add(pid)
            uid = passage_to_unit.get(pid)
            if uid:
                covered_units.add(uid)
        rt = (ex.get("reason_type") or "").strip().lower()
        if rt:
            reasons.add(rt)
        if (ex.get("scores", {}) or {}).get("unknown"):
            unknown += 1

    return {
        "num_examples": len(data),
        "doccov": (len(covered_passages) / total_passages) if total_passages else 0.0,
        "unitcov": (len(covered_units) / total_units) if total_units else 0.0,
        "reason_types": sorted(reasons),
        "unknown_frac": (unknown / len(data)) if data else 0.0,
        "covered_passages": len(covered_passages),
        "total_passages": total_passages,
        "covered_units": len(covered_units),
        "total_units": total_units,
    }

