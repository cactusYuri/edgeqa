from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edgeqa.jsonl import read_jsonl
from edgeqa.config import load_config
from edgeqa.pipeline.paths import artifacts_root, corpus_dir, run_dir
from edgeqa.pipeline.select import greedy_select


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def _passage_to_unit(passages: Iterable[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for p in passages:
        pid = str(p.get("passage_id") or "")
        if not pid:
            continue
        uid = p.get("unit_id")
        out[pid] = str(uid) if uid else None
    return out


def _coverage_for_prefix(
    selected: List[Dict[str, Any]],
    *,
    n: int,
    total_passages: int,
    total_units: int,
    passage_to_unit: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    covered_passages: Set[str] = set()
    covered_units: Set[str] = set()
    reasons: Counter[str] = Counter()

    for ex in selected[:n]:
        ev = ex.get("evidence") or []
        if isinstance(ev, list):
            for pid in ev:
                pid = str(pid or "")
                if not pid:
                    continue
                covered_passages.add(pid)
                uid = passage_to_unit.get(pid)
                if uid:
                    covered_units.add(uid)
        rt = str(ex.get("reason_type") or "").strip().lower() or "unknown"
        reasons[rt] += 1

    doccov = (len(covered_passages) / total_passages) if total_passages else 0.0
    unitcov = (len(covered_units) / total_units) if total_units else 0.0
    return {
        "num_examples": int(n),
        "doccov": float(doccov),
        "unitcov": float(unitcov),
        "covered_passages": int(len(covered_passages)),
        "total_passages": int(total_passages),
        "covered_units": int(len(covered_units)),
        "total_units": int(total_units),
        "reason_type_counts": dict(reasons),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--budgets", type=str, default="1000,2000,5000,10000")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["run_name"] = str(args.run_name)

    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]
    budgets = sorted({int(x.strip()) for x in str(args.budgets).split(",") if x.strip()})
    if not budgets:
        budgets = [10000]
    max_budget = max(budgets)

    out_path = (
        Path(args.out)
        if args.out
        else artifacts_root(cfg) / "runs" / str(args.run_name) / "selection_ablations.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lambdas_grid: Dict[str, Dict[str, float]] = {
        "main": {"doc": 1.0, "unit": 1.0, "reason": 0.5, "redundancy": 0.2},
        "doc_only": {"doc": 1.0, "unit": 0.0, "reason": 0.0, "redundancy": 0.2},
        "unit_only": {"doc": 0.0, "unit": 1.0, "reason": 0.0, "redundancy": 0.2},
        "no_reason": {"doc": 1.0, "unit": 1.0, "reason": 0.0, "redundancy": 0.2},
        "no_redundancy": {"doc": 1.0, "unit": 1.0, "reason": 0.5, "redundancy": 0.0},
    }

    report: Dict[str, Any] = {
        "run_name": str(args.run_name),
        "generated_at": int(time.time()),
        "budgets": budgets,
        "max_budget": max_budget,
        "lambdas": lambdas_grid,
        "per_corpus": {},
    }

    for corpus in corpora:
        rdir = run_dir(cfg, corpus=corpus)
        cdir = corpus_dir(cfg, corpus=corpus)
        pool_path = rdir / "edgeqa_pool.jsonl"
        passages_path = cdir / "passages.jsonl"
        units_path = cdir / "units.jsonl"

        if not pool_path.exists():
            raise SystemExit(f"missing pool: {pool_path}")
        if not passages_path.exists():
            raise SystemExit(f"missing passages: {passages_path}")
        if not units_path.exists():
            raise SystemExit(f"missing units: {units_path}")

        passages = _read_jsonl(passages_path)
        units = _read_jsonl(units_path)
        p2u = _passage_to_unit(passages)
        total_passages = len(passages)
        total_units = len(units)

        pool = _read_jsonl(pool_path)
        report["per_corpus"][corpus] = {
            "pool": len(pool),
            "total_passages": total_passages,
            "total_units": total_units,
            "by_setting": {},
        }

        for name, lambdas in lambdas_grid.items():
            selected = greedy_select(pool, passage_to_unit=p2u, N=max_budget, lambdas=lambdas, unknownness_min_frac=0.5)
            by_budget: Dict[str, Any] = {}
            for b in budgets:
                by_budget[str(b)] = _coverage_for_prefix(
                    selected,
                    n=b,
                    total_passages=total_passages,
                    total_units=total_units,
                    passage_to_unit=p2u,
                )
            report["per_corpus"][corpus]["by_setting"][name] = {
                "selected": len(selected),
                "coverage_by_budget": by_budget,
            }

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
