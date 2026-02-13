from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edgeqa.config import load_config
from edgeqa.eval.coverage import compute_edgeqa_coverage
from edgeqa.jsonl import read_jsonl, write_jsonl
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


def _write_prefix_jsonl(in_path: Path, out_path: Path, n: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with in_path.open("r", encoding="utf-8", errors="ignore") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if written >= n:
                break
            line = (line or "").strip()
            if not line:
                continue
            fout.write(line + "\n")
            written += 1
    return written


def _reason_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        rt = str(r.get("reason_type") or "").strip().lower() or "unknown"
        c[rt] += 1
    return dict(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--budgets", type=str, default="1000,2000,5000,10000")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default=None, help="Optional JSON report path.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["run_name"] = str(args.run_name)

    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]
    budgets = sorted({int(x.strip()) for x in str(args.budgets).split(",") if x.strip()})
    if not budgets:
        budgets = [10000]
    max_budget = max(budgets)

    # B1 definition (paper §Experiments): maximize DocCov, no unknownness constraint.
    lambdas = {"doc": 1.0, "unit": 0.0, "reason": 0.0, "redundancy": 0.2}
    unknownness_min_frac = 0.0

    out_report = (
        Path(args.out)
        if args.out
        else artifacts_root(cfg) / "runs" / str(args.run_name) / "baselines" / "B1_coverage_first.json"
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    report: Dict[str, Any] = {
        "run_name": str(args.run_name),
        "generated_at": int(time.time()),
        "baseline_id": "B1",
        "baseline_name": "coverage_first",
        "definition": "Greedy select from EdgeQA pool to maximize DocCov (lambda_doc=1) without unknownness constraint.",
        "lambdas": dict(lambdas),
        "unknownness_min_frac": float(unknownness_min_frac),
        "budgets": budgets,
        "max_budget": max_budget,
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

        pool = _read_jsonl(pool_path)
        passages = _read_jsonl(passages_path)
        p2u = _passage_to_unit(passages)

        # Shuffle pool as a tie-breaker so the selection is stable across runs given a seed.
        # (greedy_select is deterministic given item order when gains tie).
        pool_idx = list(range(len(pool)))
        rng.shuffle(pool_idx)
        pool_shuf = [pool[i] for i in pool_idx]

        selected = greedy_select(
            pool_shuf,
            passage_to_unit=p2u,
            N=max_budget,
            lambdas=lambdas,
            unknownness_min_frac=unknownness_min_frac,
        )

        base_dir = rdir / "baselines" / "B1_coverage_first"
        base_dir.mkdir(parents=True, exist_ok=True)
        full_path = base_dir / f"edgeqa_N{max_budget}.jsonl"
        write_jsonl(full_path, selected)

        cov_by_budget: Dict[str, Any] = {}
        quality_by_budget: Dict[str, Any] = {}
        reason_by_budget: Dict[str, Any] = {}
        for b in budgets:
            if b == max_budget:
                b_path = full_path
            else:
                b_path = base_dir / f"edgeqa_N{b}.jsonl"
                _write_prefix_jsonl(full_path, b_path, b)
            cov = compute_edgeqa_coverage(passages_path=passages_path, units_path=units_path, edgeqa_path=b_path)
            cov_by_budget[str(b)] = cov

            rows = _read_jsonl(b_path)
            unknown = sum(1 for r in rows if bool((r.get("scores") or {}).get("unknown")))
            mh = sum(1 for r in rows if isinstance(r.get("evidence"), list) and len(r.get("evidence")) >= 2)
            quality_by_budget[str(b)] = {
                "rows": len(rows),
                "unknown_frac": (unknown / len(rows)) if rows else 0.0,
                "multihop_frac": (mh / len(rows)) if rows else 0.0,
            }
            reason_by_budget[str(b)] = _reason_counts(rows)

        report["per_corpus"][corpus] = {
            "pool": len(pool),
            "selected": len(selected),
            "out_dir": str(base_dir),
            "coverage_by_budget": cov_by_budget,
            "quality_by_budget": quality_by_budget,
            "reason_type_counts_by_budget": reason_by_budget,
        }

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
