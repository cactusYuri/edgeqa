from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def _is_multihop(ex: Dict[str, Any]) -> bool:
    ev = ex.get("evidence") or []
    return isinstance(ev, list) and len(ev) >= 2


def _reason_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        rt = str(r.get("reason_type") or "").strip().lower() or "unknown"
        c[rt] += 1
    return dict(c)


def _mean(xs: List[bool]) -> float:
    if not xs:
        return 0.0
    return sum(1 for x in xs if x) / len(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--budgets", type=str, default="1000,2000,5000,10000")
    ap.add_argument("--model", type=str, default="qwen-plus")
    ap.add_argument(
        "--qwen-pool-eval-dir",
        type=str,
        default=None,
        help="Directory containing edgeqa_pool_qwen_rows.jsonl (default: artifacts/runs/<run>/evals/<model>_pool_full).",
    )
    ap.add_argument("--unknown-min-frac", type=float, default=0.5)
    ap.add_argument("--out", type=str, default=None, help="Optional output JSON report path.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["run_name"] = str(args.run_name)

    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]
    budgets = sorted({int(x.strip()) for x in str(args.budgets).split(",") if x.strip()})
    budgets = [b for b in budgets if b > 0]
    if not budgets:
        budgets = [10000]
    max_budget = max(budgets)

    qwen_eval_dir = (
        Path(args.qwen_pool_eval_dir)
        if args.qwen_pool_eval_dir
        else artifacts_root(cfg) / "runs" / str(args.run_name) / "evals" / f"{args.model}_pool_full"
    )
    qwen_rows_path = qwen_eval_dir / "edgeqa_pool_qwen_rows.jsonl"
    if not qwen_rows_path.exists():
        raise SystemExit(f"missing qwen pool eval rows: {qwen_rows_path}")

    qwen_rows = _read_jsonl(qwen_rows_path)
    qwen_by_id: Dict[str, Dict[str, Any]] = {str(r.get("id") or ""): r for r in qwen_rows if str(r.get("id") or "")}
    if not qwen_by_id:
        raise SystemExit(f"empty qwen pool eval rows: {qwen_rows_path}")

    lambdas = {"doc": 1.0, "unit": 1.0, "reason": 0.5, "redundancy": 0.2}
    unknown_min_frac = max(0.0, min(1.0, float(args.unknown_min_frac)))

    out_report = (
        Path(args.out)
        if args.out
        else artifacts_root(cfg)
        / "runs"
        / str(args.run_name)
        / "baselines"
        / "selection_qwen_target_from_pool.json"
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "run_name": str(args.run_name),
        "generated_at": int(time.time()),
        "model": str(args.model),
        "qwen_pool_eval_dir": str(qwen_eval_dir),
        "budgets": budgets,
        "max_budget": max_budget,
        "lambdas": lambdas,
        "unknownness_min_frac": unknown_min_frac,
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

        scored_pool: List[Dict[str, Any]] = []
        missing = 0
        for ex in pool:
            ex_id = str(ex.get("id") or "")
            q = qwen_by_id.get(ex_id)
            if not q:
                missing += 1

            scores = dict(ex.get("scores") or {})
            scores["unknown_deepseek"] = bool(scores.get("unknown", False))
            scores["qwen_cb_correct"] = bool((q or {}).get("cb_correct", False))
            scores["qwen_ctx_correct"] = bool((q or {}).get("ctx_correct", False))
            scores["unknown_qwen"] = bool((q or {}).get("unknown", False))
            # Greedy selector reads `scores["unknown"]`.
            scores["unknown"] = bool(scores["unknown_qwen"])

            ex2 = dict(ex)
            ex2["scores"] = scores
            scored_pool.append(ex2)

        base_dir = rdir / "baselines" / f"sel_qwen_target_rho{int(round(unknown_min_frac * 100)):02d}"
        base_dir.mkdir(parents=True, exist_ok=True)

        selected = greedy_select(
            scored_pool,
            passage_to_unit=p2u,
            N=max_budget,
            lambdas=lambdas,
            unknownness_min_frac=unknown_min_frac,
        )
        full_path = base_dir / f"edgeqa_N{max_budget}.jsonl"
        write_jsonl(full_path, selected[:max_budget])

        cov_by_budget: Dict[str, Any] = {}
        qual_by_budget: Dict[str, Any] = {}
        reason_by_budget: Dict[str, Any] = {}
        for b in budgets:
            if b == max_budget:
                b_path = full_path
            else:
                b_path = base_dir / f"edgeqa_N{b}.jsonl"
                _write_prefix_jsonl(full_path, b_path, b)

            cov_by_budget[str(b)] = compute_edgeqa_coverage(
                passages_path=passages_path, units_path=units_path, edgeqa_path=b_path
            )

            rows = _read_jsonl(b_path)
            qwen_unknown = [bool((r.get("scores") or {}).get("unknown_qwen", False)) for r in rows]
            qwen_cb_ok = [bool((r.get("scores") or {}).get("qwen_cb_correct", False)) for r in rows]
            qwen_ctx_ok = [bool((r.get("scores") or {}).get("qwen_ctx_correct", False)) for r in rows]
            mh = [bool(_is_multihop(r)) for r in rows]
            qual_by_budget[str(b)] = {
                "rows": len(rows),
                "qwen_unknown_frac": _mean(qwen_unknown),
                "qwen_cb_acc": _mean(qwen_cb_ok),
                "qwen_ctx_acc": _mean(qwen_ctx_ok),
                "multihop_frac": _mean(mh),
            }
            reason_by_budget[str(b)] = _reason_counts(rows)

        report["per_corpus"][corpus] = {
            "pool": len(pool),
            "pool_qwen_unknown_frac": _mean([bool((qwen_by_id.get(str(x.get("id") or "")) or {}).get("unknown", False)) for x in pool]),
            "missing_in_qwen_rows": int(missing),
            "out_dir": str(base_dir),
            "selected": int(min(max_budget, len(selected))),
            "coverage_by_budget": cov_by_budget,
            "quality_by_budget": qual_by_budget,
            "reason_type_counts_by_budget": reason_by_budget,
        }

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

