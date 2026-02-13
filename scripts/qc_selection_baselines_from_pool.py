from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edgeqa.config import load_config
from edgeqa.eval.coverage import compute_edgeqa_coverage
from edgeqa.jsonl import read_jsonl, write_jsonl
from edgeqa.pipeline.paths import artifacts_root, corpus_dir, run_dir
from edgeqa.pipeline.select import greedy_select


_WORD_RE = re.compile(r"[A-Za-z]{3,}")


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


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _rarity_scores(passages: List[Dict[str, Any]]) -> Dict[str, float]:
    freq: Counter[str] = Counter()
    toks_by_pid: Dict[str, List[str]] = {}
    for p in passages:
        pid = str(p.get("passage_id") or "")
        if not pid:
            continue
        toks = _tokenize(str(p.get("text") or ""))
        toks_by_pid[pid] = toks
        freq.update(toks)

    scores: Dict[str, float] = {}
    for pid, toks in toks_by_pid.items():
        if not toks:
            scores[pid] = 0.0
            continue
        # Mean inverse sqrt frequency (lightweight rarity proxy).
        s = 0.0
        for w in toks:
            s += 1.0 / math.sqrt(max(1, int(freq[w])))
        scores[pid] = float(s / len(toks))
    return scores


def _pool_score_long_tail(ex: Dict[str, Any], rarity_by_pid: Dict[str, float]) -> float:
    ev = ex.get("evidence") or []
    if not isinstance(ev, list) or not ev:
        return 0.0
    vals = [float(rarity_by_pid.get(str(pid), 0.0)) for pid in ev if pid is not None]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _pool_score_paraphrase(ex: Dict[str, Any]) -> float:
    sc = ex.get("scores") or {}
    try:
        pa = float(sc.get("paraphrase_agreement"))
    except Exception:
        pa = 1.0
    if not math.isfinite(pa):
        pa = 1.0
    pa = min(1.0, max(0.0, pa))
    return float(1.0 - pa)


def _pool_unknown(ex: Dict[str, Any]) -> bool:
    sc = ex.get("scores") or {}
    return bool(sc.get("unknown", False))


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

    out_report = (
        Path(args.out)
        if args.out
        else artifacts_root(cfg) / "runs" / str(args.run_name) / "baselines" / "selection_from_pool.json"
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    baseline_defs: Dict[str, Dict[str, Any]] = {
        "random": {"kind": "order", "description": "Uniform random selection from the existing EdgeQA pool."},
        "long_tail": {"kind": "score", "description": "Select items whose evidence passages contain rarer terms (avg inv-sqrt-freq)."},
        "paraphrase_only": {"kind": "score", "description": "Select items with highest paraphrase inconsistency (1 - paraphrase_agreement)."},
        "coverage_first": {
            "kind": "greedy",
            "description": "Greedy maximize DocCov (lambda_doc=1) without unknownness constraint (B1).",
            "lambdas": {"doc": 1.0, "unit": 0.0, "reason": 0.0, "redundancy": 0.2},
            "unknownness_min_frac": 0.0,
        },
        "unit_first": {
            "kind": "greedy",
            "description": "Greedy maximize UnitCov (lambda_unit=1) without unknownness constraint.",
            "lambdas": {"doc": 0.0, "unit": 1.0, "reason": 0.0, "redundancy": 0.2},
            "unknownness_min_frac": 0.0,
        },
    }

    report: Dict[str, Any] = {
        "run_name": str(args.run_name),
        "generated_at": int(time.time()),
        "seed": int(args.seed),
        "budgets": budgets,
        "max_budget": max_budget,
        "baselines": baseline_defs,
        "per_corpus": {},
    }

    for corpus in corpora:
        rdir = run_dir(cfg, corpus=corpus)
        cdir = corpus_dir(cfg, corpus=corpus)

        pool_path = rdir / "edgeqa_pool.jsonl"
        passages_path = cdir / "passages.jsonl"
        units_path = cdir / "units.jsonl"
        main_path = rdir / f"edgeqa_N{max_budget}.jsonl"

        if not pool_path.exists():
            raise SystemExit(f"missing pool: {pool_path}")
        if not passages_path.exists():
            raise SystemExit(f"missing passages: {passages_path}")
        if not units_path.exists():
            raise SystemExit(f"missing units: {units_path}")
        if not main_path.exists():
            raise SystemExit(f"missing main EdgeQA file: {main_path}")

        pool = _read_jsonl(pool_path)
        passages = _read_jsonl(passages_path)
        p2u = _passage_to_unit(passages)
        rarity_by_pid = _rarity_scores(passages)

        report["per_corpus"][corpus] = {"pool": len(pool), "by_baseline": {}}

        def _write_and_eval(*, baseline_id: str, selected: List[Dict[str, Any]]) -> None:
            base_dir = rdir / "baselines" / f"sel_{baseline_id}"
            base_dir.mkdir(parents=True, exist_ok=True)

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

                cov = compute_edgeqa_coverage(passages_path=passages_path, units_path=units_path, edgeqa_path=b_path)
                cov_by_budget[str(b)] = cov

                rows = _read_jsonl(b_path)
                unk = sum(1 for r in rows if _pool_unknown(r))
                mh = sum(1 for r in rows if _is_multihop(r))
                qual_by_budget[str(b)] = {
                    "rows": len(rows),
                    "unknown_frac": (unk / len(rows)) if rows else 0.0,
                    "multihop_frac": (mh / len(rows)) if rows else 0.0,
                }
                reason_by_budget[str(b)] = _reason_counts(rows)

            report["per_corpus"][corpus]["by_baseline"][baseline_id] = {
                "selected": int(min(max_budget, len(selected))),
                "out_dir": str(base_dir),
                "coverage_by_budget": cov_by_budget,
                "quality_by_budget": qual_by_budget,
                "reason_type_counts_by_budget": reason_by_budget,
            }

        # Baseline: main (already produced by our pipeline selection).
        main_rows = _read_jsonl(main_path)
        _write_and_eval(baseline_id="main", selected=main_rows)

        # Random.
        pool_idx = list(range(len(pool)))
        rng.shuffle(pool_idx)
        random_sel = [pool[i] for i in pool_idx[:max_budget]]
        _write_and_eval(baseline_id="random", selected=random_sel)

        # Long-tail.
        lt_sorted = sorted(pool, key=lambda r: _pool_score_long_tail(r, rarity_by_pid), reverse=True)
        _write_and_eval(baseline_id="long_tail", selected=lt_sorted)

        # Paraphrase-only.
        para_sorted = sorted(pool, key=_pool_score_paraphrase, reverse=True)
        _write_and_eval(baseline_id="paraphrase_only", selected=para_sorted)

        # Greedy objectives.
        cov_first = greedy_select(
            pool,
            passage_to_unit=p2u,
            N=max_budget,
            lambdas=baseline_defs["coverage_first"]["lambdas"],
            unknownness_min_frac=float(baseline_defs["coverage_first"]["unknownness_min_frac"]),
        )
        _write_and_eval(baseline_id="coverage_first", selected=cov_first)

        unit_first = greedy_select(
            pool,
            passage_to_unit=p2u,
            N=max_budget,
            lambdas=baseline_defs["unit_first"]["lambdas"],
            unknownness_min_frac=float(baseline_defs["unit_first"]["unknownness_min_frac"]),
        )
        _write_and_eval(baseline_id="unit_first", selected=unit_first)

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

