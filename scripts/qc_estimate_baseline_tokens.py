from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edgeqa.config import load_config
from edgeqa.pipeline.paths import artifacts_root


def _get(report: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int((report.get(key) or 0))
    except Exception:
        return default


def _tok(usage_by_ns: Dict[str, Any], ns: str) -> int:
    u = usage_by_ns.get(ns) or {}
    try:
        return int(u.get("total_tokens") or 0)
    except Exception:
        return 0


def _calls(call_counts_by_ns: Dict[str, Any], ns: str) -> int:
    c = call_counts_by_ns.get(ns) or {}
    try:
        return int(c.get("n") or 0)
    except Exception:
        return 0


def _avg(tokens: int, calls: int) -> float:
    if calls <= 0:
        return 0.0
    return float(tokens) / float(calls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Path to single_model_report.json (default: artifacts/runs/<run>/single_model_report.json).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: artifacts/runs/<run>/baselines/token_estimates.json).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["run_name"] = str(args.run_name)

    report_path = (
        Path(args.report_path)
        if args.report_path
        else artifacts_root(cfg) / "runs" / str(args.run_name) / "single_model_report.json"
    )
    if not report_path.exists():
        raise SystemExit(f"missing report: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8", errors="ignore"))
    usage_by_ns: Dict[str, Any] = report.get("usage_by_namespace") or {}
    call_counts_by_ns: Dict[str, Any] = report.get("call_counts_by_namespace") or {}

    # Core observed stats.
    candidates = _calls(call_counts_by_ns, "closed_book")  # ~= #question candidates processed
    ctx_calls = _calls(call_counts_by_ns, "ctx_answer")
    para_gen_calls = _calls(call_counts_by_ns, "paraphrase_gen")
    equiv_calls = _calls(call_counts_by_ns, "equiv")

    qa_gen_tok = _tok(usage_by_ns, "qa_gen")
    qa_gen_mh_tok = _tok(usage_by_ns, "qa_gen_mh")
    ctx_tok = _tok(usage_by_ns, "ctx_answer")
    ver_tok = _tok(usage_by_ns, "verify_fast")
    equiv_tok = _tok(usage_by_ns, "equiv")

    ctx_avg = _avg(ctx_tok, ctx_calls)
    ver_avg = _avg(ver_tok, _calls(call_counts_by_ns, "verify_fast"))
    equiv_avg = _avg(equiv_tok, equiv_calls)

    # Heuristic: in the current pipeline, LLM equiv is used for multiple comparisons:
    # - closed_book answer vs reference (~1 per candidate)
    # - ctx answer vs reference (~1 per ctx call)
    # - paraphrase answer pair agreement (~1 per paraphrase_gen call; approximate)
    # We estimate how many "comparisons" were attempted, then assume LLM equiv calls scale linearly with comparisons.
    comparisons_est = int(candidates + ctx_calls + para_gen_calls)
    if comparisons_est <= 0:
        comparisons_est = 1

    # Baseline estimates (token units).
    estimates: Dict[str, Any] = {}

    # B1: selection-only from existing pool (no new LLM calls).
    estimates["B1_coverage_first_from_pool"] = {
        "assumption": "Reuse existing EdgeQA pool; selection/coverage only.",
        "estimated_total_tokens": 0,
    }

    # Random/salience QG without edge scoring (still enforce ctx_correct + verifier).
    # Assumptions:
    # - we keep the same QA generation volume (qa_gen + qa_gen_mh)
    # - we run ctx_answer + verify_fast for *all* candidates (use closed_book call count as proxy)
    # - we keep ctx-vs-reference equivalence checks, with LLM equiv usage scaled by comparisons.
    ctx_all_tok = int(round(ctx_avg * candidates))
    ver_all_tok = int(round(ver_avg * candidates))

    # If we keep LLM equiv for ctx correctness only, baseline comparisons ~= candidates (one per candidate).
    equiv_calls_ctx_only = int(round(equiv_calls * (float(candidates) / float(comparisons_est))))
    equiv_tok_ctx_only = int(round(equiv_avg * equiv_calls_ctx_only))

    estimates["QG_no_edge_scoring_keep_ctx_verify_equiv"] = {
        "assumption": {
            "qa_gen_tokens": "same as report",
            "qa_gen_mh_tokens": "same as report",
            "ctx_answer_tokens": "scaled to all candidates (~closed_book calls)",
            "verify_fast_tokens": "scaled to all candidates (~closed_book calls)",
            "equiv_tokens": "scaled to ctx-only comparisons (rough)",
        },
        "inputs": {
            "candidates_closed_book_calls": candidates,
            "comparisons_est": comparisons_est,
            "equiv_calls_total_observed": equiv_calls,
        },
        "estimated_total_tokens": int(qa_gen_tok + qa_gen_mh_tok + ctx_all_tok + ver_all_tok + equiv_tok_ctx_only),
        "components": {
            "qa_gen": int(qa_gen_tok),
            "qa_gen_mh": int(qa_gen_mh_tok),
            "ctx_answer_scaled": int(ctx_all_tok),
            "verify_fast_scaled": int(ver_all_tok),
            "equiv_scaled_ctx_only": int(equiv_tok_ctx_only),
        },
    }

    # Same baseline, but using *string-only* equivalence (no LLM equiv).
    estimates["QG_no_edge_scoring_keep_ctx_verify_no_llm_equiv"] = {
        "assumption": "As above, but equivalence is non-LLM (quick normalize) => equiv tokens ~ 0.",
        "estimated_total_tokens": int(qa_gen_tok + qa_gen_mh_tok + ctx_all_tok + ver_all_tok),
        "components": {
            "qa_gen": int(qa_gen_tok),
            "qa_gen_mh": int(qa_gen_mh_tok),
            "ctx_answer_scaled": int(ctx_all_tok),
            "verify_fast_scaled": int(ver_all_tok),
            "equiv": 0,
        },
    }

    # Self-knowledge heuristic (selection): add one "self-knowledge" closed-book style query per candidate.
    # We approximate per-call tokens using the observed closed_book average.
    cb_tok = _tok(usage_by_ns, "closed_book")
    cb_calls = _calls(call_counts_by_ns, "closed_book")
    cb_avg = _avg(cb_tok, cb_calls)
    estimates["Self_knowledge_extra_cost_est"] = {
        "assumption": "Add 1 extra short call per candidate; per-call tokens ~= closed_book average.",
        "inputs": {"candidates": candidates, "closed_book_avg_tokens": cb_avg},
        "estimated_extra_tokens": int(round(cb_avg * candidates)),
    }

    out_path = (
        Path(args.out)
        if args.out
        else artifacts_root(cfg) / "runs" / str(args.run_name) / "baselines" / "token_estimates.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "run_name": str(args.run_name),
        "source_report": str(report_path),
        "notes": [
            "These are rough estimates based on observed call logs; they are meant for budgeting, not exact accounting.",
            "If a baseline changes filtering/acceptance rates materially, costs can shift (especially ctx/verify).",
        ],
        "observed": {
            "total_tokens": int((report.get("usage_total") or {}).get("total_tokens") or 0),
            "edgeqa_tokens": int(sum(v.get("total_tokens") or 0 for k, v in usage_by_ns.items() if not str(k).startswith("ecb_"))),
            "edgecoverbench_tokens": int(sum(v.get("total_tokens") or 0 for k, v in usage_by_ns.items() if str(k).startswith("ecb_"))),
            "candidates_closed_book_calls": int(candidates),
        },
        "estimates": estimates,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

