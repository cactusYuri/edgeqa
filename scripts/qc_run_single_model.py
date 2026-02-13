from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LLMapi_service.gptservice import GPT, close_session

from edgeqa.config import load_config
from edgeqa.eval.coverage import compute_edgeqa_coverage
from edgeqa.ir.beir import export_beir
from edgeqa.ir.bm25 import run_bm25
from edgeqa.jsonl import dump_json, read_jsonl
from edgeqa.logging_utils import get_logger, setup_logging
from edgeqa.pipeline.edgecoverbench import build_edgecoverbench
from edgeqa.pipeline.edgeqa import build_edgeqa
from edgeqa.pipeline.ingest import ingest_corpus
from edgeqa.pipeline.mine import mine_corpus_passages
from edgeqa.pipeline.paths import artifacts_root, cache_root, corpus_dir, run_dir


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _stats(xs: Iterable[float]) -> Dict[str, Any]:
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return {"n": 0}
    vals.sort()
    n = len(vals)
    mid = n // 2
    median = vals[mid] if (n % 2 == 1) else 0.5 * (vals[mid - 1] + vals[mid])
    p90 = vals[max(0, int(0.9 * n) - 1)]
    return {
        "n": n,
        "mean": sum(vals) / n,
        "median": median,
        "p90": p90,
        "min": vals[0],
        "max": vals[-1],
    }


def _edgeqa_quality(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"rows": 0}
    unknown = 0
    cb_correct = 0
    multihop = 0
    reason = Counter()
    edge_scores: List[float] = []
    qe: List[float] = []

    rows = 0
    for r in read_jsonl(path):
        rows += 1
        sc = r.get("scores") or {}
        if sc.get("unknown"):
            unknown += 1
        if sc.get("closed_book_correct"):
            cb_correct += 1
        ev = r.get("evidence") or []
        if isinstance(ev, list) and len(ev) >= 2:
            multihop += 1
        rt = str(r.get("reason_type") or "").strip().lower() or "unknown"
        reason[rt] += 1
        try:
            edge_scores.append(float(sc.get("edge_score") or 0.0))
        except Exception:
            pass
        flt = r.get("filters") or {}
        try:
            qe.append(float(flt.get("question_evidence_jaccard") or 0.0))
        except Exception:
            pass

    if rows <= 0:
        return {"rows": 0}
    return {
        "rows": rows,
        "unknown_frac": unknown / rows,
        "closed_book_correct_frac": cb_correct / rows,
        "multihop_frac": multihop / rows,
        "reason_type_counts": dict(reason),
        "edge_score_stats": _stats(edge_scores),
        "question_evidence_jaccard_stats": _stats(qe),
    }


def _edgecoverbench_quality(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"rows": 0}
    rows = 0
    types = Counter()
    labels = Counter()
    for r in read_jsonl(path):
        rows += 1
        types[str(r.get("type") or "unknown")] += 1
        labels[str(r.get("label") or "unknown")] += 1
    return {"rows": rows, "type_counts": dict(types), "label_counts": dict(labels)}


def _write_prefix_jsonl(in_path: Path, out_path: Path, n: int) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if written >= n:
                break
            line = (line or "").strip()
            if not line:
                continue
            fout.write(line + "\n")
            written += 1
    return written


def _sum_usage_stream(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_reasoning = 0
    for r in rows:
        u = r.get("usage") or {}
        if not isinstance(u, dict):
            continue
        total_prompt += int(u.get("prompt_tokens") or 0)
        total_completion += int(u.get("completion_tokens") or 0)
        total_tokens += int(u.get("total_tokens") or 0)
        details = u.get("completion_tokens_details") or {}
        if isinstance(details, dict):
            total_reasoning += int(details.get("reasoning_tokens") or 0)
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "reasoning_tokens": total_reasoning,
    }


def _update_usage(dst: Dict[str, int], rec: Dict[str, Any]) -> None:
    u = rec.get("usage") or {}
    if not isinstance(u, dict):
        return
    dst["prompt_tokens"] = int(dst.get("prompt_tokens") or 0) + int(u.get("prompt_tokens") or 0)
    dst["completion_tokens"] = int(dst.get("completion_tokens") or 0) + int(u.get("completion_tokens") or 0)
    dst["total_tokens"] = int(dst.get("total_tokens") or 0) + int(u.get("total_tokens") or 0)
    details = u.get("completion_tokens_details") or {}
    if isinstance(details, dict):
        dst["reasoning_tokens"] = int(dst.get("reasoning_tokens") or 0) + int(details.get("reasoning_tokens") or 0)


def _iter_call_logs(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        if not p.exists():
            continue
        # Call logs can contain a partially-written last line if the process was interrupted.
        # Be robust here so report generation never fails on a single bad JSON line.
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    yield row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--budgets", type=str, default="1000,2000,5000,10000")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingest step if corpus files already exist (avoids git fetch/checkout).",
    )
    ap.add_argument(
        "--skip-mine",
        action="store_true",
        help="Skip mining step if mined file already exists.",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="Only aggregate existing artifacts and write single_model_report.json; do not run the pipeline.",
    )
    ap.add_argument(
        "--repair-edgeqa",
        action="store_true",
        help="Repair mode: re-run EdgeQA only for sources with no pool items (by temporarily clearing edgeqa_done_sources.txt).",
    )
    ap.add_argument("--llm-concurrency", type=int, default=None, help="Override cfg.llm.concurrency for this run.")
    ap.add_argument("--passage-workers", type=int, default=None, help="Override cfg.edgeqa.passage_workers for this run.")
    ap.add_argument("--unit-workers", type=int, default=None, help="Override cfg.edgecoverbench.unit_workers for this run.")
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip a quick LLM auth/connectivity check before running (default: run preflight).",
    )
    ap.add_argument(
        "--reset-logs",
        action="store_true",
        help="Delete existing llm call logs under this run-name before running (default: keep/append).",
    )
    args = ap.parse_args()

    setup_logging(args.log_level)
    log = get_logger("qc_run_single_model")

    cfg = load_config(args.config)

    if args.run_name and str(args.run_name).strip():
        cfg["run_name"] = str(args.run_name).strip()
    else:
        base = str(cfg.get("run_name", "run") or "run")
        cfg["run_name"] = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"

    corpora = [c.strip() for c in (args.corpora or "").split(",") if c.strip()]
    if not corpora:
        corpora = ["osp"]

    budgets = sorted({int(x.strip()) for x in str(args.budgets).split(",") if x.strip()})
    if not budgets:
        budgets = [1000]

    max_budget = max(budgets)
    cfg.setdefault("edgeqa", {})
    cfg["edgeqa"]["final_N"] = max_budget
    if args.llm_concurrency is not None:
        cfg.setdefault("llm", {})
        cfg["llm"]["concurrency"] = int(args.llm_concurrency)
    if args.passage_workers is not None:
        cfg.setdefault("edgeqa", {})
        cfg["edgeqa"]["passage_workers"] = int(args.passage_workers)
    if args.unit_workers is not None:
        cfg.setdefault("edgecoverbench", {})
        cfg["edgecoverbench"]["unit_workers"] = int(args.unit_workers)

    cache_llm_dir = cache_root(cfg) / "llm"
    step_times: Dict[str, float] = {}
    per_corpus: Dict[str, Any] = {}

    call_logs: List[Path] = []
    for c in corpora:
        rdir = run_dir(cfg, c)
        call_logs.extend([rdir / "llm_calls_edgeqa.jsonl", rdir / "llm_calls_edgecoverbench.jsonl"])

    if bool(args.reset_logs):
        # Fresh accounting for this run_name (call logs are append-only).
        for p in call_logs:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    import asyncio

    if not bool(args.report_only) and not bool(args.skip_preflight) and str(os.getenv("EDGEQA_SKIP_PREFLIGHT", "")).strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        model_chat = str(cfg.get("llm", {}).get("model_chat", "deepseek-chat"))

        async def _preflight() -> None:
            try:
                await GPT(
                    [{"role": "system", "content": "Reply with exactly 'OK'."}, {"role": "user", "content": "ping"}],
                    selected_model=model_chat,
                    temperature=0.0,
                    max_tokens=1,
                    raise_on_error=True,
                )
            finally:
                # Keep the run's sessions clean; the pipeline will create a fresh one.
                await close_session()

        try:
            asyncio.run(_preflight())
        except Exception as e:
            log.error("Preflight failed; aborting to avoid corrupting outputs. Error: %s", repr(e))
            return 2

    t0 = time.time()

    def _collect_existing(corpus: str) -> None:
        rdir = run_dir(cfg, corpus)
        cdir = corpus_dir(cfg, corpus)

        edgeqa_path = rdir / "edgeqa.jsonl"
        edgecover_path = rdir / "edgecoverbench.jsonl"

        edgeqa_budget_paths: Dict[int, str] = {}
        budget_cov: Dict[int, Any] = {}
        budget_bm25: Dict[int, Any] = {}
        budget_quality: Dict[int, Any] = {}

        for b in budgets:
            out_b = rdir / f"edgeqa_N{b}.jsonl"
            if not out_b.exists() and edgeqa_path.exists():
                written = _write_prefix_jsonl(edgeqa_path, out_b, b)
                log.info("[%s] Wrote prefix N=%d written=%d", corpus, b, written)

            if out_b.exists():
                edgeqa_budget_paths[b] = str(out_b)
                budget_quality[b] = _edgeqa_quality(out_b)

            cov_path = rdir / f"coverage_N{b}.json"
            if cov_path.exists():
                budget_cov[b] = _read_json(cov_path)

            bm25_path = rdir / f"beir_N{b}" / "bm25_metrics.json"
            if bm25_path.exists():
                budget_bm25[b] = _read_json(bm25_path)

        per_corpus[corpus] = {
            "ingest_meta": None,
            "edgeqa_summary": _read_json(rdir / "edgeqa_summary.json") if (rdir / "edgeqa_summary.json").exists() else None,
            "edgecoverbench_summary": _read_json(rdir / "edgecoverbench_summary.json")
            if (rdir / "edgecoverbench_summary.json").exists()
            else None,
            "edgeqa_paths": edgeqa_budget_paths,
            "coverage_by_budget": budget_cov,
            "bm25_by_budget": budget_bm25,
            "quality_by_budget": budget_quality,
            "edgecoverbench_quality": _edgecoverbench_quality(edgecover_path),
            "elapsed_s": None,
            "paths": {
                "corpus_dir": str(cdir),
                "run_dir": str(rdir),
                "edgeqa": str(edgeqa_path) if edgeqa_path.exists() else None,
                "edgecoverbench": str(edgecover_path) if edgecover_path.exists() else None,
            },
        }

    def _maybe_repair_edgeqa_done(corpus: str) -> Optional[str]:
        if not bool(args.repair_edgeqa):
            return None
        rdir = run_dir(cfg, corpus)
        pool_path = rdir / "edgeqa_pool.jsonl"
        done_path = rdir / "edgeqa_done_sources.txt"
        if not pool_path.exists():
            return None
        if not done_path.exists():
            # Create an empty file so build_edgeqa sees it.
            try:
                done_path.parent.mkdir(parents=True, exist_ok=True)
                done_path.write_text("", encoding="utf-8")
            except Exception:
                pass
            return None
        backup = done_path.with_suffix(f".before_repair_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            done_path.replace(backup)
        except Exception:
            # If rename fails, fall back to copying then truncating.
            try:
                backup.write_text(done_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            except Exception:
                backup = None
        try:
            done_path.write_text("", encoding="utf-8")
        except Exception:
            pass
        return str(backup) if backup is not None else None

    async def run_one(corpus: str) -> None:
        c_start = time.time()

        meta = None
        if not bool(args.skip_ingest):
            s = time.time()
            meta = ingest_corpus(cfg, corpus)
            step_times[f"{corpus}.ingest_s"] = time.time() - s
        else:
            # Best-effort: load existing ingest meta if present.
            try:
                meta_path = corpus_dir(cfg, corpus) / "meta.json"
                meta = _read_json(meta_path) if meta_path.exists() else None
            except Exception:
                meta = None

        if not bool(args.skip_mine):
            s = time.time()
            mine_corpus_passages(cfg, corpus)
            step_times[f"{corpus}.mine_s"] = time.time() - s
        else:
            # Only mine if the expected file is missing.
            try:
                cdir = corpus_dir(cfg, corpus)
                mined_k = int(cfg.get("edgeqa", {}).get("candidate_passages", 200))
                mined_path = cdir / f"mined_passages_k{mined_k}.jsonl"
                if not mined_path.exists():
                    s = time.time()
                    mine_corpus_passages(cfg, corpus)
                    step_times[f"{corpus}.mine_s"] = time.time() - s
            except Exception:
                pass

        backup_done = _maybe_repair_edgeqa_done(corpus)

        s = time.time()
        edgeqa_path = await build_edgeqa(cfg, corpus)
        step_times[f"{corpus}.edgeqa_s"] = time.time() - s

        cdir = corpus_dir(cfg, corpus)
        rdir = run_dir(cfg, corpus)

        # Derive budget prefixes (coverage/cost curves) without extra LLM calls.
        edgeqa_budget_paths: Dict[int, str] = {}
        budget_cov: Dict[int, Any] = {}
        budget_bm25: Dict[int, Any] = {}
        budget_quality: Dict[int, Any] = {}
        for b in budgets:
            out_b = rdir / f"edgeqa_N{b}.jsonl"
            written = _write_prefix_jsonl(Path(edgeqa_path), out_b, b)
            edgeqa_budget_paths[b] = str(out_b)

            s = time.time()
            beir_dir = rdir / f"beir_N{b}"
            beir_meta = export_beir(passages_path=cdir / "passages.jsonl", edgeqa_path=out_b, out_dir=beir_dir)
            dump_json(beir_dir / "meta.json", beir_meta)
            step_times[f"{corpus}.beir_export_N{b}_s"] = time.time() - s

            s = time.time()
            bm25_metrics = run_bm25(beir_dir=beir_dir)
            budget_bm25[b] = bm25_metrics
            step_times[f"{corpus}.bm25_N{b}_s"] = time.time() - s

            s = time.time()
            cov = compute_edgeqa_coverage(
                passages_path=cdir / "passages.jsonl",
                units_path=cdir / "units.jsonl",
                edgeqa_path=out_b,
            )
            dump_json(rdir / f"coverage_N{b}.json", cov)
            budget_cov[b] = cov
            step_times[f"{corpus}.coverage_N{b}_s"] = time.time() - s

            budget_quality[b] = _edgeqa_quality(out_b)

            log.info("[%s] EdgeQA budget N=%d written=%d", corpus, b, written)

        s = time.time()
        ecb_path = await build_edgecoverbench(cfg, corpus)
        step_times[f"{corpus}.edgecoverbench_s"] = time.time() - s

        per_corpus[corpus] = {
            "ingest_meta": meta,
            "edgeqa_summary": _read_json(rdir / "edgeqa_summary.json") if (rdir / "edgeqa_summary.json").exists() else None,
            "edgecoverbench_summary": _read_json(rdir / "edgecoverbench_summary.json")
            if (rdir / "edgecoverbench_summary.json").exists()
            else None,
            "edgeqa_done_backup": backup_done,
            "edgeqa_paths": edgeqa_budget_paths,
            "coverage_by_budget": budget_cov,
            "bm25_by_budget": budget_bm25,
            "quality_by_budget": budget_quality,
            "edgecoverbench_quality": _edgecoverbench_quality(Path(ecb_path)),
            "elapsed_s": time.time() - c_start,
        }

    async def _run_all() -> None:
        # Run corpora sequentially by default to simplify accounting and avoid competing writer overhead.
        for c in corpora:
            await run_one(c)

        # Ensure shared aiohttp session is closed at the very end if caller opted to keep it open.
        if not bool(cfg.get("llm", {}).get("close_session", True)):
            await close_session()

    if bool(args.report_only):
        for c in corpora:
            _collect_existing(c)
    else:
        asyncio.run(_run_all())

    elapsed_s = time.time() - t0

    # Usage aggregation (streaming, exact totals).
    usage_total: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    usage_by_namespace: DefaultDict[str, Dict[str, int]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    )
    usage_by_model: DefaultDict[str, Dict[str, int]] = defaultdict(
        lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    )

    # Call/usage stats (single pass over all call logs).
    call_counts: Dict[str, Any] = {"n": 0, "ok": 0, "fail": 0}
    call_counts_by_namespace: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0, "fail": 0})
    call_counts_by_model: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "ok": 0, "fail": 0})
    calls_ts_min: Optional[float] = None
    calls_ts_max: Optional[float] = None

    def _bucket_http(err: str) -> str:
        if not err:
            return "other"
        for code in (
            "400",
            "401",
            "402",
            "404",
            "405",
            "408",
            "413",
            "422",
            "429",
            "499",
            "500",
            "502",
            "503",
            "504",
        ):
            if f"HTTP {code}" in err:
                return code
        return "other"

    errors_by_code: DefaultDict[str, int] = defaultdict(int)
    errors_by_namespace: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors_by_model: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in _iter_call_logs(call_logs):
        call_counts["n"] += 1
        ok = bool(r.get("ok"))
        if ok:
            call_counts["ok"] += 1
        else:
            call_counts["fail"] += 1

        _update_usage(usage_total, r)
        ns = str(r.get("namespace") or "unknown")
        call_counts_by_namespace[ns]["n"] += 1
        call_counts_by_namespace[ns]["ok" if ok else "fail"] += 1
        _update_usage(usage_by_namespace[ns], r)
        m = str(r.get("response_model") or (r.get("request") or {}).get("model") or r.get("request_model") or "unknown")
        call_counts_by_model[m]["n"] += 1
        call_counts_by_model[m]["ok" if ok else "fail"] += 1
        _update_usage(usage_by_model[m], r)

        ts = r.get("ts")
        if isinstance(ts, (int, float)):
            tsf = float(ts)
            calls_ts_min = tsf if calls_ts_min is None else min(calls_ts_min, tsf)
            calls_ts_max = tsf if calls_ts_max is None else max(calls_ts_max, tsf)

        if not ok:
            code = _bucket_http(str(r.get("error") or ""))
            errors_by_code[code] += 1
            errors_by_namespace[ns][code] += 1
            errors_by_model[m][code] += 1

    report = {
        "started_at": t0,
        "ended_at": time.time(),
        "elapsed_s": elapsed_s,
        "config_path": cfg.get("_config_path"),
        "cache_llm_dir": str(cache_llm_dir),
        "llm_call_logs": [str(p) for p in call_logs if p.exists()],
        "usage_total": usage_total,
        "usage_by_namespace": dict(usage_by_namespace),
        "usage_by_model": dict(usage_by_model),
        "call_counts": call_counts,
        "call_counts_by_namespace": dict(call_counts_by_namespace),
        "call_counts_by_model": dict(call_counts_by_model),
        "calls_ts_min": calls_ts_min,
        "calls_ts_max": calls_ts_max,
        "calls_ts_span_s": (calls_ts_max - calls_ts_min) if (calls_ts_min is not None and calls_ts_max is not None) else None,
        "errors_by_code": dict(errors_by_code),
        "errors_by_namespace": {k: dict(v) for k, v in errors_by_namespace.items()},
        "errors_by_model": {k: dict(v) for k, v in errors_by_model.items()},
        "step_times": step_times,
        "per_corpus": per_corpus,
        "env_defaults": {
            "EDGEQA_OPENAI_BASE_URL": os.getenv("EDGEQA_OPENAI_BASE_URL", "https://aiping.cn/api"),
            "EDGEQA_PER_KEY_CONCURRENCY": os.getenv("EDGEQA_PER_KEY_CONCURRENCY", "3"),
            "EDGEQA_PER_KEY_MIN_INTERVAL_SEC": os.getenv("EDGEQA_PER_KEY_MIN_INTERVAL_SEC", "0.0"),
            "EDGEQA_GATEWAY_MAX_INFLIGHT": os.getenv("EDGEQA_GATEWAY_MAX_INFLIGHT", ""),
            "EDGEQA_GATEWAY_CALL_DEADLINE_SEC": os.getenv("EDGEQA_GATEWAY_CALL_DEADLINE_SEC", ""),
            "EDGEQA_405_MAX_BACKOFF_SEC": os.getenv("EDGEQA_405_MAX_BACKOFF_SEC", ""),
            "EDGEQA_429_MAX_BACKOFF_SEC": os.getenv("EDGEQA_429_MAX_BACKOFF_SEC", ""),
            "EDGEQA_DEEPSEEK_MAX_TOKENS": os.getenv("EDGEQA_DEEPSEEK_MAX_TOKENS", "8000"),
            "EDGEQA_DEEPSEEK_MIN_INTERVAL_SEC": os.getenv("EDGEQA_DEEPSEEK_MIN_INTERVAL_SEC", ""),
            "EDGEQA_DEEPSEEK_THINKING_BUDGET": os.getenv("EDGEQA_DEEPSEEK_THINKING_BUDGET", "2048"),
        },
    }

    runs_root = artifacts_root(cfg) / "runs" / str(cfg.get("run_name", "run"))
    runs_root.mkdir(parents=True, exist_ok=True)
    out_path = runs_root / "single_model_report.json"
    dump_json(out_path, report)

    log.info("Done in %.1fs; total_tokens=%d", elapsed_s, usage_total.get("total_tokens", 0))
    log.info("Report: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
