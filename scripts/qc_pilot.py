from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def _sum_usage(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_reasoning = 0

    for r in records:
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


def _sample_jsonl(path: Path, n: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for row in read_jsonl(path):
        out.append(row)
        if len(out) >= n:
            break
    return out


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
    rows = list(read_jsonl(path))
    if not rows:
        return {"rows": 0}

    unknown = 0
    cb_correct = 0
    multihop = 0
    reason = Counter()
    edge_scores: List[float] = []
    qe: List[float] = []

    for r in rows:
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

    return {
        "rows": len(rows),
        "unknown_frac": unknown / len(rows),
        "closed_book_correct_frac": cb_correct / len(rows),
        "multihop_frac": multihop / len(rows),
        "reason_type_counts": dict(reason),
        "edge_score_stats": _stats(edge_scores),
        "question_evidence_jaccard_stats": _stats(qe),
    }


def _edgecoverbench_quality(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"rows": 0}
    rows = list(read_jsonl(path))
    if not rows:
        return {"rows": 0}

    types = Counter(str(r.get("type") or "unknown") for r in rows)
    labels = Counter(str(r.get("label") or "unknown") for r in rows)
    return {"rows": len(rows), "type_counts": dict(types), "label_counts": dict(labels)}


def _load_call_logs(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            rows.extend(list(read_jsonl(p)))
        except Exception:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_pilot.yaml")
    ap.add_argument("--run-name", type=str, default=None, help="Override cfg.run_name")
    ap.add_argument("--corpora", type=str, default="osp,olp", help="Comma-separated: osp,olp")
    ap.add_argument("--limit-passages", type=int, default=5)
    ap.add_argument("--limit-units", type=int, default=5)
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    log = get_logger("qc_pilot")

    cfg = load_config(args.config)

    if args.run_name and str(args.run_name).strip():
        cfg["run_name"] = str(args.run_name).strip()
    else:
        base = str(cfg.get("run_name", "run") or "run")
        if base == "qc_pilot":
            cfg["run_name"] = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}"

    corpora = [c.strip() for c in (args.corpora or "").split(",") if c.strip()]
    if not corpora:
        corpora = ["osp"]

    cache_llm_dir = cache_root(cfg) / "llm"

    call_logs: List[Path] = []
    for c in corpora:
        rdir = run_dir(cfg, c)
        call_logs.extend(
            [
                rdir / "llm_calls_edgeqa.jsonl",
                rdir / "llm_calls_edgecoverbench.jsonl",
            ]
        )

    # Ensure a clean pilot accounting run (logs are append-only).
    for p in call_logs:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    t0 = time.time()
    step_times: Dict[str, float] = {}
    per_corpus: Dict[str, Any] = {}

    async def run_one(corpus: str) -> None:
        c_start = time.time()

        s = time.time()
        meta = ingest_corpus(cfg, corpus)
        step_times[f"{corpus}.ingest_s"] = time.time() - s

        s = time.time()
        mine_corpus_passages(cfg, corpus)
        step_times[f"{corpus}.mine_s"] = time.time() - s

        s = time.time()
        edgeqa_path = await build_edgeqa(cfg, corpus, limit_passages=args.limit_passages)
        step_times[f"{corpus}.edgeqa_s"] = time.time() - s

        cdir = corpus_dir(cfg, corpus)
        rdir = run_dir(cfg, corpus)

        s = time.time()
        beir_dir = rdir / "beir"
        beir_meta = export_beir(passages_path=cdir / "passages.jsonl", edgeqa_path=edgeqa_path, out_dir=beir_dir)
        dump_json(beir_dir / "meta.json", beir_meta)
        step_times[f"{corpus}.beir_export_s"] = time.time() - s

        s = time.time()
        bm25_metrics = run_bm25(beir_dir=beir_dir)
        step_times[f"{corpus}.bm25_s"] = time.time() - s

        s = time.time()
        cov = compute_edgeqa_coverage(
            passages_path=cdir / "passages.jsonl",
            units_path=cdir / "units.jsonl",
            edgeqa_path=edgeqa_path,
        )
        dump_json(rdir / "coverage.json", cov)
        step_times[f"{corpus}.coverage_s"] = time.time() - s

        s = time.time()
        ecb_path = await build_edgecoverbench(cfg, corpus, limit_units=args.limit_units)
        step_times[f"{corpus}.edgecoverbench_s"] = time.time() - s

        per_corpus[corpus] = {
            "ingest_meta": meta,
            "edgeqa_summary": _read_json(rdir / "edgeqa_summary.json") if (rdir / "edgeqa_summary.json").exists() else None,
            "coverage": cov,
            "bm25": bm25_metrics,
            "edgecoverbench_summary": _read_json(rdir / "edgecoverbench_summary.json") if (rdir / "edgecoverbench_summary.json").exists() else None,
            "quality": {
                "edgeqa": _edgeqa_quality(Path(edgeqa_path)),
                "edgecoverbench": _edgecoverbench_quality(Path(ecb_path)),
            },
            "samples": {
                "edgeqa": _sample_jsonl(Path(edgeqa_path), 2),
                "edgecoverbench": _sample_jsonl(Path(ecb_path), 2),
            },
            "elapsed_s": time.time() - c_start,
        }

    import asyncio

    async def _run_all() -> None:
        await asyncio.gather(*(run_one(c) for c in corpora))

    asyncio.run(_run_all())

    elapsed_s = time.time() - t0

    llm_call_rows = _load_call_logs(call_logs)

    by_namespace: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in llm_call_rows:
        ns = str(r.get("namespace") or "unknown")
        by_namespace[ns].append(r)
        m = str(r.get("response_model") or (r.get("request") or {}).get("model") or r.get("request_model") or "unknown")
        by_model[m].append(r)

    usage_total = _sum_usage(llm_call_rows)
    usage_by_namespace = {k: _sum_usage(v) for k, v in by_namespace.items()}
    usage_by_model = {k: _sum_usage(v) for k, v in by_model.items()}

    report = {
        "started_at": t0,
        "ended_at": time.time(),
        "elapsed_s": elapsed_s,
        "config_path": cfg.get("_config_path"),
        "cache_llm_dir": str(cache_llm_dir),
        "llm_call_logs": [str(p) for p in call_logs if p.exists()],
        "llm_call_rows": len(llm_call_rows),
        "usage_total": usage_total,
        "usage_by_namespace": usage_by_namespace,
        "usage_by_model": usage_by_model,
        "step_times": step_times,
        "per_corpus": per_corpus,
        "env_defaults": {
            "EDGEQA_OPENAI_BASE_URL": os.getenv("EDGEQA_OPENAI_BASE_URL", "https://aiping.cn/api"),
            "EDGEQA_PER_KEY_CONCURRENCY": os.getenv("EDGEQA_PER_KEY_CONCURRENCY", "3"),
            "EDGEQA_PER_KEY_MIN_INTERVAL_SEC": os.getenv("EDGEQA_PER_KEY_MIN_INTERVAL_SEC", "0.0"),
            "EDGEQA_DEEPSEEK_MAX_TOKENS": os.getenv("EDGEQA_DEEPSEEK_MAX_TOKENS", "8000"),
            "EDGEQA_DEEPSEEK_THINKING_BUDGET": os.getenv("EDGEQA_DEEPSEEK_THINKING_BUDGET", "2048"),
        },
    }

    runs_root = artifacts_root(cfg) / "runs" / str(cfg.get("run_name", "run"))
    runs_root.mkdir(parents=True, exist_ok=True)
    out_path = runs_root / "pilot_report.json"
    dump_json(out_path, report)

    log.info(
        "Pilot done in %.1fs; llm_call_rows=%d; total_tokens=%d",
        elapsed_s,
        len(llm_call_rows),
        usage_total["total_tokens"],
    )
    log.info("Report: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
