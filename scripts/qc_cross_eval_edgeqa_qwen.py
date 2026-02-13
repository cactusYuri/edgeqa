from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LLMapi_service.qwen_batch import QwenBatchClient, parse_batch_output, write_jsonl

from edgeqa.config import load_config
from edgeqa.jsonl import read_jsonl
from edgeqa.pipeline.paths import artifacts_root, run_dir
from edgeqa.pipeline.prompts import equiv_prompt
from edgeqa.text_utils import normalize_for_match


@dataclass(frozen=True)
class ExampleMeta:
    idx: int
    ex_id: str
    corpus: str
    reason_type: str
    is_multihop: bool
    question: str
    evidence_span: str
    gold_answer: str


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    return read_jsonl(path)


def _short(s: str, n: int = 600) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n] + "…")


def _parse_equiv_json(text: str) -> Optional[bool]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "equivalent" in obj:
            return bool(obj["equivalent"])
    except Exception:
        pass
    # try best-effort substring
    try:
        i = raw.find("{")
        j = raw.rfind("}")
        if i >= 0 and j > i:
            obj = json.loads(raw[i : j + 1])
            if isinstance(obj, dict) and "equivalent" in obj:
                return bool(obj["equivalent"])
    except Exception:
        pass
    return None


async def _run_batch_with_retries(
    client: QwenBatchClient,
    *,
    requests: Dict[str, Dict[str, Any]],
    out_dir: Path,
    job_name: str,
    retries: int = 1,
    chunk_size: int = 0,
    batch_parallel: int = 1,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    poll_s: float = 5.0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = dict(requests)
    results: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {
        "job_name": job_name,
        "endpoint": endpoint,
        "completion_window": completion_window,
        "poll_s": poll_s,
        "attempts": [],
    }

    chunk_size = int(chunk_size or 0)
    if chunk_size < 0:
        chunk_size = 0
    batch_parallel = max(1, int(batch_parallel or 1))

    def _chunks(d: Dict[str, Dict[str, Any]]) -> List[Dict[str, Dict[str, Any]]]:
        if not chunk_size:
            return [d]
        items = list(d.items())
        out: List[Dict[str, Dict[str, Any]]] = []
        for i in range(0, len(items), chunk_size):
            out.append(dict(items[i : i + chunk_size]))
        return out

    for attempt in range(int(retries) + 1):
        if not pending:
            break
        chunks = _chunks(pending)
        sem = asyncio.Semaphore(batch_parallel)

        async def _run_chunk(chunk_idx: int, chunk: Dict[str, Dict[str, Any]]) -> Tuple[int, Any, Path]:
            req_path = out_dir / f"requests_{job_name}_attempt{attempt}_chunk{chunk_idx:04d}.jsonl"
            n_written = write_jsonl(req_path, chunk.values())
            attempt_dir = out_dir / f"{job_name}_attempt{attempt}_chunk{chunk_idx:04d}"
            async with sem:
                run = await client.run_job(
                    requests_path=req_path,
                    out_dir=attempt_dir,
                    endpoint=endpoint,
                    completion_window=completion_window,
                    poll_s=poll_s,
                    metadata={"job": job_name, "attempt": attempt, "chunk": chunk_idx, "count": n_written},
                )
            return chunk_idx, run, attempt_dir

        tasks = [asyncio.create_task(_run_chunk(i, c)) for i, c in enumerate(chunks)]
        done = await asyncio.gather(*tasks)
        for chunk_idx, run, attempt_dir in done:
            meta["attempts"].append(
                {
                    "attempt": attempt,
                    "chunk": chunk_idx,
                    "batch_id": run.batch_id,
                    "status": run.status,
                    "count": int((run.raw or {}).get("request_counts", {}).get("total") or 0) or None,
                    "dir": str(attempt_dir),
                }
            )
            if run.status != "completed":
                raise RuntimeError(f"Batch {job_name} failed: status={run.status} batch_id={run.batch_id}")

            out_path = attempt_dir / "output.jsonl"
            if not out_path.exists():
                raise RuntimeError(f"Missing batch output: {out_path}")

            for cid, body, err in parse_batch_output(out_path):
                if cid in results:
                    continue
                if body is None:
                    continue
                results[cid] = {"body": body, "error": err}

        # update pending
        next_pending: Dict[str, Dict[str, Any]] = {}
        for cid, req in pending.items():
            if cid not in results:
                next_pending[cid] = req
        pending = next_pending

    meta["pending_after"] = len(pending)
    meta["done"] = len(results)
    if pending:
        meta["pending_custom_ids"] = sorted(list(pending.keys()))[:50]
    return results, meta


def _body_to_text(body: Dict[str, Any]) -> str:
    try:
        return str((((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or "")
    except Exception:
        return ""


def _body_usage(body: Dict[str, Any]) -> Dict[str, int]:
    u = body.get("usage") or {}
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for k in out.keys():
        try:
            out[k] = int(u.get(k) or 0)
        except Exception:
            out[k] = 0
    return out


def _sum_usage(us: Iterable[Dict[str, int]]) -> Dict[str, int]:
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in us:
        for k in out.keys():
            out[k] += int(u.get(k) or 0)
    return out


def _metrics_from_flags(flags: List[Tuple[bool, bool]]) -> Dict[str, float]:
    n = len(flags)
    if n <= 0:
        return {"n": 0}
    cb_ok = sum(1 for cb, _ in flags if cb)
    ctx_ok = sum(1 for _, ctx in flags if ctx)
    unk = sum(1 for cb, ctx in flags if (not cb) and ctx)
    return {
        "n": n,
        "cb_acc": cb_ok / n,
        "ctx_acc": ctx_ok / n,
        "unknown_purity": unk / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--model", type=str, default="qwen-plus")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--budgets", type=str, default="1000,2000,5000,10000")
    ap.add_argument("--limit", type=int, default=0, help="If >0, only eval first N examples per corpus (smoke test).")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--completion-window", type=str, default="24h")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-chunk", type=int, default=5000, help="Split batch jobs into chunks of this size (0 = one giant batch).")
    ap.add_argument("--batch-parallel", type=int, default=4, help="How many batch chunks to run in parallel.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["run_name"] = str(args.run_name)

    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]
    budgets = sorted({int(x.strip()) for x in str(args.budgets).split(",") if x.strip()})
    budgets = [b for b in budgets if b > 0]
    if not budgets:
        budgets = [10000]
    max_budget = max(budgets)

    out_dir = artifacts_root(cfg) / "runs" / str(args.run_name) / "evals" / str(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples: List[ExampleMeta] = []
    per_corpus_order: Dict[str, List[int]] = defaultdict(list)

    for corpus in corpora:
        path = run_dir(cfg, corpus=corpus) / f"edgeqa_N{max_budget}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing edgeqa file: {path}")
        for i, ex in enumerate(_iter_jsonl(path)):
            if args.limit and i >= int(args.limit):
                break
            ex_id = str(ex.get("id") or f"{corpus}:{i}")
            q = str(ex.get("question") or "")
            gold = str(ex.get("answer") or "")
            ev_span = str(ex.get("evidence_span") or "")
            ev = ex.get("evidence") or []
            is_mh = isinstance(ev, list) and len(ev) >= 2
            rt = str(ex.get("reason_type") or "").strip().lower() or ("multi-hop" if is_mh else "unknown")
            idx = len(examples)
            examples.append(
                ExampleMeta(
                    idx=idx,
                    ex_id=ex_id,
                    corpus=corpus,
                    reason_type=rt,
                    is_multihop=is_mh,
                    question=q,
                    evidence_span=ev_span,
                    gold_answer=gold,
                )
            )
            per_corpus_order[corpus].append(idx)

    # Build answer requests.
    ans_requests: Dict[str, Dict[str, Any]] = {}
    ans_meta: Dict[str, Any] = {"examples": len(examples), "corpora": corpora, "max_budget": max_budget}

    def _add_req(custom_id: str, messages: List[Dict[str, str]]) -> None:
        body: Dict[str, Any] = {
            "model": str(args.model),
            "messages": messages,
            "temperature": float(args.temperature),
            "max_tokens": int(args.max_tokens),
        }
        if args.enable_thinking:
            body["enable_thinking"] = True
        if args.thinking_budget is not None:
            body["thinking_budget"] = int(args.thinking_budget)
        ans_requests[custom_id] = {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}

    for ex in examples:
        cb_id = f"{ex.corpus}_{ex.idx}_cb"
        ctx_id = f"{ex.corpus}_{ex.idx}_ctx"
        _add_req(
            cb_id,
            [
                {"role": "system", "content": "Answer with a short, exact answer only. No explanation."},
                {"role": "user", "content": ex.question},
            ],
        )
        _add_req(
            ctx_id,
            [
                {
                    "role": "system",
                    "content": "Use the provided evidence to answer the question. Answer with a short, exact answer only. If the evidence is insufficient, output UNKNOWN. No explanation.",
                },
                {"role": "user", "content": f"Question: {ex.question}\n\nEvidence:\n{ex.evidence_span}"},
            ],
        )

    (out_dir / "edgeqa_answer_meta.json").write_text(json.dumps(ans_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _run() -> Dict[str, Any]:
        client = QwenBatchClient()
        try:
            ans_out, ans_run_meta = await _run_batch_with_retries(
                client,
                requests=ans_requests,
                out_dir=out_dir / "answers",
                job_name="answers",
                retries=int(args.retries),
                chunk_size=int(args.batch_chunk),
                batch_parallel=int(args.batch_parallel),
                completion_window=str(args.completion_window),
                poll_s=float(args.poll_s),
            )

            # Build equiv requests for all answers (cb+ctx).
            eq_requests: Dict[str, Dict[str, Any]] = {}
            for ex in examples:
                for kind in ("cb", "ctx"):
                    cid = f"{ex.corpus}_{ex.idx}_{kind}"
                    body = ans_out.get(cid, {}).get("body")
                    pred = _body_to_text(body) if isinstance(body, dict) else ""
                    # Quick deterministic shortcut: if normalized match, skip calling equiv.
                    if (
                        normalize_for_match(pred) == normalize_for_match(ex.gold_answer)
                        and pred.strip()
                        and ex.gold_answer.strip()
                    ):
                        continue
                    eq_id = f"{cid}_eq"
                    eq_body: Dict[str, Any] = {
                        "model": str(args.model),
                        "messages": [
                            {"role": "system", "content": "Return JSON only."},
                            {"role": "user", "content": equiv_prompt(pred, ex.gold_answer)},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 48,
                    }
                    if args.enable_thinking:
                        eq_body["enable_thinking"] = True
                    if args.thinking_budget is not None:
                        eq_body["thinking_budget"] = int(args.thinking_budget)
                    eq_requests[eq_id] = {"custom_id": eq_id, "method": "POST", "url": "/v1/chat/completions", "body": eq_body}

            eq_out, eq_run_meta = await _run_batch_with_retries(
                client,
                requests=eq_requests,
                out_dir=out_dir / "equiv",
                job_name="equiv",
                retries=int(args.retries),
                chunk_size=int(args.batch_chunk),
                batch_parallel=int(args.batch_parallel),
                completion_window=str(args.completion_window),
                poll_s=float(args.poll_s),
            )
        finally:
            await client.close()

        # Build per-example results.
        per_example: List[Dict[str, Any]] = []
        usage_rows: List[Dict[str, int]] = []

        for ex in examples:
            cb_cid = f"{ex.corpus}_{ex.idx}_cb"
            ctx_cid = f"{ex.corpus}_{ex.idx}_ctx"

            cb_body = (ans_out.get(cb_cid) or {}).get("body") or {}
            ctx_body = (ans_out.get(ctx_cid) or {}).get("body") or {}
            cb_pred = _body_to_text(cb_body)
            ctx_pred = _body_to_text(ctx_body)

            usage_rows.append(_body_usage(cb_body))
            usage_rows.append(_body_usage(ctx_body))

            def _equiv_flag(kind: str, pred: str) -> bool:
                if normalize_for_match(pred) == normalize_for_match(ex.gold_answer) and pred.strip() and ex.gold_answer.strip():
                    return True
                eq_id = f"{ex.corpus}_{ex.idx}_{kind}_eq"
                eq_body = (eq_out.get(eq_id) or {}).get("body") or {}
                eq_txt = _body_to_text(eq_body)
                v = _parse_equiv_json(eq_txt)
                return bool(v) if v is not None else False

            cb_ok = _equiv_flag("cb", cb_pred)
            ctx_ok = _equiv_flag("ctx", ctx_pred)

            per_example.append(
                {
                    "corpus": ex.corpus,
                    "idx": ex.idx,
                    "id": ex.ex_id,
                    "reason_type": ex.reason_type,
                    "is_multihop": ex.is_multihop,
                    "gold": ex.gold_answer,
                    "cb_pred": cb_pred,
                    "ctx_pred": ctx_pred,
                    "cb_correct": bool(cb_ok),
                    "ctx_correct": bool(ctx_ok),
                }
            )

        eq_usage_rows: List[Dict[str, int]] = []
        for eq_id in eq_requests.keys():
            body = (eq_out.get(eq_id) or {}).get("body") or {}
            eq_usage_rows.append(_body_usage(body))

        (out_dir / "edgeqa_cross_eval_rows.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in per_example) + "\n", encoding="utf-8"
        )

        # Aggregate metrics.
        report: Dict[str, Any] = {
            "run_name": str(args.run_name),
            "model": str(args.model),
            "enable_thinking": bool(args.enable_thinking),
            "thinking_budget": args.thinking_budget,
            "max_budget": max_budget,
            "budgets": budgets,
            "usage_answers_total": _sum_usage(usage_rows),
            "usage_equiv_total": _sum_usage(eq_usage_rows),
            "usage_total": _sum_usage(list(usage_rows) + list(eq_usage_rows)),
            "batch_meta": {"answers": ans_run_meta, "equiv": eq_run_meta},
            "per_corpus": {},
        }

        rows_by_corpus: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in per_example:
            rows_by_corpus[str(r["corpus"])].append(r)

        for corpus, rows in rows_by_corpus.items():
            # Sort by idx to support prefix budgets.
            rows = sorted(rows, key=lambda x: int(x["idx"]))
            per_b: Dict[str, Any] = {}
            for b in budgets:
                sub = rows[:b]
                flags = [(bool(x["cb_correct"]), bool(x["ctx_correct"])) for x in sub]
                per_b[str(b)] = _metrics_from_flags(flags)

            # breakdowns on full max budget
            full = rows[:max_budget]
            by_reason: Dict[str, Dict[str, float]] = {}
            by_hop: Dict[str, Dict[str, float]] = {}

            buckets: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
            hop_buckets: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
            for x in full:
                buckets[str(x.get("reason_type") or "unknown")].append((bool(x["cb_correct"]), bool(x["ctx_correct"])))
                hop_buckets["multi-hop" if bool(x.get("is_multihop")) else "single-hop"].append(
                    (bool(x["cb_correct"]), bool(x["ctx_correct"]))
                )
            for k, v in sorted(buckets.items()):
                by_reason[k] = _metrics_from_flags(v)
            for k, v in sorted(hop_buckets.items()):
                by_hop[k] = _metrics_from_flags(v)

            report["per_corpus"][corpus] = {"by_budget": per_b, "by_reason_type": by_reason, "by_hop": by_hop}

        (out_dir / "edgeqa_cross_eval_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    report = asyncio.run(_run())
    print(json.dumps(report.get("per_corpus") or {}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
