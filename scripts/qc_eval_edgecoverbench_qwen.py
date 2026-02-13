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
class ECBRow:
    idx: int
    corpus: str
    row_id: str
    canonical_id: str
    unit_id: str
    typ: str
    label: str
    question: str
    gold_answer: str
    evidence_span: str


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    return read_jsonl(path)


def _parse_pred_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"answer": "", "abstain": True, "confidence": 0.0, "_raw": ""}

    def _finish(obj: Dict[str, Any]) -> Dict[str, Any]:
        ans = str(obj.get("answer") or "")
        abst = bool(obj.get("abstain")) if "abstain" in obj else False
        conf = obj.get("confidence")
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.0
        if math.isnan(conf_f) or math.isinf(conf_f):
            conf_f = 0.0
        conf_f = min(1.0, max(0.0, conf_f))

        ans_norm = normalize_for_match(ans)
        if ans_norm in {"unknown", "unanswerable", "not enough info", "not_enough_info"}:
            abst = True
            ans = ""
        if abst and ans.strip():
            # enforce contract: abstain => empty answer
            ans = ""
        return {"answer": ans, "abstain": bool(abst), "confidence": float(conf_f), "_raw": raw}

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return _finish(obj)
    except Exception:
        pass

    # best-effort substring
    try:
        i = raw.find("{")
        j = raw.rfind("}")
        if i >= 0 and j > i:
            obj = json.loads(raw[i : j + 1])
            if isinstance(obj, dict):
                return _finish(obj)
    except Exception:
        pass

    # fallback: treat as plain answer
    ans = raw.splitlines()[0].strip()
    return _finish({"answer": ans, "abstain": False, "confidence": 0.5})


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--model", type=str, default="qwen-plus")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--thinking-budget", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only eval first N rows per corpus (smoke test).")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--completion-window", type=str, default="24h")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-chunk", type=int, default=5000, help="Split batch jobs into chunks of this size (0 = one giant batch).")
    ap.add_argument("--batch-parallel", type=int, default=4, help="How many batch chunks to run in parallel.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["run_name"] = str(args.run_name)

    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]
    out_dir = artifacts_root(cfg) / "runs" / str(args.run_name) / "evals" / str(args.model) / "edgecoverbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[ECBRow] = []
    by_corpus_order: Dict[str, List[int]] = defaultdict(list)

    for corpus in corpora:
        path = run_dir(cfg, corpus=corpus) / "edgecoverbench.jsonl"
        if not path.exists():
            raise SystemExit(f"missing edgecoverbench file: {path}")
        for i, r in enumerate(_iter_jsonl(path)):
            if args.limit and i >= int(args.limit):
                break
            idx = len(rows)
            row_id = str(r.get("id") or f"ecb:{corpus}:{i}")
            rows.append(
                ECBRow(
                    idx=idx,
                    corpus=corpus,
                    row_id=row_id,
                    canonical_id=str(r.get("canonical_id") or r.get("id") or ""),
                    unit_id=str(r.get("unit_id") or ""),
                    typ=str(r.get("type") or "unknown"),
                    label=str(r.get("label") or "unknown"),
                    question=str(r.get("question") or ""),
                    gold_answer=str(r.get("answer") or ""),
                    evidence_span=str(r.get("evidence_span") or r.get("evidence") or ""),
                )
            )
            by_corpus_order[corpus].append(idx)

    pred_requests: Dict[str, Dict[str, Any]] = {}

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
        pred_requests[custom_id] = {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}

    system = (
        "Answer the question using ONLY the evidence span.\n"
        "If the evidence does not fully support a single short answer, you MUST abstain.\n"
        "Return JSON only: {\"answer\": string, \"abstain\": true/false, \"confidence\": number between 0 and 1}.\n"
        "If abstain is true, answer MUST be an empty string.\n"
        "No extra keys. No explanation."
    )

    for row in rows:
        cid = f"{row.corpus}_{row.idx}"
        _add_req(
            cid,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {row.question}\n\nEvidence span:\n{row.evidence_span}"},
            ],
        )

    async def _run() -> Dict[str, Any]:
        client = QwenBatchClient()
        try:
            pred_out, pred_meta = await _run_batch_with_retries(
                client,
                requests=pred_requests,
                out_dir=out_dir / "preds",
                job_name="preds",
                retries=int(args.retries),
                chunk_size=int(args.batch_chunk),
                batch_parallel=int(args.batch_parallel),
                completion_window=str(args.completion_window),
                poll_s=float(args.poll_s),
            )

            # Parse predictions.
            preds: Dict[str, Dict[str, Any]] = {}
            usage_rows: List[Dict[str, int]] = []
            for row in rows:
                cid = f"{row.corpus}_{row.idx}"
                body = (pred_out.get(cid) or {}).get("body") or {}
                txt = _body_to_text(body)
                usage_rows.append(_body_usage(body))
                preds[cid] = _parse_pred_json(txt)

            # Build equivalence requests for ANSWERABLE where not abstained.
            eq_requests: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                if row.label != "ANSWERABLE":
                    continue
                cid = f"{row.corpus}_{row.idx}"
                pred = preds[cid]
                if bool(pred.get("abstain")):
                    continue
                ans = str(pred.get("answer") or "")
                if normalize_for_match(ans) == normalize_for_match(row.gold_answer) and ans.strip() and row.gold_answer.strip():
                    continue
                eq_id = f"{cid}_eq"
                eq_body: Dict[str, Any] = {
                    "model": str(args.model),
                    "messages": [
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": equiv_prompt(ans, row.gold_answer)},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 48,
                }
                if args.enable_thinking:
                    eq_body["enable_thinking"] = True
                if args.thinking_budget is not None:
                    eq_body["thinking_budget"] = int(args.thinking_budget)
                eq_requests[eq_id] = {"custom_id": eq_id, "method": "POST", "url": "/v1/chat/completions", "body": eq_body}

            eq_out, eq_meta = await _run_batch_with_retries(
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

        eq_usage_rows: List[Dict[str, int]] = []
        for eq_id in eq_requests.keys():
            body = (eq_out.get(eq_id) or {}).get("body") or {}
            eq_usage_rows.append(_body_usage(body))

        # Build correctness flags and per-item rows.
        per_item: List[Dict[str, Any]] = []
        correct: Dict[str, bool] = {}

        for row in rows:
            cid = f"{row.corpus}_{row.idx}"
            pred = preds[cid]
            ans = str(pred.get("answer") or "")
            abst = bool(pred.get("abstain"))
            conf = float(pred.get("confidence") or 0.0)

            is_ok: bool
            if row.label == "ANSWERABLE":
                if abst:
                    is_ok = False
                elif normalize_for_match(ans) == normalize_for_match(row.gold_answer) and ans.strip() and row.gold_answer.strip():
                    is_ok = True
                else:
                    eq_id = f"{cid}_eq"
                    eq_body = (eq_out.get(eq_id) or {}).get("body") or {}
                    eq_txt = _body_to_text(eq_body)
                    v = _parse_equiv_json(eq_txt)
                    is_ok = bool(v) if v is not None else False
            else:
                # For near-miss/unanswerable, abstaining is correct.
                is_ok = abst or (not ans.strip())

            correct[cid] = bool(is_ok)
            per_item.append(
                {
                    "corpus": row.corpus,
                    "idx": row.idx,
                    "id": row.row_id,
                    "canonical_id": row.canonical_id,
                    "unit_id": row.unit_id,
                    "type": row.typ,
                    "label": row.label,
                    "gold": row.gold_answer,
                    "pred": ans,
                    "abstain": abst,
                    "confidence": conf,
                    "correct": bool(is_ok),
                }
            )

        (out_dir / "edgecoverbench_eval_rows.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in per_item) + "\n", encoding="utf-8"
        )

        # Aggregate metrics.
        type_counts = Counter(r.typ for r in rows)
        label_counts = Counter(r.label for r in rows)

        def _acc(filter_fn) -> Dict[str, Any]:
            cids = [f"{r.corpus}_{r.idx}" for r in rows if filter_fn(r)]
            n = len(cids)
            if n == 0:
                return {"n": 0, "acc": 0.0}
            ok = sum(1 for cid in cids if correct.get(cid, False))
            return {"n": n, "acc": ok / n}

        # Paraphrase consistency: compare to canonical prediction.
        canon_pred: Dict[str, Dict[str, Any]] = {}
        canon_correct: Dict[str, bool] = {}
        for r in rows:
            if r.typ == "canonical":
                cid = f"{r.corpus}_{r.idx}"
                canon_pred[r.row_id] = preds[cid]
                canon_correct[r.row_id] = correct.get(cid, False)

        para_cons_total = 0
        para_cons_ok = 0
        para_both_correct = 0
        for r in rows:
            if r.typ != "paraphrase":
                continue
            para_cons_total += 1
            cid = f"{r.corpus}_{r.idx}"
            p = preds[cid]
            c = canon_pred.get(r.canonical_id)
            if not c:
                continue
            if bool(p.get("abstain")) or bool(c.get("abstain")):
                continue
            if normalize_for_match(str(p.get("answer") or "")) == normalize_for_match(str(c.get("answer") or "")):
                para_cons_ok += 1
            if correct.get(cid, False) and canon_correct.get(r.canonical_id, False):
                para_both_correct += 1

        # Risk-coverage curve (threshold over confidence, abstain excluded by definition).
        curve = []
        thresholds = [round(x * 0.05, 2) for x in range(0, 21)]
        for t in thresholds:
            included = [
                r
                for r in per_item
                if (not bool(r.get("abstain"))) and float(r.get("confidence") or 0.0) >= float(t)
            ]
            cov = len(included) / len(per_item) if per_item else 0.0
            if not included:
                curve.append({"t": t, "coverage": cov, "risk": 0.0, "n": 0})
                continue
            ok = sum(1 for r in included if bool(r.get("correct")))
            acc = ok / len(included)
            curve.append({"t": t, "coverage": cov, "risk": 1.0 - acc, "n": len(included)})

        near_miss_n = sum(1 for r in rows if r.typ == "near_miss")
        near_miss_fp = 0
        for r in rows:
            if r.typ != "near_miss":
                continue
            cid = f"{r.corpus}_{r.idx}"
            p = preds.get(cid) or {}
            if (not bool(p.get("abstain"))) and str(p.get("answer") or "").strip():
                near_miss_fp += 1

        report: Dict[str, Any] = {
            "run_name": str(args.run_name),
            "model": str(args.model),
            "enable_thinking": bool(args.enable_thinking),
            "thinking_budget": args.thinking_budget,
            "rows": len(rows),
            "type_counts": dict(type_counts),
            "label_counts": dict(label_counts),
            "usage_pred_total": _sum_usage(usage_rows),
            "usage_equiv_total": _sum_usage(eq_usage_rows),
            "usage_total": _sum_usage(list(usage_rows) + list(eq_usage_rows)),
            "batch_meta": {"preds": pred_meta, "equiv": eq_meta},
            "metrics": {
                "overall": _acc(lambda r: True),
                "canonical": _acc(lambda r: r.typ == "canonical"),
                "paraphrase": _acc(lambda r: r.typ == "paraphrase"),
                "near_miss": _acc(lambda r: r.typ == "near_miss"),
                "near_miss_false_positive_rate": {
                    "n": int(near_miss_n),
                    "fp": int(near_miss_fp),
                    "rate": (near_miss_fp / near_miss_n) if near_miss_n else 0.0,
                },
                "paraphrase_consistency_exact": {
                    "n": int(para_cons_total),
                    "consistent": int(para_cons_ok),
                    "rate": (para_cons_ok / para_cons_total) if para_cons_total else 0.0,
                },
                "paraphrase_both_correct": {
                    "n": int(para_cons_total),
                    "both_correct": int(para_both_correct),
                    "rate": (para_both_correct / para_cons_total) if para_cons_total else 0.0,
                },
            },
            "risk_coverage": curve,
        }

        (out_dir / "edgecoverbench_eval_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return report

    report = asyncio.run(_run())
    print(json.dumps(report.get("metrics") or {}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
