from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edgeqa.config import load_config
from edgeqa.jsonl import read_jsonl, write_jsonl
from edgeqa.llm.cache import DiskCache
from edgeqa.llm.client import LLMClient
from edgeqa.llm.json_parse import parse_json_loose
from edgeqa.logging_utils import get_logger, setup_logging
from edgeqa.pipeline.edgeqa import _entropy_norm
from edgeqa.pipeline.equiv import answer_equiv
from edgeqa.pipeline.paths import cache_root, run_dir
from edgeqa.pipeline.prompts import paraphrase_prompt


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _edge_score(*, cb_correct: bool, sample_entropy: float, paraphrase_agree: float, w1: float, w2: float, w3: float) -> float:
    return (w1 * (0.0 if cb_correct else 1.0)) + (w2 * float(sample_entropy)) + (w3 * (1.0 - float(paraphrase_agree)))


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


async def _backfill_one(
    *,
    row: Dict[str, Any],
    client: LLMClient,
    model_chat: str,
    m: int,
    k: int,
    sample_temp: float,
    sample_max_tokens: int,
    para_temp: float,
    para_max_tokens: int,
    cb_temp: float,
    cb_max_tokens: int,
    use_llm_equiv: bool,
    w1: float,
    w2: float,
    w3: float,
) -> Dict[str, Any]:
    q = str(row.get("question") or "").strip()
    a_ref = str(row.get("answer") or "").strip()
    if not q or not a_ref:
        return row

    scores = row.get("scores") or {}
    if not isinstance(scores, dict):
        scores = {}
        row["scores"] = scores

    cb_correct = bool(scores.get("closed_book_correct", False))

    sample_entropy = 0.0
    if m > 0:
        sample_answers: List[str] = []
        for i in range(m):
            r = await client.chat(
                model=model_chat,
                messages=[
                    {"role": "system", "content": "Answer the question briefly."},
                    {"role": "user", "content": q},
                ],
                temperature=float(sample_temp),
                max_tokens=int(sample_max_tokens),
                cache_namespace=f"sample_{i}",
            )
            sample_answers.append((r.get("response") or {}).get("content") or "")
        sample_entropy = float(_entropy_norm(sample_answers))

    paraphrase_agree = 1.0
    if k > 0:
        pr = await client.chat(
            model=model_chat,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": paraphrase_prompt(q, k=k)},
            ],
            temperature=float(para_temp),
            max_tokens=int(para_max_tokens),
            cache_namespace="paraphrase_gen",
        )
        raw = (pr.get("response") or {}).get("content") or ""

        paraphrases: List[str] = []
        try:
            parsed = parse_json_loose(raw)
            if isinstance(parsed, list):
                paraphrases = [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            paraphrases = []

        para_answers: List[str] = []
        for pi, pq in enumerate(paraphrases[:k]):
            rr = await client.chat(
                model=model_chat,
                messages=[
                    {"role": "system", "content": "Answer the question briefly."},
                    {"role": "user", "content": pq},
                ],
                temperature=float(cb_temp),
                max_tokens=int(cb_max_tokens),
                cache_namespace=f"paraphrase_answer_{pi}",
            )
            para_answers.append((rr.get("response") or {}).get("content") or "")

        if len(para_answers) >= 2:
            paraphrase_agree = (
                1.0
                if await answer_equiv(client, para_answers[0], para_answers[1], model=model_chat, use_llm=use_llm_equiv)
                else 0.0
            )

    scores["sample_entropy"] = float(sample_entropy)
    scores["paraphrase_agreement"] = float(paraphrase_agree)
    scores["edge_score"] = float(
        _edge_score(
            cb_correct=cb_correct,
            sample_entropy=sample_entropy,
            paraphrase_agree=paraphrase_agree,
            w1=w1,
            w2=w2,
            w3=w3,
        )
    )
    return row


async def backfill_corpus(
    *,
    cfg: Dict[str, Any],
    corpus: str,
    budgets: List[int],
    only_edge1: bool,
    max_items: Optional[int],
) -> Tuple[int, int]:
    log = get_logger("qc_backfill")
    rdir = run_dir(cfg, corpus)
    edgeqa_path = rdir / "edgeqa.jsonl"
    if not edgeqa_path.exists():
        log.warning("[%s] Missing %s; skip", corpus, edgeqa_path)
        return (0, 0)

    rows = list(read_jsonl(edgeqa_path))
    if not rows:
        log.warning("[%s] Empty %s; skip", corpus, edgeqa_path)
        return (0, 0)

    idxs: List[int] = []
    for i, r in enumerate(rows):
        scores = r.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        edge = _safe_float(scores.get("edge_score") or 0.0)
        if only_edge1 and not math.isfinite(edge):
            continue
        if only_edge1:
            if abs(edge - 1.0) < 1e-9:
                idxs.append(i)
        else:
            # Backfill anything with missing brittleness fields or suspicious default edge score.
            if ("sample_entropy" not in scores) or ("paraphrase_agreement" not in scores) or (abs(edge - 1.0) < 1e-9):
                idxs.append(i)

    if max_items is not None:
        idxs = idxs[: int(max_items)]

    if not idxs:
        log.info("[%s] No rows need backfill", corpus)
        return (len(rows), 0)

    llm_cfg = cfg.get("llm", {}) or {}
    dec = cfg.get("decoding", {}) or {}
    cb_cfg = dec.get("closed_book", {}) or {}
    samp_cfg = dec.get("sample", {}) or {}
    para_cfg = dec.get("paraphrase", {}) or {}

    model_chat = str(llm_cfg.get("model_chat", "deepseek-chat"))
    use_llm_equiv = bool(cfg.get("equivalence", {}).get("use_llm", True))

    m = int(samp_cfg.get("m", 4) or 0)
    k = int(para_cfg.get("k", 2) or 0)

    sample_temp = float(samp_cfg.get("temperature", 0.8) or 0.8)
    sample_max_tokens = int(samp_cfg.get("max_tokens", 64) or 64)
    para_temp = float(para_cfg.get("temperature", 0.8) or 0.8)
    para_max_tokens = int(para_cfg.get("max_tokens", 192) or 192)
    cb_temp = float(cb_cfg.get("temperature", 0.0) or 0.0)
    cb_max_tokens = int(cb_cfg.get("max_tokens", 64) or 64)

    weights = cfg.get("edgeqa", {}).get("weights", {}) or {}
    w1 = float(weights.get("w1", 1.0))
    w2 = float(weights.get("w2", 1.0))
    w3 = float(weights.get("w3", 1.0))

    cache = DiskCache(cache_root(cfg) / "llm")
    client = LLMClient(
        cache=cache,
        concurrency=int(llm_cfg.get("concurrency", 64) or 64),
        force_refresh=bool(llm_cfg.get("force_refresh", False)),
        call_log_path=rdir / "llm_calls_edgeqa.jsonl",
    )

    log.info(
        "[%s] Backfilling %d/%d rows (m=%d,k=%d, model=%s)",
        corpus,
        len(idxs),
        len(rows),
        m,
        k,
        model_chat,
    )

    async def _run_idx(i: int) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        try:
            updated = await _backfill_one(
                row=rows[i],
                client=client,
                model_chat=model_chat,
                m=m,
                k=k,
                sample_temp=sample_temp,
                sample_max_tokens=sample_max_tokens,
                para_temp=para_temp,
                para_max_tokens=para_max_tokens,
                cb_temp=cb_temp,
                cb_max_tokens=cb_max_tokens,
                use_llm_equiv=use_llm_equiv,
                w1=w1,
                w2=w2,
                w3=w3,
            )
            return (i, updated, None)
        except Exception as e:
            return (i, None, repr(e))

    tasks = [asyncio.create_task(_run_idx(i)) for i in idxs]
    updated = 0
    failed = 0
    try:
        for fut in asyncio.as_completed(tasks):
            i, row_u, err = await fut
            if row_u is not None:
                rows[i] = row_u
                updated += 1
            else:
                failed += 1
                log.warning("[%s] backfill failed idx=%d err=%s", corpus, i, err)
            if (updated + failed) % 50 == 0:
                log.info("[%s] progress %d/%d (ok=%d fail=%d)", corpus, updated + failed, len(idxs), updated, failed)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()

    # Write back (with backup)
    backup = edgeqa_path.with_suffix(f".before_brittleness_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    try:
        edgeqa_path.replace(backup)
    except Exception:
        # If rename fails, best-effort copy.
        try:
            backup.write_text(edgeqa_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        except Exception:
            pass

    write_jsonl(edgeqa_path, rows)

    # Refresh budget prefixes (no extra LLM calls).
    for b in budgets:
        out_b = rdir / f"edgeqa_N{b}.jsonl"
        written = _write_prefix_jsonl(edgeqa_path, out_b, int(b))
        log.info("[%s] wrote prefix N=%d written=%d", corpus, int(b), written)

    log.info("[%s] backfill done: updated=%d failed=%d backup=%s", corpus, updated, failed, backup)
    return (len(rows), updated)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--budgets", type=str, default="1000,2000,5000,10000")
    ap.add_argument("--llm-concurrency", type=int, default=None, help="Override cfg.llm.concurrency for backfill.")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument(
        "--only-edge1",
        action="store_true",
        help="Only backfill rows with scores.edge_score == 1.0 (default: backfill missing fields too).",
    )
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    setup_logging(args.log_level)
    log = get_logger("qc_backfill")

    cfg = load_config(args.config)
    cfg["run_name"] = str(args.run_name).strip()
    if args.llm_concurrency is not None:
        cfg.setdefault("llm", {})
        cfg["llm"]["concurrency"] = int(args.llm_concurrency)

    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]
    budgets = sorted({int(x.strip()) for x in str(args.budgets).split(",") if x.strip()})
    if not budgets:
        budgets = [1000]

    async def _run_all() -> None:
        for c in corpora:
            await backfill_corpus(
                cfg=cfg,
                corpus=c,
                budgets=budgets,
                only_edge1=bool(args.only_edge1),
                max_items=args.max_items,
            )

    asyncio.run(_run_all())
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
