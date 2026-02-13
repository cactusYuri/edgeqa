from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from edgeqa.hash_utils import sha256_json
from edgeqa.jsonl import dump_json, read_jsonl, write_jsonl
from edgeqa.llm.cache import DiskCache
from edgeqa.llm.client import LLMClient
from edgeqa.llm.json_parse import parse_json_loose
from edgeqa.logging_utils import get_logger
from edgeqa.pipeline.equiv import answer_equiv
from edgeqa.pipeline.ingest import ingest_corpus
from edgeqa.pipeline.paths import cache_root, corpus_dir, run_dir
from edgeqa.pipeline.prompts import near_miss_prompt, paraphrase_prompt, qa_generation_prompt, verifier_prompt
from edgeqa.text_utils import normalize_for_match


def _parse_verdict(text: str) -> str:
    t = normalize_for_match(text)
    if "entailed" in t:
        return "ENTAILED"
    if "contradicted" in t:
        return "CONTRADICTED"
    if "not_enough_info" in t or "not enough info" in t or "insufficient" in t:
        return "NOT_ENOUGH_INFO"
    return (text or "").strip().split()[0].upper() if (text or "").strip() else "UNKNOWN"


def _append_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line))
            f.write("\n")


def _load_done_units(done_path: Path, out_path: Path) -> set[str]:
    done: set[str] = set()
    if done_path.exists():
        try:
            for raw in done_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                uid = (raw or "").strip()
                if uid:
                    done.add(uid)
        except Exception:
            pass
        return done

    # Fallback: infer from existing output (best-effort).
    if out_path.exists():
        try:
            for row in read_jsonl(out_path):
                uid = str(row.get("unit_id") or "").strip()
                if uid:
                    done.add(uid)
        except Exception:
            pass
        if done:
            try:
                done_path.write_text("\n".join(sorted(done)) + "\n", encoding="utf-8")
            except Exception:
                pass
    return done


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            n += 1
    return n


async def build_edgecoverbench(
    cfg: Dict[str, Any],
    corpus: str,
    *,
    limit_units: Optional[int] = None,
) -> Path:
    log = get_logger("edgeqa.edgecoverbench")
    cdir = corpus_dir(cfg, corpus)
    if not (cdir / "units.jsonl").exists():
        ingest_corpus(cfg, corpus)

    units = list(read_jsonl(cdir / "units.jsonl"))
    if limit_units is not None:
        units = units[: int(limit_units)]

    rdir = run_dir(cfg, corpus)
    rdir.mkdir(parents=True, exist_ok=True)
    out_path = rdir / "edgecoverbench.jsonl"
    done_path = rdir / "edgecoverbench_done_units.txt"
    done_units = _load_done_units(done_path, out_path)
    if done_units:
        log.info("Resuming EdgeCoverBench: %d units already processed", len(done_units))

    cache = DiskCache(cache_root(cfg) / "llm")
    client = LLMClient(
        cache=cache,
        concurrency=int(cfg.get("llm", {}).get("concurrency", 16)),
        force_refresh=bool(cfg.get("llm", {}).get("force_refresh", False)),
        call_log_path=rdir / "llm_calls_edgecoverbench.jsonl",
    )
    close_shared_session = bool(cfg.get("llm", {}).get("close_session", True))

    model_chat = str(cfg.get("llm", {}).get("model_chat", "deepseek-chat"))
    model_reasoner = str(cfg.get("llm", {}).get("model_reasoner", "deepseek-reasoner"))

    dec = cfg.get("decoding", {}) or {}
    gen_cfg = dec.get("gen", {}) or {}
    cb_cfg = dec.get("closed_book", {}) or {}
    para_cfg = dec.get("paraphrase", {}) or {}
    v_fast = dec.get("verify_fast", {}) or {}
    v_strict = dec.get("verify_strict", {}) or {}

    use_llm_equiv = bool(cfg.get("equivalence", {}).get("use_llm", True))
    strict_verify = bool(cfg.get("filtering", {}).get("strict_verify", False))
    strict_verify_rate = float(cfg.get("filtering", {}).get("strict_verify_rate", 0.0))

    write_lock = asyncio.Lock()
    llm_calls = 0
    llm_failures = 0

    async def safe_chat(*, step: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        nonlocal llm_calls, llm_failures
        llm_calls += 1
        try:
            return await client.chat(**kwargs)
        except Exception as e:
            llm_failures += 1
            log.warning("LLM call failed (%s): %s", step, repr(e))
            return None

    async def verify_supported(question: str, answer: str, evidence: str) -> str:
        v = await safe_chat(
            step="ecb_verify_fast",
            model=model_chat,
            messages=[
                {"role": "system", "content": "Return only the label."},
                {"role": "user", "content": verifier_prompt(question, answer, evidence)},
            ],
            temperature=float(v_fast.get("temperature", 0.0)),
            max_tokens=int(v_fast.get("max_tokens", 64)),
            cache_namespace="ecb_verify_fast",
        )
        if v is None:
            return "ERROR"
        verdict = _parse_verdict((v.get("response") or {}).get("content") or "")

        if strict_verify and (strict_verify_rate <= 0.0 or random.random() < strict_verify_rate):
            v2 = await safe_chat(
                step="ecb_verify_strict",
                model=model_reasoner,
                messages=[
                    {"role": "system", "content": "Return only the label."},
                    {"role": "user", "content": verifier_prompt(question, answer, evidence)},
                ],
                temperature=float(v_strict.get("temperature", 0.0)),
                max_tokens=int(v_strict.get("max_tokens", 128)),
                cache_namespace="ecb_verify_strict",
            )
            if v2 is None:
                return "ERROR"
            verdict2 = _parse_verdict((v2.get("response") or {}).get("content") or "")
            return verdict2

        return verdict

    try:
        unit_items = []
        for u in units:
            unit_id = str(u.get("unit_id") or "")
            evidence = str(u.get("text") or "").strip()
            if not unit_id or not evidence or len(evidence) < 40:
                continue
            if unit_id in done_units:
                continue
            unit_items.append(u)

        workers = int(cfg.get("edgecoverbench", {}).get("unit_workers", 0) or 0)
        if workers <= 0:
            workers = min(8, max(1, len(unit_items)))

        q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        for u in unit_items:
            q.put_nowait(u)
        for _ in range(workers):
            q.put_nowait(None)

        async def process_unit(u: Dict[str, Any]) -> List[Dict[str, Any]]:
            unit_id = str(u.get("unit_id") or "")
            evidence = str(u.get("text") or "").strip()
            out_rows: List[Dict[str, Any]] = []

            gen = await safe_chat(
                step="ecb_q_gen",
                model=model_chat,
                messages=[
                    {"role": "system", "content": "You are a careful benchmark builder. Return JSON only."},
                    {"role": "user", "content": qa_generation_prompt(evidence, num_questions=1)},
                ],
                temperature=float(gen_cfg.get("temperature", 0.7)),
                max_tokens=int(gen_cfg.get("max_tokens", 256)),
                cache_namespace="ecb_q_gen",
            )
            if gen is None:
                return out_rows
            raw = (gen.get("response") or {}).get("content") or ""
            try:
                parsed = parse_json_loose(raw)
            except Exception:
                return out_rows

            if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                candidates = [x for x in parsed["questions"] if isinstance(x, dict)]
            elif isinstance(parsed, dict):
                candidates = [parsed]
            elif isinstance(parsed, list):
                candidates = [x for x in parsed if isinstance(x, dict)]
            else:
                candidates = []

            for cand in candidates[:1]:
                qtext = str(cand.get("question") or "").strip()
                a_ref = str(cand.get("answer") or "").strip()
                ev_span = str(cand.get("evidence_span") or "").strip()
                if not qtext or not a_ref:
                    continue

                verdict = await verify_supported(qtext, a_ref, ev_span or evidence)
                if verdict != "ENTAILED":
                    continue

                cid = sha256_json({"unit_id": unit_id, "q": qtext, "a": a_ref})[:16]
                canonical_id = f"ecb:{corpus}:{cid}"
                out_rows.append(
                    {
                        "id": canonical_id,
                        "type": "canonical",
                        "corpus": corpus,
                        "unit_id": unit_id,
                        "doc_id": u.get("doc_id"),
                        "unit_type": u.get("unit_type"),
                        "question": qtext,
                        "answer": a_ref,
                        "evidence": evidence,
                        "evidence_span": ev_span,
                        "reason_type": str(cand.get("reason_type") or "").strip().lower(),
                        "label": "ANSWERABLE",
                        "scores": {"verdict": verdict},
                        "ts": time.time(),
                    }
                )

                # Paraphrases
                k = int(para_cfg.get("k", 2))
                pr = await safe_chat(
                    step="ecb_paraphrase_gen",
                    model=model_chat,
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": paraphrase_prompt(qtext, k=k)},
                    ],
                    temperature=float(para_cfg.get("temperature", 0.8)),
                    max_tokens=int(para_cfg.get("max_tokens", 192)),
                    cache_namespace="ecb_paraphrase_gen",
                )
                if pr is None:
                    return out_rows
                raw_p = (pr.get("response") or {}).get("content") or ""
                paraphrases: List[str] = []
                try:
                    parsed_p = parse_json_loose(raw_p)
                    if isinstance(parsed_p, list):
                        paraphrases = [str(x).strip() for x in parsed_p if str(x).strip()]
                except Exception:
                    paraphrases = []

                for pi, pq in enumerate(paraphrases[:k]):
                    pid = sha256_json({"canonical_id": canonical_id, "p": pq})[:16]
                    out_rows.append(
                        {
                            "id": f"{canonical_id}:p{pi}:{pid}",
                            "type": "paraphrase",
                            "corpus": corpus,
                            "unit_id": unit_id,
                            "canonical_id": canonical_id,
                            "question": pq,
                            "answer": a_ref,
                            "evidence": evidence,
                            "evidence_span": ev_span,
                            "label": "ANSWERABLE",
                            "ts": time.time(),
                        }
                    )

                # Near-miss (single)
                nm = await safe_chat(
                    step="ecb_near_miss_gen",
                    model=model_chat,
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": near_miss_prompt(qtext)},
                    ],
                    temperature=float(gen_cfg.get("temperature", 0.7)),
                    max_tokens=int(gen_cfg.get("max_tokens", 256)),
                    cache_namespace="ecb_near_miss_gen",
                )
                if nm is None:
                    return out_rows
                raw_nm = (nm.get("response") or {}).get("content") or ""
                try:
                    nm_obj = parse_json_loose(raw_nm)
                except Exception:
                    nm_obj = None

                if isinstance(nm_obj, dict):
                    mq = str(nm_obj.get("modified_question") or "").strip()
                    lbl = str(nm_obj.get("label") or "").strip().upper()
                    if mq and lbl in ("UNANSWERABLE", "CONTRADICTED"):
                        v = await verify_supported(mq, a_ref, ev_span or evidence)
                        if (lbl == "UNANSWERABLE" and v == "NOT_ENOUGH_INFO") or (lbl == "CONTRADICTED" and v == "CONTRADICTED"):
                            nid = sha256_json({"canonical_id": canonical_id, "mq": mq, "lbl": lbl})[:16]
                            out_rows.append(
                                {
                                    "id": f"{canonical_id}:nm:{nid}",
                                    "type": "near_miss",
                                    "corpus": corpus,
                                    "unit_id": unit_id,
                                    "canonical_id": canonical_id,
                                    "question": mq,
                                    "answer": "",
                                    "evidence": evidence,
                                    "evidence_span": ev_span,
                                    "label": lbl,
                                    "scores": {"verdict": v},
                                    "ts": time.time(),
                                }
                            )
            return out_rows

        async def worker(wid: int) -> None:
            while True:
                u = await q.get()
                try:
                    if u is None:
                        return
                    unit_id = str(u.get("unit_id") or "").strip()
                    out_rows = await process_unit(u)
                    if unit_id:
                        async with write_lock:
                            if out_rows:
                                await asyncio.to_thread(write_jsonl, out_path, out_rows, append=True)
                            await asyncio.to_thread(_append_lines, done_path, [unit_id])
                finally:
                    q.task_done()

        tasks = [asyncio.create_task(worker(i)) for i in range(workers)]
        await q.join()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        if close_shared_session:
            await client.close()

    rows_n = _count_jsonl_rows(out_path)
    dump_json(
        rdir / "edgecoverbench_summary.json",
        {
            "corpus": corpus,
            "rows": rows_n,
            "units_total": len(units),
            "units_planned": len(done_units) + len(unit_items),
            "llm_calls": llm_calls,
            "llm_failures": llm_failures,
        },
    )
    log.info("Wrote %s (%d rows)", out_path, rows_n)
    return out_path
