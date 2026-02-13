from __future__ import annotations

import asyncio
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from edgeqa.hash_utils import sha256_json
from edgeqa.jsonl import dump_json, read_jsonl, write_jsonl
from edgeqa.llm.client import LLMClient
from edgeqa.llm.cache import DiskCache
from edgeqa.llm.json_parse import parse_json_loose
from edgeqa.logging_utils import get_logger
from edgeqa.pipeline.equiv import answer_equiv
from edgeqa.pipeline.ingest import ingest_corpus
from edgeqa.pipeline.mine import mine_corpus_passages
from edgeqa.pipeline.paths import cache_root, corpus_dir, run_dir
from edgeqa.pipeline.prompts import paraphrase_prompt, qa_generation_multihop_prompt, qa_generation_prompt, verifier_prompt
from edgeqa.pipeline.select import greedy_select
from edgeqa.text_utils import jaccard, normalize_for_match


def _entropy_norm(labels: List[str]) -> float:
    if not labels:
        return 0.0
    counts: Dict[str, int] = {}
    for x in labels:
        k = normalize_for_match(x) or ""
        counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log(p + 1e-12)
    return float(h / math.log(total))


def _parse_verdict(text: str) -> str:
    t = normalize_for_match(text)
    if "entailed" in t:
        return "ENTAILED"
    if "contradicted" in t:
        return "CONTRADICTED"
    if "not_enough_info" in t or "not enough info" in t or "insufficient" in t:
        return "NOT_ENOUGH_INFO"
    # fallback: try first token
    return (text or "").strip().split()[0].upper() if (text or "").strip() else "UNKNOWN"


def _token_set(text: str) -> set[str]:
    return set((normalize_for_match(text) or "").split())


async def build_edgeqa(cfg: Dict[str, Any], corpus: str, *, limit_passages: Optional[int] = None) -> Path:
    log = get_logger("edgeqa.build")
    cdir = corpus_dir(cfg, corpus)
    passages_path = cdir / "passages.jsonl"
    if not passages_path.exists():
        ingest_corpus(cfg, corpus)
    mined_k = int(cfg.get("edgeqa", {}).get("candidate_passages", 200))
    mined_path = cdir / f"mined_passages_k{mined_k}.jsonl"
    if not mined_path.exists():
        mined_path = mine_corpus_passages(cfg, corpus)

    passage_rows = list(read_jsonl(passages_path))
    passage_text: Dict[str, str] = {p["passage_id"]: p.get("text", "") for p in passage_rows}
    passage_to_unit: Dict[str, Optional[str]] = {p["passage_id"]: p.get("unit_id") for p in passage_rows}
    passage_meta: Dict[str, Dict[str, Any]] = {
        p["passage_id"]: {"doc_id": p.get("doc_id"), "section": p.get("section"), "unit_id": p.get("unit_id")}
        for p in passage_rows
    }

    mined_all = list(read_jsonl(mined_path))
    mined_primary = mined_all
    if limit_passages is not None:
        mined_primary = mined_all[: int(limit_passages)]

    rdir = run_dir(cfg, corpus)
    rdir.mkdir(parents=True, exist_ok=True)
    pool_path = rdir / "edgeqa_pool.jsonl"
    done_path = rdir / "edgeqa_done_sources.txt"
    out_path = rdir / "edgeqa.jsonl"

    existing_pool: List[Dict[str, Any]] = []
    processed_sources: set[str] = set()
    pool_sources_loaded = 0
    done_sources_loaded = 0
    if done_path.exists():
        try:
            for raw in done_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = (raw or "").strip()
                if not line:
                    continue
                if line not in processed_sources:
                    processed_sources.add(line)
                    done_sources_loaded += 1
        except Exception as e:
            log.warning("Failed to load done sources: %s", repr(e))
    if pool_path.exists():
        for row in read_jsonl(pool_path):
            existing_pool.append(row)
            sk = row.get("source_key") or row.get("source_passage_id")
            if sk:
                sk_s = str(sk)
                if sk_s not in processed_sources:
                    processed_sources.add(sk_s)
                    pool_sources_loaded += 1
    if pool_path.exists() or done_path.exists():
        log.info(
            "Resuming: loaded %d pool items; skip_sources=%d (pool=%d, done_file=%d)",
            len(existing_pool),
            len(processed_sources),
            pool_sources_loaded,
            done_sources_loaded,
        )

    cache = DiskCache(cache_root(cfg) / "llm")
    client = LLMClient(
        cache=cache,
        concurrency=int(cfg.get("llm", {}).get("concurrency", 16)),
        force_refresh=bool(cfg.get("llm", {}).get("force_refresh", False)),
        call_log_path=rdir / "llm_calls_edgeqa.jsonl",
    )
    close_shared_session = bool(cfg.get("llm", {}).get("close_session", True))

    model_chat = str(cfg.get("llm", {}).get("model_chat", "deepseek-chat"))
    model_reasoner = str(cfg.get("llm", {}).get("model_reasoner", "deepseek-reasoner"))
    tau = float(cfg.get("edgeqa", {}).get("edge_tau", 0.6))
    multi_hop_rate = float(cfg.get("edgeqa", {}).get("multi_hop_rate", 0.0) or 0.0)
    enforce_multi_hop = bool(cfg.get("edgeqa", {}).get("enforce_multi_hop", False))
    qa_single = int(cfg.get("edgeqa", {}).get("qa_per_passage", 2) or 2)
    qa_mh = int(cfg.get("edgeqa", {}).get("multi_hop_questions", 1) or 1)
    weights = cfg.get("edgeqa", {}).get("weights", {}) or {}
    w1 = float(weights.get("w1", 1.0))
    w2 = float(weights.get("w2", 1.0))
    w3 = float(weights.get("w3", 1.0))

    dec = cfg.get("decoding", {}) or {}
    gen_cfg = dec.get("gen", {}) or {}
    cb_cfg = dec.get("closed_book", {}) or {}
    samp_cfg = dec.get("sample", {}) or {}
    para_cfg = dec.get("paraphrase", {}) or {}
    v_cfg = dec.get("verify_fast", {}) or {}
    v_strict_cfg = dec.get("verify_strict", {}) or {}

    use_llm_equiv = bool(cfg.get("equivalence", {}).get("use_llm", True))
    max_qe_jacc = float(cfg.get("filtering", {}).get("max_question_evidence_jaccard", 0.65))
    strict_verify = bool(cfg.get("filtering", {}).get("strict_verify", False))
    strict_verify_rate = float(cfg.get("filtering", {}).get("strict_verify_rate", 0.0))

    pool_new: List[Dict[str, Any]] = []
    pool_q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    done_q: asyncio.Queue[Optional[str]] = asyncio.Queue()
    processed_lock = asyncio.Lock()
    llm_calls = 0
    llm_failures = 0

    def _append_lines(path: Path, lines: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for ln in lines:
                f.write(str(ln).strip() + "\n")

    async def pool_writer() -> None:
        batch = int(cfg.get("edgeqa", {}).get("pool_write_batch", 32) or 32)
        batch = max(1, batch)
        buf: List[Dict[str, Any]] = []
        try:
            while True:
                item = await pool_q.get()
                try:
                    if item is None:
                        break
                    buf.append(item)
                    if len(buf) >= batch:
                        # Keep in-memory copy for selection even if disk write fails.
                        pool_new.extend(buf)
                        try:
                            await asyncio.to_thread(write_jsonl, pool_path, buf, append=True)
                        except Exception as e:
                            log.warning("Pool writer flush failed: %s", repr(e))
                        buf = []
                finally:
                    pool_q.task_done()
        finally:
            if buf:
                pool_new.extend(buf)
                try:
                    await asyncio.to_thread(write_jsonl, pool_path, buf, append=True)
                except Exception as e:
                    log.warning("Pool writer final flush failed: %s", repr(e))

    async def done_writer() -> None:
        batch = int(cfg.get("edgeqa", {}).get("done_write_batch", 256) or 256)
        batch = max(1, batch)
        buf: List[str] = []
        try:
            while True:
                item = await done_q.get()
                try:
                    if item is None:
                        break
                    s = str(item).strip()
                    if not s:
                        continue
                    buf.append(s)
                    if len(buf) >= batch:
                        try:
                            await asyncio.to_thread(_append_lines, done_path, buf)
                        except Exception as e:
                            log.warning("Done writer flush failed: %s", repr(e))
                        buf = []
                finally:
                    done_q.task_done()
        finally:
            if buf:
                try:
                    await asyncio.to_thread(_append_lines, done_path, buf)
                except Exception as e:
                    log.warning("Done writer final flush failed: %s", repr(e))

    async def mark_source_done(source_key: str) -> None:
        s = str(source_key or "").strip()
        if not s:
            return
        async with processed_lock:
            if s in processed_sources:
                return
            processed_sources.add(s)
        await done_q.put(s)

    async def safe_chat(*, step: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        nonlocal llm_calls, llm_failures
        llm_calls += 1
        try:
            return await client.chat(**kwargs)
        except Exception as e:
            llm_failures += 1
            log.warning("LLM call failed (%s): %s", step, repr(e))
            return None

    async def process_candidate(
        *,
        source_key: str,
        evidence_pids: List[str],
        evidence_text: str,
        evidence_span: str,
        evidence_spans: List[str],
        cand: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        q = str(cand.get("question") or "").strip()
        a_ref = str(cand.get("answer") or "").strip()
        if not q or not a_ref:
            return None

        is_multihop = len(evidence_pids) >= 2

        # Closed-book answer
        cb = await safe_chat(
            step="closed_book",
            model=model_chat,
            messages=[
                {"role": "system", "content": "Answer the question briefly."},
                {"role": "user", "content": q},
            ],
            temperature=float(cb_cfg.get("temperature", 0.0)),
            max_tokens=int(cb_cfg.get("max_tokens", 64)),
            cache_namespace="closed_book",
        )
        if cb is None:
            return None
        a_cb = (cb.get("response") or {}).get("content") or ""
        try:
            cb_correct = await answer_equiv(client, a_cb, a_ref, model=model_chat, use_llm=use_llm_equiv)
        except Exception:
            cb_correct = False

        sample_entropy = 0.0
        paraphrase_agree = 1.0
        paraphrases: List[str] = []

        if not cb_correct:
            m = int(samp_cfg.get("m", 4) or 0)
            if m > 0:
                sample_answers: List[str] = []
                for i in range(m):
                    r = await safe_chat(
                        step=f"sample_{i}",
                        model=model_chat,
                        messages=[
                            {"role": "system", "content": "Answer the question briefly."},
                            {"role": "user", "content": q},
                        ],
                        temperature=float(samp_cfg.get("temperature", 0.8)),
                        max_tokens=int(samp_cfg.get("max_tokens", 64)),
                        cache_namespace=f"sample_{i}",
                    )
                    if r is None:
                        continue
                    sample_answers.append((r.get("response") or {}).get("content") or "")
                sample_entropy = _entropy_norm(sample_answers)

            k = int(para_cfg.get("k", 2) or 0)
            if k > 0:
                pr = await safe_chat(
                    step="paraphrase_gen",
                    model=model_chat,
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": paraphrase_prompt(q, k=k)},
                    ],
                    temperature=float(para_cfg.get("temperature", 0.8)),
                    max_tokens=int(para_cfg.get("max_tokens", 192)),
                    cache_namespace="paraphrase_gen",
                )
                if pr is None:
                    return None
                raw = (pr.get("response") or {}).get("content") or ""
                try:
                    parsed = parse_json_loose(raw)
                    if isinstance(parsed, list):
                        paraphrases = [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    paraphrases = []

                para_answers: List[str] = []
                for pi, pq in enumerate(paraphrases[:k]):
                    rr = await safe_chat(
                        step=f"paraphrase_answer_{pi}",
                        model=model_chat,
                        messages=[
                            {"role": "system", "content": "Answer the question briefly."},
                            {"role": "user", "content": pq},
                        ],
                        temperature=float(cb_cfg.get("temperature", 0.0)),
                        max_tokens=int(cb_cfg.get("max_tokens", 64)),
                        cache_namespace=f"paraphrase_answer_{pi}",
                    )
                    if rr is None:
                        continue
                    para_answers.append((rr.get("response") or {}).get("content") or "")

                if len(para_answers) >= 2:
                    # Pairwise agreement for k=2 by default.
                    try:
                        paraphrase_agree = (
                            1.0
                            if await answer_equiv(
                                client,
                                para_answers[0],
                                para_answers[1],
                                model=model_chat,
                                use_llm=use_llm_equiv,
                            )
                            else 0.0
                        )
                    except Exception:
                        paraphrase_agree = 0.0

        edge_score = w1 * (0.0 if cb_correct else 1.0) + w2 * float(sample_entropy) + w3 * (1.0 - float(paraphrase_agree))
        if edge_score < tau:
            return None

        if not evidence_text:
            return None

        # Copy / triviality filter
        qe = jaccard(_token_set(q), _token_set(evidence_span or evidence_text))
        if qe > max_qe_jacc:
            return None

        # Evidence-conditioned answerability
        ctx = await safe_chat(
            step="ctx_answer",
            model=model_chat,
            messages=[
                {"role": "system", "content": "Answer using only the provided evidence. If not answerable, say NOT_ANSWERABLE."},
                {"role": "user", "content": f"Evidence:\n{evidence_span or evidence_text}\n\nQuestion:\n{q}"},
            ],
            temperature=0.0,
            max_tokens=int(cb_cfg.get("max_tokens", 64)),
            cache_namespace="ctx_answer",
        )
        if ctx is None:
            return None
        a_ctx = (ctx.get("response") or {}).get("content") or ""
        try:
            ctx_correct = await answer_equiv(client, a_ctx, a_ref, model=model_chat, use_llm=use_llm_equiv)
        except Exception:
            ctx_correct = False

        # Multi-hop enforcement (optional): require that neither single passage alone suffices.
        if enforce_multi_hop and is_multihop and ctx_correct:
            from edgeqa.pipeline.equiv import quick_equiv

            for i, pid in enumerate(evidence_pids[:2]):
                single_ev = passage_text.get(pid, "") or ""
                if not single_ev:
                    return None
                single = await safe_chat(
                    step=f"mh_single_ctx_{i}",
                    model=model_chat,
                    messages=[
                        {"role": "system", "content": "Answer using only the provided evidence. If not answerable, say NOT_ANSWERABLE."},
                        {"role": "user", "content": f"Evidence:\n{single_ev}\n\nQuestion:\n{q}"},
                    ],
                    temperature=0.0,
                    max_tokens=int(cb_cfg.get("max_tokens", 64)),
                    cache_namespace=f"mh_single_ctx_{i}",
                )
                if single is None:
                    return None
                a_single = (single.get("response") or {}).get("content") or ""
                if quick_equiv(a_single, a_ref):
                    return None

        # Verifier (fast)
        ver = await safe_chat(
            step="verify_fast",
            model=model_chat,
            messages=[
                {"role": "system", "content": "Return only the label."},
                {"role": "user", "content": verifier_prompt(q, a_ref, evidence_span or evidence_text)},
            ],
            temperature=float(v_cfg.get("temperature", 0.0)),
            max_tokens=int(v_cfg.get("max_tokens", 64)),
            cache_namespace="verify_fast",
        )
        if ver is None:
            return None
        verdict = _parse_verdict((ver.get("response") or {}).get("content") or "")

        unknown = (not cb_correct) and ctx_correct
        if verdict != "ENTAILED" or not ctx_correct:
            return None

        strict_verdict = None
        if strict_verify and (strict_verify_rate <= 0.0 or random.random() < strict_verify_rate):
            ver2 = await safe_chat(
                step="verify_strict",
                model=model_reasoner,
                messages=[
                    {"role": "system", "content": "Return only the label."},
                    {"role": "user", "content": verifier_prompt(q, a_ref, evidence_span or evidence_text)},
                ],
                temperature=float(v_strict_cfg.get("temperature", 0.0)),
                max_tokens=int(v_strict_cfg.get("max_tokens", 128)),
                cache_namespace="verify_strict",
            )
            if ver2 is None:
                return None
            strict_verdict = _parse_verdict((ver2.get("response") or {}).get("content") or "")
            if strict_verdict != "ENTAILED":
                return None

        rid = sha256_json({"corpus": corpus, "pids": evidence_pids, "q": q, "a": a_ref})[:16]
        meta0 = passage_meta.get(evidence_pids[0], {}) if evidence_pids else {}
        metas = [passage_meta.get(pid, {}) for pid in evidence_pids]

        reason_type = "multi-hop" if is_multihop else str(cand.get("reason_type") or "").strip().lower()

        return {
            "id": f"edgeqa:{corpus}:{rid}",
            "corpus": corpus,
            "question": q,
            "answer": a_ref,
            "evidence": list(evidence_pids),
            "evidence_span": str(evidence_span or ""),
            "evidence_spans": [str(s or "") for s in (evidence_spans or [])],
            "reason_type": reason_type,
            "source_key": str(source_key),
            "source_passage_id": evidence_pids[0] if evidence_pids else None,
            "source_passage_ids": list(evidence_pids),
            "source_doc_id": meta0.get("doc_id"),
            "source_doc_ids": [m.get("doc_id") for m in metas],
            "source_section": meta0.get("section"),
            "source_sections": [m.get("section") for m in metas],
            "source_unit_id": meta0.get("unit_id"),
            "source_unit_ids": [m.get("unit_id") for m in metas],
            "scores": {
                "closed_book_correct": bool(cb_correct),
                "sample_entropy": float(sample_entropy),
                "paraphrase_agreement": float(paraphrase_agree),
                "edge_score": float(edge_score),
                "ctx_correct": bool(ctx_correct),
                "unknown": bool(unknown),
                "verdict": verdict,
                "strict_verdict": strict_verdict,
            },
            "filters": {
                "question_evidence_jaccard": float(qe),
            },
            "ts": time.time(),
        }

    try:
        primary_ids: List[str] = []
        for mp in mined_primary:
            pid = mp.get("passage_id")
            if not pid:
                continue
            if not passage_text.get(pid, ""):
                continue
            primary_ids.append(pid)

        candidate_ids: List[str] = []
        for mp in mined_all:
            pid = mp.get("passage_id")
            if not pid:
                continue
            if not passage_text.get(pid, ""):
                continue
            candidate_ids.append(pid)

        workers = int(cfg.get("edgeqa", {}).get("passage_workers", 0) or 0)
        if workers <= 0:
            workers = min(8, max(1, len(primary_ids)))

        # Build per-source specs (single-hop or multi-hop evidence chains).
        unit_to_pids: Dict[str, List[str]] = {}
        doc_to_pids: Dict[str, List[str]] = {}
        for pid in candidate_ids:
            uid = passage_to_unit.get(pid)
            if uid:
                unit_to_pids.setdefault(str(uid), []).append(pid)
            meta = passage_meta.get(pid) or {}
            doc_id = meta.get("doc_id")
            if doc_id:
                doc_to_pids.setdefault(str(doc_id), []).append(pid)

        pid_tokens: Dict[str, set[str]] = {pid: _token_set(passage_text.get(pid, "")) for pid in candidate_ids}

        def pick_second(pid1: str, rng: random.Random) -> Optional[str]:
            uid = passage_to_unit.get(pid1)
            if uid:
                peers = [p for p in unit_to_pids.get(str(uid), []) if p != pid1]
                if peers:
                    t1 = pid_tokens.get(pid1) or set()
                    if t1:
                        scored: List[Tuple[float, str]] = []
                        for pid2 in peers:
                            t2 = pid_tokens.get(pid2) or set()
                            if not t2:
                                continue
                            inter = len(t1 & t2)
                            if inter <= 0:
                                continue
                            union = len(t1 | t2)
                            sim = (inter / union) if union else 0.0
                            if sim > 0.0:
                                scored.append((sim, pid2))
                        scored.sort(key=lambda x: x[0], reverse=True)
                        if scored:
                            top = [pid for _, pid in scored[: min(10, len(scored))]]
                            return rng.choice(top)
                    return rng.choice(peers)

            meta = passage_meta.get(pid1) or {}
            doc_id = meta.get("doc_id")
            if doc_id:
                peers = [p for p in doc_to_pids.get(str(doc_id), []) if p != pid1]
                if peers:
                    t1 = pid_tokens.get(pid1) or set()
                    if t1:
                        scored: List[Tuple[float, str]] = []
                        for pid2 in peers:
                            t2 = pid_tokens.get(pid2) or set()
                            if not t2:
                                continue
                            inter = len(t1 & t2)
                            if inter <= 0:
                                continue
                            union = len(t1 | t2)
                            sim = (inter / union) if union else 0.0
                            if sim > 0.0:
                                scored.append((sim, pid2))
                        scored.sort(key=lambda x: x[0], reverse=True)
                        if scored:
                            top = [pid for _, pid in scored[: min(10, len(scored))]]
                            return rng.choice(top)
                    return rng.choice(peers)

            t1 = pid_tokens.get(pid1) or set()
            scored: List[Tuple[float, str]] = []
            for pid2 in candidate_ids:
                if pid2 == pid1:
                    continue
                t2 = pid_tokens.get(pid2) or set()
                if not t1 or not t2:
                    continue
                inter = len(t1 & t2)
                if inter <= 0:
                    continue
                union = len(t1 | t2)
                sim = (inter / union) if union else 0.0
                if sim > 0.0:
                    scored.append((sim, pid2))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                top = [pid for _, pid in scored[: min(10, len(scored))]]
                return rng.choice(top)
            return None

        sources: List[Dict[str, Any]] = []
        run_salt = str(cfg.get("run_name") or "")
        for pid1 in primary_ids:
            seed = int(sha256_json({"corpus": corpus, "pid": pid1, "run": run_salt})[:16], 16)
            rng = random.Random(seed)

            # Always generate single-hop questions for pid1.
            source_key = pid1
            if source_key not in processed_sources:
                sources.append({"source_key": source_key, "evidence_pids": [pid1]})

            # Optionally add a multi-hop chain (0--1) anchored at pid1.
            if multi_hop_rate > 0.0 and rng.random() < multi_hop_rate:
                pid2 = pick_second(pid1, rng)
                if pid2 and pid2 != pid1:
                    mh_key = f"{pid1}|{pid2}"
                    if mh_key not in processed_sources:
                        sources.append({"source_key": mh_key, "evidence_pids": [pid1, pid2]})

        q: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        for spec in sources:
            q.put_nowait(spec)
        for _ in range(workers):
            q.put_nowait(None)

        async def worker(wid: int) -> None:
            while True:
                spec = await q.get()
                try:
                    if spec is None:
                        return

                    source_key = str(spec.get("source_key") or "")
                    evidence_pids = list(spec.get("evidence_pids") or [])
                    if not source_key or not evidence_pids:
                        return

                    is_multihop = len(evidence_pids) >= 2
                    if not is_multihop:
                        passage = passage_text.get(evidence_pids[0], "")
                        prompt = qa_generation_prompt(
                            passage,
                            num_questions=qa_single,
                        )
                        cache_ns = "qa_gen"
                    else:
                        passage_a = passage_text.get(evidence_pids[0], "")
                        passage_b = passage_text.get(evidence_pids[1], "")
                        prompt = qa_generation_multihop_prompt(
                            passage_a,
                            passage_b,
                            num_questions=qa_mh,
                        )
                        cache_ns = "qa_gen_mh"

                    gen = await safe_chat(
                        step=cache_ns,
                        model=model_chat,
                        messages=[
                            {"role": "system", "content": "You are a careful dataset builder. Return JSON only."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=float(gen_cfg.get("temperature", 0.7)),
                        max_tokens=int(gen_cfg.get("max_tokens", 256)),
                        cache_namespace=cache_ns,
                    )
                    if gen is None:
                        await mark_source_done(source_key)
                        continue
                    raw = (gen.get("response") or {}).get("content") or ""
                    try:
                        parsed = parse_json_loose(raw)
                    except Exception:
                        await mark_source_done(source_key)
                        continue

                    if isinstance(parsed, dict):
                        if isinstance(parsed.get("questions"), list):
                            candidates = [x for x in parsed["questions"] if isinstance(x, dict)]
                        elif isinstance(parsed.get("items"), list):
                            candidates = [x for x in parsed["items"] if isinstance(x, dict)]
                        else:
                            candidates = [parsed]
                    elif isinstance(parsed, list):
                        candidates = [x for x in parsed if isinstance(x, dict)]
                    else:
                        candidates = []

                    max_cands = qa_mh if is_multihop else qa_single
                    if max_cands > 0 and len(candidates) > max_cands:
                        candidates = candidates[:max_cands]

                    evidence_text = "\n\n".join(passage_text.get(ep, "") for ep in evidence_pids if passage_text.get(ep, ""))

                    tasks = []
                    for c in candidates:
                        span_text = ""
                        spans: List[str] = []
                        if is_multihop:
                            span_a = str(c.get("evidence_span_a") or c.get("evidence_span_1") or "").strip()
                            span_b = str(c.get("evidence_span_b") or c.get("evidence_span_2") or "").strip()
                            spans = [span_a, span_b]
                            if span_a or span_b:
                                span_text = f"[A]\n{span_a}\n\n[B]\n{span_b}".strip()
                        else:
                            span_text = str(c.get("evidence_span") or "").strip()
                            spans = [span_text] if span_text else []
                        tasks.append(
                            process_candidate(
                                source_key=source_key,
                                evidence_pids=evidence_pids,
                                evidence_text=evidence_text,
                                evidence_span=span_text,
                                evidence_spans=spans,
                                cand=c,
                            )
                        )
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    accepted = 0
                    for r in results:
                        if isinstance(r, dict):
                            await pool_q.put(r)
                            accepted += 1

                    await mark_source_done(source_key)
                    log.info("Source done: %s accepted=%d", source_key, accepted)
                finally:
                    q.task_done()

        writer_task = asyncio.create_task(pool_writer())
        done_task = asyncio.create_task(done_writer())
        tasks = [asyncio.create_task(worker(i)) for i in range(workers)]
        try:
            await q.join()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            if not writer_task.done():
                await pool_q.put(None)
                await pool_q.join()
                await writer_task
            else:
                await asyncio.gather(writer_task, return_exceptions=True)

            if not done_task.done():
                await done_q.put(None)
                await done_q.join()
                await done_task
            else:
                await asyncio.gather(done_task, return_exceptions=True)
    finally:
        if close_shared_session:
            await client.close()

    pool_all = existing_pool + pool_new
    log.info("Pool size: %d (new %d)", len(pool_all), len(pool_new))

    N = int(cfg.get("edgeqa", {}).get("final_N", 200))
    lambdas = cfg.get("selection", {}).get("lambdas", {}) or {}
    rho = float(cfg.get("selection", {}).get("unknownness_min_frac", 0.0))
    selected = greedy_select(
        pool_all,
        passage_to_unit=passage_to_unit,
        N=N,
        lambdas=lambdas,
        unknownness_min_frac=rho,
    )
    write_jsonl(out_path, selected)

    summary = {
        "corpus": corpus,
        "run_name": cfg.get("run_name"),
        "mined_passages": len(mined_primary),
        "pool": len(pool_all),
        "selected": len(selected),
        "selected_unknown": sum(1 for x in selected if (x.get("scores", {}) or {}).get("unknown")),
        "llm_calls": llm_calls,
        "llm_failures": llm_failures,
    }
    dump_json(rdir / "edgeqa_summary.json", summary)
    log.info("Wrote %s (%d items)", out_path, len(selected))
    return out_path
