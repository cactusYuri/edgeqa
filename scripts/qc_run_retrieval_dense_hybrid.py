from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edgeqa.ir.metrics import load_qrels_tsv, mrr_at_k, ndcg_at_k, recall_at_k
from edgeqa.jsonl import read_jsonl


_TOK_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOK_RE.findall((text or "").lower())


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    return read_jsonl(path)


def _load_beir(beir_dir: Path) -> Tuple[List[str], List[str], List[str], List[str]]:
    corpus_rows = list(_iter_jsonl(beir_dir / "corpus.jsonl"))
    query_rows = list(_iter_jsonl(beir_dir / "queries.jsonl"))
    doc_ids = [str(r.get("_id") or "") for r in corpus_rows]
    doc_texts = [str(r.get("text") or "") for r in corpus_rows]
    q_ids = [str(r.get("_id") or "") for r in query_rows]
    q_texts = [str(r.get("text") or "") for r in query_rows]
    return doc_ids, doc_texts, q_ids, q_texts


def _pick_device(name: str) -> torch.device:
    n = (name or "").strip().lower()
    if not n or n == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(n)


def _encode(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
    normalize: bool = True,
) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize),
        device=str(device),
    )
    if not isinstance(emb, np.ndarray):
        emb = np.asarray(emb, dtype=np.float32)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32, copy=False)
    return emb


def _dense_topk(
    q_emb: np.ndarray,
    d_emb: np.ndarray,
    *,
    topk: int,
    device: torch.device,
    query_chunk: int = 256,
) -> np.ndarray:
    """
    Returns indices into d_emb for each query (shape [Q, topk]).
    Assumes embeddings are already normalized if cosine is desired.
    """
    q = torch.from_numpy(q_emb).to(device)
    d = torch.from_numpy(d_emb).to(device)
    topk = int(topk)
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if topk > d.shape[0]:
        topk = int(d.shape[0])

    out = torch.empty((q.shape[0], topk), dtype=torch.int64, device="cpu")
    with torch.inference_mode():
        for i in range(0, q.shape[0], int(query_chunk)):
            qs = q[i : i + int(query_chunk)]
            scores = qs @ d.T
            idx = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True).indices.to("cpu")
            out[i : i + idx.shape[0], :] = idx
    return out.numpy()


def _bm25_topk(
    bm25: BM25Okapi,
    doc_ids: List[str],
    query_texts: List[str],
    *,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (idx, scores) arrays of shape [Q, topk].
    """
    topk = int(topk)
    n_docs = len(doc_ids)
    if topk <= 0:
        raise ValueError("topk must be > 0")
    if topk > n_docs:
        topk = n_docs

    idx_out = np.empty((len(query_texts), topk), dtype=np.int32)
    sc_out = np.empty((len(query_texts), topk), dtype=np.float32)

    for qi, q in enumerate(query_texts):
        scores = bm25.get_scores(_tok(q))
        if not isinstance(scores, np.ndarray):
            scores = np.asarray(scores, dtype=np.float32)
        if scores.dtype != np.float32:
            scores = scores.astype(np.float32, copy=False)
        # partial top-k
        part = np.argpartition(scores, -topk)[-topk:]
        part = part[np.argsort(scores[part])[::-1]]
        idx_out[qi, :] = part
        sc_out[qi, :] = scores[part]
    return idx_out, sc_out


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo = float(x.min())
    hi = float(x.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32, copy=False)


def _eval_metrics(
    *,
    results: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
) -> Dict[str, float]:
    return {
        "recall@5": float(recall_at_k(results, qrels, 5)),
        "recall@10": float(recall_at_k(results, qrels, 10)),
        "recall@20": float(recall_at_k(results, qrels, 20)),
        "ndcg@10": float(ndcg_at_k(results, qrels, 10)),
        "mrr@10": float(mrr_at_k(results, qrels, 10)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument("--artifacts-dir", type=str, default="artifacts")
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--budgets", type=str, default="1000,2000,5000,10000")
    ap.add_argument("--dense-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--query-chunk", type=int, default=256)
    ap.add_argument("--topk", type=int, default=20, help="How many docs to keep per query for metric computation.")
    ap.add_argument("--hybrid-bm25-topk", type=int, default=200, help="BM25 candidate pool size for hybrid.")
    ap.add_argument("--hybrid-alpha", type=float, default=0.5, help="Score = alpha*BM25 + (1-alpha)*dense (both min-max normalized per query).")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]
    budgets = sorted({int(x.strip()) for x in str(args.budgets).split(",") if x.strip()})
    if not budgets:
        budgets = [10000]
    max_budget = max(budgets)

    device = _pick_device(args.device)
    topk = int(args.topk)
    bm25_topk = int(args.hybrid_bm25_topk)
    alpha = float(args.hybrid_alpha)
    alpha = max(0.0, min(1.0, alpha))

    runs_root = Path(args.artifacts_dir) / "runs" / str(args.run_name)
    out_path = Path(args.out) if args.out else (runs_root / "retrieval_baselines_dense_hybrid.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = SentenceTransformer(str(args.dense_model), device=str(device))

    report: Dict[str, Any] = {
        "run_name": str(args.run_name),
        "generated_at": int(time.time()),
        "dense_model": str(args.dense_model),
        "device": str(device),
        "topk": topk,
        "hybrid": {"bm25_topk": bm25_topk, "alpha": alpha},
        "budgets": budgets,
        "max_budget": max_budget,
        "per_corpus": {},
    }

    for corpus in corpora:
        c_start = time.time()
        beir_max = runs_root / corpus / f"beir_N{max_budget}"
        if not beir_max.exists():
            raise SystemExit(f"missing BEIR dir: {beir_max}")

        doc_ids, doc_texts, q_ids_max, q_texts_max = _load_beir(beir_max)

        # Cache embeddings per corpus+model to avoid recomputation.
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(args.dense_model))[:120]
        cache_dir = runs_root / corpus / "retrieval" / safe_model
        cache_dir.mkdir(parents=True, exist_ok=True)
        doc_cache = cache_dir / "doc_emb.npz"
        q_cache = cache_dir / "query_emb_Nmax.npz"

        if doc_cache.exists():
            z = np.load(str(doc_cache))
            d_emb = z["emb"]
        else:
            d_emb = _encode(model, doc_texts, batch_size=int(args.batch_size), device=device, normalize=True)
            np.savez_compressed(str(doc_cache), emb=d_emb, doc_ids=np.asarray(doc_ids, dtype=object))

        if q_cache.exists():
            z = np.load(str(q_cache))
            q_emb = z["emb"]
        else:
            q_emb = _encode(model, q_texts_max, batch_size=int(args.batch_size), device=device, normalize=True)
            np.savez_compressed(str(q_cache), emb=q_emb, query_ids=np.asarray(q_ids_max, dtype=object))

        # Dense retrieval (top-k over full corpus).
        dense_idx = _dense_topk(
            q_emb,
            d_emb,
            topk=topk,
            device=device,
            query_chunk=int(args.query_chunk),
        )
        dense_results: Dict[str, List[str]] = {
            qid: [doc_ids[int(j)] for j in dense_idx[i].tolist()] for i, qid in enumerate(q_ids_max)
        }

        # Hybrid retrieval: BM25 candidates -> combine with dense.
        tokenized_docs = [_tok(t) for t in doc_texts]
        bm25 = BM25Okapi(tokenized_docs)
        bm_idx, bm_sc = _bm25_topk(bm25, doc_ids, q_texts_max, topk=bm25_topk)

        # Preload doc embeddings on CPU for gather.
        d_emb_cpu = d_emb  # np.float32
        hybrid_results: Dict[str, List[str]] = {}
        for i, qid in enumerate(q_ids_max):
            cand_idx = bm_idx[i]  # [K]
            bm_s = bm_sc[i].astype(np.float32, copy=False)
            dense_s = (d_emb_cpu[cand_idx] @ q_emb[i].astype(np.float32, copy=False)).astype(np.float32, copy=False)

            bm_n = _minmax(bm_s)
            dense_n = _minmax(dense_s)
            comb = (alpha * bm_n) + ((1.0 - alpha) * dense_n)
            order = np.argsort(comb)[::-1]
            top = cand_idx[order][:topk]
            hybrid_results[qid] = [doc_ids[int(j)] for j in top.tolist()]

        # Metrics per budget (by reading qrels/queries from each budget dir).
        per_b: Dict[str, Any] = {}
        for b in budgets:
            beir_b = runs_root / corpus / f"beir_N{b}"
            qrels = load_qrels_tsv(str(beir_b / "qrels.tsv"))
            q_rows = list(_iter_jsonl(beir_b / "queries.jsonl"))
            qids = [str(r.get("_id") or "") for r in q_rows]

            dense_sub = {qid: dense_results.get(qid, []) for qid in qids}
            hybrid_sub = {qid: hybrid_results.get(qid, []) for qid in qids}

            # Load existing BM25 metrics if present (computed elsewhere).
            bm25_metrics = None
            bm25_path = beir_b / "bm25_metrics.json"
            if bm25_path.exists():
                try:
                    bm25_metrics = json.loads(bm25_path.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    bm25_metrics = None

            per_b[str(b)] = {
                "bm25": bm25_metrics,
                "dense": _eval_metrics(results=dense_sub, qrels=qrels),
                "hybrid": _eval_metrics(results=hybrid_sub, qrels=qrels),
            }

        report["per_corpus"][corpus] = {
            "num_docs": len(doc_ids),
            "num_queries_max": len(q_ids_max),
            "embedding_cache_dir": str(cache_dir),
            "by_budget": per_b,
            "elapsed_s": time.time() - c_start,
        }

    report["elapsed_s"] = time.time() - t0
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

