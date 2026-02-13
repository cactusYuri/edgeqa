from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi

from edgeqa.ir.metrics import load_qrels_tsv, mrr_at_k, ndcg_at_k, recall_at_k
from edgeqa.jsonl import read_jsonl, dump_json


_TOK_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOK_RE.findall((text or "").lower())


def run_bm25(*, beir_dir: str | Path, k_values: List[int] | None = None) -> Dict[str, Any]:
    beir = Path(beir_dir)
    corpus = list(read_jsonl(beir / "corpus.jsonl"))
    queries = list(read_jsonl(beir / "queries.jsonl"))
    qrels = load_qrels_tsv(str(beir / "qrels.tsv"))

    doc_ids = [d["_id"] for d in corpus]
    tokenized = [_tok(d.get("text", "")) for d in corpus]
    bm25 = BM25Okapi(tokenized)

    results: Dict[str, List[str]] = {}
    for q in queries:
        qid = q["_id"]
        scores = bm25.get_scores(_tok(q.get("text", "")))
        # top-k by score
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results[qid] = [doc_ids[i] for i in ranked_idx]

    k_values = k_values or [5, 10, 20]
    metrics: Dict[str, Any] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(results, qrels, k)
    metrics["mrr@10"] = mrr_at_k(results, qrels, 10)
    metrics["ndcg@10"] = ndcg_at_k(results, qrels, 10)

    dump_json(beir / "bm25_metrics.json", metrics)
    return metrics

