from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def recall_at_k(results: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int) -> float:
    total = 0
    acc = 0.0
    for qid, rels in qrels.items():
        if not rels:
            continue
        total += 1
        got = set((results.get(qid) or [])[:k]) & rels
        acc += len(got) / len(rels)
    return acc / total if total else 0.0


def mrr_at_k(results: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int) -> float:
    total = 0
    acc = 0.0
    for qid, rels in qrels.items():
        if not rels:
            continue
        total += 1
        rr = 0.0
        for rank, doc_id in enumerate((results.get(qid) or [])[:k], start=1):
            if doc_id in rels:
                rr = 1.0 / rank
                break
        acc += rr
    return acc / total if total else 0.0


def ndcg_at_k(results: Dict[str, List[str]], qrels: Dict[str, Set[str]], k: int) -> float:
    def dcg(rels_ranked: Sequence[int]) -> float:
        s = 0.0
        for i, rel in enumerate(rels_ranked, start=1):
            if rel:
                s += 1.0 / math.log2(i + 1)
        return s

    total = 0
    acc = 0.0
    for qid, rels in qrels.items():
        if not rels:
            continue
        total += 1
        ranked = (results.get(qid) or [])[:k]
        gains = [1 if d in rels else 0 for d in ranked]
        ideal = [1] * min(k, len(rels))
        denom = dcg(ideal)
        acc += (dcg(gains) / denom) if denom else 0.0
    return acc / total if total else 0.0


def load_qrels_tsv(path: str) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.lower().startswith("query-id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            qid, doc_id = parts[0], parts[1]
            qrels.setdefault(qid, set()).add(doc_id)
    return qrels

