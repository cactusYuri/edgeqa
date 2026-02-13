from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from edgeqa.jsonl import read_jsonl, write_jsonl


def export_beir(
    *,
    passages_path: str | Path,
    edgeqa_path: str | Path,
    out_dir: str | Path,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    passages = list(read_jsonl(passages_path))
    examples = list(read_jsonl(edgeqa_path))

    corpus_rows: List[Dict[str, Any]] = []
    for p in passages:
        corpus_rows.append(
            {
                "_id": p["passage_id"],
                "title": str(p.get("section") or p.get("doc_id") or ""),
                "text": str(p.get("text") or ""),
            }
        )

    query_rows: List[Dict[str, Any]] = []
    qrels_lines: List[str] = ["query-id\tcorpus-id\tscore"]
    for ex in examples:
        qid = str(ex.get("id") or "")
        if not qid:
            continue
        query_rows.append({"_id": qid, "text": str(ex.get("question") or "")})
        for pid in ex.get("evidence", []) or []:
            qrels_lines.append(f"{qid}\t{pid}\t1")

    write_jsonl(out / "corpus.jsonl", corpus_rows)
    write_jsonl(out / "queries.jsonl", query_rows)
    (out / "qrels.tsv").write_text("\n".join(qrels_lines) + "\n", encoding="utf-8")

    return {"out_dir": str(out), "num_corpus": len(corpus_rows), "num_queries": len(query_rows)}

