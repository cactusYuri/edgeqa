from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from edgeqa.corpora.download import ensure_git_repo
from edgeqa.corpora.olp import parse_repo as parse_olp
from edgeqa.corpora.osp import parse_repo as parse_osp
from edgeqa.hash_utils import sha256_text
from edgeqa.jsonl import dump_json, write_jsonl
from edgeqa.logging_utils import get_logger
from edgeqa.pipeline.paths import corpus_dir
from edgeqa.text_utils import normalize_ws


def ingest_corpus(cfg: Dict[str, Any], corpus: str) -> Dict[str, Any]:
    log = get_logger("edgeqa.ingest")
    cdir = corpus_dir(cfg, corpus)
    cdir.mkdir(parents=True, exist_ok=True)

    src = cfg.get("corpora", {}).get(corpus, {}).get("source", {})
    if src.get("type") != "git":
        raise ValueError(f"Unsupported corpus source type: {src.get('type')}")

    raw_dir = cdir / "raw"
    commit = ensure_git_repo(url=src["url"], dest_dir=raw_dir, revision=src.get("revision"))
    log.info("Corpus %s checked out at %s", corpus, commit)

    if corpus == "olp":
        passages, units, _ = parse_olp(raw_dir)
    elif corpus == "osp":
        passages, units, _ = parse_osp(raw_dir)
    else:
        raise ValueError(f"Unknown corpus: {corpus}")

    out_passages = cdir / "passages.jsonl"
    out_units = cdir / "units.jsonl"

    # Exact dedup by normalized text hash (keeps first occurrence).
    total_passages = len(passages)
    seen: set[str] = set()
    passage_rows = []
    for p in passages:
        text_hash = sha256_text(normalize_ws(p.text))
        if text_hash in seen:
            continue
        seen.add(text_hash)
        row = {**p.__dict__, "text_hash": text_hash}
        passage_rows.append(row)

    unit_rows = []
    for u in units:
        unit_rows.append({**u.__dict__, "text_hash": sha256_text(normalize_ws(u.text))})

    write_jsonl(out_passages, passage_rows)
    write_jsonl(out_units, unit_rows)

    meta: Dict[str, Any] = {
        "corpus": corpus,
        "commit": commit,
        "num_passages": total_passages,
        "num_units": len(units),
        "num_passages_deduped": len(passage_rows),
    }
    dump_json(cdir / "meta.json", meta)
    log.info("Wrote %s and %s", out_passages, out_units)
    return meta
