from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edgeqa.config import load_config
from edgeqa.jsonl import read_jsonl
from edgeqa.pipeline.paths import artifacts_root, corpus_dir, run_dir


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _truncate(s: str, n: int = 4000) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[:n] + "…")


def _is_multihop(ex: Dict[str, Any]) -> bool:
    ev = ex.get("evidence") or []
    return isinstance(ev, list) and len(ev) >= 2


def _sample(items: List[Dict[str, Any]], *, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if n <= 0 or not items:
        return []
    items = list(items)
    rng.shuffle(items)
    return items[:n]


def _sample_stratified(
    items: List[Dict[str, Any]],
    *,
    n: int,
    key_fn: Callable[[Dict[str, Any]], str],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if n <= 0 or not items:
        return []
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        buckets[str(key_fn(it) or "unknown")].append(it)
    keys = sorted(buckets.keys())

    total = sum(len(v) for v in buckets.values())
    if total <= 0:
        return _sample(items, n=n, rng=rng)

    # Proportional quota with rounding, then fill remaining.
    alloc: Dict[str, int] = {}
    remaining = int(n)
    for k in keys:
        frac = len(buckets[k]) / total
        want = int(round(frac * n))
        want = max(0, min(want, len(buckets[k])))
        alloc[k] = want
        remaining -= want

    while remaining > 0:
        made = False
        for k in keys:
            if remaining <= 0:
                break
            if alloc[k] < len(buckets[k]):
                alloc[k] += 1
                remaining -= 1
                made = True
        if not made:
            break

    out: List[Dict[str, Any]] = []
    for k in keys:
        b = list(buckets[k])
        rng.shuffle(b)
        out.extend(b[: alloc[k]])

    rng.shuffle(out)
    return out[:n]


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/qc_paper_single_full.yaml")
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument("--corpora", type=str, default="osp,olp")
    ap.add_argument("--n-edgeqa", type=int, default=100)
    ap.add_argument("--n-ecb", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["run_name"] = str(args.run_name)

    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else artifacts_root(cfg) / "runs" / str(args.run_name) / "human_audit_C2"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# C2 Human Audit Template",
                "",
                f"Run: `{args.run_name}`",
                "",
                "This folder contains audit slices for manual inspection to mitigate LLM-as-a-judge bias.",
                "",
                "## What to fill",
                "",
                "### EdgeQA rows",
                "- `fill_answer_correct`: {yes,no,partial,unclear}",
                "- `fill_evidence_support`: {yes,no,partial,unclear}",
                "- `fill_ambiguous`: {no,yes}",
                "- `fill_notes`: free text",
                "",
                "### EdgeCoverBench rows",
                "- `fill_label_valid`: {yes,no,unclear}",
                "- `fill_answer_correct`: {yes,no,partial,unclear}",
                "- `fill_evidence_support`: {yes,no,partial,unclear}",
                "- `fill_paraphrase_preserves_meaning`: {yes,no,unclear} (only for type=paraphrase)",
                "- `fill_notes`: free text",
                "",
            ]
        ),
        encoding="utf-8",
    )

    meta: Dict[str, Any] = {"run_name": str(args.run_name), "generated_at": int(time.time()), "per_corpus": {}}

    for corpus in corpora:
        rdir = run_dir(cfg, corpus=corpus)
        cdir = corpus_dir(cfg, corpus=corpus)

        passages_path = cdir / "passages.jsonl"
        edgeqa_path = rdir / "edgeqa_N10000.jsonl"
        ecb_path = rdir / "edgecoverbench.jsonl"

        if not passages_path.exists():
            raise SystemExit(f"missing passages: {passages_path}")
        if not edgeqa_path.exists():
            raise SystemExit(f"missing EdgeQA: {edgeqa_path}")
        if not ecb_path.exists():
            raise SystemExit(f"missing EdgeCoverBench: {ecb_path}")

        passages = _read_jsonl(passages_path)
        pid_to_text = {
            str(p.get("passage_id") or ""): str(p.get("text") or "")
            for p in passages
            if str(p.get("passage_id") or "").strip()
        }

        # ---- EdgeQA audit ----
        edgeqa = _read_jsonl(edgeqa_path)
        edgeqa_s = _sample_stratified(
            edgeqa,
            n=int(args.n_edgeqa),
            key_fn=lambda x: "multi-hop" if _is_multihop(x) else "single-hop",
            rng=rng,
        )

        edgeqa_rows: List[Dict[str, Any]] = []
        for ex in edgeqa_s:
            ev = ex.get("evidence") or []
            ev_ids = [str(x) for x in ev] if isinstance(ev, list) else []
            ev_texts = [{"passage_id": pid, "text": _truncate(pid_to_text.get(pid, ""), 4000)} for pid in ev_ids]
            edgeqa_rows.append(
                {
                    "dataset": "edgeqa",
                    "corpus": corpus,
                    "id": str(ex.get("id") or ""),
                    "reason_type": str(ex.get("reason_type") or ""),
                    "is_multihop": bool(_is_multihop(ex)),
                    "question": str(ex.get("question") or ""),
                    "gold_answer": str(ex.get("answer") or ""),
                    "evidence_ids": ",".join(ev_ids),
                    "evidence_span": _truncate(str(ex.get("evidence_span") or ""), 8000),
                    "evidence_texts": ev_texts,
                    "fill_answer_correct": "",
                    "fill_evidence_support": "",
                    "fill_ambiguous": "",
                    "fill_notes": "",
                }
            )

        edgeqa_jsonl = out_dir / f"edgeqa_{corpus}.jsonl"
        edgeqa_csv = out_dir / f"edgeqa_{corpus}.csv"
        _write_jsonl(edgeqa_jsonl, edgeqa_rows)
        _write_csv(
            edgeqa_csv,
            edgeqa_rows,
            fieldnames=[
                "dataset",
                "corpus",
                "id",
                "reason_type",
                "is_multihop",
                "question",
                "gold_answer",
                "evidence_ids",
                "evidence_span",
                "fill_answer_correct",
                "fill_evidence_support",
                "fill_ambiguous",
                "fill_notes",
            ],
        )

        # ---- EdgeCoverBench audit ----
        ecb = _read_jsonl(ecb_path)
        canon_by_id: Dict[str, Dict[str, Any]] = {
            str(r.get("id") or ""): r for r in ecb if str(r.get("type") or "") == "canonical" and r.get("id")
        }

        canon = [r for r in ecb if str(r.get("type") or "") == "canonical"]
        para = [r for r in ecb if str(r.get("type") or "") == "paraphrase"]
        nm = [r for r in ecb if str(r.get("type") or "") == "near_miss"]

        n_total = int(args.n_ecb)
        n_nm = min(len(nm), max(0, int(round(0.2 * n_total))))
        n_each = max(0, (n_total - n_nm) // 2)
        n_canon = min(len(canon), n_each)
        n_para = min(len(para), max(0, n_total - n_nm - n_canon))

        picked: List[Dict[str, Any]] = []
        picked.extend(_sample(canon, n=n_canon, rng=rng))
        picked.extend(_sample(para, n=n_para, rng=rng))
        picked.extend(_sample(nm, n=n_nm, rng=rng))

        if len(picked) < n_total:
            rest = [r for r in ecb if r not in picked]
            picked.extend(_sample(rest, n=n_total - len(picked), rng=rng))

        ecb_rows: List[Dict[str, Any]] = []
        for r in picked[:n_total]:
            typ = str(r.get("type") or "")
            cid = str(r.get("canonical_id") or "")
            canon_row = canon_by_id.get(cid) if cid else None
            ecb_rows.append(
                {
                    "dataset": "edgecoverbench",
                    "corpus": corpus,
                    "id": str(r.get("id") or ""),
                    "type": typ,
                    "label": str(r.get("label") or ""),
                    "canonical_id": cid,
                    "canonical_question": str((canon_row or {}).get("question") or ""),
                    "canonical_answer": str((canon_row or {}).get("answer") or ""),
                    "question": str(r.get("question") or ""),
                    "gold_answer": str(r.get("answer") or ""),
                    "evidence": _truncate(str(r.get("evidence") or ""), 8000),
                    "evidence_span": _truncate(str(r.get("evidence_span") or ""), 8000),
                    "fill_label_valid": "",
                    "fill_answer_correct": "",
                    "fill_evidence_support": "",
                    "fill_paraphrase_preserves_meaning": "" if typ == "paraphrase" else "",
                    "fill_notes": "",
                }
            )

        ecb_jsonl = out_dir / f"edgecoverbench_{corpus}.jsonl"
        ecb_csv = out_dir / f"edgecoverbench_{corpus}.csv"
        _write_jsonl(ecb_jsonl, ecb_rows)
        _write_csv(
            ecb_csv,
            ecb_rows,
            fieldnames=[
                "dataset",
                "corpus",
                "id",
                "type",
                "label",
                "canonical_id",
                "canonical_question",
                "canonical_answer",
                "question",
                "gold_answer",
                "evidence",
                "evidence_span",
                "fill_label_valid",
                "fill_answer_correct",
                "fill_evidence_support",
                "fill_paraphrase_preserves_meaning",
                "fill_notes",
            ],
        )

        meta["per_corpus"][corpus] = {
            "edgeqa": {"n": len(edgeqa_rows), "jsonl": str(edgeqa_jsonl), "csv": str(edgeqa_csv)},
            "edgecoverbench": {"n": len(ecb_rows), "jsonl": str(ecb_jsonl), "csv": str(ecb_csv)},
        }

    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
