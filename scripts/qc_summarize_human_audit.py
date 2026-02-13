from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _as_bool(v: Any) -> Optional[bool]:
    s = _norm(v)
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    if s in {"", "na", "n/a", "none"}:
        return None
    if s in {"partial", "unclear"}:
        return None
    return None


def _rate_true(rows: Iterable[Dict[str, Any]], field: str) -> float:
    vals: List[Optional[bool]] = []
    for r in rows:
        vals.append(_as_bool(r.get(field)))
    xs = [v for v in vals if v is not None]
    if not xs:
        return float("nan")
    return sum(1 for v in xs if v) / len(xs)


def _count_non_na(rows: Iterable[Dict[str, Any]], field: str) -> int:
    n = 0
    for r in rows:
        v = _as_bool(r.get(field))
        if v is not None:
            n += 1
    return n


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)


def _fmt_rate(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{float(x):.2f}"


def _write_tex(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(input_dir: Path) -> Dict[str, Any]:
    edgeqa_files = {
        "osp": input_dir / "edgeqa_osp_filled.csv",
        "olp": input_dir / "edgeqa_olp_filled.csv",
    }
    ecb_files = {
        "osp": input_dir / "edgecoverbench_osp_filled.csv",
        "olp": input_dir / "edgecoverbench_olp_filled.csv",
    }

    for p in list(edgeqa_files.values()) + list(ecb_files.values()):
        if not p.exists():
            raise FileNotFoundError(str(p))

    out: Dict[str, Any] = {"generated_at": int(time.time()), "input_dir": str(input_dir), "edgeqa": {}, "edgecoverbench": {}}

    # ---- EdgeQA ----
    for corpus, path in edgeqa_files.items():
        rows = _read_csv(path)
        out["edgeqa"][corpus] = {
            "n": len(rows),
            "answer_correct_rate": _rate_true(rows, "fill_answer_correct"),
            "evidence_support_rate": _rate_true(rows, "fill_evidence_support"),
            "ambiguous_rate": _rate_true(rows, "fill_ambiguous"),
            "non_na": {
                "fill_answer_correct": _count_non_na(rows, "fill_answer_correct"),
                "fill_evidence_support": _count_non_na(rows, "fill_evidence_support"),
                "fill_ambiguous": _count_non_na(rows, "fill_ambiguous"),
            },
        }

    # ---- EdgeCoverBench ----
    for corpus, path in ecb_files.items():
        rows = _read_csv(path)
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_type[_norm(r.get("type"))].append(r)

        can = by_type.get("canonical") or []
        para = by_type.get("paraphrase") or []
        nm = by_type.get("near_miss") or []

        out["edgecoverbench"][corpus] = {
            "n": len(rows),
            "type_counts": {k: len(v) for k, v in sorted(by_type.items())},
            "label_valid_rate": _rate_true(rows, "fill_label_valid"),
            "canonical": {
                "n": len(can),
                "answer_correct_rate": _rate_true(can, "fill_answer_correct"),
                "evidence_support_rate": _rate_true(can, "fill_evidence_support"),
            },
            "paraphrase": {
                "n": len(para),
                "answer_correct_rate": _rate_true(para, "fill_answer_correct"),
                "evidence_support_rate": _rate_true(para, "fill_evidence_support"),
                "meaning_preserved_rate": _rate_true(para, "fill_paraphrase_preserves_meaning"),
            },
            "near_miss": {
                "n": len(nm),
                "label_valid_rate": _rate_true(nm, "fill_label_valid"),
            },
        }

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, default="artifacts/human")
    ap.add_argument("--out-json", type=str, default="artifacts/human/human_audit_summary.json")
    ap.add_argument("--out-edgeqa-tex", type=str, default="edgeqa_sigir/tables/tab_human_audit_edgeqa.tex")
    ap.add_argument("--out-ecb-tex", type=str, default="edgeqa_sigir/tables/tab_human_audit_edgecoverbench.tex")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    report = summarize(input_dir)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- EdgeQA LaTeX ----
    edgeqa = report["edgeqa"]
    rows_edgeqa = []
    for corpus in ["olp", "osp"]:
        r = edgeqa.get(corpus) or {}
        rows_edgeqa.append(
            " & ".join(
                [
                    corpus.upper(),
                    str(int(r.get("n") or 0)),
                    _fmt_rate(r.get("answer_correct_rate")),
                    _fmt_rate(r.get("evidence_support_rate")),
                    _fmt_rate(r.get("ambiguous_rate")),
                ]
            )
            + " \\\\"
        )
    edgeqa_tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "\\textbf{Corpus} & \\textbf{N} & Ans. correct & Evidence support & Ambiguous \\\\",
        "\\midrule",
        *rows_edgeqa,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Human audit results for EdgeQA (C2). Entries are fractions over a 100-item stratified sample per corpus.}",
        "\\label{tab:human_audit_edgeqa}",
        "\\end{table}",
        "",
    ]
    _write_tex(Path(args.out_edgeqa_tex), edgeqa_tex)

    # ---- EdgeCoverBench LaTeX ----
    ecb = report["edgecoverbench"]
    rows_ecb = []
    for corpus in ["olp", "osp"]:
        r = ecb.get(corpus) or {}
        tc = r.get("type_counts") or {}
        can_n = int((r.get("canonical") or {}).get("n") or tc.get("canonical") or 0)
        para_n = int((r.get("paraphrase") or {}).get("n") or tc.get("paraphrase") or 0)
        nm_n = int((r.get("near_miss") or {}).get("n") or tc.get("near_miss") or 0)
        rows_ecb.append(
            " & ".join(
                [
                    corpus.upper(),
                    f"{can_n}/{para_n}/{nm_n}",
                    _fmt_rate((r.get("canonical") or {}).get("answer_correct_rate")),
                    _fmt_rate((r.get("paraphrase") or {}).get("meaning_preserved_rate")),
                    _fmt_rate((r.get("near_miss") or {}).get("label_valid_rate")),
                ]
            )
            + " \\\\"
        )
    ecb_tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "\\textbf{Corpus} & \\textbf{N (can/para/nm)} & Can. Ans. & Para. meaning & Near-miss valid \\\\",
        "\\midrule",
        *rows_ecb,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Human audit results for EdgeCoverBench (C2). Can. Ans. is answer correctness on canonical items; Para. meaning is meaning preservation on paraphrase items; Near-miss valid is whether the near-miss label is valid (unanswerable/contradicted as intended).}",
        "\\label{tab:human_audit_ecb}",
        "\\end{table}",
        "",
    ]
    _write_tex(Path(args.out_ecb_tex), ecb_tex)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {args.out_edgeqa_tex}")
    print(f"Wrote: {args.out_ecb_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
