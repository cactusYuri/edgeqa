from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class EdgeQABrittleness:
    sample_entropy: List[float]
    paraphrase_agreement: List[float]
    multihop_mask: List[bool]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _read_json_maybe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fmt_k(n: int) -> str:
    if n >= 1000 and n % 1000 == 0:
        return f"{n//1000}k"
    return str(n)


def _load_brittleness(edgeqa_path: Path) -> EdgeQABrittleness:
    se: List[float] = []
    pa: List[float] = []
    mh: List[bool] = []

    for ex in _iter_jsonl(edgeqa_path):
        scores = ex.get("scores") or {}
        try:
            se.append(float(scores.get("sample_entropy")))
        except Exception:
            se.append(float("nan"))
        try:
            pa.append(float(scores.get("paraphrase_agreement")))
        except Exception:
            pa.append(float("nan"))

        ev = ex.get("evidence") or []
        mh.append(isinstance(ev, list) and len(ev) >= 2)

    return EdgeQABrittleness(sample_entropy=se, paraphrase_agreement=pa, multihop_mask=mh)


def _cdf(ax: plt.Axes, values: List[float], *, label: str, color: str) -> None:
    xs = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if xs.size == 0:
        return
    xs.sort()
    ys = np.arange(1, xs.size + 1, dtype=float) / xs.size
    ax.plot(xs, ys, label=label, color=color, linewidth=1.6)


def plot_coverage_curves(report: Dict[str, Any], out_dir: Path) -> None:
    per_corpus = report.get("per_corpus") or {}
    corpora = [c for c in ["olp", "osp"] if c in per_corpus] + [c for c in per_corpus.keys() if c not in {"olp", "osp"}]

    budgets: List[int] = []
    for corp in corpora:
        cov = (per_corpus[corp].get("coverage_by_budget") or {}).keys()
        budgets.extend([int(x) for x in cov if str(x).isdigit()])
    budgets = sorted({b for b in budgets if b > 0})
    if not budgets:
        return

    fig, axes = plt.subplots(1, len(corpora), figsize=(4.6 * len(corpora), 3.3), sharey=True)
    if len(corpora) == 1:
        axes = [axes]

    for ax, corp in zip(axes, corpora):
        cov_by_budget = per_corpus[corp].get("coverage_by_budget") or {}
        xs = [b for b in budgets if str(b) in cov_by_budget]
        doccov = [float(cov_by_budget[str(b)]["doccov"]) for b in xs]
        unitcov = [float(cov_by_budget[str(b)]["unitcov"]) for b in xs]
        ax.plot(xs, doccov, marker="o", linewidth=1.8, label="DocCov")
        ax.plot(xs, unitcov, marker="s", linewidth=1.8, label="UnitCov")
        ax.set_title(corp.upper())
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(xs, labels=[_fmt_k(b) for b in xs])
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel("Budget N")

    axes[0].set_ylabel("Coverage")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.tight_layout()

    out_pdf = out_dir / "coverage_curves.pdf"
    out_png = out_dir / "coverage_curves.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_brittleness(report: Dict[str, Any], out_dir: Path) -> None:
    per_corpus = report.get("per_corpus") or {}
    corpora = [c for c in ["olp", "osp"] if c in per_corpus] + [c for c in per_corpus.keys() if c not in {"olp", "osp"}]

    if not corpora:
        return

    fig, axes = plt.subplots(2, len(corpora), figsize=(4.6 * len(corpora), 5.6), sharey="row")
    if len(corpora) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, corp in enumerate(corpora):
        edgeqa_paths = per_corpus[corp].get("edgeqa_paths") or {}
        path_10k = edgeqa_paths.get("10000") or edgeqa_paths.get(10000) or edgeqa_paths.get("1000") or None
        if not path_10k:
            continue
        britt = _load_brittleness(Path(path_10k))

        mh_idx = [i for i, is_mh in enumerate(britt.multihop_mask) if is_mh]
        sh_idx = [i for i, is_mh in enumerate(britt.multihop_mask) if not is_mh]

        def _sel(xs: List[float], idx: List[int]) -> List[float]:
            return [xs[i] for i in idx]

        ax_se = axes[0, col]
        ax_pa = axes[1, col]
        _cdf(ax_se, _sel(britt.sample_entropy, sh_idx), label="single-hop", color="#1f77b4")
        _cdf(ax_se, _sel(britt.sample_entropy, mh_idx), label="multi-hop", color="#d62728")
        ax_se.set_title(corp.upper())
        ax_se.grid(True, linestyle=":", alpha=0.5)
        ax_se.set_xlabel("Sample entropy $H_s$")
        ax_se.set_ylabel("CDF")

        # Plot paraphrase *disagreement* for readability (higher = more brittle).
        pa_dis_sh = [1.0 - v for v in _sel(britt.paraphrase_agreement, sh_idx)]
        pa_dis_mh = [1.0 - v for v in _sel(britt.paraphrase_agreement, mh_idx)]
        _cdf(ax_pa, pa_dis_sh, label="single-hop", color="#1f77b4")
        _cdf(ax_pa, pa_dis_mh, label="multi-hop", color="#d62728")
        ax_pa.grid(True, linestyle=":", alpha=0.5)
        ax_pa.set_xlabel("Paraphrase disagreement $1-A_p$")
        ax_pa.set_ylabel("CDF")

    axes[0, -1].legend(loc="lower right", frameon=False)
    fig.tight_layout()

    out_pdf = out_dir / "brittleness_cdfs.pdf"
    out_png = out_dir / "brittleness_cdfs.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_token_breakdown(report: Dict[str, Any], out_dir: Path) -> None:
    u = report.get("usage_by_namespace") or {}
    if not isinstance(u, dict) or not u:
        return

    def _tok(ns: str) -> int:
        try:
            return int((u.get(ns) or {}).get("total_tokens") or 0)
        except Exception:
            return 0

    sampling = sum(_tok(f"sample_{i}") for i in range(4))
    paraphrase_answers = sum(_tok(f"paraphrase_answer_{i}") for i in range(2))
    edgecoverbench = sum(
        _tok(ns) for ns in ["ecb_q_gen", "ecb_paraphrase_gen", "ecb_near_miss_gen", "ecb_verify_fast"]
    )

    groups: List[Tuple[str, int]] = [
        ("equiv", _tok("equiv")),
        ("qa_gen", _tok("qa_gen")),
        ("qa_gen_mh", _tok("qa_gen_mh")),
        ("closed_book", _tok("closed_book")),
        ("ctx_answer", _tok("ctx_answer")),
        ("sampling (m=4)", sampling),
        ("paraphrase_gen", _tok("paraphrase_gen")),
        ("paraphrase_ans (k=2)", paraphrase_answers),
        ("verify_fast", _tok("verify_fast")),
        ("edgecoverbench", edgecoverbench),
    ]
    groups = [(k, v) for k, v in groups if v > 0]
    groups.sort(key=lambda kv: kv[1], reverse=True)

    labels = [k for k, _ in groups]
    vals = np.asarray([v for _, v in groups], dtype=float) / 1e6  # millions

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    y = np.arange(len(labels))
    ax.barh(y, vals, color="#4c78a8")
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Total tokens (M)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.set_title("Token breakdown (all call logs)")
    fig.tight_layout()

    out_pdf = out_dir / "token_breakdown.pdf"
    out_png = out_dir / "token_breakdown.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selection_baselines(selection_report: Dict[str, Any], out_dir: Path) -> None:
    per_corpus = selection_report.get("per_corpus") or {}
    corpora = [c for c in ["olp", "osp"] if c in per_corpus] + [
        c for c in per_corpus.keys() if c not in {"olp", "osp"}
    ]
    if not corpora:
        return

    budgets = selection_report.get("budgets") or []
    try:
        budgets = sorted({int(b) for b in budgets})
    except Exception:
        budgets = []
    if not budgets:
        # Fallback to the defaults used in the paper.
        budgets = [1000, 2000, 5000, 10000]

    baseline_order = ["main", "coverage_first", "unit_first", "long_tail", "paraphrase_only", "random"]
    baseline_label = {
        "main": "EdgeQA",
        "coverage_first": "Coverage-first",
        "unit_first": "Unit-first",
        "long_tail": "Long-tail",
        "paraphrase_only": "Paraphrase-only",
        "random": "Random",
    }
    baseline_style = {
        "main": {"color": "#000000", "marker": "o"},
        "coverage_first": {"color": "#1f77b4", "marker": "s"},
        "unit_first": {"color": "#2ca02c", "marker": "D"},
        "long_tail": {"color": "#ff7f0e", "marker": "^"},
        "paraphrase_only": {"color": "#9467bd", "marker": "v"},
        "random": {"color": "#7f7f7f", "marker": "x"},
    }

    fig, axes = plt.subplots(1, len(corpora), figsize=(4.6 * len(corpora), 3.3), sharey=True)
    if len(corpora) == 1:
        axes = [axes]

    for ax, corp in zip(axes, corpora):
        by_baseline = (per_corpus.get(corp) or {}).get("by_baseline") or {}
        for bl in baseline_order:
            b = by_baseline.get(bl)
            if not isinstance(b, dict):
                continue
            cov_by_budget = b.get("coverage_by_budget") or {}
            xs = [n for n in budgets if str(n) in cov_by_budget]
            if not xs:
                continue
            ys = [float((cov_by_budget[str(n)] or {}).get("doccov") or 0.0) for n in xs]
            style = baseline_style.get(bl) or {}
            ax.plot(
                xs,
                ys,
                linewidth=1.8,
                label=baseline_label.get(bl, bl),
                color=style.get("color"),
                marker=style.get("marker"),
            )
        ax.set_title(corp.upper())
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(budgets, labels=[_fmt_k(int(b)) for b in budgets])
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlabel("Budget N")

    axes[0].set_ylabel("DocCov")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.tight_layout()

    out_pdf = out_dir / "selection_baselines.pdf"
    out_png = out_dir / "selection_baselines.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _risk_coverage_curve(rows: List[Dict[str, Any]], *, thresholds: List[float]) -> List[Dict[str, Any]]:
    curve: List[Dict[str, Any]] = []
    for t in thresholds:
        included = [
            r
            for r in rows
            if (not bool(r.get("abstain"))) and float(r.get("confidence") or 0.0) >= float(t)
        ]
        cov = len(included) / len(rows) if rows else 0.0
        if not included:
            curve.append({"t": float(t), "coverage": float(cov), "risk": 0.0, "n": 0})
            continue
        ok = sum(1 for r in included if bool(r.get("correct")))
        acc = ok / len(included)
        curve.append({"t": float(t), "coverage": float(cov), "risk": float(1.0 - acc), "n": int(len(included))})
    return curve


def plot_edgecoverbench_risk_coverage_qwen_plus(run_dir: Path, out_dir: Path) -> None:
    rows_path = run_dir / "evals" / "qwen-plus" / "edgecoverbench" / "edgecoverbench_eval_rows.jsonl"
    if not rows_path.exists():
        return

    rows = list(_iter_jsonl(rows_path))
    if not rows:
        return

    thresholds = [round(x * 0.05, 2) for x in range(0, 21)]
    curves = {
        "ALL": _risk_coverage_curve(rows, thresholds=thresholds),
    }

    by_corpus: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        c = str(r.get("corpus") or "").strip().lower()
        if not c:
            continue
        by_corpus.setdefault(c, []).append(r)
    for c, sub in sorted(by_corpus.items()):
        curves[c.upper()] = _risk_coverage_curve(sub, thresholds=thresholds)

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    styles = {
        "ALL": {"color": "#000000", "marker": "o"},
        "OLP": {"color": "#1f77b4", "marker": "s"},
        "OSP": {"color": "#d62728", "marker": "D"},
    }
    for name in ["ALL", "OLP", "OSP"]:
        if name not in curves:
            continue
        xs = [float(p.get("coverage") or 0.0) for p in curves[name]]
        ys = [float(p.get("risk") or 0.0) for p in curves[name]]
        st = styles.get(name) or {}
        ax.plot(
            xs,
            ys,
            linewidth=1.8,
            label=name,
            color=st.get("color"),
            marker=st.get("marker"),
            markersize=4,
        )

    ax.set_xlabel("Coverage (non-abstained above threshold)")
    ax.set_ylabel("Risk (1 - accuracy)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    out_pdf = out_dir / "edgecoverbench_risk_coverage_qwen_plus.pdf"
    out_png = out_dir / "edgecoverbench_risk_coverage_qwen_plus.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _latex_escape(s: str) -> str:
    # Minimal escaping for our table values.
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )

def write_tables(
    report: Dict[str, Any],
    out_dir: Path,
    *,
    selection_report: Optional[Dict[str, Any]] = None,
    retrieval_report: Optional[Dict[str, Any]] = None,
) -> None:
    per_corpus = report.get("per_corpus") or {}
    corpora = [c for c in ["olp", "osp"] if c in per_corpus] + [c for c in per_corpus.keys() if c not in {"olp", "osp"}]
    budgets = ["1000", "2000", "5000", "10000"]

    # Coverage / composition.
    rows: List[str] = []
    for corp in corpora:
        cov_by_budget = per_corpus[corp].get("coverage_by_budget") or {}
        qual_by_budget = per_corpus[corp].get("quality_by_budget") or {}
        for b in budgets:
            if b not in cov_by_budget:
                continue
            cov = cov_by_budget[b]
            qual = qual_by_budget.get(b) or {}
            rows.append(
                " & ".join(
                    [
                        _latex_escape(corp.upper()),
                        _fmt_k(int(b)),
                        f"{float(cov['doccov']):.3f}",
                        f"{float(cov['unitcov']):.3f}",
                        f"{float(qual.get('multihop_frac') or 0.0):.3f}",
                    ]
                )
                + " \\\\"
            )

    cov_table = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "\\textbf{Corpus} & \\textbf{N} & \\textbf{DocCov} & \\textbf{UnitCov} & \\textbf{Multi-hop} \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Coverage and multi-hop composition for the released DeepSeek-V3.2 (chat) instantiations. DocCov/UnitCov are computed over corpus passages/units (Sec.~\\ref{sec:unitcov}).}",
            "\\label{tab:coverage_main}",
            "\\end{table}",
            "",
        ]
    )
    (out_dir / "tab_coverage_main.tex").write_text(cov_table, encoding="utf-8")

    # Retrieval baselines (prefer the explicit dense/hybrid report if available).
    retrieval_rows: List[str] = []
    dense_model = None

    if retrieval_report:
        dense_model = retrieval_report.get("dense_model")
        retr_per_corpus = retrieval_report.get("per_corpus") or {}
        method_order = [("bm25", "BM25"), ("dense", "Dense"), ("hybrid", "Hybrid")]
        for corp in corpora:
            c = retr_per_corpus.get(corp) or {}
            by_budget = c.get("by_budget") or {}
            for b in budgets:
                m_all = by_budget.get(str(b))
                if not isinstance(m_all, dict):
                    continue
                for mk, mlabel in method_order:
                    m = m_all.get(mk) or {}
                    retrieval_rows.append(
                        " & ".join(
                            [
                                _latex_escape(corp.upper()),
                                _fmt_k(int(b)),
                                _latex_escape(mlabel),
                                f"{float(m.get('recall@5') or 0.0):.3f}",
                                f"{float(m.get('recall@10') or 0.0):.3f}",
                                f"{float(m.get('recall@20') or 0.0):.3f}",
                                f"{float(m.get('ndcg@10') or 0.0):.3f}",
                            ]
                        )
                        + " \\\\"
                    )
    else:
        # Back-compat: BM25 from the single-model report.
        for corp in corpora:
            bm_by_budget = per_corpus[corp].get("bm25_by_budget") or {}
            for b in budgets:
                if b not in bm_by_budget:
                    continue
                m = bm_by_budget[b]
                retrieval_rows.append(
                    " & ".join(
                        [
                            _latex_escape(corp.upper()),
                            _fmt_k(int(b)),
                            "BM25",
                            f"{float(m.get('recall@5') or 0.0):.3f}",
                            f"{float(m.get('recall@10') or 0.0):.3f}",
                            f"{float(m.get('recall@20') or 0.0):.3f}",
                            f"{float(m.get('ndcg@10') or 0.0):.3f}",
                        ]
                    )
                    + " \\\\"
                )

    caption = "Retrieval results on the BEIR-style exports of EdgeQA (questions as queries; evidence passages as relevant)."
    if dense_model:
        caption = f"{caption} Dense uses \\texttt{{{_latex_escape(str(dense_model))}}}."

    retrieval_table = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{@{}lllrrrr@{}}",
            "\\toprule",
            "\\textbf{Corpus} & \\textbf{N} & \\textbf{Method} & R@5 & R@10 & R@20 & nDCG@10 \\\\",
            "\\midrule",
            *retrieval_rows,
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            "\\label{tab:bm25}",
            "\\end{table}",
            "",
        ]
    )
    (out_dir / "tab_bm25.tex").write_text(retrieval_table, encoding="utf-8")

    # Selection baselines (from existing pool; cost-free).
    if selection_report:
        sel_per_corpus = selection_report.get("per_corpus") or {}
        max_budget = str(int(selection_report.get("max_budget") or 10000))
        baseline_order = ["main", "coverage_first", "unit_first", "long_tail", "paraphrase_only", "random"]
        baseline_label = {
            "main": "EdgeQA",
            "coverage_first": "Coverage-first",
            "unit_first": "Unit-first",
            "long_tail": "Long-tail",
            "paraphrase_only": "Paraphrase-only",
            "random": "Random",
        }

        sel_rows: List[str] = []
        for corp in corpora:
            c = sel_per_corpus.get(corp) or {}
            by_baseline = c.get("by_baseline") or {}
            for bl in baseline_order:
                b = by_baseline.get(bl)
                if not isinstance(b, dict):
                    continue
                cov = (b.get("coverage_by_budget") or {}).get(max_budget) or {}
                sel_rows.append(
                    " & ".join(
                        [
                            _latex_escape(corp.upper()),
                            _latex_escape(baseline_label.get(bl, bl)),
                            f"{float(cov.get('doccov') or 0.0):.3f}",
                            f"{float(cov.get('unitcov') or 0.0):.3f}",
                        ]
                    )
                    + " \\\\"
                )

        sel_table = "\n".join(
            [
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                "\\begin{tabular}{@{}llrr@{}}",
                "\\toprule",
                "\\textbf{Corpus} & \\textbf{Baseline} & \\textbf{DocCov} & \\textbf{UnitCov} \\\\",
                "\\midrule",
                *sel_rows,
                "\\bottomrule",
                "\\end{tabular}",
                f"\\caption{{Selection baseline coverage at $N={_fmt_k(int(max_budget))}$ using the same candidate pool (no additional API calls).}}",
                "\\label{tab:selection_baselines}",
                "\\end{table}",
                "",
            ]
        )
        (out_dir / "tab_selection_baselines_10k.tex").write_text(sel_table, encoding="utf-8")

    # EdgeCoverBench composition.
    ecb_rows: List[str] = []
    for corp in corpora:
        ecbq = per_corpus[corp].get("edgecoverbench_quality") or {}
        if not ecbq:
            continue
        tc = ecbq.get("type_counts") or {}
        lc = ecbq.get("label_counts") or {}
        ecb_rows.append(
            " & ".join(
                [
                    _latex_escape(corp.upper()),
                    str(int(ecbq.get("rows") or 0)),
                    str(int(tc.get("canonical") or 0)),
                    str(int(tc.get("paraphrase") or 0)),
                    str(int(tc.get("near_miss") or 0)),
                    str(int(lc.get("UNANSWERABLE") or 0)),
                ]
            )
            + " \\\\"
        )

    ecb_table = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{@{}lrrrrr@{}}",
            "\\toprule",
            "\\textbf{Corpus} & \\textbf{Rows} & Canon. & Para. & Near-miss & Unans. \\\\",
            "\\midrule",
            *ecb_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{EdgeCoverBench composition for the released corpora. Near-miss items include both contradicted and unanswerable variants.}",
            "\\label{tab:ecb_comp}",
            "\\end{table}",
            "",
        ]
    )
    (out_dir / "tab_edgecoverbench_comp.tex").write_text(ecb_table, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True, help="Run directory name under artifacts/runs/.")
    ap.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts root (default: artifacts).")
    ap.add_argument("--out-figures", type=str, default="edgeqa_sigir/figures", help="Output directory for figures.")
    ap.add_argument("--out-tables", type=str, default="edgeqa_sigir/tables", help="Output directory for LaTeX tables.")
    args = ap.parse_args()

    run_dir = Path(args.artifacts_dir) / "runs" / args.run_name
    report_path = run_dir / "single_model_report.json"
    if not report_path.exists():
        raise SystemExit(f"missing report: {report_path}")

    report = _read_json(report_path)

    selection_report = _read_json_maybe(run_dir / "baselines" / "selection_from_pool.json")
    retrieval_report = _read_json_maybe(run_dir / "retrieval_baselines_dense_hybrid.json")

    fig_dir = Path(args.out_figures)
    tab_dir = Path(args.out_tables)
    _ensure_dir(fig_dir)
    _ensure_dir(tab_dir)

    plot_coverage_curves(report, fig_dir)
    plot_brittleness(report, fig_dir)
    plot_token_breakdown(report, fig_dir)
    plot_edgecoverbench_risk_coverage_qwen_plus(run_dir, fig_dir)
    if selection_report:
        plot_selection_baselines(selection_report, fig_dir)
    write_tables(report, tab_dir, selection_report=selection_report, retrieval_report=retrieval_report)

    print(f"Wrote figures to: {fig_dir}")
    print(f"Wrote tables to: {tab_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
