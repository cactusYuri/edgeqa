from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _add_file(items: List[Tuple[Path, Path]], src: Path, dst_rel: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(str(src))
    items.append((src, dst_rel))


def _add_glob(items: List[Tuple[Path, Path]], src_glob: Iterable[Path], dst_dir_rel: Path) -> None:
    for p in sorted(src_glob):
        if p.is_file():
            items.append((p, dst_dir_rel / p.name))


def build_manifest_from_dir(*, out_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name in {"MANIFEST.json", "SHA256SUMS.txt"}:
            continue
        rel = p.relative_to(out_dir)
        files.append(
            {
                "path": str(rel).replace("\\", "/"),
                "bytes": int(p.stat().st_size),
                "sha256": _sha256(p),
            }
        )
    return {"meta": meta, "files": sorted(files, key=lambda x: x["path"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="qc_paper_single_full_20260210_233827")
    ap.add_argument("--corpora", type=str, default="olp,osp")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--include-pool", action="store_true", help="Also copy edgeqa_pool.jsonl (large).")
    ap.add_argument("--include-llm-calls", action="store_true", help="Also copy llm_calls_*.jsonl (very large).")
    args = ap.parse_args()

    run_name = str(args.run_name)
    corpora = [c.strip() for c in str(args.corpora).split(",") if c.strip()]

    default_out = Path("artifacts/release") / f"edgeqa_resource_release_{time.strftime('%Y%m%d')}"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: List[Tuple[Path, Path]] = []

    # ---- Human audit (filled + summary) ----
    _add_glob(pairs, Path("artifacts/human").glob("*.csv"), Path("human_audit"))
    _add_file(pairs, Path("artifacts/human/human_audit_summary.json"), Path("human_audit/human_audit_summary.json"))

    # ---- Reports (small, helpful for reviewers/users) ----
    run_dir = Path("artifacts/runs") / run_name
    for p in [
        run_dir / "paper_report.md",
        run_dir / "single_model_report.json",
        run_dir / "selection_ablations.json",
        run_dir / "retrieval_baselines_dense_hybrid.json",
    ]:
        if p.exists():
            _add_file(pairs, p, Path("reports") / p.name)

    # ---- Config snapshot ----
    cfg_path = Path("configs/qc_paper_single_full.yaml")
    if cfg_path.exists():
        _add_file(pairs, cfg_path, Path("configs") / cfg_path.name)

    # ---- Licenses / attribution helpers ----
    # Code license (paper claims Apache-2.0); include it in the packaged snapshot.
    code_license = Path("LICENSE")
    if code_license.exists():
        _add_file(pairs, code_license, Path("licenses") / "CODE_LICENSE")

    # Upstream corpus license texts (CC BY 4.0) live under raw/.
    for corpus in corpora:
        raw_dir = Path("artifacts/corpora") / corpus / "raw"
        if not raw_dir.exists():
            continue
        for lic in sorted(raw_dir.glob("LICENSE*")):
            if lic.is_file():
                _add_file(pairs, lic, Path("licenses") / corpus / lic.name)

    # ---- Corpora + datasets ----
    for corpus in corpora:
        # Base corpora artifacts (redistributable passages + units)
        corpus_dir = Path("artifacts/corpora") / corpus
        for p in ["meta.json", "passages.jsonl", "units.jsonl"]:
            _add_file(pairs, corpus_dir / p, Path("corpora") / corpus / p)

        # Released EdgeQA budgets (exclude pool / full logs by default)
        for p in sorted((run_dir / corpus).glob("edgeqa_N*.jsonl")):
            _add_file(pairs, p, Path("edgeqa") / corpus / p.name)
        if args.include_pool:
            pool = run_dir / corpus / "edgeqa_pool.jsonl"
            if pool.exists():
                _add_file(pairs, pool, Path("edgeqa") / corpus / pool.name)

        # EdgeCoverBench
        _add_file(pairs, run_dir / corpus / "edgecoverbench.jsonl", Path("edgecoverbench") / corpus / "edgecoverbench.jsonl")

        # Coverage + summaries (small)
        _add_glob(pairs, (run_dir / corpus).glob("coverage_N*.json"), Path("eval") / corpus)
        for p in ["edgeqa_summary.json", "edgecoverbench_summary.json"]:
            sp = run_dir / corpus / p
            if sp.exists():
                _add_file(pairs, sp, Path("eval") / corpus / p)

        # BEIR exports
        for beir_dir in sorted((run_dir / corpus).glob("beir_N*")):
            if not beir_dir.is_dir():
                continue
            for f in ["corpus.jsonl", "queries.jsonl", "qrels.tsv", "meta.json", "bm25_metrics.json"]:
                fp = beir_dir / f
                if fp.exists():
                    _add_file(
                        pairs,
                        fp,
                        Path("beir_exports") / corpus / beir_dir.name / f,
                    )

        # Optional: full call logs (very large; generally not required for the resource data)
        if args.include_llm_calls:
            for p in sorted((run_dir / corpus).glob("llm_calls_*.jsonl")):
                _add_file(pairs, p, Path("llm_calls") / corpus / p.name)

    # ---- Copy ----
    copied: List[Tuple[Path, Path]] = []
    for src, rel in pairs:
        dst = out_dir / rel
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            copied.append((src, rel))
            continue
        _copy(src, dst)
        copied.append((src, rel))

    # ---- README + attribution ----
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# EdgeQA resource release (local package)",
                "",
                f"- Source run: `{run_name}`",
                f"- Packaged at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
                "",
                "## Contents",
                "",
                "- `corpora/`: redistributable passages + structured units for OLP/OSP",
                "- `edgeqa/`: released EdgeQA exports at multiple budgets",
                "- `edgecoverbench/`: EdgeCoverBench JSONL per corpus",
                "- `beir_exports/`: BEIR-style retrieval collections per corpus and budget",
                "- `human_audit/`: filled C2 audit CSVs + summarized rates",
                "- `eval/`: small JSON summaries/coverage files",
                "- `reports/`: run-level reports used to generate paper tables/figures",
                "- `licenses/`: code + upstream corpus license texts",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "ATTRIBUTION.md").write_text(
        "\n".join(
            [
                "# Attribution and licenses",
                "",
                "This package redistributes evidence text derived from two CC BY 4.0 corpora:",
                "",
                "- Open Logic Project (OLP) — CC BY 4.0 (license text under `licenses/olp/`).",
                "- OpenStax University Physics (OSP) — CC BY 4.0 (license text under `licenses/osp/`).",
                "",
                "Please ensure the final public release includes any additional attribution statements required by the upstream projects.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # ---- Manifest + checksums ----
    meta: Dict[str, Any] = {
        "run_name": run_name,
        "corpora": corpora,
        "include_pool": bool(args.include_pool),
        "include_llm_calls": bool(args.include_llm_calls),
        "generated_at_unix": int(time.time()),
    }
    manifest = build_manifest_from_dir(out_dir=out_dir, meta=meta)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sums_lines = [f"{f['sha256']}  {f['path']}" for f in manifest["files"]]
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_dir}")
    print(f"- files: {len(manifest['files'])}")
    print(f"- manifest: {out_dir / 'MANIFEST.json'}")
    print(f"- sha256: {out_dir / 'SHA256SUMS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
