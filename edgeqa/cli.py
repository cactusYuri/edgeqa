from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, Dict

from edgeqa.config import load_config
from edgeqa.logging_utils import setup_logging
from edgeqa.pipeline.edgeqa import build_edgeqa
from edgeqa.pipeline.edgecoverbench import build_edgecoverbench
from edgeqa.pipeline.ingest import ingest_corpus
from edgeqa.pipeline.mine import mine_corpus_passages
from edgeqa.pipeline.paths import corpus_dir, run_dir
from edgeqa.eval.coverage import compute_edgeqa_coverage
from edgeqa.ir.beir import export_beir
from edgeqa.ir.bm25 import run_bm25
from edgeqa.jsonl import dump_json


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--log-level", type=str, default="INFO")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="edgeqa")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Download + parse corpus into passages/units.")
    _add_common(p_ingest)
    p_ingest.add_argument("--corpus", type=str, choices=["olp", "osp"], required=True)

    p_mine = sub.add_parser("mine", help="Mine knowledge-rich passages.")
    _add_common(p_mine)
    p_mine.add_argument("--corpus", type=str, choices=["olp", "osp"], required=True)

    p_build = sub.add_parser("build-edgeqa", help="Build EdgeQA dataset (end-to-end).")
    _add_common(p_build)
    p_build.add_argument("--corpus", type=str, choices=["olp", "osp"], required=True)
    p_build.add_argument("--limit-passages", type=int, default=None, help="For smoke tests: only process first K mined passages.")

    p_beir = sub.add_parser("export-beir", help="Export EdgeQA as BEIR-style retrieval collection.")
    _add_common(p_beir)
    p_beir.add_argument("--corpus", type=str, choices=["olp", "osp"], required=True)

    p_bm25 = sub.add_parser("bm25", help="Run BM25 baseline on exported BEIR collection.")
    _add_common(p_bm25)
    p_bm25.add_argument("--corpus", type=str, choices=["olp", "osp"], required=True)

    p_cov = sub.add_parser("coverage", help="Compute DocCov/UnitCov/ReasonCov for EdgeQA.")
    _add_common(p_cov)
    p_cov.add_argument("--corpus", type=str, choices=["olp", "osp"], required=True)

    p_ecb = sub.add_parser("build-edgecoverbench", help="Build EdgeCoverBench (unit-anchored stress test).")
    _add_common(p_ecb)
    p_ecb.add_argument("--corpus", type=str, choices=["olp", "osp"], required=True)
    p_ecb.add_argument("--limit-units", type=int, default=None, help="For smoke tests: only process first K units.")

    p_all = sub.add_parser("run-all", help="Run ingest→mine→EdgeQA→BEIR→BM25→coverage→EdgeCoverBench.")
    _add_common(p_all)
    p_all.add_argument("--corpora", type=str, default="all", choices=["all", "olp", "osp"])
    p_all.add_argument("--limit-passages", type=int, default=None)
    p_all.add_argument("--limit-units", type=int, default=None)

    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    cfg: Dict[str, Any] = load_config(args.config)

    if args.cmd == "ingest":
        ingest_corpus(cfg, args.corpus)
        return 0
    if args.cmd == "mine":
        mine_corpus_passages(cfg, args.corpus)
        return 0
    if args.cmd == "build-edgeqa":
        asyncio.run(build_edgeqa(cfg, args.corpus, limit_passages=args.limit_passages))
        return 0
    if args.cmd == "export-beir":
        cdir = corpus_dir(cfg, args.corpus)
        rdir = run_dir(cfg, args.corpus)
        beir_dir = rdir / "beir"
        meta = export_beir(passages_path=cdir / "passages.jsonl", edgeqa_path=rdir / "edgeqa.jsonl", out_dir=beir_dir)
        dump_json(beir_dir / "meta.json", meta)
        return 0
    if args.cmd == "bm25":
        rdir = run_dir(cfg, args.corpus)
        beir_dir = rdir / "beir"
        metrics = run_bm25(beir_dir=beir_dir)
        print(metrics)
        return 0
    if args.cmd == "coverage":
        cdir = corpus_dir(cfg, args.corpus)
        rdir = run_dir(cfg, args.corpus)
        metrics = compute_edgeqa_coverage(passages_path=cdir / "passages.jsonl", units_path=cdir / "units.jsonl", edgeqa_path=rdir / "edgeqa.jsonl")
        dump_json(rdir / "coverage.json", metrics)
        print(metrics)
        return 0
    if args.cmd == "build-edgecoverbench":
        asyncio.run(build_edgecoverbench(cfg, args.corpus, limit_units=args.limit_units))
        return 0
    if args.cmd == "run-all":
        corpora = ["olp", "osp"] if args.corpora == "all" else [args.corpora]

        async def _run() -> None:
            for corpus in corpora:
                ingest_corpus(cfg, corpus)
                mine_corpus_passages(cfg, corpus)
                await build_edgeqa(cfg, corpus, limit_passages=args.limit_passages)

                cdir = corpus_dir(cfg, corpus)
                rdir = run_dir(cfg, corpus)
                beir_dir = rdir / "beir"
                meta = export_beir(passages_path=cdir / "passages.jsonl", edgeqa_path=rdir / "edgeqa.jsonl", out_dir=beir_dir)
                dump_json(beir_dir / "meta.json", meta)

                run_bm25(beir_dir=beir_dir)

                cov = compute_edgeqa_coverage(passages_path=cdir / "passages.jsonl", units_path=cdir / "units.jsonl", edgeqa_path=rdir / "edgeqa.jsonl")
                dump_json(rdir / "coverage.json", cov)

                await build_edgecoverbench(cfg, corpus, limit_units=args.limit_units)

        asyncio.run(_run())
        return 0

    ap.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
