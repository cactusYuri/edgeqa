# Experiment checklist (paper ↔ artifacts)

This note helps track which artifacts from the main run are reflected in the LaTeX paper (`edgeqa_sigir/`) versus which remain as raw outputs only.

## Primary runs / sources

- Main single-model run: `artifacts/runs/qc_paper_single_full_20260210_233827/`
- Human audit (filled): `artifacts/human/`
- Human audit templates: `artifacts/runs/qc_paper_single_full_20260210_233827/human_audit_C2/`

## Reflected in the paper (main body)

- Coverage curves + summary: `edgeqa_sigir/figures/coverage_curves.pdf`, `edgeqa_sigir/tables/tab_coverage_main.tex`
- Cross-model evaluation (qwen-plus): `edgeqa_sigir/tables/tab_cross_eval_qwen_plus.tex`
- EdgeCoverBench composition + baseline: `edgeqa_sigir/tables/tab_edgecoverbench_comp.tex`, `edgeqa_sigir/tables/tab_edgecoverbench_qwen_plus.tex`
- Human audit (C2): `edgeqa_sigir/tables/tab_human_audit_edgeqa.tex`, `edgeqa_sigir/tables/tab_human_audit_edgecoverbench.tex`

## Reflected in the paper (appendix)

- Selection baselines: `edgeqa_sigir/tables/tab_selection_baselines_10k.tex`, `edgeqa_sigir/figures/selection_baselines.pdf`
- Cheap retargeting from pool: `edgeqa_sigir/tables/tab_retarget_qwen_pool.tex`
- Retrieval baselines (BM25/Dense/Hybrid): `edgeqa_sigir/tables/tab_bm25.tex`
- Brittleness + token accounting: `edgeqa_sigir/figures/brittleness_cdfs.pdf`, `edgeqa_sigir/figures/token_breakdown.pdf`
- EdgeCoverBench risk–coverage curve: `edgeqa_sigir/figures/edgecoverbench_risk_coverage_qwen_plus.pdf`

## Produced by the run but not explicitly summarized in the paper

- Selection ablations across objective weights: `artifacts/runs/qc_paper_single_full_20260210_233827/selection_ablations.json`
- Budgeting estimates for additional baselines: `artifacts/runs/qc_paper_single_full_20260210_233827/baselines/token_estimates.json`
- Raw per-item and cache logs (intentionally not embedded): `artifacts/runs/qc_paper_single_full_20260210_233827/**/llm_calls_*.jsonl`

