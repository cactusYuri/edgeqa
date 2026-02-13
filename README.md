# EdgeQA (SIGIR Resources Track)

This repository contains the **code** and **paper sources** for the EdgeQA resource release.

- Dataset snapshot (HuggingFace Datasets): `https://huggingface.co/datasets/cactusYuri/edgeqa-resource-release`
- Paper sources: `edgeqa_sigir/`

## Repo layout

- `edgeqa/`: generation + evaluation pipeline
- `scripts/`: experiment runners and analysis utilities
- `LLMapi_service/`: OpenAI-compatible LLM clients + Qwen batch helper
- `configs/`: experiment configs
- `edgeqa_sigir/`: LaTeX sources (`main_resources_6p.tex` is the 6-page Resources-track version)

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce figures/tables (from an existing run directory)

If you have an experiment run under `artifacts/runs/<RUN_NAME>/` with `single_model_report.json`, you can generate the paper figures/tables via:

```bash
python scripts/make_sigir_figures.py --run-name <RUN_NAME>
```

## API keys

This repo does **not** ship any API keys.

`LLMapi_service/api_keys.py` loads keys from local files (or env var overrides):

- QC gateway: `1_qc200(1).txt` (or `EDGEQA_QC_KEYS_FILE`)
- DeepSeek official: `deepseekkey.txt` (or `EDGEQA_DEEPSEEK_KEYS_FILE`)
- Qwen: `qwenkey.txt` (or `EDGEQA_QWEN_KEYS_FILE`)

## License

- Code: see `LICENSE`
- Data: see the dataset card and `licenses/` in the HuggingFace dataset repo
