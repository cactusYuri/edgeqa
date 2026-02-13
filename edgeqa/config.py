from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "run_name": "dev",
    "artifacts_dir": "artifacts",
    "cache_dir": "cache",
    "llm": {
        "concurrency": 16,
        "force_refresh": False,
        "model_chat": "deepseek-chat",
        "model_reasoner": "deepseek-reasoner",
    },
    "edgeqa": {
        "candidate_passages": 200,
        "qa_per_passage": 2,
        "multi_hop_rate": 0.25,
        "edge_tau": 0.60,
        "weights": {"w1": 1.0, "w2": 1.0, "w3": 1.0},
        "final_N": 200,
    },
    "decoding": {
        "gen": {"temperature": 0.7, "max_tokens": 256},
        "closed_book": {"temperature": 0.0, "max_tokens": 64},
        "sample": {"temperature": 0.8, "max_tokens": 64, "m": 4},
        "paraphrase": {"temperature": 0.8, "max_tokens": 192, "k": 2},
        "verify_fast": {"temperature": 0.0, "max_tokens": 64},
        "verify_strict": {"temperature": 0.0, "max_tokens": 128},
    },
    "filtering": {
        "max_question_evidence_jaccard": 0.65,
        "min_verifier_entailment": 0.8,
    },
    "equivalence": {
        "use_llm": True,
    },
    "selection": {
        "lambdas": {"doc": 1.0, "unit": 1.0, "reason": 0.5, "redundancy": 0.2},
        "unknownness_min_frac": 0.0,
    },
    "corpora": {
        "olp": {
            "source": {"type": "git", "url": "https://github.com/OpenLogicProject/OpenLogic.git", "revision": "master"},
        },
        "osp": {
            "source": {"type": "git", "url": "https://github.com/openstax/university-physics.git", "revision": "main"},
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg = _deep_merge(DEFAULT_CONFIG, raw or {})

    base_dir = p.parent.resolve()
    cfg["_config_path"] = str(p.resolve())
    cfg["_config_dir"] = str(base_dir)

    # Resolve top-level dirs relative to config file location.
    for key in ("artifacts_dir", "cache_dir"):
        val = cfg.get(key)
        if isinstance(val, str):
            cfg[key] = str((base_dir / val).resolve())
    return cfg
