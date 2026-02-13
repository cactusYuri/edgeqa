from __future__ import annotations

import os
from pathlib import Path
from typing import List


def _load_keys_from_file(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return []
    except Exception:
        return []

    keys: List[str] = []
    for raw in text.splitlines():
        line = (raw or "").strip()
        if not line or line.startswith("#"):
            continue
        keys.append(line)
    return list(dict.fromkeys(keys))  # stable dedup


_REPO_ROOT = Path(__file__).resolve().parents[1]

# 清程极智（OpenAI-compatible）key 池：默认读取仓库根目录的 1_qc200(1).txt
_QC_KEYS_FILE = os.getenv("EDGEQA_QC_KEYS_FILE") or os.getenv("QC_KEYS_FILE") or str(
    _REPO_ROOT / "1_qc200(1).txt"
)
open_ai_keys = _load_keys_from_file(Path(_QC_KEYS_FILE))

# DeepSeek 官方 key（可选；本项目默认走清程极智的 OpenAI-compatible 网关）
_DEEPSEEK_KEYS_FILE = (os.getenv("EDGEQA_DEEPSEEK_KEYS_FILE") or os.getenv("DEEPSEEK_KEYS_FILE") or "").strip()
_DEFAULT_DEEPSEEK_KEYS_PATH = _REPO_ROOT / "deepseekkey.txt"
if not _DEEPSEEK_KEYS_FILE and _DEFAULT_DEEPSEEK_KEYS_PATH.exists():
    _DEEPSEEK_KEYS_FILE = str(_DEFAULT_DEEPSEEK_KEYS_PATH)
deepseek_api_keys = _load_keys_from_file(Path(_DEEPSEEK_KEYS_FILE)) if _DEEPSEEK_KEYS_FILE else []

# Qwen (DashScope OpenAI-compatible) key（可选）
_QWEN_KEYS_FILE = (os.getenv("EDGEQA_QWEN_KEYS_FILE") or os.getenv("QWEN_KEYS_FILE") or "").strip()
_DEFAULT_QWEN_KEYS_PATH = _REPO_ROOT / "qwenkey.txt"
if not _QWEN_KEYS_FILE and _DEFAULT_QWEN_KEYS_PATH.exists():
    _QWEN_KEYS_FILE = str(_DEFAULT_QWEN_KEYS_PATH)
qwen_api_keys = _load_keys_from_file(Path(_QWEN_KEYS_FILE)) if _QWEN_KEYS_FILE else []

# 其它 provider（本项目不使用；保留为空以兼容导入）
gemini_keys: List[str] = []
vapi_api_keys: List[str] = []
openrouter_api_keys: List[str] = []
