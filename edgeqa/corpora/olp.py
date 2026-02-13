from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from edgeqa.hash_utils import sha256_text
from edgeqa.text_utils import normalize_ws


_COMMENT_RE = re.compile(r"(?<!\\)%.*$")
_BEGIN_RE = re.compile(r"\\begin\{([a-zA-Z*]+)\}")
_END_RE = re.compile(r"\\end\{([a-zA-Z*]+)\}")

_STRIP_CMD_SIMPLE = [
    r"\\label\{[^}]*\}",
    r"\\ref\{[^}]*\}",
    r"\\cite\{[^}]*\}",
    r"\\citep\{[^}]*\}",
    r"\\citet\{[^}]*\}",
    r"\\footnote\{[^}]*\}",
]
_STRIP_CMD_SIMPLE_RE = re.compile("|".join(_STRIP_CMD_SIMPLE))

# Commands where we want to keep the argument text.
_KEEP_ARG_CMDS = [
    "textbf",
    "textit",
    "emph",
    "underline",
    "texttt",
    "textrm",
]
_KEEP_ARG_RE = re.compile(r"\\(" + "|".join(_KEEP_ARG_CMDS) + r")\{([^}]*)\}")


def _strip_comments(text: str) -> str:
    out_lines: List[str] = []
    for line in text.splitlines():
        out_lines.append(_COMMENT_RE.sub("", line))
    return "\n".join(out_lines)


def latex_to_text(text: str) -> str:
    t = _strip_comments(text)
    # Drop some non-content commands.
    t = _STRIP_CMD_SIMPLE_RE.sub(" ", t)
    # Keep content of styling commands.
    while True:
        m = _KEEP_ARG_RE.search(t)
        if not m:
            break
        t = t[: m.start()] + m.group(2) + t[m.end() :]
    # Remove remaining control sequences (\foo or \foo*).
    t = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", t)
    # Remove braces and math delimiters.
    t = t.replace("{", " ").replace("}", " ")
    t = t.replace("$", " ")
    # Normalize whitespace.
    t = normalize_ws(t)
    return t


def _split_paragraphs(text: str) -> List[str]:
    # We assume text already has comments removed; keep blank lines as paragraph boundaries.
    paras: List[str] = []
    buf: List[str] = []
    for raw in text.splitlines():
        line = (raw or "").strip()
        if not line:
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        paras.append(" ".join(buf).strip())
    return [p for p in (normalize_ws(p) for p in paras) if p]


@dataclass(frozen=True)
class OLPUnit:
    unit_id: str
    doc_id: str
    unit_type: str
    text: str


@dataclass(frozen=True)
class OLPPassage:
    passage_id: str
    doc_id: str
    section: str
    text: str
    unit_id: Optional[str]


def iter_tex_files(repo_dir: str | Path) -> Iterator[Path]:
    root = Path(repo_dir)
    for p in root.rglob("*.tex"):
        # Skip obvious build dirs.
        if any(part in ("build", "out", ".git") for part in p.parts):
            continue
        yield p


def parse_tex_file(path: Path, *, repo_root: Path) -> Tuple[List[OLPPassage], List[OLPUnit]]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _strip_comments(raw)

    doc_id = str(path.relative_to(repo_root)).replace("\\", "/")
    section = doc_id

    env_counters: Dict[str, int] = {}
    env_stack: List[Tuple[str, str]] = []  # (env_type, unit_id)

    passages: List[OLPPassage] = []
    unit_text: Dict[str, str] = {}
    unit_type: Dict[str, str] = {}

    cur_lines: List[str] = []
    cur_unit_id: Optional[str] = None

    def flush_block(lines: List[str], unit_id: Optional[str]) -> None:
        if not lines:
            return
        text = "\n".join(lines).strip()
        if not text:
            return
        cleaned = latex_to_text(text)
        if not cleaned:
            return
        if unit_id:
            prev = unit_text.get(unit_id, "")
            unit_text[unit_id] = (prev + "\n" + cleaned).strip() if prev else cleaned
        for i, para in enumerate(_split_paragraphs(text)):
            para_text = latex_to_text(para)
            if not para_text:
                continue
            pid = sha256_text(f"{doc_id}::{unit_id or 'none'}::{i}::{para_text}")[:16]
            passages.append(
                OLPPassage(
                    passage_id=f"olp:{pid}",
                    doc_id=doc_id,
                    section=section,
                    text=para_text,
                    unit_id=unit_id,
                )
            )

    for line in raw.splitlines():
        begin = _BEGIN_RE.search(line)
        end = _END_RE.search(line)

        if begin:
            # flush previous block outside env boundary
            flush_block(cur_lines, cur_unit_id)
            cur_lines = []
            env_type = begin.group(1).strip()
            counter = env_counters.get(env_type, 0) + 1
            env_counters[env_type] = counter
            uid = f"olp_unit:{doc_id}:{env_type}:{counter}"
            unit_type[uid] = env_type
            env_stack.append((env_type, uid))
            cur_unit_id = uid
            continue

        if end:
            flush_block(cur_lines, cur_unit_id)
            cur_lines = []
            env_type = end.group(1).strip()
            # finalize unit text for the env we just closed (best-effort)
            if env_stack and env_stack[-1][0] == env_type:
                env_stack.pop()
            cur_unit_id = env_stack[-1][1] if env_stack else None
            continue

        cur_lines.append(line)

    flush_block(cur_lines, cur_unit_id)
    units: List[OLPUnit] = []
    for uid, utype in unit_type.items():
        units.append(
            OLPUnit(
                unit_id=uid,
                doc_id=doc_id,
                unit_type=utype,
                text=unit_text.get(uid, ""),
            )
        )
    return passages, units


def parse_repo(repo_dir: str | Path) -> Tuple[List[OLPPassage], List[OLPUnit], str]:
    repo_root = Path(repo_dir)
    passages: List[OLPPassage] = []
    units: List[OLPUnit] = []
    for tex in iter_tex_files(repo_root):
        p, u = parse_tex_file(tex, repo_root=repo_root)
        passages.extend(p)
        units.extend(u)
    return passages, units, str(repo_root)
