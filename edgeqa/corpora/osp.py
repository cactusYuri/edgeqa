from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from edgeqa.hash_utils import sha256_text
from edgeqa.text_utils import normalize_ws


@dataclass(frozen=True)
class OSPUnit:
    unit_id: str
    doc_id: str
    unit_type: str
    text: str


@dataclass(frozen=True)
class OSPPassage:
    passage_id: str
    doc_id: str
    section: str
    text: str
    unit_id: Optional[str]


_COL_NS = {
    "col": "http://cnx.rice.edu/collxml",
    "md": "http://cnx.rice.edu/mdml",
}

_CNX_NS = {
    "cnx": "http://cnx.rice.edu/cnxml",
    "m": "http://www.w3.org/1998/Math/MathML",
    "md": "http://cnx.rice.edu/mdml",
}


def _t(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _elem_text(el: ET.Element) -> str:
    return normalize_ws(" ".join((t or "").strip() for t in el.itertext()))


def _parse_collection_module_paths(collection_xml: Path) -> Tuple[str, Dict[str, List[str]]]:
    root = ET.fromstring(collection_xml.read_text(encoding="utf-8", errors="ignore"))

    # Collection title (Volume 1/2/3)
    vol_title = None
    md_title = root.find(".//md:title", _COL_NS)
    if md_title is not None and (md_title.text or "").strip():
        vol_title = (md_title.text or "").strip()
    if not vol_title:
        vol_title = collection_xml.stem

    module_to_path: Dict[str, List[str]] = {}

    def walk(node: ET.Element, path_titles: List[str]) -> None:
        # Subcollection title
        if node.tag.endswith("subcollection"):
            title_el = node.find("./md:title", _COL_NS)
            if title_el is not None and (title_el.text or "").strip():
                path_titles = path_titles + [(title_el.text or "").strip()]

        # Modules
        for mod in node.findall(".//col:module", _COL_NS):
            mid = mod.attrib.get("document")
            if not mid:
                continue
            # Prefer the first path we see.
            module_to_path.setdefault(mid, [vol_title] + path_titles)

        # Recurse into direct subcollections only to preserve hierarchy
        for child in list(node):
            if child.tag.endswith("subcollection"):
                walk(child, path_titles)

    content = root.find("./col:content", _COL_NS)
    if content is not None:
        walk(content, [])
    return vol_title, module_to_path


def _parse_module_cnxml(
    cnxml_path: Path,
    *,
    base_path_titles: List[str],
) -> Tuple[List[OSPPassage], List[OSPUnit]]:
    module_id = cnxml_path.parent.name
    root = ET.fromstring(cnxml_path.read_text(encoding="utf-8", errors="ignore"))
    doc_title_el = root.find("./cnx:title", _CNX_NS)
    doc_title = (doc_title_el.text or "").strip() if doc_title_el is not None else module_id

    passages: List[OSPPassage] = []
    units: Dict[str, OSPUnit] = {}

    unit_counters: Dict[str, int] = defaultdict(int)

    def make_section(section_stack: List[str]) -> str:
        parts = [p for p in (base_path_titles + [doc_title] + section_stack) if p]
        return " / ".join(parts)

    def make_unit_id(unit_type: str, elem: ET.Element) -> str:
        raw_id = (elem.attrib.get("id") or "").strip()
        if raw_id:
            return f"osp_unit:{module_id}:{unit_type}:{raw_id}"
        unit_counters[unit_type] += 1
        return f"osp_unit:{module_id}:{unit_type}:{unit_counters[unit_type]}"

    def make_unit_passage_id(unit_id: str) -> str:
        # Stable ID derived from the unit_id. These passages are meant to make all units "coverable"
        # by passage-level pipelines (EdgeQA), not to replace existing paragraph passages.
        return f"osp_passage_unit:{unit_id}"

    def make_passage_id(elem: ET.Element, fallback_text: str) -> str:
        raw_id = (elem.attrib.get("id") or "").strip()
        if raw_id:
            return f"osp:{module_id}:{raw_id}"
        h = sha256_text(f"{module_id}::{fallback_text}")[:16]
        return f"osp:{module_id}:{h}"

    def walk(el: ET.Element, *, unit_id: Optional[str], section_stack: List[str]) -> None:
        tag = el.tag
        if tag == _t(_CNX_NS["cnx"], "section"):
            title_el = el.find("./cnx:title", _CNX_NS)
            title = (title_el.text or "").strip() if title_el is not None else ""
            if title:
                section_stack = section_stack + [title]
            for child in list(el):
                if child is title_el:
                    continue
                walk(child, unit_id=unit_id, section_stack=section_stack)
            return

        if tag in (
            _t(_CNX_NS["cnx"], "definition"),
            _t(_CNX_NS["cnx"], "example"),
            _t(_CNX_NS["cnx"], "equation"),
        ):
            unit_type = tag.rsplit("}", 1)[-1]
            uid = make_unit_id(unit_type, el)
            if uid not in units:
                units[uid] = OSPUnit(
                    unit_id=uid,
                    doc_id=module_id,
                    unit_type=unit_type,
                    text=_elem_text(el),
                )
            # Add an explicit unit-level passage so coverage over `units.jsonl` is meaningful for OSP.
            # Many unit elements do not contain `<para>` children; without this, only a small subset of
            # units ever appear in `passages.jsonl` (unit_id=None for most passages), making unit coverage
            # curves appear artificially flat.
            unit_text = units[uid].text
            if unit_text and len(unit_text) >= 40:
                passages.append(
                    OSPPassage(
                        passage_id=make_unit_passage_id(uid),
                        doc_id=module_id,
                        section=make_section(section_stack),
                        text=unit_text,
                        unit_id=uid,
                    )
                )
            for child in list(el):
                walk(child, unit_id=uid, section_stack=section_stack)
            return

        if tag == _t(_CNX_NS["cnx"], "para"):
            text = _elem_text(el)
            if text:
                pid = make_passage_id(el, text)
                passages.append(
                    OSPPassage(
                        passage_id=pid,
                        doc_id=module_id,
                        section=make_section(section_stack),
                        text=text,
                        unit_id=unit_id,
                    )
                )
            return

        # Default: recurse
        for child in list(el):
            walk(child, unit_id=unit_id, section_stack=section_stack)

    content = root.find("./cnx:content", _CNX_NS)
    if content is not None:
        walk(content, unit_id=None, section_stack=[])
    return passages, list(units.values())


def parse_repo(repo_dir: str | Path) -> Tuple[List[OSPPassage], List[OSPUnit], str]:
    repo_root = Path(repo_dir)
    collections_dir = repo_root / "collections"
    modules_dir = repo_root / "modules"

    module_paths: Dict[str, List[str]] = {}
    if collections_dir.exists():
        for cxml in sorted(collections_dir.glob("*.collection.xml")):
            _, m = _parse_collection_module_paths(cxml)
            module_paths.update(m)

    passages: List[OSPPassage] = []
    units: List[OSPUnit] = []

    for module_dir in sorted(modules_dir.glob("m*")):
        cnxml = module_dir / "index.cnxml"
        if not cnxml.exists():
            continue
        base_path_titles = module_paths.get(module_dir.name, ["OpenStax University Physics"])
        p, u = _parse_module_cnxml(cnxml, base_path_titles=base_path_titles)
        passages.extend(p)
        units.extend(u)

    return passages, units, str(repo_root)
