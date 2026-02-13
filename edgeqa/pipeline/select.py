from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from edgeqa.text_utils import jaccard, normalize_for_match


_TOK_RE = re.compile(r"[a-z0-9]+")


def _q_tokens(q: str) -> Set[str]:
    return set(_TOK_RE.findall(normalize_for_match(q)))


@dataclass(frozen=True)
class Candidate:
    item: Dict[str, Any]
    docs: Set[str]
    units: Set[str]
    reason: str
    unknown: bool
    q_tokens: Set[str]


def _build_candidates(
    pool: Sequence[Dict[str, Any]],
    *,
    passage_to_unit: Dict[str, Optional[str]],
) -> List[Candidate]:
    out: List[Candidate] = []
    for it in pool:
        ev = it.get("evidence", []) or []
        docs = set(ev)
        units: Set[str] = set()
        for pid in ev:
            uid = passage_to_unit.get(pid)
            if uid:
                units.add(uid)
        reason = str(it.get("reason_type") or "")
        scores = it.get("scores", {}) or {}
        unknown = bool(scores.get("unknown", False))
        out.append(
            Candidate(
                item=it,
                docs=docs,
                units=units,
                reason=reason,
                unknown=unknown,
                q_tokens=_q_tokens(str(it.get("question") or "")),
            )
        )
    return out


def greedy_select(
    pool: Sequence[Dict[str, Any]],
    *,
    passage_to_unit: Dict[str, Optional[str]],
    N: int,
    lambdas: Dict[str, float],
    unknownness_min_frac: float = 0.0,
) -> List[Dict[str, Any]]:
    candidates = _build_candidates(pool, passage_to_unit=passage_to_unit)
    N = max(0, int(N))
    if N == 0 or not candidates:
        return []

    need_unknown = 0
    if unknownness_min_frac and unknownness_min_frac > 0:
        need_unknown = int(math.ceil(N * float(unknownness_min_frac)))

    selected: List[Candidate] = []
    covered_docs: Set[str] = set()
    covered_units: Set[str] = set()
    covered_reasons: Set[str] = set()

    w_doc = float(lambdas.get("doc", 1.0))
    w_unit = float(lambdas.get("unit", 1.0))
    w_reason = float(lambdas.get("reason", 0.5))
    w_dup = float(lambdas.get("redundancy", 0.2))

    # Redundancy helper: index selected questions by token to avoid O(N^2) scans.
    selected_q_tokens: List[Set[str]] = []
    token_to_selected: Dict[str, List[int]] = {}
    max_postings = 200  # ignore overly-common tokens when computing redundancy

    def _max_sim(c: Candidate) -> float:
        if w_dup <= 0.0 or not selected_q_tokens or not c.q_tokens:
            return 0.0
        idxs: Set[int] = set()
        for t in c.q_tokens:
            postings = token_to_selected.get(t)
            if not postings or len(postings) > max_postings:
                continue
            idxs.update(postings)
        if not idxs:
            return 0.0
        best = 0.0
        for j in idxs:
            sim = jaccard(c.q_tokens, selected_q_tokens[j])
            if sim > best:
                best = sim
        return best

    def marginal_gain(c: Candidate) -> float:
        new_docs = 0
        for d in c.docs:
            if d not in covered_docs:
                new_docs += 1
        new_units = 0
        for u in c.units:
            if u not in covered_units:
                new_units += 1
        new_reason = 1 if (c.reason and c.reason not in covered_reasons) else 0
        return (w_doc * new_docs) + (w_unit * new_units) + (w_reason * new_reason) - (w_dup * _max_sim(c))

    remaining: Set[int] = set(range(len(candidates)))

    def _select_idx(idx: int) -> None:
        c = candidates[idx]
        selected.append(c)
        covered_docs.update(c.docs)
        covered_units.update(c.units)
        if c.reason:
            covered_reasons.add(c.reason)

        if w_dup > 0.0 and c.q_tokens:
            pos = len(selected_q_tokens)
            selected_q_tokens.append(c.q_tokens)
            for t in c.q_tokens:
                token_to_selected.setdefault(t, []).append(pos)

    def _lazy_pick(*, target_total: int, require_unknown: bool) -> None:
        # Lazy greedy with upper bounds: exact greedy selection but fewer gain recomputations.
        heap: List[Tuple[float, int]] = []
        for i in remaining:
            if require_unknown and not candidates[i].unknown:
                continue
            g0 = marginal_gain(candidates[i])
            heapq.heappush(heap, (-g0, i))

        eps = 1e-12
        while heap and remaining and len(selected) < target_total:
            neg_g, i = heapq.heappop(heap)
            if i not in remaining:
                continue
            c = candidates[i]
            if require_unknown and not c.unknown:
                continue

            g = marginal_gain(c)
            best_upper = -heap[0][0] if heap else float("-inf")
            if g + eps >= best_upper:
                remaining.remove(i)
                _select_idx(i)
            else:
                heapq.heappush(heap, (-g, i))

    # Phase 1: satisfy unknownness constraint if requested.
    if need_unknown > 0 and remaining and len(selected) < N:
        _lazy_pick(target_total=min(N, need_unknown), require_unknown=True)

    # Phase 2: fill the rest.
    if remaining and len(selected) < N:
        _lazy_pick(target_total=N, require_unknown=False)

    return [c.item for c in selected[:N]]
