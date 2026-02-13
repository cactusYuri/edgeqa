from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from LLMapi_service.gptservice import GPT, close_session


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(p * len(xs)) - 1))
    return float(xs[k])


async def _run_level(*, concurrency: int, total: int, model: str, max_tokens: int) -> Dict[str, Any]:
    q: asyncio.Queue[int | None] = asyncio.Queue()
    for i in range(total):
        q.put_nowait(i)
    for _ in range(max(1, concurrency)):
        q.put_nowait(None)

    lock = asyncio.Lock()
    ok = 0
    err = Counter()
    lats: List[float] = []

    async def worker(wid: int) -> None:
        nonlocal ok
        while True:
            item = await q.get()
            try:
                if item is None:
                    return
                t0 = time.time()
                try:
                    await GPT(
                        [
                            {
                                "role": "system",
                                "content": "Output exactly the single character A. No punctuation, no whitespace.",
                            },
                            {"role": "user", "content": f"ping {item}"},
                        ],
                        selected_model=model,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        raise_on_error=True,
                    )
                    code = "OK"
                except Exception as e:
                    msg = repr(e)
                    m = re.search(r"HTTP\\s+(\\d{3})", msg)
                    code = m.group(1) if m else "ERR"
                lat = time.time() - t0
                async with lock:
                    lats.append(lat)
                    if code == "OK":
                        ok += 1
                    else:
                        err[code] += 1
            finally:
                q.task_done()

    tasks = [asyncio.create_task(worker(i)) for i in range(max(1, concurrency))]
    started = time.time()
    await q.join()
    elapsed = time.time() - started

    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await close_session()

    total_done = ok + sum(err.values())
    return {
        "concurrency": concurrency,
        "total": total,
        "done": total_done,
        "elapsed_s": elapsed,
        "rps": (total_done / elapsed) if elapsed > 0 else 0.0,
        "ok": ok,
        "errors": dict(err),
        "latency_s": {
            "n": len(lats),
            "mean": statistics.mean(lats) if lats else 0.0,
            "p50": _pct(lats, 0.50),
            "p90": _pct(lats, 0.90),
            "max": max(lats) if lats else 0.0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=str, default="200", help='Either "600" or "200,400,600"')
    ap.add_argument("--total", type=int, default=300)
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--max-tokens", type=int, default=8)
    args = ap.parse_args()

    levels = []
    for part in str(args.concurrency).split(","):
        part = part.strip()
        if not part:
            continue
        levels.append(int(part))
    if not levels:
        levels = [200]

    async def _run_all() -> List[Dict[str, Any]]:
        out = []
        for c in levels:
            out.append(await _run_level(concurrency=c, total=int(args.total), model=str(args.model), max_tokens=int(args.max_tokens)))
        return out

    rows = asyncio.run(_run_all())
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

