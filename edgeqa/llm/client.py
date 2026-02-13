from __future__ import annotations

import json
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from LLMapi_service.gptservice import GPT, close_session

from edgeqa.hash_utils import sha256_json
from edgeqa.llm.cache import DiskCache
from edgeqa.logging_utils import get_logger


def _append_text_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


class LLMClient:
    def __init__(
        self,
        *,
        cache: DiskCache,
        concurrency: int = 16,
        force_refresh: bool = False,
        call_log_path: str | Path | None = None,
    ):
        self._cache = cache
        self._sem = asyncio.Semaphore(max(1, int(concurrency)))
        self._force_refresh = bool(force_refresh)
        self._log = get_logger("edgeqa.llm")
        self._call_log_path = Path(call_log_path) if call_log_path else None
        self._call_log_lock = asyncio.Lock()

    async def close(self) -> None:
        await close_session()

    async def _append_call_log(self, rec: Dict[str, Any]) -> None:
        if self._call_log_path is None:
            return
        row = {
            "ts": rec.get("ts"),
            "elapsed_s": rec.get("elapsed_s"),
            "namespace": rec.get("namespace"),
            "key": rec.get("key"),
            "ok": bool(rec.get("error") is None),
            "error": rec.get("error"),
            "request": {
                "model": (rec.get("request") or {}).get("model"),
                "temperature": (rec.get("request") or {}).get("temperature"),
                "max_tokens": (rec.get("request") or {}).get("max_tokens"),
            },
            "response_model": rec.get("response_model"),
            "usage": rec.get("usage"),
        }
        try:
            if isinstance(row.get("error"), str) and len(row["error"]) > 500:
                row["error"] = row["error"][:500]
            line = json.dumps(row, ensure_ascii=False) + "\n"
            async with self._call_log_lock:
                await asyncio.to_thread(_append_text_line, self._call_log_path, line)
        except Exception as e:
            self._log.warning("Failed to append call log: %s", repr(e))

    async def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        cache_namespace: str = "chat",
        force_refresh: Optional[bool] = None,
        raise_on_error: bool = True,
    ) -> Dict[str, Any]:
        req: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key = sha256_json({"ns": cache_namespace, "req": req})
        do_refresh = self._force_refresh if force_refresh is None else bool(force_refresh)
        if not do_refresh:
            cached = await asyncio.to_thread(self._cache.get, key)
            if cached is not None:
                cached["_cached"] = True
                return cached

        exc: BaseException | None = None
        async with self._sem:
            started = time.time()
            try:
                msg = await GPT(
                    messages,
                    selected_model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    raise_on_error=raise_on_error,
                )
            except BaseException as e:
                msg = None
                exc = e
            elapsed_s = time.time() - started

        usage = None
        resp_model = None
        if isinstance(msg, dict):
            usage = msg.get("_usage") or msg.get("usage")
            resp_model = msg.get("_model") or msg.get("model")

        rec: Dict[str, Any] = {
            "_cached": False,
            "key": key,
            "ts": time.time(),
            "elapsed_s": elapsed_s,
            "namespace": cache_namespace,
            "request": req,
            "response": msg,
            "error": (repr(exc) if exc is not None else None),
            "usage": usage,
            "response_model": resp_model,
        }
        await self._append_call_log(rec)
        if exc is not None:
            raise exc
        await asyncio.to_thread(self._cache.set, key, rec)
        return rec
