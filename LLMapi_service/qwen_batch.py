from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import aiohttp

from LLMapi_service.api_keys import qwen_api_keys
from LLMapi_service.gptservice import QwenBaseUrl, _retry_async, check_proxy, close_session, get_session


_DEFAULT_POLL_S = 5.0
_DEFAULT_UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=30, sock_connect=30, sock_read=240)
_DEFAULT_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=600, connect=30, sock_connect=30, sock_read=540)


def _first_qwen_key() -> str:
    if not qwen_api_keys:
        return ""
    return str(qwen_api_keys[0] or "").strip()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


@dataclass(frozen=True)
class BatchRun:
    batch_id: str
    status: str
    input_file_id: str
    output_file_id: Optional[str]
    error_file_id: Optional[str]
    raw: Dict[str, Any]


class QwenBatchClient:
    def __init__(self, *, api_key: Optional[str] = None, base_url: str = QwenBaseUrl):
        self._api_key = (api_key or _first_qwen_key()).strip()
        if not self._api_key:
            raise ValueError("Missing Qwen API key (expected qwenkey.txt or EDGEQA_QWEN_KEYS_FILE).")
        self._base_url = str(base_url).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def close(self) -> None:
        # Reuse gptservice's global session lifecycle.
        await close_session()

    async def upload_file(self, *, path: Path, purpose: str = "batch") -> str:
        if not path.exists():
            raise FileNotFoundError(str(path))

        url = f"{self._base_url}/files"
        proxy = await check_proxy()

        async def _do() -> str:
            session = await get_session()
            form = aiohttp.FormData()
            form.add_field("purpose", purpose)
            with path.open("rb") as f:
                form.add_field("file", f, filename=path.name, content_type="application/jsonl")
                async with session.post(url, data=form, headers=self._headers(), proxy=proxy, timeout=_DEFAULT_UPLOAD_TIMEOUT) as resp:
                    resp.raise_for_status()
                    obj = await resp.json()
            fid = str(obj.get("id") or "").strip()
            if not fid:
                raise RuntimeError(f"upload_file: missing id (resp keys={list(obj.keys())})")
            return fid

        return await _retry_async(_do, retries=2, base_delay=1.0, max_delay=20.0)

    async def create_batch(
        self,
        *,
        input_file_id: str,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}/batches"
        proxy = await check_proxy()
        payload: Dict[str, Any] = {
            "input_file_id": str(input_file_id),
            "endpoint": str(endpoint),
            "completion_window": str(completion_window),
        }
        if metadata:
            payload["metadata"] = metadata

        async def _do() -> Dict[str, Any]:
            session = await get_session()
            async with session.post(url, json=payload, headers=self._headers(), proxy=proxy, timeout=_DEFAULT_UPLOAD_TIMEOUT) as resp:
                resp.raise_for_status()
                return await resp.json()

        return await _retry_async(_do, retries=2, base_delay=1.0, max_delay=20.0)

    async def get_batch(self, *, batch_id: str) -> Dict[str, Any]:
        url = f"{self._base_url}/batches/{batch_id}"
        proxy = await check_proxy()

        async def _do() -> Dict[str, Any]:
            session = await get_session()
            async with session.get(url, headers=self._headers(), proxy=proxy, timeout=_DEFAULT_UPLOAD_TIMEOUT) as resp:
                resp.raise_for_status()
                return await resp.json()

        return await _retry_async(_do, retries=2, base_delay=1.0, max_delay=20.0)

    async def download_file_content(self, *, file_id: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self._base_url}/files/{file_id}/content"
        proxy = await check_proxy()

        async def _do() -> None:
            session = await get_session()
            async with session.get(url, headers=self._headers(), proxy=proxy, timeout=_DEFAULT_DOWNLOAD_TIMEOUT) as resp:
                resp.raise_for_status()
                data = await resp.read()
            out_path.write_bytes(data)

        await _retry_async(_do, retries=2, base_delay=1.0, max_delay=30.0)

    async def wait(
        self,
        *,
        batch_id: str,
        poll_s: float = _DEFAULT_POLL_S,
        timeout_s: Optional[float] = None,
    ) -> BatchRun:
        started = time.time()
        last_status = None
        while True:
            obj = await self.get_batch(batch_id=batch_id)
            status = str(obj.get("status") or "").strip()
            if status and status != last_status:
                last_status = status
            if status in {"completed", "failed", "cancelled", "expired"}:
                return BatchRun(
                    batch_id=str(obj.get("id") or batch_id),
                    status=status,
                    input_file_id=str(obj.get("input_file_id") or ""),
                    output_file_id=(str(obj.get("output_file_id")) if obj.get("output_file_id") else None),
                    error_file_id=(str(obj.get("error_file_id")) if obj.get("error_file_id") else None),
                    raw=obj,
                )
            if timeout_s is not None and (time.time() - started) > float(timeout_s):
                raise TimeoutError(f"Batch {batch_id} timed out after {timeout_s}s (last status={status})")
            await asyncio.sleep(max(0.5, float(poll_s)))

    async def run_job(
        self,
        *,
        requests_path: Path,
        out_dir: Path,
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        poll_s: float = _DEFAULT_POLL_S,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchRun:
        out_dir.mkdir(parents=True, exist_ok=True)

        submitted_path = out_dir / "batch_submitted.json"
        meta_path = out_dir / "batch_meta.json"
        out_path = out_dir / "output.jsonl"
        err_path = out_dir / "error.jsonl"

        # If already completed and outputs exist, return quickly.
        if meta_path.exists() and out_path.exists():
            try:
                obj = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
                return BatchRun(
                    batch_id=str(obj.get("id") or ""),
                    status=str(obj.get("status") or "completed"),
                    input_file_id=str(obj.get("input_file_id") or ""),
                    output_file_id=(str(obj.get("output_file_id")) if obj.get("output_file_id") else None),
                    error_file_id=(str(obj.get("error_file_id")) if obj.get("error_file_id") else None),
                    raw=obj if isinstance(obj, dict) else {},
                )
            except Exception:
                pass

        batch_id: str | None = None
        input_file_id: str | None = None
        created: Dict[str, Any] | None = None

        # Resume if we already submitted.
        if submitted_path.exists():
            try:
                sub = json.loads(submitted_path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(sub, dict):
                    batch_id = str(sub.get("batch_id") or sub.get("id") or "").strip() or None
                    input_file_id = str(sub.get("input_file_id") or "").strip() or None
            except Exception:
                batch_id = None
                input_file_id = None

        # Otherwise submit a new batch and persist its ID immediately (so we can resume after interruption).
        if not batch_id:
            input_file_id = await self.upload_file(path=requests_path, purpose="batch")
            created = await self.create_batch(
                input_file_id=input_file_id, endpoint=endpoint, completion_window=completion_window, metadata=metadata
            )
            batch_id = str(created.get("id") or "").strip() or None
            if not batch_id:
                raise RuntimeError(f"create_batch: missing id (resp keys={list(created.keys())})")
            submitted_payload = {
                "batch_id": batch_id,
                "input_file_id": input_file_id,
                "endpoint": endpoint,
                "completion_window": completion_window,
                "metadata": metadata,
                "created": created,
                "requests_path": str(requests_path),
                "submitted_at": time.time(),
            }
            submitted_path.write_text(json.dumps(submitted_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        run = await self.wait(batch_id=batch_id, poll_s=poll_s)

        meta_path.write_text(json.dumps(run.raw, ensure_ascii=False, indent=2), encoding="utf-8")

        if run.output_file_id:
            await self.download_file_content(file_id=run.output_file_id, out_path=out_path)
        if run.error_file_id:
            await self.download_file_content(file_id=run.error_file_id, out_path=err_path)

        return run


def parse_batch_output(path: Path) -> Iterator[Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    """
    Yields (custom_id, response_body, error_obj).

    DashScope OpenAI-compatible Batch output lines look like:
      {"custom_id": "...", "response": {"status_code": 200, "body": {...}}, "error": null}
    """
    for row in iter_jsonl(path):
        cid = str(row.get("custom_id") or "")
        resp = row.get("response") or None
        err = row.get("error") or None
        body = None
        if isinstance(resp, dict):
            body = resp.get("body")
        yield cid, (body if isinstance(body, dict) else None), (err if isinstance(err, dict) else err)
