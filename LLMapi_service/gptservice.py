import json
import asyncio
import aiohttp
import asyncio
import random
import os
import builtins as _builtins
from typing import List, Dict, Any, Optional, Union


def _safe_print(*args, **kwargs):
    try:
        _builtins.print(*args, **kwargs)
    except Exception:
        pass


print = _safe_print


def _compact_error_snippet(txt: str, *, limit: int = 240) -> str:
    raw = (txt or "").strip()
    if not raw:
        return ""

    code = None
    msg = None
    trace_id = None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("msg")
                code = err.get("code") or err.get("type")
                trace_id = err.get("trace_id") or obj.get("trace_id")
            else:
                msg = obj.get("msg") or obj.get("message")
                code = obj.get("code")
                trace_id = obj.get("trace_id")
    except Exception:
        pass

    parts = []
    if code is not None and str(code).strip():
        parts.append(f"code={str(code).strip()}")
    if trace_id is not None and str(trace_id).strip():
        parts.append(f"trace_id={str(trace_id).strip()}")
    if msg is not None and str(msg).strip():
        parts.append(f"message={str(msg).strip()}")

    if parts:
        snippet = "; ".join(parts)
    else:
        snippet = raw.replace("\r", " ").replace("\n", " ").strip()

    if len(snippet) > limit:
        snippet = snippet[:limit]
    return snippet

BaseUrl = str(os.getenv("EDGEQA_OPENAI_BASE_URL", "https://aiping.cn/api"))
DeepseekBaseUrl = 'https://api.deepseek.com'
GeminiBaseUrl = 'https://generativelanguage.googleapis.com/v1beta'
QwenBaseUrl = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
VAPIBaseUrl = 'https://api.gpt.ge/v1'

ApiLoadCount = 0
# Import API keys from configuration file
try:
    from .api_keys import gemini_keys, open_ai_keys, deepseek_api_keys, qwen_api_keys, vapi_api_keys
except ImportError:
    # Fallback if configuration file is not available
    gemini_keys = ['']
    open_ai_keys = ['']
    deepseek_api_keys = ['']
    qwen_api_keys = ['']
    vapi_api_keys = ['']

# --- 并发安全锁 ---
_api_key_lock = asyncio.Lock()
_session_lock = asyncio.Lock()
_proxy_check_lock = asyncio.Lock() # 确保代理检查只进行一次的锁

models = {
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4o': 'chatgpt-4o-latest',
    'o3': 'o3',
    'o4-mini-high': 'o4-mini-high',
    'claude-3-5-sonnet-20240620': 'claude-3-5-sonnet-20240620',
    'claude-3-7-sonnet': 'claude-3-7-sonnet-20250219',
    'chatgpt-4o-latest': 'chatgpt-4o-latest',
    'gemini-2.0-flash': 'gemini-2.0-flash',
    'gemini-2.5-flash': 'gemini-2.5-flash',
    'gemini-2.5-pro': 'gemini-2.5-pro',
    # 清程极智（OpenAI-compatible）上的 DeepSeek 模型名：可用环境变量覆盖
    'deepseek-chat': str(os.getenv("EDGEQA_DEEPSEEK_CHAT_MODEL", "DeepSeek-V3.2")),
    'deepseek-reasoner': str(os.getenv("EDGEQA_DEEPSEEK_REASONER_MODEL", "DeepSeek-V3.2")),
    'qwen-flash': 'qwen-flash',
    'qwen-turbo': 'qwen-turbo',
    'qwen-plus': 'qwen-plus',
    'qwen-max': 'qwen-max',
    'gpt-4o-mini-search-preview': 'gpt-4o-mini-search-preview',
    'gpt-4o-search-preview': 'gpt-4o-search-preview',
    'openrouter/horizon-beta': 'openrouter/horizon-beta',
    'openrouter/horizon-alpha': 'openrouter/horizon-alpha',
    'deepseek/deepseek-r1-0528:free': 'deepseek/deepseek-r1-0528:free',
    'google/gemini-2.5-pro-exp-03-25': 'google/gemini-2.5-pro-exp-03-25',
}

_session: Optional[aiohttp.ClientSession] = None
_direct_session: Optional[aiohttp.ClientSession] = None
# Deepseek-chat 一般应在几十秒内返回；将超时设为“足够大但不过分”，避免网关/坏 key 异常时长时间挂起占用并发。
_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=90, connect=15, sock_connect=15, sock_read=75)
_DEEPSEEK_CHAT_MAX_TOKENS = 8000
# 用户要求：deepseek-reasoner 单次输出上限 64000（对应 chat 的 8000）
_DEEPSEEK_REASONER_MAX_TOKENS = 64000
# reasoner 可能生成很长推理，适当拉长请求超时，减少 missing_final
_DEEPSEEK_REASONER_TIMEOUT = aiohttp.ClientTimeout(total=3600, connect=30, sock_connect=30, sock_read=3550)
_DEEPSEEK_FORCE_ENV_PROXY: bool = False
_session_reset_lock = asyncio.Lock()
_last_session_reset_at: float = 0.0
_session_reset_min_interval_sec: float = 5.0

# 是否允许使用代理；对 deepseek 默认禁用
USE_PROXY = False
_rate_limit_lock = asyncio.Lock()
_last_request_ts: float = 0.0
_min_interval_sec: float = 3.0  # 全局最小间隔
_ds_rate_limit_lock = asyncio.Lock()
_ds_last_request_ts: float = 0.0
_ds_min_interval_sec: float = float(os.getenv("EDGEQA_DEEPSEEK_MIN_INTERVAL_SEC", "0.0") or "0.0")
_key_block_until: Dict[str, float] = {}
_key_last_used_at: Dict[str, float] = {}  # 单 key 最近一次“占用”时间
_per_key_min_interval_sec: float = float(os.getenv("EDGEQA_PER_KEY_MIN_INTERVAL_SEC", "0.0") or "0.0")
_per_key_max_concurrency: int = int(os.getenv("EDGEQA_PER_KEY_CONCURRENCY", "3") or "3")
_key_inflight: Dict[str, int] = {}
_proxy_status: Optional[str] = None # 用于缓存代理检查结果
_proxy_checked: bool = False      # 标记是否已执行过检查
_gateway_overload_until: float = 0.0
_gateway_overload_lock = asyncio.Lock()
_gateway_inflight_sem = asyncio.Semaphore(
    max(1, int(os.getenv("EDGEQA_GATEWAY_MAX_INFLIGHT", "5") or "5"))
)

# DeepSeek 调用路径选择：
# - 若设置了 EDGEQA_DEEPSEEK_VIA_BIANXIE，则显式遵从（1=走清程极智网关，0=直连 DeepSeek 官方）。
# - 否则：若存在官方 DeepSeek key（例如仓库根目录 deepseekkey.txt），默认直连官方；没有则默认走清程极智网关。
def _deepseek_via_bianxie() -> bool:
    v = os.getenv("EDGEQA_DEEPSEEK_VIA_BIANXIE")
    if v is not None and str(v).strip() != "":
        return str(v).lower() not in ("0", "false", "no")
    return not bool(deepseek_api_keys)
_DEEPSEEK_MAX_TOKENS: int = int(os.getenv("EDGEQA_DEEPSEEK_MAX_TOKENS", "8000") or "8000")
_DEEPSEEK_THINKING_BUDGET: int = int(os.getenv("EDGEQA_DEEPSEEK_THINKING_BUDGET", "2048") or "2048")

# 判断是否为Deepseek模型
def is_deepseek_model(model_name: str) -> bool:
    return model_name.startswith('deepseek-')
# 判断是否为 Gemini 模型
def is_gemini_model(model_name: str) -> bool:
    return model_name.startswith('gemini-')
# 判断是否为 VAPI 模型（本地区分前缀，实际请求需去掉前缀）
def is_vapi_model(model_name: str) -> bool:
    return model_name.startswith('VAPI/')

# 判断是否为 Qwen 模型（DashScope OpenAI-compatible）
def is_qwen_model(model_name: str) -> bool:
    m = (model_name or "").lower()
    return m.startswith("qwen-") or m.startswith("qwen2") or m.startswith("qwen3")

def _deepseek_max_tokens(selected_model: str) -> int:
    if selected_model == "deepseek-reasoner":
        return _DEEPSEEK_REASONER_MAX_TOKENS
    return _DEEPSEEK_CHAT_MAX_TOKENS


async def _wait_gateway_overload() -> None:
    """When the gateway signals overload (e.g., HTTP 405 '并行请求量超限'),
    pause new attempts globally to let in-flight requests drain."""
    global _gateway_overload_until
    try:
        now = asyncio.get_event_loop().time()
    except Exception:
        return
    until = _gateway_overload_until
    if until <= now:
        return
    await asyncio.sleep(max(0.0, until - now))


async def _extend_gateway_overload(seconds: float) -> None:
    global _gateway_overload_until
    try:
        s = float(seconds)
    except Exception:
        return
    if s <= 0:
        return
    now = asyncio.get_event_loop().time()
    async with _gateway_overload_lock:
        _gateway_overload_until = max(_gateway_overload_until, now + s)

async def get_session() -> aiohttp.ClientSession:
    """获取或创建一个 aiohttp.ClientSession 实例 (线程安全)"""
    global _session
    async with _session_lock:
        if _session is None or _session.closed:
            _session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    ssl=False,
                    limit=0,  # rely on caller-side semaphores for concurrency control
                    enable_cleanup_closed=True,
                    ttl_dns_cache=300,
                ),
                timeout=_DEFAULT_TIMEOUT,
                trust_env=True,
            )
    return _session

async def get_direct_session() -> aiohttp.ClientSession:
    """获取或创建一个直连（不使用环境变量代理）的 aiohttp.ClientSession (线程安全)"""
    global _direct_session
    async with _session_lock:
        if _direct_session is None or _direct_session.closed:
            _direct_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    ssl=False,
                    limit=0,  # rely on caller-side semaphores for concurrency control
                    enable_cleanup_closed=True,
                    ttl_dns_cache=300,
                ),
                timeout=_DEFAULT_TIMEOUT,
                trust_env=False,  # avoid env HTTP(S)_PROXY for deepseek by default
            )
    return _direct_session

async def close_session():
    """关闭共享的 aiohttp.ClientSession"""
    global _session, _direct_session
    async with _session_lock:
        closed_any = False
        if _session and not _session.closed:
            await _session.close()
            closed_any = True
        if _direct_session and not _direct_session.closed:
            await _direct_session.close()
            closed_any = True
        if closed_any:
            # 让底层清理任务有机会完成，避免 pending 警告
            try:
                await asyncio.sleep(0.05)
            except Exception:
                pass
        _session = None
        _direct_session = None


async def _reset_session_throttled(reason: str) -> None:
    """Reset the shared session on severe transport errors (throttled to avoid stampedes)."""
    global _last_session_reset_at
    async with _session_reset_lock:
        try:
            now = asyncio.get_event_loop().time()
        except Exception:
            # Best-effort fallback
            now = _last_session_reset_at + _session_reset_min_interval_sec + 1.0
        if now - _last_session_reset_at < _session_reset_min_interval_sec:
            return
        _last_session_reset_at = now

        try:
            print(f"[gptservice] resetting aiohttp session due to: {reason}")
        except Exception:
            pass

        old = None
        old_direct = None
        async with _session_lock:
            global _session, _direct_session
            old = _session
            old_direct = _direct_session
            _session = None
            _direct_session = None

        try:
            delay_s = float(os.getenv("EDGEQA_SESSION_RESET_CLOSE_DELAY_SEC", "120") or "120")
        except Exception:
            delay_s = 120.0
        delay_s = max(0.0, delay_s)

        async def _close_later(s: aiohttp.ClientSession) -> None:
            try:
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
            except Exception:
                pass
            try:
                if s and (not s.closed):
                    await s.close()
            except Exception:
                pass

        for s in (old, old_direct):
            if s is None:
                continue
            try:
                asyncio.create_task(_close_later(s))
            except Exception:
                pass

async def check_proxy():
    """仅检查一次代理并缓存结果，避免并发时的性能风暴。"""
    if not USE_PROXY:
        return None
    global _proxy_status, _proxy_checked
    if _proxy_checked:
        return _proxy_status
    
    async with _proxy_check_lock:
        # 再次检查，因为在等待锁的时候可能已经被其他协程检查过了
        if _proxy_checked:
            return _proxy_status
        
        print("首次检查代理连接...")
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                 async with session.get("http://127.0.0.1:33210", timeout=1) as response:
                    if response.status < 400:
                        print("代理可用，将为后续请求设置代理。")
                        _proxy_status = "http://127.0.0.1:33210"
                    else:
                         _proxy_status = None
        except Exception:
            print("代理不可用，使用直接连接。")
            _proxy_status = None
        
        _proxy_checked = True
    return _proxy_status

async def _wait_rate_limit():
    """简单全局速率限制：相邻请求至少间隔 _min_interval_sec 秒。"""
    global _last_request_ts
    async with _rate_limit_lock:
        loop = asyncio.get_event_loop()
        now = loop.time()
        delta = now - _last_request_ts
        if delta < _min_interval_sec:
            await asyncio.sleep(_min_interval_sec - delta)
        _last_request_ts = loop.time()

async def _wait_deepseek_rate_limit() -> None:
    """DeepSeek(OpenAI-compatible) 网关的全局速率限制（可选）。

    说明：清程极智网关在长时间高压下可能出现持续 429/405；通过设置
    EDGEQA_DEEPSEEK_MIN_INTERVAL_SEC（例如 0.05~0.2）可显著降低长尾重试风暴。
    """
    if _ds_min_interval_sec <= 0.0:
        return
    global _ds_last_request_ts
    async with _ds_rate_limit_lock:
        loop = asyncio.get_event_loop()
        now = loop.time()
        delta = now - _ds_last_request_ts
        if delta < _ds_min_interval_sec:
            await asyncio.sleep(_ds_min_interval_sec - delta)
        _ds_last_request_ts = loop.time()

async def _pick_gemini_key() -> str:
    """选择一个当前未被临时封禁的 Gemini key；若全部封禁则等待到最早解封时间。"""
    if not gemini_keys:
        return ''
    while True:
        async with _api_key_lock:
            now = asyncio.get_event_loop().time()
            # round-robin 起点
            global ApiLoadCount
            ApiLoadCount += 1
            start = ApiLoadCount % len(gemini_keys)
            ordered = gemini_keys[start:] + gemini_keys[:start]
            candidate: Optional[str] = None
            earliest_ready = now + 3600.0
            for k in ordered:
                blocked_until = _key_block_until.get(k, 0.0)
                last_used = _key_last_used_at.get(k, 0.0)
                ready_at = max(blocked_until, last_used + _per_key_min_interval_sec)
                if now >= ready_at:
                    candidate = k
                    break
                earliest_ready = min(earliest_ready, ready_at)
            if candidate is not None:
                # 立刻占位，防止并发下重复命中同一 key
                _key_last_used_at[candidate] = now
                return candidate
            # 全部仍被封禁：等待最早解封
            wait_s = max(0.1, earliest_ready - now)
        await asyncio.sleep(wait_s)

async def _pick_openai_key() -> str:
    if not open_ai_keys:
        return ''
    while True:
        async with _api_key_lock:
            now = asyncio.get_event_loop().time()
            global ApiLoadCount
            ApiLoadCount += 1
            start = ApiLoadCount % len(open_ai_keys)
            ordered = open_ai_keys[start:] + open_ai_keys[:start]
            candidate: Optional[str] = None
            earliest_ready = now + 3600.0
            for k in ordered:
                blocked_until = _key_block_until.get(k, 0.0)
                last_used = _key_last_used_at.get(k, 0.0)
                inflight = _key_inflight.get(k, 0)
                if inflight >= _per_key_max_concurrency:
                    earliest_ready = min(earliest_ready, now + 0.05)
                    continue
                ready_at = max(blocked_until, last_used + _per_key_min_interval_sec)
                if now >= ready_at:
                    candidate = k
                    break
                earliest_ready = min(earliest_ready, ready_at)
            if candidate is not None:
                _key_last_used_at[candidate] = now
                _key_inflight[candidate] = _key_inflight.get(candidate, 0) + 1
                return candidate
            wait_s = max(0.1, earliest_ready - now)
        await asyncio.sleep(wait_s)


async def _release_openai_key(key: str) -> None:
    if not key:
        return
    async with _api_key_lock:
        cur = _key_inflight.get(key, 0)
        if cur <= 1:
            _key_inflight.pop(key, None)
        else:
            _key_inflight[key] = cur - 1

async def _pick_deepseek_key() -> str:
    if not deepseek_api_keys:
        return ''
    if len(deepseek_api_keys) == 1:
        k = deepseek_api_keys[0]
        while True:
            now = asyncio.get_event_loop().time()
            blocked_until = _key_block_until.get(k, 0.0)
            last_used = _key_last_used_at.get(k, 0.0)
            ready_at = max(blocked_until, last_used)
            if now >= ready_at:
                _key_last_used_at[k] = now
                return k
            await asyncio.sleep(max(0.05, ready_at - now))
    while True:
        async with _api_key_lock:
            now = asyncio.get_event_loop().time()
            global ApiLoadCount
            ApiLoadCount += 1
            start = ApiLoadCount % len(deepseek_api_keys)
            ordered = deepseek_api_keys[start:] + deepseek_api_keys[:start]
            candidate: Optional[str] = None
            earliest_ready = now + 3600.0
            for k in ordered:
                blocked_until = _key_block_until.get(k, 0.0)
                last_used = _key_last_used_at.get(k, 0.0)
                # Deepseek 官方允许高并发，这里不再强制 per-key 0.6s 间隔
                ready_at = max(blocked_until, last_used)
                if now >= ready_at:
                    candidate = k
                    break
                earliest_ready = min(earliest_ready, ready_at)
            if candidate is not None:
                _key_last_used_at[candidate] = now
                return candidate
            wait_s = max(0.1, earliest_ready - now)
        await asyncio.sleep(wait_s)

async def _pick_qwen_key() -> str:
    if not qwen_api_keys:
        return ''
    while True:
        async with _api_key_lock:
            now = asyncio.get_event_loop().time()
            global ApiLoadCount
            ApiLoadCount += 1
            start = ApiLoadCount % len(qwen_api_keys)
            ordered = qwen_api_keys[start:] + qwen_api_keys[:start]
            candidate: Optional[str] = None
            earliest_ready = now + 3600.0
            for k in ordered:
                blocked_until = _key_block_until.get(k, 0.0)
                last_used = _key_last_used_at.get(k, 0.0)
                # Qwen 官方允许高并发，这里不强制 per-key 间隔
                ready_at = max(blocked_until, last_used)
                if now >= ready_at:
                    candidate = k
                    break
                earliest_ready = min(earliest_ready, ready_at)
            if candidate is not None:
                _key_last_used_at[candidate] = now
                return candidate
            wait_s = max(0.1, earliest_ready - now)
        await asyncio.sleep(wait_s)

async def GPT(
    input_data,
    selected_model: str = 'gpt-4o-mini',
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    thinking_budget: Optional[int] = None,
    raise_on_error: bool = False,
):
    try:
        # VAPI 前缀模型不参与 models 校验
        if (selected_model not in models) and (not is_vapi_model(selected_model)) and (not is_qwen_model(selected_model)):
            raise ValueError(f"未知的模型: {selected_model}")

        if is_vapi_model(selected_model):
            return await call_vapi_api(input_data, selected_model, temperature=temperature, max_tokens=max_tokens)
        elif is_gemini_model(selected_model):
            return await call_gemini_api(input_data, selected_model, temperature=temperature, max_tokens=max_tokens)
        elif is_deepseek_model(selected_model):
            if _deepseek_via_bianxie():
                ds_enable_thinking = enable_thinking
                if ds_enable_thinking is None:
                    ds_enable_thinking = (selected_model == "deepseek-reasoner")
                ds_budget = thinking_budget
                if (ds_budget is None) and ds_enable_thinking:
                    ds_budget = _DEEPSEEK_THINKING_BUDGET
                # Heuristic: when thinking is enabled, cap thinking_budget to leave room for the final answer.
                # Otherwise DeepSeek-V3.2 may spend all output tokens on reasoning_content and return empty content.
                if ds_enable_thinking and (ds_budget is not None) and (max_tokens is not None):
                    try:
                        mt = int(max_tokens)
                        if mt > 0:
                            reserve = int(os.getenv("EDGEQA_DEEPSEEK_THINKING_RESERVE_TOKENS", "32") or "32")
                            if reserve < 0:
                                reserve = 32
                            cap_reserve = max(1, mt - reserve)
                            cap_half = max(1, mt // 2)
                            ds_budget = min(int(ds_budget), cap_reserve, cap_half)
                    except Exception:
                        pass
                return await call_bianxie_api(
                    input_data,
                    selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=ds_enable_thinking,
                    thinking_budget=ds_budget,
                )
            return await call_deepseek_api(input_data, selected_model, temperature=temperature, max_tokens=max_tokens)
        elif is_qwen_model(selected_model):
            return await call_qwen_api(
                input_data,
                selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
        else:
            return await call_bianxie_api(input_data, selected_model, temperature=temperature, max_tokens=max_tokens)

    except Exception as error:
        safe_msg = repr(error)
        if isinstance(error, aiohttp.ClientResponseError):
            try:
                safe_msg = f"HTTP {error.status} {error.message} url={error.request_info.real_url}"
            except Exception:
                safe_msg = f"HTTP {getattr(error, 'status', '?')} {getattr(error, 'message', '')}"
        print(f"GPT主函数出错: {safe_msg}")
        if raise_on_error:
            raise RuntimeError(safe_msg) from None
        return {"role": "assistant", "content": f"请求失败: {safe_msg}"}

# 重试次数：训练阶段用户要求尽量少，这里默认 1 次重试（共 2 次尝试）
async def _retry_async(
    fn,
    *,
    retries: int = 1,
    base_delay: float = 1.2,
    max_delay: float = 30.0,
    jitter: float = 0.25,
    retry_on_429: bool = True,
):
    """通用重试工具：指数退避+抖动。对可重试异常进行多次尝试。"""
    attempt = 0
    last_exc = None
    while attempt <= retries:
        try:
            return await fn()
        except aiohttp.ClientResponseError as e:
            last_exc = e
            # 仅对 408/429/5xx 进行重试
            if (e.status == 429) and (not retry_on_429):
                raise
            if e.status not in (408, 429) and (e.status < 500 or e.status >= 600):
                raise
            if attempt == retries:
                break
            # transport may be unhealthy after gateway/proxy errors
            if e.status in (502, 503, 504):
                await _reset_session_throttled(f"HTTP {e.status}")
            # honor Retry-After header if provided
            retry_after = 0.0
            try:
                ra = e.headers.get('Retry-After') if hasattr(e, 'headers') and e.headers else None
                if ra:
                    retry_after = float(ra)
            except Exception:
                retry_after = 0.0
            base_backoff = min(max_delay, base_delay * (2 ** attempt))
            sleep_s = max(retry_after, base_backoff)
            sleep_s = sleep_s * (1.0 + random.uniform(-jitter, jitter))
            await asyncio.sleep(max(0.05, sleep_s))
            attempt += 1
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            last_exc = e
            if attempt == retries:
                break
            # Reset the shared session on severe I/O errors to avoid reusing a broken connection pool.
            if isinstance(
                e,
                (
                    aiohttp.ClientPayloadError,
                    aiohttp.ServerDisconnectedError,
                    aiohttp.ClientConnectorError,
                    aiohttp.ClientOSError,
                    asyncio.TimeoutError,
                    ConnectionResetError,
                ),
            ):
                await _reset_session_throttled(type(e).__name__)
            sleep_s = min(max_delay, base_delay * (2 ** attempt))
            # 抖动
            sleep_s = sleep_s * (1.0 + random.uniform(-jitter, jitter))
            await asyncio.sleep(max(0.05, sleep_s))
            attempt += 1
        except Exception as e:
            # 非预期异常：不重试，立刻抛出
            raise
    raise last_exc

async def call_bianxie_api(
    input_data,
    selected_model,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    thinking_budget: Optional[int] = None,
):
    # NOTE: OpenAI-compatible 网关：请求不应泄露 key 到日志/异常。

    data = {
        "model": models[selected_model],
        "messages": input_data,
        "stream": False,
    }
    if temperature is not None:
        try:
            data["temperature"] = float(temperature)
        except Exception:
            pass

    # DeepSeek-V3.2：默认关闭思考时也给足 max_tokens（上限 8k），避免个别推理任务被截断。
    if is_deepseek_model(selected_model):
        max_eff = _DEEPSEEK_MAX_TOKENS
        if max_tokens is not None:
            try:
                mt = int(max_tokens)
                if mt > 0:
                    max_eff = min(mt, _DEEPSEEK_MAX_TOKENS)
            except Exception:
                pass
        data["max_tokens"] = int(max_eff)
        if enable_thinking is not None:
            data["enable_thinking"] = bool(enable_thinking)
        if thinking_budget is not None:
            try:
                tb = int(thinking_budget)
                if tb > 0:
                    data["thinking_budget"] = tb
            except Exception:
                pass
    else:
        if max_tokens is not None:
            try:
                mt = int(max_tokens)
                if mt > 0:
                    data["max_tokens"] = mt
            except Exception:
                pass

    # deepseek 请求不走代理，减少时延与错误
    proxy = None

    open_ai_key = await _pick_openai_key()
    try:
        consecutive_405 = 0
        consecutive_429 = 0
        consecutive_io = 0
        n_405 = 0
        n_429 = 0
        n_io = 0
        # Prevent infinite retry loops when the gateway is degraded for an extended period.
        try:
            deadline_s = float(os.getenv("EDGEQA_GATEWAY_CALL_DEADLINE_SEC", "600") or "600")
        except Exception:
            deadline_s = 600.0
        if selected_model == "deepseek-reasoner":
            deadline_s = max(deadline_s, 3600.0)
        started_at = asyncio.get_event_loop().time()
        while True:
            if deadline_s > 0 and (asyncio.get_event_loop().time() - started_at) > deadline_s:
                raise RuntimeError(
                    f"Gateway retry deadline exceeded ({deadline_s:.0f}s), 405={n_405}, 429={n_429}, io={n_io}"
                )
            # If the gateway is overloaded, pause all attempts for a while to avoid retry storms.
            await _wait_gateway_overload()
            headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {open_ai_key}'}

            async def _do():
                if is_deepseek_model(selected_model):
                    await _wait_deepseek_rate_limit()
                else:
                    await _wait_rate_limit()
                # DeepSeek via OpenAI-compatible gateway should NOT use env proxies by default.
                session = await (get_direct_session() if is_deepseek_model(selected_model) else get_session())
                async with _gateway_inflight_sem:
                    async with session.post(
                        f"{BaseUrl}/v1/chat/completions",
                        json=data,
                        headers=headers,
                        proxy=proxy,
                        timeout=_DEEPSEEK_REASONER_TIMEOUT if selected_model == "deepseek-reasoner" else _DEFAULT_TIMEOUT,
                    ) as response:
                        if response.status >= 400:
                            # Best-effort read response text for debugging / error attribution.
                            try:
                                txt = await response.text()
                            except Exception:
                                txt = ""
                            snippet = _compact_error_snippet(txt, limit=240)
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=snippet,
                                headers=response.headers,
                            )
                        return await response.json()

            try:
                resp_json = await _retry_async(_do, retry_on_429=False)
                msg = resp_json.get("choices", [{}])[0].get("message") or {}
                if not isinstance(msg, dict):
                    msg = {"role": "assistant", "content": str(msg)}
                # attach usage/model for downstream accounting (do NOT include key)
                msg["_id"] = resp_json.get("id")
                msg["_model"] = resp_json.get("model")
                msg["_usage"] = resp_json.get("usage")
                return msg
            except aiohttp.ClientResponseError as cre:
                if cre.status == 429:
                    n_429 += 1
                    consecutive_429 += 1
                    consecutive_405 = 0
                    consecutive_io = 0

                    # Gateways often return 429 under global throttling; avoid rapid key-rotation storms
                    # that can temporarily block the whole key pool. Use exponential backoff + shared cooldown.
                    retry_after = 1.0
                    try:
                        if cre.headers and 'Retry-After' in cre.headers:
                            retry_after = float(cre.headers['Retry-After'])
                    except Exception:
                        retry_after = 1.0

                    try:
                        max_backoff = float(os.getenv("EDGEQA_429_MAX_BACKOFF_SEC", "30") or "30")
                    except Exception:
                        max_backoff = 30.0
                    max_backoff = max(1.0, min(300.0, max_backoff))

                    base = min(max_backoff, max(0.5, retry_after) * (2 ** min(8, consecutive_429 - 1)))
                    sleep_s = base * (1.0 + random.uniform(-0.2, 0.2))
                    sleep_s = max(0.1, min(max_backoff, sleep_s))

                    # Briefly block the current key to reduce immediate reuse; global cooldown handles throttling.
                    _key_block_until[open_ai_key] = asyncio.get_event_loop().time() + sleep_s
                    try:
                        if consecutive_429 in (1, 3, 5, 10, 20):
                            print(f"[gptservice] 429 throttled; cooling down {sleep_s:.1f}s")
                    except Exception:
                        pass
                    await _release_openai_key(open_ai_key)
                    await _extend_gateway_overload(sleep_s)
                    await asyncio.sleep(sleep_s)
                    open_ai_key = await _pick_openai_key()
                    continue
                if cre.status == 405:
                    n_405 += 1
                    consecutive_405 += 1
                    consecutive_429 = 0
                    consecutive_io = 0
                    msg = str(cre.message or "")
                    msg_l = msg.lower()
                    is_parallel_limit = ("并行" in msg) or ("parallel" in msg_l) or ("concurrent" in msg_l)
                    # Optional debug: surface the gateway's response snippet once in a while.
                    if consecutive_405 in (1, 5, 20) and str(os.getenv("EDGEQA_DEBUG_405", "0")).lower() in (
                        "1",
                        "true",
                        "yes",
                    ):
                        try:
                            print(f"[gptservice] HTTP 405 body: {cre.message}")
                        except Exception:
                            pass
                    # Some gateways return 405 under overload / edge routing issues.
                    # Treat as transient and rotate key with a short backoff.
                    retry_after = 1.0
                    try:
                        if cre.headers and 'Retry-After' in cre.headers:
                            retry_after = float(cre.headers['Retry-After'])
                    except Exception:
                        retry_after = 1.0
                    _key_block_until[open_ai_key] = asyncio.get_event_loop().time() + retry_after
                    await _release_openai_key(open_ai_key)
                    # Exponential backoff with jitter to avoid retry storms.
                    # For "并行请求量超限" errors, the gateway can keep returning 405 even for single requests
                    # until existing in-flight requests time out; allow a much larger cooldown in that case.
                    max_backoff = 10.0
                    if is_parallel_limit:
                        try:
                            max_backoff = float(os.getenv("EDGEQA_405_MAX_BACKOFF_SEC", "240") or "240")
                        except Exception:
                            max_backoff = 240.0
                        max_backoff = max(10.0, min(600.0, max_backoff))
                    base = min(max_backoff, max(0.5, retry_after) * (2 ** min(8, consecutive_405 - 1)))
                    sleep_s = base * (1.0 + random.uniform(-0.2, 0.2))
                    sleep_s = max(0.1, min(max_backoff, sleep_s))
                    # Keep this key out of rotation for the full backoff window to avoid repeatedly
                    # picking a key while the gateway is still overloaded.
                    _key_block_until[open_ai_key] = asyncio.get_event_loop().time() + sleep_s
                    if is_parallel_limit and sleep_s >= 30.0 and consecutive_405 in (1, 3, 5, 10, 20):
                        try:
                            print(f"[gptservice] gateway overload (405); cooling down {sleep_s:.1f}s")
                        except Exception:
                            pass
                    await _extend_gateway_overload(sleep_s)
                    await asyncio.sleep(sleep_s)
                    open_ai_key = await _pick_openai_key()
                    continue
                if cre.status in (408, 500, 502, 503, 504):
                    n_io += 1
                    consecutive_io += 1
                    consecutive_429 = 0
                    consecutive_405 = 0

                    retry_after = 1.0
                    try:
                        if cre.headers and 'Retry-After' in cre.headers:
                            retry_after = float(cre.headers['Retry-After'])
                    except Exception:
                        retry_after = 1.0

                    try:
                        max_backoff = float(os.getenv("EDGEQA_5XX_MAX_BACKOFF_SEC", "30") or "30")
                    except Exception:
                        max_backoff = 30.0
                    max_backoff = max(1.0, min(300.0, max_backoff))

                    base = min(max_backoff, max(0.5, retry_after) * (2 ** min(6, consecutive_io - 1)))
                    sleep_s = base * (1.0 + random.uniform(-0.2, 0.2))
                    sleep_s = max(0.1, min(max_backoff, sleep_s))

                    _key_block_until[open_ai_key] = asyncio.get_event_loop().time() + sleep_s
                    await _release_openai_key(open_ai_key)
                    await _extend_gateway_overload(min(sleep_s, 5.0))
                    await asyncio.sleep(sleep_s)
                    open_ai_key = await _pick_openai_key()
                    continue
                raise
            except RuntimeError as rte:
                msg_l = str(rte or "").lower()
                if ("session is closed" in msg_l) or ("connector is closed" in msg_l):
                    n_io += 1
                    consecutive_io += 1
                    consecutive_429 = 0
                    consecutive_405 = 0
                    try:
                        await _reset_session_throttled(type(rte).__name__)
                    except Exception:
                        pass
                    await _release_openai_key(open_ai_key)
                    base = min(2.0, 0.25 * (2 ** min(4, consecutive_io - 1)))
                    sleep_s = base * (1.0 + random.uniform(-0.2, 0.2))
                    await asyncio.sleep(max(0.05, min(2.0, sleep_s)))
                    open_ai_key = await _pick_openai_key()
                    continue
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as ioe:
                n_io += 1
                consecutive_io += 1
                consecutive_429 = 0
                consecutive_405 = 0
                # Transport issues: rotate key and apply a short backoff to avoid retry storms.
                retry_after = 1.0
                try:
                    await _reset_session_throttled(type(ioe).__name__)
                except Exception:
                    pass
                await _release_openai_key(open_ai_key)
                base = min(10.0, max(0.5, retry_after) * (2 ** min(4, consecutive_io - 1)))
                sleep_s = base * (1.0 + random.uniform(-0.2, 0.2))
                # Quarantine flaky keys longer to bias selection toward responsive ones.
                # In practice some keys may hang/timeout repeatedly; short blocks cause the pool to churn.
                _key_block_until[open_ai_key] = asyncio.get_event_loop().time() + max(300.0, min(3600.0, float(sleep_s)))
                await asyncio.sleep(max(0.1, min(10.0, sleep_s)))
                open_ai_key = await _pick_openai_key()
                continue
            except Exception as error:
                print(f"Bianxie API request failed after retries: {repr(error)}")
                raise
    finally:
        await _release_openai_key(open_ai_key)

async def call_deepseek_api(input_data, selected_model, *, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    deepseek_api_key = await _pick_deepseek_key()
    
    # 显式提高生成上限，避免 4k 默认截断；deepseek 官方支持设置 max_tokens
    max_out = _deepseek_max_tokens(selected_model)
    max_tokens_eff = max_out
    if max_tokens is not None:
        try:
            mt = int(max_tokens)
            if mt > 0:
                max_tokens_eff = min(mt, max_out)
        except Exception:
            max_tokens_eff = max_out
    data = {
        "model": selected_model,
        "messages": input_data,
        "stream": False,
        "max_tokens": int(max_tokens_eff),
    }
    if temperature is not None:
        try:
            data["temperature"] = float(temperature)
        except Exception:
            pass
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {deepseek_api_key}'}

    proxy = await check_proxy()

    async def _do():
        global _DEEPSEEK_FORCE_ENV_PROXY
        use_env = bool(_DEEPSEEK_FORCE_ENV_PROXY)
        session = await (get_session() if use_env else get_direct_session())
        try:
            async with session.post(
                f"{DeepseekBaseUrl}/v1/chat/completions",
                json=data,
                headers=headers,
                proxy=proxy,
                timeout=_DEEPSEEK_REASONER_TIMEOUT if selected_model == "deepseek-reasoner" else _DEFAULT_TIMEOUT,
            ) as response:
                if response.status >= 400:
                    try:
                        txt = await response.text()
                    except Exception:
                        txt = ""
                    snippet = _compact_error_snippet(txt, limit=240)
                    if not snippet:
                        try:
                            snippet = str(response.reason or "")
                        except Exception:
                            snippet = ""
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=snippet,
                        headers=response.headers,
                    )

                resp_data = await response.json()
                msg = resp_data.get("choices", [{}])[0].get("message") or {}
                if not isinstance(msg, dict):
                    msg = {"role": "assistant", "content": str(msg)}
                msg["_id"] = resp_data.get("id")
                msg["_model"] = resp_data.get("model")
                msg["_usage"] = resp_data.get("usage")
                if "role" not in msg:
                    msg["role"] = "assistant"
                if "content" not in msg:
                    msg["content"] = ""
                return msg
        except (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, OSError):
            # deepseek 默认走直连；如果直连失败（例如网络环境限制），自动回退到环境代理（HTTP(S)_PROXY）。
            if (not use_env) and (not USE_PROXY):
                _DEEPSEEK_FORCE_ENV_PROXY = True
                try:
                    print("[gptservice] deepseek direct connect failed; fallback to env proxy for subsequent requests")
                except Exception:
                    pass
            raise
    try:
        return await _retry_async(_do)
    except aiohttp.ClientResponseError as cre:
        if cre.status == 429:
            retry_after = 20.0
            try:
                if cre.headers and 'Retry-After' in cre.headers:
                    retry_after = float(cre.headers['Retry-After'])
            except Exception:
                pass
            _key_block_until[deepseek_api_key] = asyncio.get_event_loop().time() + retry_after
            print(f"Deepseek key 封禁 {retry_after:.1f}s (429)")
            return await call_deepseek_api(input_data, selected_model, temperature=temperature, max_tokens=max_tokens)
        raise
    except Exception as error:
        print(f"Deepseek request failed after retries: {repr(error)}")
        raise

async def call_qwen_api(
    input_data,
    selected_model,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    thinking_budget: Optional[int] = None,
):
    """Call Qwen (DashScope OpenAI-compatible) chat completions API."""
    qwen_api_key = await _pick_qwen_key()

    data = {
        "model": selected_model,
        "messages": input_data,
        "stream": False,
    }
    if enable_thinking is not None:
        data["enable_thinking"] = bool(enable_thinking)
    if thinking_budget is not None:
        try:
            tb = int(thinking_budget)
            if tb > 0:
                data["thinking_budget"] = tb
        except Exception:
            pass
    if temperature is not None:
        try:
            data["temperature"] = float(temperature)
        except Exception:
            pass
    if max_tokens is not None:
        try:
            mt = int(max_tokens)
            if mt > 0:
                data["max_tokens"] = mt
        except Exception:
            pass

    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {qwen_api_key}'}
    proxy = await check_proxy()

    url = f"{QwenBaseUrl}/chat/completions"

    async def _do():
        session = await get_session()
        async with session.post(url, json=data, headers=headers, proxy=proxy) as response:
            response.raise_for_status()
            resp_data = await response.json()
            return {
                "role": resp_data.get("choices", [{}])[0].get("message", {}).get("role", "assistant"),
                "content": resp_data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            }

    try:
        return await _retry_async(_do)
    except aiohttp.ClientResponseError as cre:
        if cre.status == 429:
            retry_after = 20.0
            try:
                if cre.headers and 'Retry-After' in cre.headers:
                    retry_after = float(cre.headers['Retry-After'])
            except Exception:
                pass
            _key_block_until[qwen_api_key] = asyncio.get_event_loop().time() + retry_after
            print(f"Qwen key 封禁 {retry_after:.1f}s (429)")
            return await call_qwen_api(
                input_data,
                selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
        raise
    except Exception as error:
        print(f"Qwen request failed after retries: {repr(error)}")
        raise

async def call_gemini_api(input_data, selected_model, *, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    gemini_api_key = await _pick_gemini_key()
    
    # 将 system 信息放入 systemInstruction，其余只保留 user/assistant 对话
    system_texts = [str(msg.get("content", "")) for msg in input_data if msg.get("role") == "system"]
    conversation_msgs = []
    for msg in input_data:
        role = msg.get("role")
        if role == "system":
            continue
        mapped_role = "user" if role == "user" else "model"
        conversation_msgs.append({
            "role": mapped_role,
            "parts": [{"text": str(msg.get("content", ""))}]
        })
    data = {"contents": conversation_msgs}
    if system_texts:
        data["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
    # 生成配置：统一限制输出长度，并给出温度，便于服务端接受
    max_out = 512
    if max_tokens is not None:
        try:
            mt = int(max_tokens)
            if mt > 0:
                max_out = mt
        except Exception:
            max_out = 512
    temp = 0.4
    if temperature is not None:
        try:
            temp = float(temperature)
        except Exception:
            temp = 0.4
    data["generationConfig"] = {"maxOutputTokens": int(max_out), "temperature": float(temp), "topP": 0.95}
    
    headers = {'Content-Type': 'application/json'}
    proxy = await check_proxy()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={gemini_api_key}"

    async def _do():
        await _wait_rate_limit()
        session = await get_session()
        async with session.post(url, json=data, headers=headers, proxy=proxy) as response:
            response.raise_for_status()
            resp_data = await response.json()
            if "candidates" in resp_data and resp_data["candidates"]:
                candidate = resp_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    text = ''.join(part.get("text", "") for part in candidate["content"]["parts"] if "text" in part)
                    return {"role": "assistant", "content": text}
            # 错误响应解析
            if "error" in resp_data:
                try:
                    code = resp_data["error"].get("code")
                    status = resp_data["error"].get("status")
                    message = resp_data["error"].get("message")
                    print(f"Gemini API 错误: code={code} status={status} message={message}")
                except Exception:
                    pass
            # 处理空或不可解析：返回统一结构
            return {"role": "assistant", "content": ""}
    try:
        resp = await _retry_async(_do)
        if not resp.get("content"):
            print("Gemini API 响应格式不正确或无内容。")
            return {"role": "assistant", "content": "无法解析Gemini响应或无有效内容。"}
        return resp
    except aiohttp.ClientResponseError as cre:
        # 如果模型不可用/不存在，尝试自动回退
        if cre.status == 429:
            # 标记当前 key 暂时不可用（使用 Retry-After 或 20s）并换下一个 key
            retry_after = 20.0
            try:
                if cre.headers and 'Retry-After' in cre.headers:
                    retry_after = float(cre.headers['Retry-After'])
            except Exception:
                pass
            _key_block_until[gemini_api_key] = asyncio.get_event_loop().time() + retry_after
            print(f"Gemini key 封禁 {retry_after:.1f}s (429)")
            return await call_gemini_api(input_data, selected_model, temperature=temperature, max_tokens=max_tokens)
        if cre.status in (400, 404):
            fallback_model = "gemini-2.0-flash" if selected_model != "gemini-2.0-flash" else "gemini-1.5-flash"
            print(f"Gemini 模型 '{selected_model}' 可能不可用(status={cre.status})，尝试回退到 '{fallback_model}'。")
            try:
                return await call_gemini_api(input_data, fallback_model, temperature=temperature, max_tokens=max_tokens)
            except Exception as e2:
                print(f"回退模型也失败: {repr(e2)}")
        print(f"Gemini API 请求失败(状态码={cre.status}): {cre.message}")
        raise
    except Exception as error:
        print(f"Gemini API 请求失败(含重试): {repr(error)}, URL: {url}")
        raise


async def call_vapi_api(input_data, selected_model, *, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    """调用 VAPI 提供商。以 VAPI/ 前缀区分本地模型名，实际请求去掉前缀。
    不使用全局速率限制与 per-key 轮询，按其官方并发设计直接请求。
    """
    # 选择第一个可用 key（无需轮询/退避占位）
    vapi_key = vapi_api_keys[0] if vapi_api_keys else ''
    # 去掉前缀，得到真实模型名
    underlying_model = selected_model.split('/', 1)[1] if is_vapi_model(selected_model) else selected_model

    data = {
        "model": underlying_model,
        "messages": input_data,
        "stream": False,
    }
    if temperature is not None:
        try:
            data["temperature"] = float(temperature)
        except Exception:
            pass
    if max_tokens is not None:
        try:
            mt = int(max_tokens)
            if mt > 0:
                data["max_tokens"] = mt
        except Exception:
            pass
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {vapi_key}',
    }

    proxy = await check_proxy()

    url = f"{VAPIBaseUrl}/chat/completions"
    #url = f"/chat/completions"

    async def _do():
        # VAPI 不调用全局 _wait_rate_limit
        session = await get_session()
        async with session.post(url, json=data, headers=headers, proxy=proxy) as response:
            response.raise_for_status()
            resp_json = await response.json()
            # OpenAI 兼容结构
            role = resp_json.get("choices", [{}])[0].get("message", {}).get("role", "assistant")
            content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            user_tail = input_data[-1:][0].get("content", "").splitlines()[-1]
            content_tail = content.splitlines()[-1][-20:] if content else ""
            print(f"VAPI {user_tail} ==> 响应: {content_tail}")
            return {"role": role, "content": content}

    try:
        req_tail = input_data[-1:][0].get("content", "").splitlines()[-1]
        print(f"VAPI 请求: data: ...{req_tail}")
        return await _retry_async(_do)
    except aiohttp.ClientResponseError as cre:
        # 简单打印，VAPI 不进行 key 封禁与轮询
        print(f"VAPI 请求失败(状态码={cre.status}): {cre.message}")
        raise
    except Exception as error:
        print(f"VAPI 请求失败(含重试): {repr(error)}, URL: {url}")
        raise
