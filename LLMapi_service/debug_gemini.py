import asyncio
import sys
from typing import List, Dict

from gptservice import call_gemini_api, is_gemini_model, close_session


async def _run(model: str) -> None:
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say only: OK"}
    ]
    try:
        if not is_gemini_model(model):
            print(f"模型不是Gemini: {model}")
            return
        resp = await call_gemini_api(msgs, model)
        print("返回：", resp)
    except Exception as e:
        print("调用失败：", repr(e))
    finally:
        try:
            await close_session()
        except Exception:
            pass


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"
    asyncio.run(_run(model))


if __name__ == "__main__":
    main()


