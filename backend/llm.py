"""NIM chat-model factory (Task 3.1).

One place to construct the generation LLM from env (model/key/base-URL all live
in ``.env`` and can be swapped with no code change). Wrapped with retry/backoff
so transient NIM free-tier 429s and 5xx errors don't break a turn.

``get_llm()`` returns a LangChain Runnable (a retry-wrapped ``ChatNVIDIA``) that
plugs directly into the history-aware retriever and the stuff-documents chain.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.runnables import Runnable
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from backend.config import settings


@lru_cache(maxsize=1)
def get_llm() -> Runnable:
    """Return the process-wide NIM chat model with retry/backoff.

    Reads ``NVIDIA_API_KEY``/``NVIDIA_MODEL``/``NVIDIA_BASE_URL`` and the
    ``LLM_*`` sampling params from settings. Raises a clear error if the key is
    missing (via ``require_nvidia_api_key``), so failures are actionable.
    """
    llm = ChatNVIDIA(
        model=settings.NVIDIA_MODEL,
        base_url=settings.NVIDIA_BASE_URL,
        api_key=settings.require_nvidia_api_key(),
        temperature=settings.LLM_TEMPERATURE,
        max_completion_tokens=settings.LLM_MAX_TOKENS,
    )
    # Exponential backoff with jitter on transient errors (429 / 5xx / network).
    return llm.with_retry(
        stop_after_attempt=4,
        wait_exponential_jitter=True,
    )


if __name__ == "__main__":  # Manual check: python -m backend.llm
    resp = get_llm().invoke("Say hi in 3 words")
    print(getattr(resp, "content", resp))
