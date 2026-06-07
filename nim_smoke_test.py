"""NVIDIA NIM connectivity smoke test (Task 0.4).

Proves that the configured NVIDIA API key, model, and base URL work end-to-end
via LangChain's ``ChatNVIDIA`` BEFORE any RAG code is built on top of them.

It deliberately exercises the same config path the application uses
(``backend/config.py``), so a success here means the app's LLM factory will work
too. The chat model is selected entirely from ``.env`` - swap ``NVIDIA_MODEL``
there and re-run, no code change required.

Run from the project root with the venv active::

    python nim_smoke_test.py

Exit codes: 0 = success, 1 = connectivity/credential/model failure,
2 = unexpected error.
"""

from __future__ import annotations

import sys
import time

# Ensure the project root is importable when run as a plain script.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import settings  # noqa: E402


PROMPT = "Say hello in 3 words"


def _redact_key(key: str) -> str:
    """Show only enough of the key to confirm which one is loaded."""
    if len(key) <= 8:
        return "****"
    return f"{key[:6]}...{key[-4:]}"


def _classify_error(exc: Exception) -> tuple[str, str]:
    """Map an exception to (short label, actionable hint) for diagnostics."""
    text = str(exc).lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)

    if status == 401 or "401" in text or "unauthorized" in text or "invalid api key" in text:
        return (
            "Authentication failed (401)",
            "The NVIDIA_API_KEY is missing or invalid. Get a fresh key at "
            "https://build.nvidia.com and update .env.",
        )
    if status == 404 or "404" in text or "not found" in text or "does not exist" in text:
        return (
            f"Model unavailable ({status or 404})",
            f"The model '{settings.NVIDIA_MODEL}' is not available to your account. "
            "Pick another from https://build.nvidia.com and change NVIDIA_MODEL in "
            ".env, then re-run (no code change needed).",
        )
    if status == 429 or "429" in text or "rate limit" in text or "too many requests" in text:
        return (
            "Rate limited (429)",
            "The NIM free tier is throttling requests. Wait a moment and re-run. "
            "Note: Phase 3/4 will add retry/backoff for this.",
        )
    if "connection" in text or "timed out" in text or "timeout" in text or "name resolution" in text:
        return (
            "Network error",
            f"Could not reach {settings.NVIDIA_BASE_URL}. Check your internet "
            "connection / proxy and the NVIDIA_BASE_URL value in .env.",
        )
    return ("Unexpected error", "See the full exception above for details.")


def main() -> int:
    print("=" * 64)
    print("NVIDIA NIM connectivity smoke test")
    print("=" * 64)

    # 1) Credentials check (clear, actionable error if missing).
    try:
        api_key = settings.require_nvidia_api_key()
    except RuntimeError as exc:
        print(f"\n[FAIL] {exc}")
        return 1

    print(f"  Model    : {settings.NVIDIA_MODEL}")
    print(f"  Base URL : {settings.NVIDIA_BASE_URL}")
    print(f"  API key  : {_redact_key(api_key)}")
    print(f"  Prompt   : {PROMPT!r}")
    print("-" * 64)

    # 2) Import the client (kept inside main so a missing dep gives a clean msg).
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
    except ImportError as exc:
        print(f"\n[FAIL] Could not import ChatNVIDIA: {exc}")
        print("       Install dependencies: pip install -r requirements.txt")
        return 2

    # 3) Build the model purely from env-driven settings.
    llm = ChatNVIDIA(
        model=settings.NVIDIA_MODEL,
        base_url=settings.NVIDIA_BASE_URL,
        api_key=api_key,
        temperature=0.1,
        max_completion_tokens=32,
    )

    # 4) Invoke and measure latency.
    print("Sending request to NVIDIA NIM ...")
    start = time.perf_counter()
    try:
        response = llm.invoke(PROMPT)
    except Exception as exc:  # noqa: BLE001 - we classify and report all failures
        elapsed = time.perf_counter() - start
        label, hint = _classify_error(exc)
        print(f"\n[FAIL] {label} after {elapsed:.2f}s")
        print(f"       Details: {exc}")
        print(f"       Hint: {hint}")
        return 1
    elapsed = time.perf_counter() - start

    content = getattr(response, "content", str(response)).strip()
    print("-" * 64)
    print(f"[OK] Completion received in {elapsed:.2f}s:")
    print(f"     {content}")
    print("=" * 64)
    print("Smoke test passed. NIM key/model/base-URL are working.")
    print("Tip: change NVIDIA_MODEL in .env and re-run to swap models (no code change).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
