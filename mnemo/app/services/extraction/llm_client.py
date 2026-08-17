"""Shared OpenAI-compatible LLM client for extraction (NVIDIA NIM / Groq)."""

from __future__ import annotations

import logging
from typing import Any

from mnemo.app.core.config import settings

logger = logging.getLogger(__name__)

_instructor_client: Any | None = None
_raw_client: Any | None = None
_connection_checked = False


def llm_api_key() -> str:
    """Return the configured LLM API key (NVIDIA preferred)."""
    return (
        settings.nvidia_api_key or settings.groq_api_key or settings.openai_api_key or ""
    ).strip()


def llm_base_url() -> str:
    if settings.nvidia_api_key.strip():
        return settings.nvidia_base_url
    if settings.groq_api_key.strip():
        return settings.groq_base_url
    return settings.nvidia_base_url


def llm_key_source() -> str:
    if settings.nvidia_api_key.strip():
        return "NVIDIA_API_KEY"
    if settings.groq_api_key.strip():
        return "GROQ_API_KEY"
    if settings.openai_api_key.strip():
        return "OPENAI_API_KEY"
    return "none"


def llm_key_format_warning() -> str | None:
    """Return a human-readable warning when the key looks misconfigured."""
    key = llm_api_key()
    if not key:
        return "No LLM API key set. Add NVIDIA_API_KEY=nvapi-... to .env."
    source = llm_key_source()
    if source == "NVIDIA_API_KEY" and not key.startswith("nvapi-"):
        return "NVIDIA_API_KEY should start with nvapi-."
    if source == "GROQ_API_KEY" and key.startswith("sk-") and not key.startswith("gsk_"):
        return (
            "GROQ_API_KEY looks like an OpenAI key (sk-...). "
            "Groq keys start with gsk_."
        )
    return None


def get_instructor_client():
    """Lazy Instructor client bound to the configured OpenAI-compatible API."""
    global _instructor_client
    if _instructor_client is None:
        import instructor
        from openai import AsyncOpenAI

        key = llm_api_key()
        if not key:
            raise RuntimeError("LLM API key not configured")
        _instructor_client = instructor.from_openai(
            AsyncOpenAI(api_key=key, base_url=llm_base_url()),
            mode=instructor.Mode.JSON,
        )
    return _instructor_client


def get_raw_client():
    """Lazy plain OpenAI-compatible client for callers that parse raw responses."""
    global _raw_client
    if _raw_client is None:
        from openai import AsyncOpenAI

        key = llm_api_key()
        if not key:
            raise RuntimeError("LLM API key not configured")
        _raw_client = AsyncOpenAI(api_key=key, base_url=llm_base_url())
    return _raw_client


def reset_llm_client() -> None:
    """Clear cached clients (e.g. after key rotation)."""
    global _instructor_client, _raw_client, _connection_checked
    _instructor_client = None
    _raw_client = None
    _connection_checked = False


def log_llm_skip(reason: str) -> None:
    logger.warning("LLM skipped: %s", reason)


def log_llm_failure(stage: str, exc: Exception) -> None:
    msg = str(exc).lower()
    if "invalid_api_key" in msg or "invalid api key" in msg or "unauthorized" in msg:
        logger.error(
            "LLM %s failed: invalid API key (%s). %s",
            stage,
            llm_key_source(),
            llm_key_format_warning() or "Check NVIDIA_API_KEY in .env.",
        )
    elif "connection" in msg or "timeout" in msg:
        logger.error("LLM %s failed: network error — %s", stage, exc)
    else:
        logger.error("LLM %s failed: %s", stage, exc)


async def verify_llm_connection() -> bool:
    """Ping the configured LLM once at startup; log result."""
    global _connection_checked
    if _connection_checked:
        return bool(llm_api_key())

    _connection_checked = True
    warn = llm_key_format_warning()
    if warn:
        logger.warning(warn)

    key = llm_api_key()
    if not key:
        logger.warning("LLM extraction disabled — no API key configured")
        return False

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=key, base_url=llm_base_url())
    try:
        response = await client.chat.completions.create(
            model=settings.extraction_model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=32,
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        logger.info(
            "LLM connection OK (%s, model=%s, probe_response=%r)",
            llm_key_source(),
            settings.extraction_model,
            text,
        )
        return True
    except Exception as exc:
        log_llm_failure("startup_probe", exc)
        return False
