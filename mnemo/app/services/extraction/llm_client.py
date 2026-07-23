"""Shared Groq/OpenAI-compatible LLM client for extraction."""

from __future__ import annotations

import logging
from typing import Any

from mnemo.app.core.config import settings

logger = logging.getLogger(__name__)

_instructor_client: Any | None = None
_connection_checked = False


def llm_api_key() -> str:
    """Return the configured LLM API key (Groq preferred, OpenAI fallback)."""
    return (settings.groq_api_key or settings.openai_api_key or "").strip()


def llm_key_source() -> str:
    if settings.groq_api_key.strip():
        return "GROQ_API_KEY"
    if settings.openai_api_key.strip():
        return "OPENAI_API_KEY"
    return "none"


def llm_key_format_warning() -> str | None:
    """Return a human-readable warning when the key looks misconfigured."""
    key = llm_api_key()
    if not key:
        return "No LLM API key set. Add GROQ_API_KEY=gsk_... to .env (get one at console.groq.com)."
    if key.startswith("sk-") and not key.startswith("gsk_"):
        return (
            f"{llm_key_source()} looks like an OpenAI key (sk-...). "
            "Groq keys start with gsk_. Set GROQ_API_KEY to your Groq key."
        )
    return None


def get_instructor_client():
    """Lazy Instructor client bound to Groq's OpenAI-compatible API."""
    global _instructor_client
    if _instructor_client is None:
        import instructor
        from openai import AsyncOpenAI

        key = llm_api_key()
        if not key:
            raise RuntimeError("LLM API key not configured")
        _instructor_client = instructor.from_openai(
            AsyncOpenAI(api_key=key, base_url=settings.groq_base_url),
            mode=instructor.Mode.JSON,
        )
    return _instructor_client


def reset_llm_client() -> None:
    """Clear cached client (e.g. after key rotation)."""
    global _instructor_client, _connection_checked
    _instructor_client = None
    _connection_checked = False


def log_llm_skip(reason: str) -> None:
    logger.warning("LLM skipped: %s", reason)


def log_llm_failure(stage: str, exc: Exception) -> None:
    msg = str(exc).lower()
    if "invalid_api_key" in msg or "invalid api key" in msg:
        logger.error(
            "LLM %s failed: invalid API key (%s). %s",
            stage,
            llm_key_source(),
            llm_key_format_warning() or "Check your key at console.groq.com.",
        )
    elif "connection" in msg or "timeout" in msg:
        logger.error("LLM %s failed: network error — %s", stage, exc)
    else:
        logger.error("LLM %s failed: %s", stage, exc)


async def verify_llm_connection() -> bool:
    """Ping Groq once at startup; log result."""
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

    client = AsyncOpenAI(api_key=key, base_url=settings.groq_base_url)
    try:
        response = await client.chat.completions.create(
            model=settings.extraction_model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=5,
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
