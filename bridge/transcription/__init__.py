"""
Voice-message transcription.

Two backends, each loaded lazily so the dependency only gets imported
when actually used:

* ``openai`` — calls the OpenAI Whisper API (no extra Python dep; uses
  aiohttp, which the bridge already pulls in). Needs an API key.
* ``local`` — runs faster-whisper on the host CPU. No key, no network.
  Adds ~50 MB of pip deps + a ~150 MB-1.5 GB model download on first
  use, so it's optional.

The chosen backend is configured per-platform under
``platforms.<name>.voice_transcription`` and resolved by
:func:`get_transcriber`. Returns ``None`` when the config is missing
required fields — callers then skip transcription silently.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from bridge.transcription.base import Transcriber
from bridge.utils.logger import get_logger

logger = get_logger("transcription")

try:
    _MAX_CONCURRENT = max(1, int(os.environ.get("BRIDGER_TRANSCRIPTION_CONCURRENCY", "1")))
except ValueError:
    _MAX_CONCURRENT = 1
_TRANSCRIPTION_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT)
_TRANSCRIBERS: dict[tuple, Transcriber] = {}


@asynccontextmanager
async def transcription_slot():
    """Bound expensive voice download/inference/API work process-wide."""
    async with _TRANSCRIPTION_SEMAPHORE:
        yield


def get_transcriber(cfg: dict) -> "Transcriber | None":
    """Resolve a :class:`Transcriber` from the parsed voice_transcription cfg.

    Returns ``None`` if the provider is unknown or its required keys
    are missing — never raises. Lazy-imports the backend module so
    OpenAI-only deployments don't pay for faster-whisper and vice versa.
    """
    provider = (cfg.get("provider") or "openai").lower()
    if provider == "openai":
        key = cfg.get("openai_api_key")
        if not key:
            logger.warning("voice_transcription provider=openai requires openai_api_key — disabled")
            return None
        from bridge.transcription.openai_whisper import OpenAITranscriber
        cache_key = ("openai", key, cfg.get("openai_model") or "whisper-1")
        if cache_key not in _TRANSCRIBERS:
            _TRANSCRIBERS[cache_key] = OpenAITranscriber(
                api_key=key,
                model=cfg.get("openai_model") or "whisper-1",
            )
        return _TRANSCRIBERS[cache_key]
    if provider in ("local", "faster-whisper", "faster_whisper", "whisper"):
        from bridge.transcription.local_whisper import LocalTranscriber
        cache_key = ("local", cfg.get("local_model") or "small")
        if cache_key not in _TRANSCRIBERS:
            _TRANSCRIBERS[cache_key] = LocalTranscriber(model_name=cache_key[1])
        return _TRANSCRIBERS[cache_key]
    logger.warning("voice_transcription: unknown provider %r — disabled", provider)
    return None


async def close_transcribers() -> None:
    """Close reusable backend resources during application shutdown."""
    instances = list(_TRANSCRIBERS.values())
    _TRANSCRIBERS.clear()
    for instance in instances:
        close = getattr(instance, "close", None)
        if close is None:
            continue
        result = close()
        if asyncio.iscoroutine(result):
            await result


__all__ = [
    "Transcriber", "get_transcriber", "transcription_slot",
    "close_transcribers",
]
