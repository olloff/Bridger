"""
Local faster-whisper backend.

The import is lazy and the model is loaded once per process (cached on
the class). The model lives in RAM for the lifetime of the bridge — the
``small`` default needs about 500 MB. Use ``tiny`` or ``base`` on a
memory-constrained host; ``medium`` / ``large`` if the host has the
spare RAM and you want better accuracy.

``faster-whisper`` is NOT pinned in requirements.txt — it pulls in
ctranslate2 and torch-like deps that bloat the image. Install it on
demand: ``pip install faster-whisper``. The bridge logs a clear error
and falls back to "no transcript" if the package is missing.
"""

import asyncio
import os

from bridge.transcription.base import Transcriber
from bridge.utils.logger import get_logger

logger = get_logger("transcription.local")

# Distribution path: a language in the priority list is treated as a real
# candidate for the clip when its detected probability clears this. We then
# pick the HIGHEST-PRIORITY plausible language, honoring list order.
_LANGUAGE_PLAUSIBLE_FLOOR = 0.2

# No-distribution fallback (older faster-whisper without all_language_probs):
# force the top-priority language only when the single auto detection is
# below this confidence. faster-whisper reports ~0.95+ on clear speech, so a
# low value here really does mean "unsure".
_LANGUAGE_CONFIDENCE_FLOOR = 0.6


class LocalTranscriber(Transcriber):
    _model = None
    _model_name = None
    _model_lock = asyncio.Lock()

    def __init__(self, model_name: str = "small"):
        self.model_name = model_name

    async def _ensure_model(self):
        async with LocalTranscriber._model_lock:
            if (
                LocalTranscriber._model is not None
                and LocalTranscriber._model_name == self.model_name
            ):
                return LocalTranscriber._model
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                logger.error(
                    "voice_transcription provider=local needs faster-whisper — "
                    "pip install faster-whisper"
                )
                return None
            # Model load is heavy and synchronous; offload to a thread so
            # the asyncio loop keeps spinning.
            try:
                model = await asyncio.to_thread(
                    WhisperModel, self.model_name, device="cpu", compute_type="int8",
                )
            except Exception as exc:
                logger.error("faster-whisper failed to load model %r: %s", self.model_name, exc)
                return None
            LocalTranscriber._model = model
            LocalTranscriber._model_name = self.model_name
            return model

    async def transcribe(self, path: str, language: "str | list[str] | None" = None) -> str | None:
        if not path or not os.path.isfile(path):
            return None
        model = await self._ensure_model()
        if model is None:
            return None

        priority = list(language) if isinstance(language, (list, tuple)) else None
        forced = None if priority else (language if (language and language != "auto") else None)

        def _run():
            if priority:
                return _transcribe_priority(model, path, priority)
            segments, _info = model.transcribe(path, language=forced, beam_size=1)
            return _join(segments)

        try:
            text = await asyncio.to_thread(_run)
        except Exception as exc:
            logger.warning("local whisper failed on %s: %s", os.path.basename(path), exc)
            return None
        return text or None


def _join(segments) -> str:
    return " ".join(s.text.strip() for s in segments if s.text).strip()


def _pick_language(priority: list, all_probs, top_prob: float) -> "str | None":
    """Choose which language to decode in, or ``None`` to keep the auto result.

    With a full per-language distribution (faster-whisper's
    ``all_language_probs``) we honor the *order* of the priority list: pick
    the highest-priority language that's a plausible candidate for the clip.
    When none is plausible we return ``None`` — that's the auto-fallback.

    Without a distribution (older faster-whisper) we fall back to a single
    signal: force the top-priority language only when the auto detection is
    low-confidence, otherwise keep the auto result."""
    if all_probs:
        pm = dict(all_probs)
        for lang in priority:
            if pm.get(lang, 0.0) >= _LANGUAGE_PLAUSIBLE_FLOOR:
                return lang
        return None
    return priority[0] if top_prob < _LANGUAGE_CONFIDENCE_FLOOR else None


def _transcribe_priority(model, path: str, priority: list) -> str:
    """Transcribe honoring a language *priority* list, decoding at most twice.

    faster-whisper detects the language during ``transcribe()`` setup and
    populates ``info`` *before* the (lazy) segment generator is consumed, so
    the first pass costs only detection until we decide. If the chosen
    language matches the auto-detected one we consume that generator;
    otherwise we start a fresh decode forcing the choice. When nothing in the
    list is a plausible candidate we keep the auto-detected result."""
    segments, info = model.transcribe(path, language=None, beam_size=1)
    detected = getattr(info, "language", None)
    top_prob = getattr(info, "language_probability", 0.0) or 0.0
    all_probs = getattr(info, "all_language_probs", None)
    chosen = _pick_language(priority, all_probs, top_prob)
    if chosen is None or chosen == detected:
        return _join(segments)
    logger.info("local whisper: preferring %s over detected %s (%.2f)", chosen, detected, top_prob)
    segments, _info = model.transcribe(path, language=chosen, beam_size=1)
    return _join(segments)
