"""
OpenAI Whisper backend — POST audio to /v1/audio/transcriptions.

Uses aiohttp (already a project dep). No need for the ``openai`` SDK.
Streams the file from disk rather than reading it into memory.
"""

import os

import aiohttp

from bridge.transcription.base import Transcriber
from bridge.utils.logger import get_logger

logger = get_logger("transcription.openai")

_API_URL = "https://api.openai.com/v1/audio/transcriptions"
_OPENAI_MAX_BYTES = 25 * 1024 * 1024  # OpenAI's documented upload cap
_TIMEOUT = 180.0


class OpenAITranscriber(Transcriber):
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.api_key = api_key
        self.model = model
        self._priority_notice_logged = False
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def transcribe(self, path: str, language: "str | list[str] | None" = None) -> str | None:
        if not path or not os.path.isfile(path):
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        if size > _OPENAI_MAX_BYTES:
            logger.warning(
                "voice file %s is %.1f MB > OpenAI cap %.0f MB — skipping",
                os.path.basename(path), size / (1024 * 1024), _OPENAI_MAX_BYTES / (1024 * 1024),
            )
            return None

        # Priority list: the OpenAI API exposes no per-language confidence,
        # so we can't do the confidence-gated forcing the local backend does.
        # We honor the spirit — prefer the listed languages — by letting
        # Whisper auto-detect (which handles all common languages well) and
        # never forcing, so a genuinely foreign clip isn't garbled. Real
        # priority/fallback is a local-backend capability.
        if isinstance(language, (list, tuple)) and language:
            if not self._priority_notice_logged:
                logger.info(
                    "OpenAI backend: language priority %s treated as auto-detect "
                    "(confidence-gated forcing is local-only)", list(language),
                )
                self._priority_notice_logged = True
            language = None

        lang = language if (language and language != "auto") else None
        return await self._post(path, lang)

    async def _post(self, path: str, language: "str | None") -> "str | None":
        """POST the audio once and return the plain transcript (or ``None``)."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with open(path, "rb") as fh:
                data = aiohttp.FormData()
                data.add_field(
                    "file", fh,
                    filename=os.path.basename(path),
                    content_type="application/octet-stream",
                )
                data.add_field("model", self.model)
                if language and language != "auto":
                    data.add_field("language", language)
                # response_format=text — server returns the plain transcript
                # as the body, no JSON parse needed.
                data.add_field("response_format", "text")
                session = self._get_session()
                async with session.post(_API_URL, data=data, headers=headers) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        logger.warning(
                            "OpenAI Whisper failed (%d): %s", resp.status, body[:300],
                        )
                        return None
                    return body.strip() or None
        except (aiohttp.ClientError, OSError) as exc:
            logger.warning("OpenAI Whisper transport error: %s", exc)
            return None
