"""
Transcriber interface — one method, async, returns plain text or None.
"""


class Transcriber:
    async def transcribe(self, path: str, language: "str | list[str] | None" = None) -> str | None:
        """Return the transcript of the audio at *path*, or ``None`` on failure.

        *language* may be an ISO code (force that language), ``None`` /
        ``"auto"`` (auto-detect), or a **list of ISO codes** — a priority
        list: transcribe in the highest-priority language that fits the
        clip, falling back to auto-detect when none does. Confidence-gated
        priority/fallback needs a per-language detection signal, which only
        the local backend has; the OpenAI backend treats a list as plain
        auto-detect.

        Implementations must not raise; a failure (missing file, network
        error, model error) returns ``None`` so the caller can fall back
        to the original audio without the transcript companion.
        """
        raise NotImplementedError
