"""
Shared yt-dlp-backed enricher base.

Facebook, Instagram, and TikTok have no convenient free JSON API, so we
lean on yt-dlp to download public media (and, where useful, the caption).
Subclasses just set a host regex, an icon, an author label, and whether
to fetch the caption.
"""

import asyncio
import os
import shutil
import uuid

from bridge.enrichers.base import BaseEnricher, EnrichResult
from bridge.utils.logger import get_logger

logger = get_logger("enrich.ytdlp")

_YTDLP_TIMEOUT = 180.0
try:
    _YTDLP_CONCURRENCY = max(1, int(os.environ.get("BRIDGER_YTDLP_CONCURRENCY", "2")))
except ValueError:
    _YTDLP_CONCURRENCY = 2
_YTDLP_SEMAPHORE = asyncio.Semaphore(_YTDLP_CONCURRENCY)


class YtDlpEnricher(BaseEnricher):
    """Base for media-first enrichers that shell out to yt-dlp."""

    name = "ytdlp"
    icon = "🔗"
    author_label = ""
    host_re = None          # subclass: compiled regex, .search() must match the URL
    fetch_caption = False   # subclass: True to also pull the title/caption

    def matches(self, url: str) -> bool:
        return bool(self.host_re.search(url)) if self.host_re else False

    async def enrich(self, url: str, cfg: dict, dest_dir: str) -> "EnrichResult | None":
        if not cfg.get("download_media"):
            return None
        m = self.host_re.search(url) if self.host_re else None
        if not m: return None
        clean_url = m.group(0)

        if shutil.which("yt-dlp") is None:
            logger.warning("yt-dlp not found on PATH — cannot enrich %s media", self.name)
            return None

        os.makedirs(dest_dir, exist_ok=True)
        max_bytes = int(float(cfg.get("max_media_mb", 50)) * 1024 * 1024)
        max_mb = max(1, int(max_bytes / (1024 * 1024)))
        # A unique prefix prevents concurrent enrichment of the same post from
        # racing over one output path.
        job_id = uuid.uuid4().hex
        out_tmpl = os.path.join(dest_dir, f"{self.name}_{job_id}_%(id)s.%(ext)s")
        args = [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--max-filesize", f"{max_mb}M",
            "-o", out_tmpl,
        ]
        # Optional cookies file — required for login-walled content
        # (most Facebook / Instagram posts). Export a Netscape-format
        # cookies.txt from a logged-in browser and point cookies_file at it.
        cookies = cfg.get("cookies_file")
        if cookies and os.path.isfile(cookies):
            args += ["--cookies", cookies]
        elif cookies:
            logger.warning("cookies_file %s not found — proceeding without auth", cookies)
        # Caption (if requested) prints at info-extraction; filepath prints
        # after the file is moved into place. yt-dlp emits them in event
        # order, so the filepath is always the LAST stdout line.
        if self.fetch_caption:
            args += ["--print", "%(title)s"]
        args += ["--print", "after_move:filepath", clean_url]

        async with _YTDLP_SEMAPHORE:
            path, caption = await self._run(args)
        if not path or not os.path.isfile(path):
            return None
        # Treat subprocess stdout as untrusted. The only acceptable result is
        # a regular file beneath the staging directory created by this app.
        abs_dir = os.path.realpath(dest_dir)
        abs_path = os.path.realpath(path)
        try:
            inside = os.path.commonpath([abs_dir, abs_path]) == abs_dir
        except ValueError:
            inside = False
        expected_prefix = f"{self.name}_{job_id}_"
        if not inside or not os.path.basename(abs_path).startswith(expected_prefix):
            logger.error("yt-dlp returned an unsafe output path for %s", self.name)
            return None
        try:
            actual_size = os.path.getsize(abs_path)
        except OSError:
            return None
        if actual_size > max_bytes:
            logger.warning(
                "yt-dlp output for %s is %.1f MB > %.1f MB cap; discarding",
                self.name, actual_size / (1024 * 1024), max_bytes / (1024 * 1024),
            )
            try: os.remove(abs_path)
            except OSError: pass
            return None
        path = abs_path
        ext = os.path.splitext(path)[1] or ".mp4"
        return EnrichResult(
            author=self.author_label,
            text=(caption or "").strip(),
            media=[(path, ext)],
            source_url=url,
            icon=self.icon,
        )

    async def _run(self, args: list[str]) -> tuple:
        """Run yt-dlp. Returns ``(filepath_or_None, caption)``."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            logger.warning("failed to launch yt-dlp: %s", exc)
            return None, ""
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_YTDLP_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("yt-dlp timed out after %.0fs (%s) — killing", _YTDLP_TIMEOUT, self.name)
            try: proc.kill()
            except ProcessLookupError: pass
            try: await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError): pass
            return None, ""
        if proc.returncode != 0:
            logger.info("yt-dlp failed for %s (rc=%d): %s", self.name, proc.returncode,
                        stderr[:300].decode("utf-8", errors="replace"))
            return None, ""
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        if not lines:
            return None, ""
        path = lines[-1].strip()
        caption = "\n".join(lines[:-1]).strip() if self.fetch_caption else ""
        return path, caption
