"""
X / Twitter enricher.

Uses the free community fxtwitter API (https://github.com/FixTweet/FxTwitter)
to fetch a post's text + media URLs without auth or the paid X API.
Media is downloaded directly from Twitter's CDN (validated host).

Thread unroll (``enrich.unroll: true``): the fxtwitter payload carries
``replying_to_status``, so from the linked tweet we can walk UP the reply
chain, collecting same-author ancestors back to the thread root, and merge
them into one chronological result. Walking up is the only direction the
free API supports — a link to the FIRST tweet of a thread has no ancestors
and degrades to today's single-post behavior; a link to the last/middle
tweet reconstructs everything above it.
"""

import os
import re
import json
from urllib.parse import urljoin, urlparse

from bridge.enrichers.base import BaseEnricher, EnrichResult
from bridge.utils.media import get_unique_filepath
from bridge.utils.logger import get_logger

logger = get_logger("enrich.twitter")

_STATUS_RE = re.compile(
    r"https?://(?:www\.|mobile\.)?(?:twitter\.com|x\.com)/([A-Za-z0-9_]+)/status/(\d+)",
    re.IGNORECASE,
)
_FX_API = "https://api.fxtwitter.com/{screen}/status/{tid}"
# Media must come from Twitter's own CDN — defense in depth so a tampered
# API response can't make us fetch an arbitrary host.
_ALLOWED_MEDIA_HOSTS = ("pbs.twimg.com", "video.twimg.com")
_FETCH_TIMEOUT = 20.0
_DOWNLOAD_TIMEOUT = 120.0
_CHUNK = 64 * 1024
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 3


class TwitterEnricher(BaseEnricher):
    name = "twitter"
    icon = "🐦"

    def __init__(self):
        self._session = None

    async def _get_session(self):
        import aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def matches(self, url: str) -> bool:
        return bool(_STATUS_RE.search(url))

    async def enrich(self, url: str, cfg: dict, dest_dir: str) -> "EnrichResult | None":
        m = _STATUS_RE.search(url)
        if not m: return None
        screen, tid = m.group(1), m.group(2)
        data = await self._fetch_json(screen, tid)
        if not data: return None
        tweet = data.get("tweet") or {}
        author_obj = tweet.get("author") or {}
        author = author_obj.get("name") or author_obj.get("screen_name") or screen

        # Chronological list of tweets to render: [root, ..., linked] when
        # unrolling, else just the linked tweet.
        tweets = [tweet]
        if cfg.get("unroll"):
            ancestors = await self._walk_up(tweet, cfg)
            tweets = ancestors + tweets

        text = "\n\n".join(
            t for t in ((tw.get("text") or "").strip() for tw in tweets) if t
        )

        media: list = []
        if cfg.get("download_media"):
            for tw in tweets:
                media.extend(await self._download_media(tw, cfg, dest_dir))

        if not text and not media:
            return None
        return EnrichResult(author=author, text=text, media=media, source_url=url, icon=self.icon)

    async def _walk_up(self, tweet: dict, cfg: dict) -> list:
        """Collect same-author ancestors of *tweet*, root first.

        Follows ``replying_to_status`` upward, stopping at the thread root,
        an author change (that's a reply into someone else's conversation,
        not a self-thread), a fetch failure (partial unroll — keep what we
        have), a repeated ID (cycle guard), or the configured cap."""
        screen = ((tweet.get("author") or {}).get("screen_name") or "").lower()
        if not screen: return []
        try: cap = int(cfg.get("max_thread_tweets", 25))
        except (TypeError, ValueError): cap = 25

        ancestors: list = []
        seen = {str(tweet.get("id") or "")}
        current = tweet
        while len(ancestors) + 1 < cap:
            parent_id = str(current.get("replying_to_status") or "")
            if not parent_id or parent_id in seen:
                break
            # Pre-check: fxtwitter names who's being replied to. A different
            # screen name means the chain leaves this author — stop before
            # spending a fetch on it.
            replying_to = (current.get("replying_to") or "").lower()
            if replying_to and replying_to != screen:
                break
            seen.add(parent_id)
            data = await self._fetch_json(screen, parent_id)
            parent = (data or {}).get("tweet") or {}
            if not parent:
                logger.info("thread walk-up stopped at %s (fetch failed) — partial unroll", parent_id)
                break
            parent_screen = ((parent.get("author") or {}).get("screen_name") or "").lower()
            if parent_screen != screen:
                break
            ancestors.append(parent)
            current = parent
        ancestors.reverse()  # fetched linked→root; render root→linked
        return ancestors

    async def _fetch_json(self, screen: str, tid: str):
        import aiohttp
        api_url = _FX_API.format(screen=screen, tid=tid)
        try:
            timeout = aiohttp.ClientTimeout(total=_FETCH_TIMEOUT)
            sess = await self._get_session()
            # The API endpoint is fixed; do not follow an unexpected redirect
            # to a different network location.
            async with sess.get(api_url, timeout=timeout, allow_redirects=False) as r:
                if r.status != 200:
                    logger.info("fxtwitter returned %d for %s/%s", r.status, screen, tid)
                    return None
                if r.content_length and r.content_length > _MAX_JSON_BYTES:
                    logger.warning("fxtwitter response exceeds JSON cap")
                    return None
                raw = await r.content.read(_MAX_JSON_BYTES + 1)
                if len(raw) > _MAX_JSON_BYTES:
                    logger.warning("fxtwitter response exceeds JSON cap")
                    return None
                return json.loads(raw)
        except Exception as exc:
            logger.warning("fxtwitter fetch failed for %s/%s: %s", screen, tid, exc)
            return None

    async def _download_media(self, tweet: dict, cfg: dict, dest_dir: str) -> list:
        import aiohttp
        media = tweet.get("media") or {}
        candidates: list[tuple[str, str]] = []
        for photo in (media.get("photos") or []):
            u = photo.get("url")
            if u: candidates.append((u, ".jpg"))
        for video in (media.get("videos") or []):
            u = video.get("url")
            if u: candidates.append((u, ".mp4"))
        if not candidates: return []

        max_bytes = int(cfg.get("max_media_mb", 50) * 1024 * 1024)
        os.makedirs(dest_dir, exist_ok=True)
        out: list = []
        timeout = aiohttp.ClientTimeout(total=_DOWNLOAD_TIMEOUT)
        sess = await self._get_session()
        for u, ext in candidates:
            if not self._allowed_media_host(u):
                logger.warning("Refusing non-CDN media URL: %s", u)
                continue
            path = get_unique_filepath(dest_dir, "x_media", ext)
            if await self._download_one(sess, u, path, max_bytes, timeout):
                out.append((path, ext))
        return out

    @staticmethod
    def _allowed_media_host(u: str) -> bool:
        host = (urlparse(u).hostname or "").lower()
        return any(host == h or host.endswith("." + h) for h in _ALLOWED_MEDIA_HOSTS)

    async def _download_one(self, sess, url: str, path: str, max_bytes: int, timeout=None) -> bool:
        try:
            current = url
            for _hop in range(_MAX_REDIRECTS + 1):
                if not self._allowed_media_host(current):
                    logger.warning("Refusing non-CDN redirect URL: %s", current)
                    return False
                async with sess.get(current, timeout=timeout, allow_redirects=False) as r:
                    if 300 <= r.status < 400 and r.headers.get("Location"):
                        current = urljoin(current, r.headers["Location"])
                        continue
                    if r.status != 200:
                        logger.info("media GET %s -> %d", current, r.status)
                        return False
                    if r.content_length and r.content_length > max_bytes:
                        logger.warning("media content-length exceeds %d bytes: %s", max_bytes, current)
                        return False
                    total = 0
                    with open(path, "wb") as fh:
                        async for chunk in r.content.iter_chunked(_CHUNK):
                            total += len(chunk)
                            if total > max_bytes:
                                logger.warning("media exceeds %d bytes, aborting: %s", max_bytes, current)
                                try: os.remove(path)
                                except OSError: pass
                                return False
                            fh.write(chunk)
                    return True
            logger.warning("Too many media redirects: %s", url)
            return False
        except Exception as exc:
            logger.warning("media download failed %s: %s", url, exc)
            try: os.remove(path)
            except OSError: pass
            return False
