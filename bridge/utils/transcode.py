"""
ffmpeg-based media fitting: shrink (and, if needed, split) oversized files
so they can be uploaded to a destination with a per-file size limit.

Public entry points:

* :func:`fit_to_limit` — returns a list of freshly-encoded files, each of
  which fits within the limit. One element when the media could be
  re-encoded into a single acceptable file; several (in playback order)
  when a single file would have needed an unwatchable bitrate and the
  media was split into parts instead.
* :func:`compress_to_fit` — single-file convenience wrapper (never splits).

Both return ``None`` if the format is unsupported, ffmpeg isn't available,
or nothing acceptable could be produced. The caller owns the returned
files and must delete them when done.

Strategy for video (the common Telegram → Discord case):

1. Work out the bitrate a single file may use (``limit / duration``).
2. Pick encode settings from a quality ladder keyed on that bitrate — the
   lower the bitrate, the lower the resolution / frame rate / audio
   bitrate, so the picture degrades gracefully instead of turning into
   full-resolution macroblock soup.
3. If the single-file bitrate falls below the "still watchable" floor,
   split the video into the fewest parts that restore a decent bitrate
   (bounded by ``max_parts``) and encode each part to fit.
4. Encoders overshoot. Any output that lands over the limit is re-encoded
   at a proportionally lower bitrate (bounded retries).
"""

import asyncio
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from bridge.utils.logger import get_logger

logger = get_logger("transcode")

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif", ".3gp", ".ts", ".wmv", ".flv", ".mpg", ".mpeg"}
AUDIO_EXTS = {".mp3", ".m4a", ".ogg", ".oga", ".wav", ".opus", ".aac", ".flac", ".wma", ".amr"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

_HEADROOM_RATIO = 0.92  # encoders overshoot bitrate; aim a bit below target
_FFPROBE_TIMEOUT = 30.0


def _env_number(name: str, default: float, minimum: float) -> float:
    try: return max(minimum, float(os.environ.get(name, default)))
    except ValueError: return default


# Per-ffmpeg-invocation bound, so a malicious file can't hang the bridge.
_FFMPEG_TIMEOUT = _env_number("BRIDGER_FFMPEG_TIMEOUT", 300.0, 10.0)
_TRANSCODE_CONCURRENCY = int(_env_number("BRIDGER_TRANSCODE_CONCURRENCY", 2, 1))
_TRANSCODE_SEMAPHORE = asyncio.Semaphore(_TRANSCODE_CONCURRENCY)

# Where transcoded files are written. Defaults to ``messages/transcoded/``
# so a single docker-compose volume mapping for ``/app/messages`` covers the
# DB, source-side downloads, AND transcoder output. Override via env var
# ``BRIDGER_TRANSCODE_DIR`` if you want them somewhere else (e.g. tmpfs).
TRANSCODE_DIR = os.environ.get("BRIDGER_TRANSCODE_DIR") or os.path.join("messages", "transcoded")

DEFAULT_MAX_PARTS = 10
_MAX_PARTS_CEILING = 30
_MAX_FIT_ATTEMPTS = 3  # encodes per output file before giving up on it

# ---- video quality ladder ---------------------------------------------------
# (minimum video bitrate, cap on the SHORT side in px, frame-rate cap).
# The short side is capped (not the height) so portrait phone videos get
# the same treatment as landscape ones.
_VIDEO_TIERS = (
    (2_500_000, 1080, 60),
    (1_200_000, 720, 30),
    (600_000, 480, 30),
    (300_000, 360, 30),
    (0, 240, 24),
)
# Below this a single file is judged not worth watching; split instead
# (when splitting is allowed).
_MIN_SINGLE_VIDEO_BPS = 300_000
# Bitrate each part aims for when we do split — comfortably 480p.
_SPLIT_VIDEO_BPS = 700_000
# Hard floor. Under this even 240p falls apart; report failure instead.
_ABS_MIN_VIDEO_BPS = 100_000

# ---- audio ------------------------------------------------------------------
_MAX_AUDIO_BPS = 192_000
_MIN_SINGLE_AUDIO_BPS = 48_000
_SPLIT_AUDIO_BPS = 96_000
_ABS_MIN_AUDIO_BPS = 24_000


@dataclass(frozen=True)
class VideoPlan:
    """Encode settings for a video that has to fit a byte budget."""
    parts: int       # 1 = single file
    video_bps: int
    audio_bps: int   # 0 = source has no audio track
    short_side: int  # px cap on the shorter dimension
    fps_cap: int


@dataclass(frozen=True)
class AudioPlan:
    parts: int
    audio_bps: int
    channels: int


@dataclass(frozen=True)
class _Probe:
    duration: float
    has_video: bool = False
    has_audio: bool = False
    fps: float = 0.0


# ------------------------------------------------------------------ planning

def _video_tier(video_bps: int) -> tuple[int, int]:
    """Return ``(short_side_cap, fps_cap)`` for a video bitrate."""
    for floor, short_side, fps in _VIDEO_TIERS:
        if video_bps >= floor:
            return short_side, fps
    return _VIDEO_TIERS[-1][1], _VIDEO_TIERS[-1][2]


def _audio_share(total_bps: float, has_audio: bool) -> int:
    """How much of a video's total bitrate the audio track gets."""
    if not has_audio: return 0
    if total_bps >= 1_000_000: return 96_000
    if total_bps >= 400_000: return 64_000
    return 48_000


def _clamp_parts(value) -> int:
    try: n = int(value)
    except (TypeError, ValueError): n = DEFAULT_MAX_PARTS
    return min(max(n, 1), _MAX_PARTS_CEILING)


def plan_video(duration: float, target_bytes: int, *, has_audio: bool = True,
               max_parts: int = DEFAULT_MAX_PARTS, min_parts: int = 1) -> VideoPlan | None:
    """Decide how to fit a *duration*-second video into *target_bytes* per file.

    Prefers a single file with lowered settings; switches to several parts
    only when a single file would drop below the watchable floor (or when
    *min_parts* forces it). Returns ``None`` when even ``max_parts`` parts
    can't reach the hard minimum bitrate."""
    if duration <= 0 or target_bytes <= 0: return None
    max_parts = _clamp_parts(max_parts)
    min_parts = min(max(int(min_parts), 1), max_parts)
    target_bits = target_bytes * 8

    def _plan(parts: int) -> VideoPlan:
        total = target_bits / (duration / parts)
        audio = _audio_share(total, has_audio)
        video = int(total - audio)
        short_side, fps = _video_tier(video)
        return VideoPlan(parts, video, audio, short_side, fps)

    single = _plan(1)
    if min_parts == 1 and (single.video_bps >= _MIN_SINGLE_VIDEO_BPS or max_parts == 1):
        return single if single.video_bps >= _ABS_MIN_VIDEO_BPS else None

    wanted_total = _SPLIT_VIDEO_BPS + _audio_share(_SPLIT_VIDEO_BPS, has_audio)
    parts = math.ceil(duration * wanted_total / target_bits)
    parts = min(max(parts, min_parts, 2), max_parts)
    plan = _plan(parts)
    return plan if plan.video_bps >= _ABS_MIN_VIDEO_BPS else None


def plan_audio(duration: float, target_bytes: int, *,
               max_parts: int = DEFAULT_MAX_PARTS) -> AudioPlan | None:
    """Decide how to fit a *duration*-second audio file into *target_bytes* per file."""
    if duration <= 0 or target_bytes <= 0: return None
    max_parts = _clamp_parts(max_parts)
    usable_bits = target_bytes * 8 * 0.97  # container overhead

    def _plan(parts: int) -> AudioPlan:
        bps = int(min(usable_bits / (duration / parts), _MAX_AUDIO_BPS))
        return AudioPlan(parts, bps, 2 if bps >= 64_000 else 1)

    single = _plan(1)
    if single.audio_bps >= _MIN_SINGLE_AUDIO_BPS or max_parts == 1:
        return single if single.audio_bps >= _ABS_MIN_AUDIO_BPS else None
    parts = math.ceil(duration * _SPLIT_AUDIO_BPS / usable_bits)
    parts = min(max(parts, 2), max_parts)
    plan = _plan(parts)
    return plan if plan.audio_bps >= _ABS_MIN_AUDIO_BPS else None


# ---------------------------------------------------------------- public API

async def fit_to_limit(file_path: str, limit_bytes: int, *,
                       max_parts: int = DEFAULT_MAX_PARTS) -> list[str] | None:
    """Re-encode *file_path* into one or more files of at most *limit_bytes* each.

    Returns the new files in playback order, or ``None`` if the file isn't
    a supported media format, ffmpeg isn't available, or no acceptable
    encode could be produced. ``max_parts=1`` disables splitting."""
    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found on PATH — cannot transcode %s", file_path)
        return None
    ext = os.path.splitext(file_path)[1].lower()
    target_bytes = max(int(limit_bytes * _HEADROOM_RATIO), 65_536)
    max_parts = _clamp_parts(max_parts)
    try:
        if ext in VIDEO_EXTS: return await _fit_video(file_path, target_bytes, max_parts)
        if ext in AUDIO_EXTS: return await _fit_audio(file_path, target_bytes, max_parts)
        if ext in IMAGE_EXTS:
            image = await _compress_image(file_path, target_bytes)
            return [image] if image else None
    except Exception as exc:
        logger.warning("Transcode failed for %s: %s", file_path, exc, exc_info=True)
    return None


async def compress_to_fit(file_path: str, target_bytes: int) -> str | None:
    """Produce a single transcoded copy of *file_path* under *target_bytes*.

    Returns the new file's path on success, or ``None`` if the file isn't a
    supported format, ffmpeg isn't available, or the result can't fit."""
    parts = await fit_to_limit(file_path, target_bytes, max_parts=1)
    return parts[0] if parts else None


# ---------------------------------------------------------------------- helpers

async def _kill(proc) -> None:
    try: proc.kill()
    except ProcessLookupError: pass
    try: await asyncio.wait_for(proc.wait(), timeout=5.0)
    except (asyncio.TimeoutError, ProcessLookupError): pass


async def _exec(*args: str, timeout: float, capture_stdout: bool = False) -> tuple[int, bytes, bytes] | None:
    """Run a subprocess with a hard timeout.

    Returns ``(returncode, stdout, stderr)``, or ``None`` on timeout. A
    malicious or malformed media file can make ffmpeg hang or burn CPU
    indefinitely; the timeout keeps the bridge responsive. The child is
    killed on timeout AND on task cancellation, so a caller that gives up
    (e.g. the router's own deadline) never leaves an orphan encode running."""
    async with _TRANSCODE_SEMAPHORE:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("%s timed out after %.0fs — killing", args[0], timeout)
            await _kill(proc)
            return None
        except asyncio.CancelledError:
            await _kill(proc)
            raise
    return proc.returncode, stdout or b"", stderr or b""


async def _run(*args: str, timeout: float | None = None) -> bool:
    """Run a subprocess; True on a zero exit status."""
    result = await _exec(*args, timeout=_FFMPEG_TIMEOUT if timeout is None else timeout)
    if result is None: return False
    returncode, _, stderr = result
    if returncode != 0:
        logger.warning("%s failed (rc=%d): %s", args[0], returncode, stderr[:500].decode("utf-8", errors="replace"))
        return False
    return True


def _parse_rate(value) -> float:
    """Parse an ffprobe rational like ``30000/1001``."""
    try:
        num, _, den = str(value).partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


async def _probe(file_path: str) -> _Probe | None:
    """Duration and stream layout of *file_path*, or ``None`` if unreadable."""
    if shutil.which("ffprobe") is None: return None
    try:
        result = await _exec(
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,avg_frame_rate:stream_disposition=attached_pic",
            "-of", "json", file_path,
            timeout=_FFPROBE_TIMEOUT, capture_stdout=True,
        )
    except OSError:
        return None
    if result is None or result[0] != 0: return None
    try:
        data = json.loads(result[1].decode("utf-8", errors="replace") or "{}")
        duration = float(data.get("format", {}).get("duration"))
    except (ValueError, TypeError):
        return None
    if not math.isfinite(duration) or duration <= 0: return None
    has_video = has_audio = False
    fps = 0.0
    for stream in data.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "audio":
            has_audio = True
        elif kind == "video" and not has_video:
            # Cover art in an audio file is reported as a video stream.
            if (stream.get("disposition") or {}).get("attached_pic"): continue
            has_video = True
            fps = _parse_rate(stream.get("avg_frame_rate"))
    return _Probe(duration, has_video, has_audio, fps)


async def _probe_duration(file_path: str) -> float | None:
    probe = await _probe(file_path)
    return probe.duration if probe else None


def _temp(suffix: str) -> str:
    """Create a unique transcoded-output file in :data:`TRANSCODE_DIR`.

    Uses :func:`tempfile.mkstemp` so the file is created O_EXCL (no race
    against a same-named file)."""
    os.makedirs(TRANSCODE_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="bridge_xcode_", dir=TRANSCODE_DIR)
    os.close(fd)
    return path


def _drop(path: str) -> None:
    try: os.remove(path)
    except OSError: pass


def _size(path: str) -> int:
    try: return os.path.getsize(path)
    except OSError: return 0


# -------------------------------------------------------------------- encoders

def _scale_filter(short_side: int) -> str:
    """Cap the shorter dimension at *short_side*, never upscale, and keep
    both dimensions even (libx264 + yuv420p rejects odd sizes)."""
    s = int(short_side) // 2 * 2
    return (
        f"scale='if(gte(iw,ih),-2,min({s},trunc(iw/2)*2))'"
        f":'if(gte(iw,ih),min({s},trunc(ih/2)*2),-2)'"
    )


def _video_args(src: str, out: str, probe: _Probe, video_bps: int, audio_bps: int,
                start: float | None, length: float | None) -> list[str]:
    short_side, fps_cap = _video_tier(video_bps)
    filters = [_scale_filter(short_side)]
    if probe.fps > fps_cap + 1: filters.append(f"fps={fps_cap}")
    args = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error"]
    # Input seeking: fast, and frame-accurate when re-encoding.
    if start: args += ["-ss", f"{start:.3f}"]
    args += ["-i", src]
    if length: args += ["-t", f"{length:.3f}"]
    # First video + (optional) first audio only — drops subtitle/data
    # streams that the mp4 muxer may refuse.
    args += [
        "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-b:v", str(video_bps),
        "-maxrate", str(int(video_bps * 1.15)), "-bufsize", str(int(video_bps * 2)),
    ]
    if audio_bps > 0:
        args += ["-c:a", "aac", "-b:a", str(audio_bps), "-ac", "2" if audio_bps >= 64_000 else "1"]
    else:
        args += ["-an"]
    args += ["-movflags", "+faststart", "-max_muxing_queue_size", "1024", out]
    return args


def _audio_args(src: str, out: str, audio_bps: int, channels: int,
                start: float | None, length: float | None) -> list[str]:
    args = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error"]
    if start: args += ["-ss", f"{start:.3f}"]
    args += ["-i", src]
    if length: args += ["-t", f"{length:.3f}"]
    args += ["-vn", "-sn", "-dn", "-c:a", "aac", "-b:a", str(audio_bps), "-ac", str(channels)]
    # AAC-LC at very low bitrates sounds better with a reduced sample rate.
    if audio_bps < 32_000: args += ["-ar", "24000"]
    args += ["-movflags", "+faststart", out]
    return args


async def _encode_until_fit(suffix: str, target_bytes: int, bitrate: int, floor: int, build_args) -> str | None:
    """Encode with ``build_args(out, bitrate)``; if the result overshoots
    *target_bytes*, retry at a proportionally lower bitrate.

    Returns the output path, or ``None`` (output removed) on failure."""
    out = _temp(suffix)
    done = False
    try:
        for _ in range(_MAX_FIT_ATTEMPTS):
            if not await _run(*build_args(out, bitrate)): return None
            size = _size(out)
            if 0 < size <= target_bytes:
                done = True
                return out
            if size <= 0: return None
            lowered = int(bitrate * target_bytes / size * 0.9)
            logger.info(
                "Encode overshot (%.2f MB > %.2f MB) — retrying at %d kbps (was %d kbps)",
                size / (1024 * 1024), target_bytes / (1024 * 1024), lowered // 1000, bitrate // 1000,
            )
            if lowered < floor: return None
            bitrate = lowered
        return None
    finally:
        # Also covers cancellation: never leave a half-written output behind.
        if not done: _drop(out)


async def _encode_parts(parts: int, duration: float, encode_one) -> list[str] | None:
    """Encode *parts* equal slices via ``encode_one(start, length)``.

    All-or-nothing: if any slice fails (or the task is cancelled) the
    slices already produced are removed."""
    if parts <= 1:
        single = await encode_one(None, None)
        return [single] if single else None
    seg = duration / parts
    outputs: list[str] = []
    complete = False
    try:
        for i in range(parts):
            # The last slice is left open-ended so rounding never clips the tail.
            piece = await encode_one(i * seg, seg if i < parts - 1 else None)
            if not piece: return None
            outputs.append(piece)
        complete = True
        return outputs
    finally:
        if not complete:
            for path in outputs: _drop(path)


async def _fit_video(file_path: str, target_bytes: int, max_parts: int) -> list[str] | None:
    probe = await _probe(file_path)
    if probe is None or not probe.has_video: return None

    async def _attempt(plan: VideoPlan) -> list[str] | None:
        logger.info(
            "Fitting video %s (%.0fs) into %.1f MB/file: %d part(s), ≤%dp @ %d kbps video + %d kbps audio",
            os.path.basename(file_path), probe.duration, target_bytes / (1024 * 1024),
            plan.parts, plan.short_side, plan.video_bps // 1000, plan.audio_bps // 1000,
        )

        async def _one(start, length):
            return await _encode_until_fit(
                ".mp4", target_bytes, plan.video_bps, _ABS_MIN_VIDEO_BPS // 2,
                lambda out, bps: _video_args(file_path, out, probe, bps, plan.audio_bps, start, length),
            )
        return await _encode_parts(plan.parts, probe.duration, _one)

    plan = plan_video(probe.duration, target_bytes, has_audio=probe.has_audio, max_parts=max_parts)
    if plan is None:
        logger.info("Video %s is too long to fit even in %d part(s)", os.path.basename(file_path), max_parts)
        return None
    result = await _attempt(plan)
    if result or plan.parts > 1 or max_parts <= 1: return result
    # The single-file encode couldn't be made to fit — fall back to splitting.
    plan = plan_video(probe.duration, target_bytes, has_audio=probe.has_audio, max_parts=max_parts, min_parts=2)
    return await _attempt(plan) if plan else None


async def _fit_audio(file_path: str, target_bytes: int, max_parts: int) -> list[str] | None:
    probe = await _probe(file_path)
    if probe is None or not probe.has_audio: return None
    plan = plan_audio(probe.duration, target_bytes, max_parts=max_parts)
    if plan is None:
        logger.info("Audio %s is too long to fit even in %d part(s)", os.path.basename(file_path), max_parts)
        return None
    logger.info(
        "Fitting audio %s (%.0fs) into %.1f MB/file: %d part(s) @ %d kbps",
        os.path.basename(file_path), probe.duration, target_bytes / (1024 * 1024),
        plan.parts, plan.audio_bps // 1000,
    )

    async def _one(start, length):
        return await _encode_until_fit(
            ".m4a", target_bytes, plan.audio_bps, _ABS_MIN_AUDIO_BPS // 2,
            lambda out, bps: _audio_args(file_path, out, bps, plan.channels, start, length),
        )
    return await _encode_parts(plan.parts, probe.duration, _one)


async def _compress_image(file_path: str, target_bytes: int) -> str | None:
    """Iteratively drop quality / resolution until the image fits."""
    out = _temp(".jpg")
    done = False
    try:
        # ffmpeg's mjpeg quality scale is 2 (best) → 31 (worst); pair with shrinking
        for q, max_dim in [(4, 4096), (6, 3000), (10, 2200), (15, 1600), (20, 1200), (28, 800)]:
            ok = await _run(
                "ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", file_path,
                "-vf", f"scale='min({max_dim},iw)':'min({max_dim},ih)':force_original_aspect_ratio=decrease",
                "-frames:v", "1", "-update", "1", "-q:v", str(q), out,
            )
            if ok and 0 < _size(out) <= target_bytes:
                done = True
                return out
        return None
    finally:
        if not done: _drop(out)
