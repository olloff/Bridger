"""
Tests for the oversized-media pipeline: fit planning (quality ladder vs.
split), the real ffmpeg encode path, Discord multi-part uploads, the
router's prepare-then-send flow, and Telegram attachment classification.

The ffmpeg-backed tests generate their own tiny clips and are skipped
when ffmpeg / ffprobe aren't on PATH.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, sentinel

sys.modules.setdefault("aiosqlite", MagicMock())

from bridge.utils import transcode  # noqa: E402
from bridge.utils.config import effective_max_split_parts, parse_max_split_parts  # noqa: E402
from bridge.utils.transcode import plan_audio, plan_video  # noqa: E402

MB = 1024 * 1024
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class TestVideoPlan(unittest.TestCase):
    def test_short_video_stays_single_at_high_quality(self):
        plan = plan_video(30, 9 * MB)
        self.assertEqual(plan.parts, 1)
        self.assertGreaterEqual(plan.short_side, 720)

    def test_lower_bitrate_lowers_resolution_before_splitting(self):
        # ~2 min into 9 MB ≈ 600 kbps total: still one file, but smaller.
        plan = plan_video(120, 9 * MB)
        self.assertEqual(plan.parts, 1)
        self.assertLessEqual(plan.short_side, 480)
        self.assertGreaterEqual(plan.video_bps, transcode._MIN_SINGLE_VIDEO_BPS)

    def test_resolution_steps_down_monotonically_with_duration(self):
        sides = [plan_video(d, 9 * MB, max_parts=1).short_side for d in (20, 50, 100, 200, 400)]
        self.assertEqual(sides, sorted(sides, reverse=True))
        self.assertGreater(sides[0], sides[-1])

    def test_long_video_splits_instead_of_going_unwatchable(self):
        plan = plan_video(600, 9 * MB)
        self.assertGreater(plan.parts, 1)
        self.assertGreaterEqual(plan.video_bps, transcode._MIN_SINGLE_VIDEO_BPS)
        # Every part must fit the budget at the planned bitrate.
        part_bits = (plan.video_bps + plan.audio_bps) * (600 / plan.parts)
        self.assertLessEqual(part_bits, 9 * MB * 8 * 1.001)

    def test_split_uses_no_more_parts_than_needed(self):
        plan = plan_video(600, 9 * MB)
        fewer = plan_video(600, 9 * MB, max_parts=plan.parts - 1)
        self.assertLess(fewer.video_bps, transcode._SPLIT_VIDEO_BPS)

    def test_max_parts_one_never_splits(self):
        plan = plan_video(330, 9 * MB, max_parts=1)
        self.assertEqual(plan.parts, 1)
        self.assertLess(plan.video_bps, transcode._MIN_SINGLE_VIDEO_BPS)

    def test_max_parts_caps_the_split(self):
        plan = plan_video(1200, 9 * MB, max_parts=4)
        self.assertEqual(plan.parts, 4)

    def test_hopeless_video_returns_none(self):
        self.assertIsNone(plan_video(6 * 3600, 9 * MB, max_parts=3))
        self.assertIsNone(plan_video(3600, 9 * MB, max_parts=1))

    def test_min_parts_forces_a_split(self):
        self.assertEqual(plan_video(30, 9 * MB, min_parts=2).parts, 2)

    def test_silent_video_gives_all_bits_to_the_picture(self):
        with_audio = plan_video(120, 9 * MB, has_audio=True)
        silent = plan_video(120, 9 * MB, has_audio=False)
        self.assertEqual(silent.audio_bps, 0)
        self.assertGreater(silent.video_bps, with_audio.video_bps)

    def test_bad_input(self):
        self.assertIsNone(plan_video(0, 9 * MB))
        self.assertIsNone(plan_video(10, 0))


class TestAudioPlan(unittest.TestCase):
    def test_single_file_caps_bitrate(self):
        plan = plan_audio(60, 9 * MB)
        self.assertEqual((plan.parts, plan.audio_bps, plan.channels), (1, 192_000, 2))

    def test_long_audio_splits(self):
        plan = plan_audio(3 * 3600, 9 * MB)
        self.assertGreater(plan.parts, 1)
        self.assertGreaterEqual(plan.audio_bps, transcode._MIN_SINGLE_AUDIO_BPS)

    def test_no_split_goes_mono_at_low_bitrate(self):
        plan = plan_audio(3600, 9 * MB * 2, max_parts=1)
        self.assertEqual(plan.parts, 1)
        self.assertEqual(plan.channels, 1)

    def test_hopeless_audio_returns_none(self):
        self.assertIsNone(plan_audio(24 * 3600, 1 * MB, max_parts=2))


class TestSplitPartsConfig(unittest.TestCase):
    def test_default_and_clamping(self):
        self.assertEqual(parse_max_split_parts(None), 10)
        self.assertEqual(parse_max_split_parts(0), 1)
        self.assertEqual(parse_max_split_parts(False), 1)
        self.assertEqual(parse_max_split_parts(4), 4)
        self.assertEqual(parse_max_split_parts(999), 30)
        self.assertEqual(parse_max_split_parts("junk", default=7), 7)

    def test_bridge_override_wins(self):
        self.assertEqual(effective_max_split_parts({"max_split_parts": 1}, 10), 1)
        self.assertEqual(effective_max_split_parts({}, 6), 6)


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", *args], check=True)


def _duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(out.strip())


def _dimensions(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    ).stdout
    width, height = out.strip().split(",")[:2]
    return int(width), int(height)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
class TestFitWithRealFfmpeg(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "clip.mp4")
        _ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=duration=24:size=1280x720:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=24",
            "-c:v", "libx264", "-preset", "ultrafast", "-b:v", "3M", "-c:a", "aac", "-shortest", cls.video,
        )
        # Odd dimensions + a non-4:2:0 pixel format: both used to break the encode.
        cls.odd = os.path.join(cls.tmp.name, "odd.mkv")
        _ffmpeg(
            "-f", "lavfi", "-i", "testsrc2=duration=6:size=405x721:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p", "-b:v", "3M", cls.odd,
        )
        cls.audio = os.path.join(cls.tmp.name, "long.mp3")
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=300:duration=240", "-c:a", "libmp3lame", "-b:a", "192k", cls.audio)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.out_dir = tempfile.TemporaryDirectory()
        patcher = patch("bridge.utils.transcode.TRANSCODE_DIR", self.out_dir.name)
        patcher.start()
        self.addCleanup(self.out_dir.cleanup)
        self.addCleanup(patcher.stop)

    async def test_single_file_with_lowered_settings(self):
        limit = 2 * MB
        self.assertGreater(os.path.getsize(self.video), limit)
        parts = await transcode.fit_to_limit(self.video, limit)
        self.assertEqual(len(parts), 1)
        self.assertLessEqual(os.path.getsize(parts[0]), limit)
        self.assertAlmostEqual(await asyncio.to_thread(_duration, parts[0]), 24, delta=0.5)
        self.assertLess(min(await asyncio.to_thread(_dimensions, parts[0])), 720)  # stepped down from 720p

    async def test_splits_when_one_file_would_be_unwatchable(self):
        limit = 600 * 1024
        parts = await transcode.fit_to_limit(self.video, limit, max_parts=10)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(os.path.getsize(part), limit)
        # Nothing lost or duplicated across the cut points.
        durations = [await asyncio.to_thread(_duration, p) for p in parts]
        self.assertAlmostEqual(sum(durations), 24, delta=0.5 * len(parts))
        self.assertEqual(sorted(os.listdir(self.out_dir.name)), sorted(os.path.basename(p) for p in parts))

    async def test_compress_to_fit_never_splits(self):
        limit = 600 * 1024
        single = await transcode.compress_to_fit(self.video, limit)
        self.assertIsNotNone(single)
        self.assertLessEqual(os.path.getsize(single), limit)
        self.assertAlmostEqual(await asyncio.to_thread(_duration, single), 24, delta=0.5)

    async def test_impossible_target_leaves_no_files_behind(self):
        self.assertIsNone(await transcode.fit_to_limit(self.video, 70 * 1024, max_parts=2))
        self.assertEqual(os.listdir(self.out_dir.name), [])

    async def test_odd_dimensions_portrait_video(self):
        limit = 300 * 1024
        parts = await transcode.fit_to_limit(self.odd, limit)
        self.assertTrue(parts)
        for part in parts:
            width, height = await asyncio.to_thread(_dimensions, part)
            self.assertEqual((width % 2, height % 2), (0, 0))
            self.assertLess(width, height)  # still portrait
            self.assertLessEqual(os.path.getsize(part), limit)

    async def test_long_audio_is_split(self):
        limit = 1 * MB
        parts = await transcode.fit_to_limit(self.audio, limit)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(os.path.getsize(part), limit)
        durations = [await asyncio.to_thread(_duration, p) for p in parts]
        self.assertAlmostEqual(sum(durations), 240, delta=1.0 * len(parts))

    async def test_unsupported_extension_returns_none(self):
        doc = os.path.join(self.tmp.name, "archive.zip")
        with open(doc, "wb") as fh: fh.write(b"x" * 4096)
        self.assertIsNone(await transcode.fit_to_limit(doc, 1024))

    async def test_cancellation_removes_partial_output(self):
        task = asyncio.create_task(transcode.fit_to_limit(self.video, 600 * 1024))
        await asyncio.sleep(0.3)
        if task.done():
            for path in task.result() or []: os.remove(path)
            self.skipTest("encode finished before it could be cancelled")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(os.listdir(self.out_dir.name), [])


class _FakeHTTPException(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


class TestDiscordSendFile(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for name in ("discord", "discord.ext", "discord.ext.commands"):
            sys.modules.setdefault(name, MagicMock())
        from bridge.platforms import discord as discord_mod
        self.mod = discord_mod
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        # Record what would be uploaded instead of opening real discord.File objects.
        file_patch = patch.object(
            discord_mod.discord, "File",
            side_effect=lambda path, filename=None: SimpleNamespace(path=path, filename=filename),
        )
        exc_patch = patch.object(discord_mod.discord, "HTTPException", _FakeHTTPException)
        file_patch.start(); exc_patch.start()
        self.addCleanup(file_patch.stop); self.addCleanup(exc_patch.stop)

        self.channel = MagicMock()
        self._ids = iter(range(100, 200))
        self.channel.send = AsyncMock(side_effect=lambda *a, **kw: MagicMock(id=next(self._ids)))
        self.channel.get_partial_message.return_value.to_reference.return_value = sentinel.reference
        self.channel.get_partial_message.return_value.reply = AsyncMock(return_value=MagicMock(id=99))

        platform = discord_mod.DiscordPlatform.__new__(discord_mod.DiscordPlatform)
        platform.name = "discord"
        platform._allowed_mentions = sentinel.allowed_mentions
        platform._destination_size_limit_bytes = 1000
        platform._max_split_parts = 10
        platform._bridge_for_chat = {}
        platform._bridge_by_name = {}
        platform._bot = MagicMock()
        platform._bot.get_channel.return_value = self.channel
        self.platform = platform

    def _file(self, name: str, size: int) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as fh: fh.write(b"\0" * size)
        return path

    def _calls(self) -> list[dict]:
        return [c.kwargs for c in self.channel.send.await_args_list]

    async def test_file_under_limit_is_uploaded_untouched(self):
        src = self._file("clip.mp4", 500)
        with patch.object(self.mod, "fit_to_limit", new=AsyncMock()) as fit:
            sent_id = await self.platform.send_file(1, src, ".mp4", "Alice")
        fit.assert_not_awaited()
        self.assertEqual(sent_id, 100)
        (call,) = self._calls()
        self.assertEqual((call["file"].path, call["file"].filename), (src, "clip.mp4"))
        self.assertEqual(call["content"], "*Alice:* [clip.mp4]")
        self.assertIs(call["allowed_mentions"], sentinel.allowed_mentions)
        self.assertTrue(os.path.isfile(src))  # the source file is the router's to delete

    async def test_oversized_file_is_compressed_and_temp_removed(self):
        src = self._file("clip.mov", 5000)
        small = self._file("bridge_xcode_a.mp4", 800)
        with patch.object(self.mod, "fit_to_limit", new=AsyncMock(return_value=[small])) as fit:
            sent_id = await self.platform.send_file(1, src, ".mov", "Alice")
        fit.assert_awaited_once_with(src, 1000, max_parts=10)
        self.assertEqual(sent_id, 100)
        (call,) = self._calls()
        # Uploaded under the original stem with the new container's extension.
        self.assertEqual((call["file"].path, call["file"].filename), (small, "clip.mp4"))
        self.assertEqual(call["content"], "*Alice:* [clip.mov] (compressed)")
        self.assertFalse(os.path.exists(small))
        self.assertTrue(os.path.isfile(src))

    async def test_split_parts_posted_in_order_and_first_id_returned(self):
        src = self._file("movie.mp4", 5000)
        parts = [self._file(f"bridge_xcode_{i}.mp4", 900) for i in range(3)]
        with patch.object(self.mod, "fit_to_limit", new=AsyncMock(return_value=list(parts))):
            sent_id = await self.platform.send_file(1, src, ".mp4", "Al*ice", reply_to_native_id=55)
        self.assertEqual(sent_id, 100)
        calls = self._calls()
        self.assertEqual([c["file"].path for c in calls], parts)
        self.assertEqual(
            [c["file"].filename for c in calls],
            ["movie_part1of3.mp4", "movie_part2of3.mp4", "movie_part3of3.mp4"],
        )
        self.assertEqual(
            [c["content"] for c in calls],
            [f"*Al\\*ice:* [movie.mp4] (part {i}/3)" for i in (1, 2, 3)],
        )
        # Only the first part is a reply — and a non-failing one.
        self.assertEqual(["reference" in c for c in calls], [True, False, False])
        self.channel.get_partial_message.return_value.to_reference.assert_called_once_with(fail_if_not_exists=False)
        for part in parts:
            self.assertFalse(os.path.exists(part))

    async def test_prepared_paths_are_used_and_left_for_the_caller(self):
        src = self._file("movie.mp4", 5000)
        parts = [self._file(f"prepared_{i}.mp4", 900) for i in range(2)]
        with patch.object(self.mod, "fit_to_limit", new=AsyncMock()) as fit:
            sent_id = await self.platform.send_file(1, src, ".mp4", "Alice", prepared_paths=parts)
        fit.assert_not_awaited()
        self.assertEqual(sent_id, 100)
        self.assertEqual(len(self._calls()), 2)
        for part in parts:
            self.assertTrue(os.path.isfile(part))

    async def test_internal_media_parts_are_labelled_without_attribution(self):
        src = self._file("post.mp4", 5000)
        parts = [self._file(f"prepared_{i}.mp4", 900) for i in range(2)]
        await self.platform.send_file(1, src, ".mp4", "bridge", trusted_internal=True, prepared_paths=parts)
        self.assertEqual([c.get("content") for c in self._calls()], ["(part 1/2)", "(part 2/2)"])

    async def test_unfittable_file_posts_notice(self):
        src = self._file("archive.zip", 5000)
        with patch.object(self.mod, "fit_to_limit", new=AsyncMock(return_value=None)):
            sent_id = await self.platform.send_file(1, src, ".zip", "Alice")
        self.assertIsNone(sent_id)
        (call,) = self.channel.send.await_args_list
        self.assertIn("couldn't be compressed or split", call.args[0])
        self.assertNotIn("file", call.kwargs)

    async def test_prepare_file_reports_fit_split_and_failure(self):
        small = self._file("ok.mp4", 500)
        big = self._file("big.mp4", 5000)
        parts = [self._file(f"bridge_xcode_{i}.mp4", 900) for i in range(2)]
        self.assertEqual(await self.platform.prepare_file(1, small, ".mp4"), [small])
        with patch.object(self.mod, "fit_to_limit", new=AsyncMock(return_value=list(parts))):
            self.assertEqual(await self.platform.prepare_file(1, big, ".mp4"), parts)
        with patch.object(self.mod, "fit_to_limit", new=AsyncMock(return_value=None)):
            self.assertEqual(await self.platform.prepare_file(1, big, ".mp4"), [])

    async def test_bridge_overrides_limit_and_split_parts(self):
        self.platform._bridge_for_chat = {1: "b"}
        self.platform._bridge_by_name = {"b": {"name": "b", "destination_size_limit_mb": 50, "max_split_parts": 1}}
        src = self._file("clip.mp4", 5000)
        self.assertEqual(self.platform._file_limits_for_chat(1), (50 * MB, 1))
        self.assertEqual(await self.platform.prepare_file(1, src, ".mp4"), [src])

    async def test_413_refits_to_fallback_limit_once(self):
        self.platform._destination_size_limit_bytes = 50 * MB
        src = self._file("clip.mp4", 100)
        refit = self._file("bridge_xcode_refit.mp4", 50)
        outcomes = [_FakeHTTPException(413), MagicMock(id=7)]

        def _send(*_a, **_kw):
            result = outcomes.pop(0)
            if isinstance(result, Exception): raise result
            return result
        self.channel.send = AsyncMock(side_effect=_send)
        fit = AsyncMock(return_value=[refit])
        # The source is under the configured 50 MB but over Discord's real limit.
        with patch.object(self.mod, "OVERSIZE_FALLBACK_LIMIT_BYTES", 60), patch.object(self.mod, "fit_to_limit", new=fit):
            sent_id = await self.platform.send_file(1, src, ".mp4", "Alice")
        self.assertEqual(sent_id, 7)
        fit.assert_awaited_once_with(src, 60, max_parts=10)
        self.assertEqual(self.channel.send.await_args_list[1].kwargs["file"].path, refit)
        self.assertFalse(os.path.exists(refit))

    async def test_413_at_fallback_limit_posts_notice(self):
        src = self._file("clip.mp4", 500)
        outcomes = [_FakeHTTPException(413), MagicMock(id=8)]

        def _send(*_a, **_kw):
            result = outcomes.pop(0)
            if isinstance(result, Exception): raise result
            return result
        self.channel.send = AsyncMock(side_effect=_send)
        self.assertIsNone(await self.platform.send_file(1, src, ".mp4", "Alice"))
        self.assertIn("couldn't be compressed or split", self.channel.send.await_args_list[1].args[0])

    async def test_failed_reply_retries_with_a_fresh_file_object(self):
        src = self._file("clip.mp4", 500)
        outcomes = [_FakeHTTPException(400), MagicMock(id=9)]

        def _send(*_a, **_kw):
            result = outcomes.pop(0)
            if isinstance(result, Exception): raise result
            return result
        self.channel.send = AsyncMock(side_effect=_send)
        sent_id = await self.platform.send_file(1, src, ".mp4", "Alice", reply_to_native_id=55)
        self.assertEqual(sent_id, 9)
        first, second = self._calls()
        self.assertIn("reference", first)
        self.assertNotIn("reference", second)
        # discord.py closes a File after the request — it must not be reused.
        self.assertIsNot(first["file"], second["file"])

    async def test_other_http_errors_propagate(self):
        src = self._file("clip.mp4", 500)
        self.channel.send = AsyncMock(side_effect=_FakeHTTPException(500))
        with self.assertRaises(_FakeHTTPException):
            await self.platform.send_file(1, src, ".mp4", "Alice")


class _PreparingPlatform:
    """Router-facing fake whose prepare_file hands back transcoder output."""

    def __init__(self, prepared=None, delay: float = 0.0):
        self.prepared = prepared
        self.delay = delay
        self.events: list[tuple] = []

    async def prepare_file(self, chat_id, file_path, file_ext):
        self.events.append(("prepare", chat_id))
        if self.delay: await asyncio.sleep(self.delay)
        return self.prepared

    async def send_file(self, chat_id, file_path, file_ext, sender, reply_to_native_id=None, *,
                        trusted_internal=False, **extra):
        self.events.append(("send", chat_id, extra.get("prepared_paths", "absent")))
        return 42

    async def send_text(self, chat_id, content, sender, reply_to_native_id=None, *, trusted_internal=False):
        self.events.append(("text", chat_id, content))
        return 43


class TestRouterPrepareThenSend(unittest.IsolatedAsyncioTestCase):
    BRIDGE = {"name": "b", "from": {"telegram": -1}, "to": {"discord": 7}}

    async def asyncSetUp(self):
        from bridge import router as router_mod
        self.router_mod = router_mod
        self._real_db = router_mod.database
        router_mod.database = MagicMock()
        router_mod.database.resolve_native_id = AsyncMock(return_value=None)
        router_mod.database.save_message_mapping = AsyncMock(return_value=None)
        self.tmp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self):
        self.router_mod.database = self._real_db
        self.tmp.cleanup()

    def _file(self, name: str) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as fh: fh.write(b"x")
        return path

    def _router(self, discord):
        return self.router_mod.Router({"telegram": MagicMock(), "discord": discord}, [self.BRIDGE], db_path=":memory:")

    async def test_prepared_parts_are_passed_on_and_cleaned_up(self):
        src = self._file("video.mp4")
        parts = [self._file("bridge_xcode_1.mp4"), self._file("bridge_xcode_2.mp4")]
        discord = _PreparingPlatform(prepared=list(parts))
        await self._router(discord).on_file("telegram", "b", 1, src, ".mp4", "Alice")
        self.assertEqual(discord.events, [("prepare", 7), ("send", 7, parts)])
        for path in parts + [src]:
            self.assertFalse(os.path.exists(path))
        self.router_mod.database.save_message_mapping.assert_awaited_once()

    async def test_no_preparation_needed_keeps_plain_send(self):
        src = self._file("photo.jpg")
        discord = _PreparingPlatform(prepared=None)
        await self._router(discord).on_file("telegram", "b", 1, src, ".jpg", "Alice")
        self.assertEqual(discord.events, [("prepare", 7), ("send", 7, "absent")])

    async def test_original_path_in_prepared_is_not_deleted_early(self):
        src = self._file("photo.jpg")
        seen: list[bool] = []
        discord = _PreparingPlatform(prepared=[src])
        real_send = discord.send_file

        async def _send(*args, **kwargs):
            seen.append(os.path.isfile(src))
            return await real_send(*args, **kwargs)
        discord.send_file = _send
        await self._router(discord).on_file("telegram", "b", 1, src, ".jpg", "Alice")
        self.assertEqual(seen, [True])

    async def test_slow_prepare_does_not_block_text_to_same_destination(self):
        src = self._file("video.mp4")
        discord = _PreparingPlatform(prepared=None, delay=0.3)
        router = self._router(discord)
        file_task = asyncio.create_task(router.on_file("telegram", "b", 1, src, ".mp4", "Alice"))
        await asyncio.sleep(0.05)
        await router.on_message("telegram", "b", 2, "hello", "Bob")
        await file_task
        self.assertEqual([e[0] for e in discord.events], ["prepare", "text", "send"])

    async def test_prepare_timeout_becomes_too_large_notice(self):
        src = self._file("video.mp4")
        discord = _PreparingPlatform(prepared=None, delay=5)
        router = self._router(discord)
        router._prepare_timeout = 0.05
        await router.on_file("telegram", "b", 1, src, ".mp4", "Alice")
        self.assertEqual(discord.events, [("prepare", 7), ("send", 7, [])])

    async def test_split_upload_gets_a_proportional_deadline(self):
        src = self._file("video.mp4")
        parts = [self._file(f"bridge_xcode_{i}.mp4") for i in range(3)]
        router = self._router(_PreparingPlatform(prepared=parts))
        seen: list[float] = []
        real = router._ordered_target_call

        async def _spy(target, chat_id, operation, timeout=None):
            seen.append(timeout)
            return await real(target, chat_id, operation, timeout=timeout)
        router._ordered_target_call = _spy
        await router.on_file("telegram", "b", 1, src, ".mp4", "Alice")
        self.assertEqual(seen, [router._send_timeout * 3])


class TestTelegramMediaClassification(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        for name in ("pyrogram", "pyrogram.enums", "pyrogram.errors"):
            sys.modules.setdefault(name, MagicMock())
        from bridge.platforms import telegram as telegram_mod
        self.mod = telegram_mod

    @staticmethod
    def _msg(**media):
        return SimpleNamespace(**media)

    def test_extensions_match_what_telegram_stores(self):
        info = self.mod._get_media_info
        self.assertEqual(info(self._msg(photo=object())), ("image", ".jpg"))
        self.assertEqual(info(self._msg(video=object())), ("video", ".mp4"))
        self.assertEqual(info(self._msg(animation=object())), ("animation", ".mp4"))
        self.assertEqual(info(self._msg(video_note=object())), ("video_note", ".mp4"))
        self.assertEqual(info(self._msg(voice=object())), ("voice", ".ogg"))

    def test_audio_keeps_its_real_container(self):
        info = self.mod._get_media_info
        self.assertEqual(info(self._msg(audio=SimpleNamespace(file_name="song.FLAC", mime_type="audio/flac"))), ("song", ".flac"))
        self.assertEqual(info(self._msg(audio=SimpleNamespace(file_name=None, mime_type="audio/mp4"))), ("audio", ".m4a"))
        self.assertEqual(info(self._msg(audio=SimpleNamespace(file_name="../../x.exe", mime_type=None))), ("x", ".mp3"))

    def test_sticker_containers(self):
        info = self.mod._get_media_info
        self.assertEqual(info(self._msg(sticker=SimpleNamespace(is_video=True, is_animated=False))), ("sticker", ".webm"))
        self.assertEqual(info(self._msg(sticker=SimpleNamespace(is_video=False, is_animated=True))), ("sticker", ".tgs"))
        self.assertEqual(info(self._msg(sticker=SimpleNamespace(is_video=False, is_animated=False))), ("sticker", ".webp"))

    def test_document_name_is_sanitized(self):
        name, ext = self.mod._get_media_info(self._msg(document=SimpleNamespace(file_name="../../etc/report.pdf")))
        self.assertEqual((name, ext), ("report", ".pdf"))

    def test_only_real_attachments_count_as_downloadable(self):
        has = self.mod._has_downloadable_media
        self.assertTrue(has(self._msg(video_note=object())))
        self.assertTrue(has(self._msg(animation=object())))
        self.assertFalse(has(self._msg(media="web_page", web_page=object(), poll=object())))

    def test_size_covers_video_notes(self):
        self.assertEqual(self.mod._get_media_size(self._msg(video_note=SimpleNamespace(file_size=123))), 123)

    def test_album_caption_found_on_any_item(self):
        items = [SimpleNamespace(caption=None), SimpleNamespace(caption="trip"), SimpleNamespace(caption=None)]
        self.assertEqual(self.mod._album_caption(items), "trip")
        self.assertIsNone(self.mod._album_caption([SimpleNamespace(caption=None)]))

    def _platform(self):
        platform = self.mod.TelegramPlatform.__new__(self.mod.TelegramPlatform)
        platform.name = "telegram"
        platform._bridge_for_source_chat = {-1: "b"}
        platform._bridge_by_name = {"b": {"name": "b"}}
        platform._mention_filters = []
        platform._destination_size_limit_bytes = 50 * MB
        platform.router = MagicMock()
        platform.router.on_message = AsyncMock()
        platform.router.on_file = AsyncMock()
        return platform

    @staticmethod
    def _incoming(**overrides):
        base = dict(
            id=10, chat=SimpleNamespace(id=-1), date=None, from_user=None,
            media_group_id=None, reply_to_message_id=None, text=None, caption=None, media=None,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    async def test_link_preview_message_is_bridged_as_text(self):
        platform = self._platform()
        client = MagicMock()
        client.download_media = AsyncMock()
        message = self._incoming(media="web_page", web_page=object(), text="look https://example.com/x")
        await platform._handle_incoming_message(client, message)
        client.download_media.assert_not_awaited()
        platform.router.on_file.assert_not_awaited()
        platform.router.on_message.assert_awaited_once()
        self.assertEqual(platform.router.on_message.await_args.args[3], "look https://example.com/x")

    async def test_textless_non_downloadable_message_is_skipped(self):
        platform = self._platform()
        await platform._handle_incoming_message(MagicMock(), self._incoming(media="poll", poll=object()))
        platform.router.on_message.assert_not_awaited()
        platform.router.on_file.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
