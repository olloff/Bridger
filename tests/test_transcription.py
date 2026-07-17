"""
Tests for the voice-transcription subsystem.

Covers the pure parts: config parsing and the provider-registry factory.
The actual transcription network/local calls are not exercised here —
they're isolated behind ``Transcriber.transcribe`` and require live deps.
"""

import sys
import unittest
from unittest.mock import MagicMock

# Stub heavy / network deps so the transcription modules import cleanly.
sys.modules.setdefault("aiohttp", MagicMock())

from bridge.transcription import get_transcriber
from bridge.transcription.openai_whisper import OpenAITranscriber
from bridge.utils.config import platform_voice_transcription_config


class TestVoiceTranscriptionConfig(unittest.TestCase):
    def test_disabled_returns_none(self):
        self.assertIsNone(platform_voice_transcription_config({}))
        self.assertIsNone(platform_voice_transcription_config({"voice_transcription": {"enabled": False}}))

    def test_no_outputs_returns_none(self):
        # If neither output toggle is on, there's nothing to do.
        cfg = platform_voice_transcription_config({
            "voice_transcription": {
                "enabled": True,
                "reply_with_transcript": False,
                "bridge_with_transcript": False,
            }
        })
        self.assertIsNone(cfg)

    def test_defaults_with_bridge_only(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "openai_api_key": "sk-x"}
        })
        # bridge_with_transcript defaults True; reply_with_transcript defaults False.
        self.assertTrue(cfg["bridge_with_transcript"])
        self.assertFalse(cfg["reply_with_transcript"])
        self.assertEqual(cfg["provider"], "openai")
        self.assertEqual(cfg["openai_model"], "whisper-1")
        self.assertEqual(cfg["local_model"], "small")
        self.assertEqual(cfg["max_duration_seconds"], 600.0)
        self.assertIsNone(cfg["language"])

    def test_provider_normalized(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "provider": "  LOCAL "}
        })
        self.assertEqual(cfg["provider"], "local")

    def test_max_duration_invalid_falls_back(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "max_duration_seconds": "abc"}
        })
        self.assertEqual(cfg["max_duration_seconds"], 600.0)

    def test_max_duration_negative_clamped(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "max_duration_seconds": -10}
        })
        self.assertEqual(cfg["max_duration_seconds"], 0.0)

    def test_language_auto_kept_as_string(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "language": "auto"}
        })
        self.assertEqual(cfg["language"], "auto")

    def test_language_scalar_forced(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "language": "  UK "}
        })
        # A scalar stays a scalar (forced single language), unchanged behavior.
        self.assertEqual(cfg["language"], "UK")

    def test_language_priority_list(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "language": ["UK", "en", " ru "]}
        })
        # A list becomes a lowercased, trimmed priority list.
        self.assertEqual(cfg["language"], ["uk", "en", "ru"])

    def test_language_list_drops_auto_and_empty(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "language": ["uk", "auto", "", "en"]}
        })
        self.assertEqual(cfg["language"], ["uk", "en"])

    def test_language_list_all_auto_becomes_none(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "language": ["auto", ""]}
        })
        self.assertIsNone(cfg["language"])

    def test_reply_only_no_bridge(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {
                "enabled": True,
                "reply_with_transcript": True,
                "bridge_with_transcript": False,
            }
        })
        self.assertTrue(cfg["reply_with_transcript"])
        self.assertFalse(cfg["bridge_with_transcript"])

    def test_api_key_passthrough_and_strip(self):
        cfg = platform_voice_transcription_config({
            "voice_transcription": {"enabled": True, "openai_api_key": "  sk-abc  "}
        })
        self.assertEqual(cfg["openai_api_key"], "sk-abc")


class TestTranscriberFactory(unittest.TestCase):
    def test_openai_requires_key(self):
        cfg = {"provider": "openai", "openai_model": "whisper-1"}
        self.assertIsNone(get_transcriber(cfg))

    def test_openai_with_key_returns_instance(self):
        cfg = {"provider": "openai", "openai_api_key": "sk-x", "openai_model": "whisper-1"}
        t = get_transcriber(cfg)
        self.assertIsInstance(t, OpenAITranscriber)
        self.assertEqual(t.api_key, "sk-x")
        self.assertEqual(t.model, "whisper-1")

    def test_unknown_provider(self):
        self.assertIsNone(get_transcriber({"provider": "elevenlabs"}))

    def test_local_provider_returns_instance_without_faster_whisper_check(self):
        # The factory itself doesn't import faster-whisper — that's lazy
        # inside the first transcribe() call. So the factory always returns
        # a LocalTranscriber object for provider=local; the import only
        # fails later if faster-whisper isn't installed.
        from bridge.transcription.local_whisper import LocalTranscriber
        t = get_transcriber({"provider": "local", "local_model": "tiny"})
        self.assertIsInstance(t, LocalTranscriber)
        self.assertEqual(t.model_name, "tiny")


class TestLocalPriority(unittest.TestCase):
    """Exercise the local backend's priority/auto-fallback selection with a
    lightweight fake model — no faster-whisper needed."""

    def _model(self, script):
        from types import SimpleNamespace

        class FakeModel:
            def __init__(self, script):
                self.script = script
                self.calls = []

            def transcribe(self, path, language=None, beam_size=1):
                self.calls.append(language)
                lang, prob, all_probs, text = self.script[len(self.calls) - 1]
                info = SimpleNamespace(
                    language=lang, language_probability=prob, all_language_probs=all_probs,
                )
                return ([SimpleNamespace(text=text)], info)

        return FakeModel(script)

    def test_distribution_honors_priority_order(self):
        from bridge.transcription.local_whisper import _transcribe_priority
        # Auto picks ru, but uk is a plausible candidate and higher priority.
        m = self._model([
            ("ru", 0.5, [("ru", 0.5), ("uk", 0.4), ("en", 0.05)], "privet"),
            ("uk", 0.99, None, "привіт"),
        ])
        self.assertEqual(_transcribe_priority(m, "x", ["uk", "ru"]), "привіт")
        self.assertEqual(m.calls, [None, "uk"])  # forced uk on second pass

    def test_distribution_keeps_auto_when_nothing_plausible(self):
        from bridge.transcription.local_whisper import _transcribe_priority
        # Confidently German — none of the listed langs is plausible → auto.
        m = self._model([
            ("de", 0.95, [("de", 0.95), ("uk", 0.01), ("en", 0.01)], "hallo welt"),
        ])
        self.assertEqual(_transcribe_priority(m, "x", ["uk", "en"]), "hallo welt")
        self.assertEqual(m.calls, [None])  # single decode, no forcing

    def test_distribution_preferred_top_no_second_pass(self):
        from bridge.transcription.local_whisper import _transcribe_priority
        m = self._model([
            ("uk", 0.9, [("uk", 0.9), ("ru", 0.05)], "привіт"),
        ])
        self.assertEqual(_transcribe_priority(m, "x", ["uk", "en"]), "привіт")
        self.assertEqual(m.calls, [None])

    def test_no_distribution_falls_back_to_confidence(self):
        from bridge.transcription.local_whisper import _transcribe_priority
        # Older faster-whisper: all_language_probs=None. Low confidence forces
        # the top-priority language.
        m = self._model([
            ("en", 0.30, None, "garbled"),
            ("uk", 0.99, None, "привіт"),
        ])
        self.assertEqual(_transcribe_priority(m, "x", ["uk", "en"]), "привіт")
        self.assertEqual(m.calls, [None, "uk"])

    def test_no_distribution_confident_keeps_auto(self):
        from bridge.transcription.local_whisper import _transcribe_priority
        m = self._model([
            ("ru", 0.97, None, "privet"),
        ])
        self.assertEqual(_transcribe_priority(m, "x", ["uk", "en"]), "privet")
        self.assertEqual(m.calls, [None])


if __name__ == "__main__":
    unittest.main()
