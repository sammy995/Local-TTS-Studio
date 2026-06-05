"""Tests for the TTS provider abstraction (pure logic — no models required)."""
import pytest

from infra.tts_provider import (
    TTSProviderRegistry,
    ProviderInfo,
    UnsupportedModeError,
    ProviderNotFoundError,
    ProviderUnavailableError,
    dispatch_generate,
    MODE_CUSTOM_VOICE,
    MODE_VOICE_DESIGN,
    MODE_VOICE_CLONE,
)


class FakeProvider:
    """A model-free stand-in that records the calls it receives."""

    def __init__(self):
        self.calls = []

    def generate_custom_voice(self, text, speaker, language=None, instruct=None, **params):
        self.calls.append(("custom_voice", text, speaker))
        return ("AUDIO_CV", 24000)

    def generate_voice_design(self, text, language, instruct, **params):
        self.calls.append(("voice_design", text))
        return ("AUDIO_VD", 24000)

    def generate_voice_clone(self, text, language, ref_audio, ref_text=None, x_vector_only=False, **params):
        self.calls.append(("voice_clone", text))
        return ("AUDIO_VC", 24000)


def make_info(pid="fake", caps=(MODE_CUSTOM_VOICE,), available=True):
    return ProviderInfo(id=pid, name=pid.title(), capabilities=tuple(caps), available=available)


def test_register_and_get_provider():
    reg = TTSProviderRegistry()
    provider = FakeProvider()
    reg.register(make_info(), provider)
    assert reg.get("fake") is provider


def test_list_returns_registered_infos():
    reg = TTSProviderRegistry()
    reg.register(make_info("a"), FakeProvider())
    reg.register(make_info("b"), FakeProvider())
    assert {info.id for info in reg.list()} == {"a", "b"}


def test_get_unknown_provider_raises():
    reg = TTSProviderRegistry()
    with pytest.raises(ProviderNotFoundError):
        reg.get("nope")


def test_get_unavailable_provider_raises():
    reg = TTSProviderRegistry()
    reg.register(make_info("piper", available=False), None)
    with pytest.raises(ProviderUnavailableError):
        reg.get("piper")


def test_supports_reflects_capabilities():
    reg = TTSProviderRegistry()
    reg.register(make_info("fake", caps=(MODE_CUSTOM_VOICE,)), FakeProvider())
    assert reg.supports("fake", MODE_CUSTOM_VOICE) is True
    assert reg.supports("fake", MODE_VOICE_CLONE) is False


def test_default_provider():
    reg = TTSProviderRegistry()
    reg.register(make_info("a"), FakeProvider(), default=True)
    reg.register(make_info("b"), FakeProvider())
    assert reg.default_id == "a"
    assert reg.get_default() is reg.get("a")


def test_dispatch_routes_to_method_and_returns_audio():
    reg = TTSProviderRegistry()
    provider = FakeProvider()
    reg.register(make_info("fake", caps=(MODE_CUSTOM_VOICE,)), provider)
    audio, sr = dispatch_generate(reg, "fake", MODE_CUSTOM_VOICE, text="hi", speaker="Serena")
    assert audio == "AUDIO_CV"
    assert sr == 24000
    assert provider.calls == [("custom_voice", "hi", "Serena")]


def test_dispatch_unsupported_mode_raises_without_calling_provider():
    reg = TTSProviderRegistry()
    provider = FakeProvider()
    reg.register(make_info("fake", caps=(MODE_CUSTOM_VOICE,)), provider)
    with pytest.raises(UnsupportedModeError) as exc:
        dispatch_generate(reg, "fake", MODE_VOICE_CLONE, text="hi", ref_audio="x")
    message = str(exc.value)
    assert "fake" in message and "voice_clone" in message
    assert provider.calls == []
