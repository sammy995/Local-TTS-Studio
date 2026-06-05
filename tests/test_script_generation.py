"""Tests for LLM-powered podcast script generation.

The LLM call is injected, so the whole pipeline (prompt -> parse -> project) is
testable with a fake model and no Ollama running.
"""
import pytest

from services.script_generation_service import (
    parse_script_response,
    script_to_project,
    generate_script,
    build_prompt,
    ScriptGenerationError,
)
from services.podcast_models import PodcastProject

SAMPLE = (
    '{"title": "AI Today", "turns": ['
    '{"speaker": "Host", "text": "Welcome to the show."}, '
    '{"speaker": "Guest", "text": "Glad to be here."}, '
    '{"speaker": "Host", "text": "Let us dive in."}]}'
)


class FakeLLM:
    """Records calls and returns a canned response (no network)."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, prompt, system=None, **opts):
        self.calls.append((system, prompt))
        return self.response


def test_parse_plain_json():
    data = parse_script_response(SAMPLE)
    assert data["title"] == "AI Today"
    assert len(data["turns"]) == 3


def test_parse_json_wrapped_in_prose_and_fences():
    raw = "Sure, here is your script:\n```json\n" + SAMPLE + "\n```\nLet me know!"
    data = parse_script_response(raw)
    assert len(data["turns"]) == 3
    assert data["turns"][1]["speaker"] == "Guest"


def test_parse_invalid_raises():
    with pytest.raises(ScriptGenerationError):
        parse_script_response("there is no json here")


def test_parse_missing_turns_raises():
    with pytest.raises(ScriptGenerationError):
        parse_script_response('{"title": "x"}')


def test_script_to_project_builds_speakers_and_segments():
    turns = [
        {"speaker": "Host", "text": "Hi"},
        {"speaker": "Guest", "text": "Hello"},
        {"speaker": "Host", "text": "Bye"},
    ]
    proj = script_to_project("My Show", turns)
    assert isinstance(proj, PodcastProject)
    assert proj.title == "My Show"
    assert [s.name for s in proj.speakers] == ["Host", "Guest"]
    assert len({s.config["voice_id"] for s in proj.speakers}) == 2
    assert all(s.mode == "custom" for s in proj.speakers)
    assert [seg.order for seg in proj.segments] == [10, 20, 30]
    assert proj.segments[0].speaker_id == proj.speakers[0].id
    assert proj.segments[0].text == "Hi"
    assert proj.segments[2].speaker_id == proj.speakers[0].id  # Host speaks again
    assert all(seg.kind == "speech" for seg in proj.segments)


def test_generate_script_uses_llm_and_returns_project():
    llm = FakeLLM(SAMPLE)
    proj = generate_script(llm, topic="The future of AI", num_speakers=2)
    assert isinstance(proj, PodcastProject)
    assert len(proj.segments) == 3
    combined = " ".join((s or "") + " " + (p or "") for s, p in llm.calls)
    assert "future of AI" in combined


def test_build_prompt_includes_topic_and_asks_for_json():
    system, user = build_prompt("Climate tech", num_speakers=3)
    assert "Climate tech" in user
    assert "json" in (system + user).lower()
