"""LLM-powered podcast script generation.

Turns a topic (and optional source document) into a ready-to-render
``PodcastProject`` by asking a local LLM for a JSON dialogue and mapping the
turns onto speakers and segments.

The LLM is injected (any object with ``complete(prompt, system=...) -> str``),
so prompt-building, response parsing, and project assembly are all pure and
unit-testable without a running model. Only the concrete provider in
``infra.llm_provider`` performs network I/O.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from services.podcast_models import PodcastProject, Segment, Speaker

# Preset voices to assign to generated speakers, round-robin.
DEFAULT_VOICE_POOL: Tuple[str, ...] = (
    "Ryan", "Serena", "Aiden", "Sohee", "Vivian", "Eric", "Dylan", "Ono_Anna", "Uncle_Fu",
)


class ScriptGenerationError(Exception):
    """Raised when a model response can't be turned into a script."""


def build_prompt(
    topic: str,
    num_speakers: int = 2,
    style: Optional[str] = None,
    document: Optional[str] = None,
) -> Tuple[str, str]:
    """Build (system, user) prompts that steer the model toward strict JSON."""
    system = (
        "You are a professional podcast script writer. "
        "You always respond with a single valid JSON object and nothing else."
    )
    parts = [
        f"Write a natural, engaging podcast conversation about: {topic}.",
        f"Use exactly {num_speakers} distinct speakers who refer to each other by name.",
        "Keep each turn to a few sentences and make it sound spoken, not written.",
    ]
    if style:
        parts.append(f"Tone and style: {style}.")
    if document:
        parts.append("Base the discussion on this source material:\n" + document)
    parts.append(
        'Respond ONLY with JSON of this exact shape: '
        '{"title": "Episode title", "turns": '
        '[{"speaker": "Name", "text": "What they say"}, ...]}'
    )
    return system, "\n".join(parts)


def _try_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def parse_script_response(raw: str) -> Dict[str, Any]:
    """Extract the JSON script object from a model response.

    Tolerates surrounding prose and ```json code fences by falling back to the
    outermost ``{ ... }`` span.
    """
    if not raw or not raw.strip():
        raise ScriptGenerationError("Empty response from the model")

    text = raw.strip()
    data = _try_json(text)
    if data is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = _try_json(text[start : end + 1])

    if data is None:
        raise ScriptGenerationError("Could not parse JSON from the model response")
    if not isinstance(data, dict) or not isinstance(data.get("turns"), list):
        raise ScriptGenerationError("Model response is missing a 'turns' list")
    return data


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "speaker"


def script_to_project(
    title: str,
    turns: List[Dict[str, Any]],
    voice_pool: Optional[Tuple[str, ...]] = None,
    project_id: Optional[str] = None,
) -> PodcastProject:
    """Assemble distinct speakers and ordered speech segments from dialogue turns."""
    if not turns:
        raise ScriptGenerationError("No dialogue turns to build a project from")

    pool = voice_pool or DEFAULT_VOICE_POOL
    speakers: Dict[str, Speaker] = {}
    order_names: List[str] = []

    for turn in turns:
        name = (str(turn.get("speaker") or "Speaker").strip()) or "Speaker"
        if name not in speakers:
            order_names.append(name)
            sid, base, i = _slug(name), _slug(name), 2
            existing = {s.id for s in speakers.values()}
            while sid in existing:
                sid = f"{base}_{i}"
                i += 1
            voice = pool[len(speakers) % len(pool)]
            speakers[name] = Speaker(id=sid, name=name, mode="custom", config={"voice_id": voice})

    segments: List[Segment] = []
    for i, turn in enumerate(turns):
        name = (str(turn.get("speaker") or "Speaker").strip()) or "Speaker"
        text = str(turn.get("text") or "").strip()
        segments.append(
            Segment(
                id=f"seg_{i + 1}",
                order=(i + 1) * 10,
                speaker_id=speakers[name].id,
                text=text,
                kind="speech",
            )
        )

    return PodcastProject(
        id=project_id or f"gen_{uuid.uuid4().hex[:8]}",
        title=title or "Generated Podcast",
        speakers=[speakers[n] for n in order_names],
        segments=segments,
    )


def generate_script(
    llm: Any,
    topic: str,
    num_speakers: int = 2,
    style: Optional[str] = None,
    document: Optional[str] = None,
    title: Optional[str] = None,
) -> PodcastProject:
    """Generate a podcast project from a topic using the given LLM provider."""
    system, user = build_prompt(topic, num_speakers=num_speakers, style=style, document=document)
    raw = llm.complete(user, system=system)
    data = parse_script_response(raw)
    return script_to_project(title or data.get("title") or topic, data["turns"])
