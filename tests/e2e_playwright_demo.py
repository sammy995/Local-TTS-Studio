"""
Local TTS Studio — Enhanced Playwright E2E Demo
================================================
Each mode carries narrative content that tells the product/architecture story:

  Custom Voice  — Ryan (Dynamic male, English) narrates the hexagonal architecture
  Voice Design  — AI-shaped "lead architect" voice explains the Voice Design feature
  Voice Clone   — Clones LJ Speech (Linda Johnson), the canonical public-domain TTS
                  reference voice, cited in hundreds of papers (public domain)
  Podcast Mode  — Three-speaker conversation between Alex (PM), Sam (Architect),
                  and Jordan (User), using Aiden / Ryan / Serena voices

Reference audio: LJ Speech dataset — Linda Johnson reading Project Gutenberg texts.
  License: public domain  |  https://keithito.com/LJ-Speech-Dataset/
  Recognised by every TTS researcher; zero copyright risk.

Usage
-----
  pip install playwright
  playwright install chromium
  python tests/e2e_playwright_demo.py

To use a real LJ Speech clip instead of the synthetic fallback:
  1. Download LJSpeech-1.1.tar.bz2 from keithito.com/LJ-Speech-Dataset/
  2. Extract any wavs/LJ001-*.wav file
  3. Copy it to  tests/fixtures/lj_reference.wav
  The script detects the file automatically.
"""

import asyncio
import math
import struct
import wave
from pathlib import Path

from playwright.async_api import async_playwright

BASE_URL = "http://localhost:8000/"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_AUDIO = FIXTURES_DIR / "lj_reference.wav"

GENERATION_TIMEOUT_MS = 120_000  # TTS synthesis can take 30-90 s


# ---------------------------------------------------------------------------
# Reference audio helper
# ---------------------------------------------------------------------------

def _ensure_reference_audio() -> Path:
    """
    Returns the path to the reference audio for voice cloning.

    Prefers tests/fixtures/lj_reference.wav (drop in an LJ Speech clip for a
    real voice-clone result).  Falls back to a 3-second 440 Hz sine-wave WAV
    that proves the upload + cloning pipeline works end-to-end.

    LJ Speech (Linda Johnson) is the de-facto public-domain benchmark voice
    in TTS research — zero copyright risk, universally recognised.
    """
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if REFERENCE_AUDIO.exists():
        return REFERENCE_AUDIO

    print(
        "[Voice Clone] lj_reference.wav not found — generating synthetic fallback.\n"
        "  For a real voice clone, place an LJ Speech WAV at:\n"
        f"  {REFERENCE_AUDIO}"
    )
    sample_rate = 22_050
    duration_s = 3
    frequency = 440  # A4 tone
    with wave.open(str(REFERENCE_AUDIO), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        for i in range(sample_rate * duration_s):
            value = int(32_767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", value))
    return REFERENCE_AUDIO


async def _wait_for_result(page, result_id: str, label: str) -> None:
    """Wait for either an audio player or an error message to appear."""
    await page.wait_for_function(
        f"() => document.querySelector('#{result_id} audio, #{result_id} .status.error') !== null",
        timeout=GENERATION_TIMEOUT_MS,
    )
    # Report what appeared
    has_audio = await page.query_selector(f"#{result_id} audio")
    has_error = await page.query_selector(f"#{result_id} .status.error")
    if has_audio:
        print(f"[{label}] ✅  Audio generated successfully.")
    elif has_error:
        error_text = await has_error.inner_text()
        print(f"[{label}] ⚠️  UI error: {error_text.strip()}")


# ---------------------------------------------------------------------------
# Test: Custom Voice
# ---------------------------------------------------------------------------

async def test_custom_voice(page) -> None:
    """
    Ryan (Dynamic male, English) narrates the hexagonal architecture of
    Local TTS Studio from both a product and an engineering perspective.
    """
    print("\n[Custom Voice] Starting…")

    await page.locator("#cv-text").fill(
        "Local TTS Studio is built on a hexagonal architecture with three distinct layers. "
        "The core layer performs pure Qwen3-TTS synthesis and contains zero I/O dependencies. "
        "The services layer orchestrates validation, storage, and multi-speaker rendering. "
        "The runtime layer exposes a FastAPI server that brings it all together. "
        "Everything runs on your hardware — your audio never leaves your machine."
    )

    await page.select_option("#cv-speaker", "Ryan")  # Dynamic male, English
    await page.select_option("#cv-language", "English")
    await page.locator("#cv-instruct").fill(
        "Speak clearly and confidently, like a senior engineer presenting an architecture diagram"
    )

    await page.locator("button", has_text="Generate Speech").click()
    await _wait_for_result(page, "cv-result", "Custom Voice")


# ---------------------------------------------------------------------------
# Test: Voice Design
# ---------------------------------------------------------------------------

async def test_voice_design(page) -> None:
    """
    An AI-shaped 'lead architect' voice explains the Voice Design feature
    from a product perspective — how it unlocks unlimited personas on demand.
    """
    print("\n[Voice Design] Starting…")

    await page.locator("button.tab", has_text="Voice Design").click()

    await page.locator("#vd-text").fill(
        "Voice Design is the most expressive mode in the studio. "
        "Instead of selecting a preset, you describe the voice in plain English: "
        "its tone, energy, accent, age, or personality. "
        "The AI interprets your description and synthesises a unique voice on demand — "
        "no reference recording required. "
        "This opens up unlimited voice personas for any content pipeline."
    )

    await page.select_option("#vd-language", "English")

    await page.locator("#vd-instruct").fill(
        "A precise, measured lead architect's voice — calm authority, slight gravitas, "
        "mid-range pitch, as if presenting a technical design to a senior engineering team"
    )

    await page.locator("button", has_text="Design & Generate").click()
    await _wait_for_result(page, "vd-result", "Voice Design")


# ---------------------------------------------------------------------------
# Test: Voice Clone  (LJ Speech — public domain)
# ---------------------------------------------------------------------------

async def test_voice_clone(page) -> None:
    """
    Clones LJ Speech (Linda Johnson) — the canonical public-domain TTS benchmark
    voice, public domain, used to evaluate synthesis quality in hundreds of papers.

    The synthesis text describes what voice cloning is and how it works locally,
    so the demo itself explains the feature in the cloned voice.
    """
    print("\n[Voice Clone] Starting…")
    reference = _ensure_reference_audio()

    await page.locator("button.tab", has_text="Voice Clone").click()

    # Upload the reference audio (hidden input, triggered via set_input_files)
    await page.locator("#vc-audio").set_input_files(str(reference))

    # Provide the transcript of the LJ Speech reference (standard opening line)
    # If using the synthetic fallback, leave this descriptive but harmless
    await page.locator("#vc-ref-text").fill(
        "Printing, in the only sense with which we are at present concerned, "
        "differs from most if not from all the arts and crafts."
    )

    await page.locator("#vc-text").fill(
        "You are hearing Local TTS Studio's voice clone feature. "
        "The model extracted this speaker's vocal characteristics from a short reference recording "
        "and applied them to this entirely new text — completely offline. "
        "The LJ Speech dataset, recorded by Linda Johnson, is one of the most cited "
        "public-domain voices in TTS research, used to benchmark synthesis quality worldwide. "
        "Your reference audio never leaves your machine."
    )

    await page.select_option("#vc-language", "English")

    # Leave X-Vector unchecked so the transcript improves clone fidelity
    xvec = page.locator("#vc-xvector")
    if await xvec.is_checked():
        await xvec.uncheck()

    await page.locator("button", has_text="Clone & Generate").click()
    await _wait_for_result(page, "vc-result", "Voice Clone")


# ---------------------------------------------------------------------------
# Test: Podcast Mode  (PM × Architect × User three-way conversation)
# ---------------------------------------------------------------------------

async def test_podcast_mode(page) -> None:
    """
    Three-speaker conversation that walks through the product from every angle:
      Alex  (PM)        → Aiden, American male, English — product vision & benefits
      Sam   (Architect) → Ryan, Dynamic male, English   — technical design decisions
      Jordan (User)     → Serena, Warm female, Chinese  — real-world usage story

    The script covers: architecture rationale, local-first privacy, one-command
    install, multi-voice rendering, and deterministic mode.
    """
    print("\n[Podcast Mode] Starting…")

    await page.locator("button.tab", has_text="Podcast Mode").click()

    # Title
    title = page.locator("#podcast-title")
    await title.triple_click()
    await title.fill("Local TTS Studio — Product, Architecture & Real-World Use")

    # -----------------------------------------------------------------------
    # Rebuild speakers via JS (the UI is dynamically rendered with innerHTML)
    # -----------------------------------------------------------------------
    await page.evaluate("""() => {
        // initPodcastMode() resets counters and arrays; then we clear its demo data
        initPodcastMode();
        speakers.length = 0;
        segments.length = 0;
        speakerCounter = 0;
        segmentCounter = 0;

        // Alex — Product Manager
        addSpeaker('Alex (PM)', 'custom', {voice_id: 'Aiden'}, 'enthusiastic and clear, like a product director presenting to stakeholders');

        // Sam — Software Architect
        addSpeaker('Sam (Architect)', 'custom', {voice_id: 'Ryan'}, 'measured and precise, like a senior engineer walking through a design');

        // Jordan — End User
        addSpeaker('Jordan (User)', 'custom', {voice_id: 'Serena'}, 'warm and genuine, sharing a real experience');
    }""")

    # Retrieve the dynamically assigned speaker IDs
    speaker_ids: list[str] = await page.evaluate(
        "() => speakers.map(s => s.id)"
    )
    alex_id, sam_id, jordan_id = speaker_ids

    # -----------------------------------------------------------------------
    # Build the script — five lines, interleaved across all three speakers
    # -----------------------------------------------------------------------
    await page.evaluate(f"""() => {{
        addSegment('{alex_id}',
            "Welcome to Local TTS Studio. I'm Alex, the product manager. " +
            "This studio was built for developers, creators, and podcasters " +
            "who want studio-quality audio without cloud API bills or privacy trade-offs. " +
            "Everything runs on your own hardware.",
            10);

        addSegment('{sam_id}',
            "I'm Sam, the lead architect. We chose a hexagonal design: " +
            "the Qwen3-TTS synthesis core has zero I/O dependencies. " +
            "Adding a new TTS engine is just a matter of implementing a provider protocol — " +
            "no changes to the business logic or the API layer needed.",
            20);

        addSegment('{jordan_id}',
            "I'm Jordan. I was spending sixty dollars a month on a cloud TTS API. " +
            "With Local TTS Studio I generate full podcast episodes, clone reference voices, " +
            "and mix in background music — all from my browser, all on my own machine. " +
            "The install was literally one command.",
            30);

        addSegment('{alex_id}',
            "And in Podcast Mode, each speaker can have a different voice type: " +
            "a preset profile, an AI-described voice, or a voice cloned from any recording. " +
            "The renderer produces a single mixed audio file in the correct order, " +
            "every time.",
            40);

        addSegment('{sam_id}',
            "Deterministic mode is my favourite detail. " +
            "Same text, same configuration, identical output on every render. " +
            "That is what makes Local TTS Studio suitable for automated content pipelines " +
            "where reproducibility actually matters.",
            50);
    }}""")

    # Confirm deterministic mode is on
    det = page.locator("#podcast-deterministic")
    if not await det.is_checked():
        await det.check()

    await page.locator("button", has_text="Render Podcast").click()
    await _wait_for_result(page, "podcast-result", "Podcast Mode")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_demo() -> None:
    reference = _ensure_reference_audio()
    print(f"Reference audio: {reference}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=250)
        page = await browser.new_page()

        await page.goto(BASE_URL)
        await page.wait_for_load_state("networkidle")
        print(f"Loaded: {BASE_URL}")

        await test_custom_voice(page)
        await test_voice_design(page)
        await test_voice_clone(page)
        await test_podcast_mode(page)

        print("\n✅  All four modes completed.")
        print("   Leave the browser open to inspect results, or press Ctrl-C to exit.")
        await page.wait_for_timeout(10_000)  # pause so results are visible
        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_demo())
