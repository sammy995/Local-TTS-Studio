"""
Capture runner for e2e_playwright_demo.py
Runs headless, takes screenshots after each mode, writes results to tests/results/.
"""

import asyncio
import math
import struct
import sys
import time
import wave
from pathlib import Path

from playwright.async_api import async_playwright, Page

BASE_URL = "http://localhost:8000/"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"
REFERENCE_AUDIO = FIXTURES_DIR / "lj_reference.wav"
GENERATION_TIMEOUT_MS = 150_000

results = {}  # mode -> {status, note, duration_s}


def _ensure_reference_audio() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if REFERENCE_AUDIO.exists():
        return REFERENCE_AUDIO
    print("[setup] Generating synthetic reference WAV (no lj_reference.wav found)…")
    sample_rate, duration_s, frequency = 22_050, 3, 440
    with wave.open(str(REFERENCE_AUDIO), "w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        for i in range(sample_rate * duration_s):
            wf.writeframes(struct.pack("<h", int(32_767 * math.sin(2 * math.pi * frequency * i / sample_rate))))
    return REFERENCE_AUDIO


async def _wait_for_result(page: Page, result_id: str, label: str) -> tuple[bool, str]:
    try:
        await page.wait_for_function(
            f"() => document.querySelector('#{result_id} audio, #{result_id} .status.error') !== null",
            timeout=GENERATION_TIMEOUT_MS,
        )
    except Exception as e:
        return False, f"Timeout waiting for result: {e}"
    has_audio = await page.query_selector(f"#{result_id} audio")
    has_error = await page.query_selector(f"#{result_id} .status.error")
    if has_audio:
        src = await has_audio.get_attribute("src") or ""
        return True, f"audio player present, src={src[:60]}"
    if has_error:
        text = (await has_error.inner_text()).strip()
        return False, f"UI error: {text}"
    return False, "Unknown state"


async def _screenshot(page: Page, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.png"
    await page.screenshot(path=str(path), full_page=False)
    return path


# ── Custom Voice ────────────────────────────────────────────────────────────

async def run_custom_voice(page: Page) -> None:
    label = "Custom Voice"
    print(f"\n[{label}] Starting…")
    t0 = time.time()
    await page.locator("#cv-text").fill(
        "Local TTS Studio is built on a hexagonal architecture with three distinct layers. "
        "The core layer performs pure Qwen3-TTS synthesis and contains zero I/O dependencies. "
        "The services layer orchestrates validation, storage, and multi-speaker rendering. "
        "The runtime layer exposes a FastAPI server that brings it all together. "
        "Everything runs on your hardware — your audio never leaves your machine."
    )
    await page.select_option("#cv-speaker", "Ryan")
    await page.select_option("#cv-language", "English")
    await page.locator("#cv-instruct").fill(
        "Speak clearly and confidently, like a senior engineer presenting an architecture diagram"
    )
    await _screenshot(page, "01_custom_voice_before")
    await page.locator("button", has_text="Generate Speech").click()
    ok, note = await _wait_for_result(page, "cv-result", label)
    await _screenshot(page, "02_custom_voice_after")
    elapsed = round(time.time() - t0, 1)
    results[label] = {"ok": ok, "note": note, "duration_s": elapsed}
    status = "✅" if ok else "❌"
    print(f"[{label}] {status} {note}  ({elapsed}s)")


# ── Voice Design ─────────────────────────────────────────────────────────────

async def run_voice_design(page: Page) -> None:
    label = "Voice Design"
    print(f"\n[{label}] Starting…")
    t0 = time.time()
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
    await _screenshot(page, "03_voice_design_before")
    await page.locator("button", has_text="Design & Generate").click()
    ok, note = await _wait_for_result(page, "vd-result", label)
    await _screenshot(page, "04_voice_design_after")
    elapsed = round(time.time() - t0, 1)
    results[label] = {"ok": ok, "note": note, "duration_s": elapsed}
    status = "✅" if ok else "❌"
    print(f"[{label}] {status} {note}  ({elapsed}s)")


# ── Voice Clone ──────────────────────────────────────────────────────────────

async def run_voice_clone(page: Page) -> None:
    label = "Voice Clone"
    print(f"\n[{label}] Starting…")
    t0 = time.time()
    reference = _ensure_reference_audio()
    await page.locator("button.tab", has_text="Voice Clone").click()
    await page.locator("#vc-audio").set_input_files(str(reference))
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
    xvec = page.locator("#vc-xvector")
    if await xvec.is_checked():
        await xvec.uncheck()
    await _screenshot(page, "05_voice_clone_before")
    await page.locator("button", has_text="Clone & Generate").click()
    ok, note = await _wait_for_result(page, "vc-result", label)
    await _screenshot(page, "06_voice_clone_after")
    elapsed = round(time.time() - t0, 1)
    results[label] = {"ok": ok, "note": note, "duration_s": elapsed}
    status = "✅" if ok else "❌"
    print(f"[{label}] {status} {note}  ({elapsed}s)")


# ── Podcast Mode ─────────────────────────────────────────────────────────────

async def run_podcast_mode(page: Page) -> None:
    label = "Podcast Mode"
    print(f"\n[{label}] Starting…")
    t0 = time.time()
    await page.locator("button.tab", has_text="Podcast Mode").click()
    title = page.locator("#podcast-title")
    await title.triple_click()
    await title.fill("Local TTS Studio — Product, Architecture & Real-World Use")

    await page.evaluate("""() => {
        initPodcastMode();
        speakers.length = 0;
        segments.length = 0;
        speakerCounter = 0;
        segmentCounter = 0;
        addSpeaker('Alex (PM)', 'custom', {voice_id: 'Aiden'}, 'enthusiastic and clear, like a product director');
        addSpeaker('Sam (Architect)', 'custom', {voice_id: 'Ryan'}, 'measured and precise, like a senior engineer');
        addSpeaker('Jordan (User)', 'custom', {voice_id: 'Serena'}, 'warm and genuine, sharing a real experience');
    }""")

    speaker_ids = await page.evaluate("() => speakers.map(s => s.id)")
    alex_id, sam_id, jordan_id = speaker_ids

    await page.evaluate(f"""() => {{
        addSegment('{alex_id}',
            "Welcome to Local TTS Studio. I'm Alex, the product manager. " +
            "This studio was built for developers, creators, and podcasters " +
            "who want studio-quality audio without cloud API bills or privacy trade-offs. " +
            "Everything runs on your own hardware.", 10);

        addSegment('{sam_id}',
            "I'm Sam, the lead architect. We chose a hexagonal design: " +
            "the Qwen3-TTS synthesis core has zero I/O dependencies. " +
            "Adding a new TTS engine is just implementing a provider protocol — " +
            "no changes to the business logic or the API layer needed.", 20);

        addSegment('{jordan_id}',
            "I'm Jordan. I was spending sixty dollars a month on a cloud TTS API. " +
            "With Local TTS Studio I generate full podcast episodes, clone reference voices, " +
            "and mix in background music — all from my browser, all on my own machine. " +
            "The install was literally one command.", 30);

        addSegment('{alex_id}',
            "And in Podcast Mode, each speaker can have a different voice type: " +
            "a preset profile, an AI-described voice, or a voice cloned from any recording. " +
            "The renderer produces a single mixed audio file in the correct order, every time.", 40);

        addSegment('{sam_id}',
            "Deterministic mode is my favourite detail. " +
            "Same text, same configuration, identical output on every render. " +
            "That is what makes Local TTS Studio suitable for automated content pipelines " +
            "where reproducibility actually matters.", 50);
    }}""")

    det = page.locator("#podcast-deterministic")
    if not await det.is_checked():
        await det.check()

    await _screenshot(page, "07_podcast_before")
    await page.locator("button", has_text="Render Podcast").click()
    ok, note = await _wait_for_result(page, "podcast-result", label)
    await _screenshot(page, "08_podcast_after")
    elapsed = round(time.time() - t0, 1)
    results[label] = {"ok": ok, "note": note, "duration_s": elapsed}
    status = "✅" if ok else "❌"
    print(f"[{label}] {status} {note}  ({elapsed}s)")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    _ensure_reference_audio()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto(BASE_URL)
        await page.wait_for_load_state("networkidle")
        print(f"App loaded: {BASE_URL}")
        await _screenshot(page, "00_home")

        await run_custom_voice(page)
        await run_voice_design(page)
        await run_voice_clone(page)
        await run_podcast_mode(page)

        await browser.close()

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v["ok"])
    total = len(results)
    for mode, r in results.items():
        icon = "✅" if r["ok"] else "❌"
        print(f"{icon}  {mode:<20} {r['duration_s']:>6.1f}s  {r['note']}")
    print("-"*60)
    print(f"{'PASS' if passed == total else 'FAIL'}  {passed}/{total} modes succeeded")
    print(f"Screenshots saved to: {RESULTS_DIR}/")
    return 0 if passed == total else 1


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
