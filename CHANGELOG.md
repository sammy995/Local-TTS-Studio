# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and this changelog.
- `pyproject.toml` with a `local-tts` console entry point and a `dev` extra.
- One-command install scripts: `install.ps1` (Windows) and `install.sh` (macOS/Linux).

### Planned
- Pluggable multi-engine TTS (Piper, Kokoro, and more) alongside Qwen3-TTS.
- Local LLM podcast-script generation (Ollama) — turn a topic or document into a multi-voice script.
- A/B engine comparison and a Settings panel.

## [3.0.0] — Timeline Studio

### Added
- **Timeline Studio**: multi-track editor with speech + music lanes.
- Music library search/import from Jamendo, Freesound, and Openverse.
- Automatic audio ducking, per-track loop / trim / fade, and live duration preview.
- MP3 output for podcasts; MP3/M4A input for voice cloning; animated progress bar.
- Fault-tolerant rendering: a failed segment becomes a silence placeholder instead of crashing.

## [2.0.0] — Podcast Mode

### Added
- **Podcast Mode**: multi-speaker, script-to-audio compiler with per-segment timing/volume/emotion.
- Deterministic rendering — the same script produces identical audio.

### Changed
- Full architectural refactor to a hexagonal (ports & adapters) layout.

## [1.0.0] — Initial release

### Added
- Local TTS Studio with Qwen3-TTS integration.
- Custom Voice (9 multilingual presets), Voice Design, and Voice Clone (ICL + x-vector).
- FastAPI backend with a single-file HTML frontend.
