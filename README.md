<div align="center">

# 🎙️ Local TTS Studio

**The self-hosted ElevenLabs.** Open-source, 100% offline speech studio — voice cloning,
multi-speaker podcasts, and timeline music mixing, all running on your own GPU.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qwen3-TTS](https://img.shields.io/badge/Qwen3--TTS-1.7B-purple?logo=huggingface)](https://huggingface.co/collections/Qwen/qwen3-tts)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![100% Local](https://img.shields.io/badge/100%25-Local%20%26%20Private-black)](#why-local-tts-studio)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick Start](#-quick-start) · [Features](#-features) · [Roadmap](#-roadmap) · [API](#-api-reference) · [Contributing](#-contributing)

</div>

---

## Why Local TTS Studio?

| Cloud TTS services | Local TTS Studio |
|---|---|
| Per-character / per-minute billing | **$0 marginal cost** — unlimited generations |
| Your audio leaves your network | **100% local & private** — nothing leaves your machine |
| Rate limits and vendor lock-in | **No API keys required** for core TTS |
| Limited voice control | **Design, clone, or pick** from 9 preset voices |
| Single voice per request | **Multi-speaker podcast studio** with music mixing |

> Think: a local, self-hosted ElevenLabs alternative you fully control.

---

## 🚀 Quick Start

**Requirements:** Python 3.10+ · NVIDIA GPU with 6 GB+ VRAM recommended (CPU works, but slowly) · ~15 GB disk.

```bash
git clone https://github.com/sammy995/Local-TTS-Studio.git
cd Local-TTS-Studio
```

Then run the installer for your OS — it creates a virtual environment, installs everything
(including ffmpeg), and launches the app:

```bash
# Windows (PowerShell)
./install.ps1

# macOS / Linux
chmod +x install.sh && ./install.sh
```

Open **http://localhost:8000** — that's it.

> The first run downloads the Qwen3-TTS model weights (~10 GB). Subsequent starts take ~30 s.

<details>
<summary><strong>📋 Manual install (pip / venv)</strong></summary>

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

pip install -e .          # installs the package + the `local-tts` command
# or: pip install -r requirements.txt

local-tts                 # or: python run_local.py
```

ffmpeg is required for MP3/M4A export (WAV works without it):

```bash
# Windows:  winget install Gyan.FFmpeg
# macOS:    brew install ffmpeg
# Linux:    sudo apt install ffmpeg
```

</details>

<details>
<summary><strong>⚙️ Hardware guide</strong></summary>

| | Minimum | Recommended |
|---|---|---|
| **GPU** | GTX 1660 (6 GB VRAM) | RTX 3060+ (8 GB+ VRAM) |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 15 GB | 20 GB |

Low on VRAM? Set `default_size: "0.6B"` in [`config.yaml`](config.yaml).

</details>

<details>
<summary><strong>⬇️ Optional: pre-download models</strong></summary>

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz        --local-dir models/Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base        --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

</details>

---

## ✨ Features

### Single-voice generation
- **Custom Voice** — 9 multilingual presets (English, Chinese, Japanese, Korean + 7 more)
- **Voice Design** — describe a voice in natural language and generate it
- **Voice Clone** — clone any voice from a short audio sample (MP3 / M4A / WAV), in ICL or x-vector mode

### Podcast Mode — a script-to-audio compiler
- Up to **10 speakers** per production, mixing preset / designed / cloned voices
- Per-segment **timing, volume, and emotion** control
- **Deterministic rendering** — the same script always produces identical audio
- **Fault-tolerant** — a failed segment becomes a silence placeholder instead of crashing the render

### Timeline Studio — multi-track mixing
- **Speech + music lanes** on a single timeline
- **Music library** — search royalty-free tracks from Jamendo, Freesound, and Openverse
- **Automatic ducking** — music lowers under speech automatically
- **Loop, trim, fade** per track, with a **live duration preview**

### Output
- WAV · MP3 · FLAC · real-time progress over Server-Sent Events

---

## 🗺️ Roadmap

Local TTS Studio is built on a clean provider architecture, and these are next:

- [ ] **Pluggable TTS engines** — Piper (fast/CPU), Kokoro, XTTS-v2, and more, alongside Qwen3-TTS
- [ ] **Local LLM script generation** — turn a topic or a pasted document into a multi-voice podcast script (Ollama) — a fully local NotebookLM
- [ ] **A/B engine comparison** — synthesize the same text across engines and compare
- [ ] **Settings panel** — pick engine / device / output directory in the UI
- [ ] **CLI** + OpenAI-compatible `/v1/audio/speech` endpoint for scripting and drop-in use

Want one of these? [Open an issue](https://github.com/sammy995/Local-TTS-Studio/issues) or jump in — see [Contributing](#-contributing).

---

## 📸 Screenshots

| Custom Voice | Voice Design |
|:---:|:---:|
| ![Custom Voice](https://github.com/user-attachments/assets/40241e98-e521-49c7-b03c-253a7b4cba4e) | ![Voice Design](https://github.com/user-attachments/assets/24281f16-3575-4e5a-950a-fc3d144345b1) |

| Voice Clone | Podcast Mode |
|:---:|:---:|
| ![Voice Clone](https://github.com/user-attachments/assets/d812fc31-266b-4bb4-b8e3-92c9ecbad1c9) | ![Podcast Mode](https://github.com/user-attachments/assets/6d3faa49-cb36-4b62-aea3-5336a2008055) |

---

## 🏗️ Architecture

Hexagonal (ports & adapters) — core logic has zero framework dependencies. Full details in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

```
Local-TTS-Studio/
├── core/                   # Pure compute (no I/O): tts_engine, model_manager, audio_pipeline
├── services/               # Stateless orchestration: tts, podcast, music
├── infra/                  # Side-effect adapters: storage
├── runtimes/               # FastAPI server + config loader
├── simple-ui.html          # Single-file frontend (no build step)
├── config.yaml             # All tunables in one place
└── run_local.py            # Entry point
```

---

## ⚙️ Configuration

All settings live in [`config.yaml`](config.yaml):

```yaml
models:
  default_size: "1.7B"     # or "0.6B" for lower VRAM

# Optional — only needed for Timeline Studio's online music search
music_apis:
  jamendo:   { client_id: "" }   # free → https://devportal.jamendo.com
  freesound: { token: "" }       # free → https://freesound.org/apiv2/apply
  openverse: { token: "" }       # optional (anonymous works)
```

> Copy `.env.example` → `.env` for secrets. Both files are read; `.env` is git-ignored.

---

## 📡 API Reference

Everything is served from `http://localhost:8000`. TTS endpoints accept multipart form fields.

### TTS

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/tts/custom-voice` | Generate with a preset speaker |
| `POST` | `/api/tts/voice-design` | Generate from a voice description |
| `POST` | `/api/tts/voice-clone` | Clone a voice from reference audio |
| `POST` | `/api/voice-prompt/save` | Save a reusable voice prompt |
| `GET`  | `/api/audio/download/{filename}` | Download generated audio |

### Podcast

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/podcast/render` | Render a multi-speaker script (v2) |
| `POST` | `/api/podcast/render/v3` | Render a timeline (speech + music, v3) |

### Music library

| Method | Endpoint | Description |
|---|---|---|
| `GET`    | `/api/music/search` | Search Jamendo / Freesound / Openverse |
| `POST`   | `/api/music/import` | Download & cache a track |
| `GET`    | `/api/assets` | List cached assets |
| `POST`   | `/api/assets/upload` | Upload your own audio |
| `DELETE` | `/api/assets/{asset_id}` | Remove a cached asset |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/system/info` | GPU / device / loaded models |
| `GET` | `/api/speakers` · `/api/languages` | Preset voices & supported languages |
| `GET` | `/api/progress/{session_id}` | Live progress (Server-Sent Events) |

<details>
<summary><strong>Example: generate speech</strong></summary>

```bash
curl -X POST http://localhost:8000/api/tts/custom-voice \
  -F "text=Hello world, this is Local TTS Studio." \
  -F "speaker=Serena" \
  -F "language=English" \
  --output hello.wav
```

</details>

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| **CUDA out of memory** | Set `default_size: "0.6B"` in `config.yaml`, or close other GPU apps |
| **MP3 / M4A export fails** | Install ffmpeg (see Manual install above). WAV always works. |
| **First generation is slow (~30 s)** | Normal — model loading. Later runs: ~8–12 s on an RTX 3060. |
| **Flash Attention warning** | Safe to ignore — it's an optional optimization. |
| **Music search returns nothing** | Add a free API key to `config.yaml` → `music_apis`. |
| **Voice clone sounds different each run** | Provide `ref_text` with `ref_audio` to use the more stable ICL mode. |

---

## 🤝 Contributing

Contributions are welcome — from bug reports to whole new TTS engines. Start with
[`CONTRIBUTING.md`](CONTRIBUTING.md) and the [open issues](https://github.com/sammy995/Local-TTS-Studio/issues).

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). Found a security issue? See [`SECURITY.md`](SECURITY.md).

---

## 📄 License

MIT — see [`LICENSE`](LICENSE).

**Model license:** Qwen3-TTS models are subject to their [own license terms](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE). This project is a UI/orchestration layer and does not claim ownership of the underlying models.

---

## 🙏 Acknowledgements

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** — the TTS model powering generation
- **[Jamendo](https://www.jamendo.com)**, **[Freesound](https://freesound.org)**, **[Openverse](https://openverse.org)** — royalty-free music APIs
- **[FastAPI](https://fastapi.tiangolo.com)** — the async Python web framework

<div align="center">

**If this project is useful to you, please ⭐ star it — it really helps.**

</div>
