<div align="center">

# 🎙️ Local TTS Studio

**Open-source, GPU-accelerated speech studio for single-voice generation and multi-speaker podcast production — 100% offline.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Qwen3-TTS](https://img.shields.io/badge/Qwen3--TTS-1.7B-purple?logo=huggingface)](https://huggingface.co/collections/Qwen/qwen3-tts)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Think: a local, self-hosted ElevenLabs alternative you fully control.*

[Getting Started](#-quick-start) · [Features](#-features) · [Architecture](#-architecture) · [Configuration](#%EF%B8%8F-configuration) · [API Reference](#-api-reference) · [Contributing](#-contributing)

</div>

---

## Why Local TTS Studio?

| Cloud TTS services | Local TTS Studio |
|---|---|
| Per-minute billing that scales with usage | **$0 marginal cost** — run unlimited generations |
| Audio leaves your network | **100% local & private** — nothing leaves your GPU |
| Rate limits and vendor lock-in | **No API keys required** for core TTS |
| Limited voice customization | **Design, clone, or pick** from 9 preset voices |

**Performance**: ~8–12s per generation on RTX 3060 · bfloat16 inference · 24 kHz output

---

## ✨ Features

### Single Voice Generation
- **Custom Voice** — 9 multilingual presets (English, Chinese, Japanese, Korean, + 7 more)
- **Voice Design** — describe a voice in natural language and generate it
- **Voice Clone** — clone any voice from a short audio sample (ICL or x-vector modes)

### Podcast Mode — Script-to-Audio Compiler
- Up to **10 speakers** per production with mixed voice types
- Per-segment **timing, volume, and emotion** control
- **Deterministic rendering** — same script produces identical audio
- **Fault-tolerant pipeline** — failed segments get silence placeholders instead of crashing the entire render

### v3: Timeline Studio *(New)*
- **Multi-track timeline editor** with speech + music lanes
- **Music Library** — search royalty-free tracks from Jamendo, Freesound, and Openverse
- **Audio ducking** — music auto-lowers under speech segments
- **Loop, trim, fade** — per-track audio manipulation
- **Live timeline preview** — estimated duration updates as you edit

---

## Screenshots

| Custom Voice | Voice Design |
|:---:|:---:|
| ![Custom Voice](https://github.com/user-attachments/assets/00e23608-f04c-45fa-ad49-54972c773118) | ![Voice Design](https://github.com/user-attachments/assets/e0e5faa8-57d1-4f00-8257-128028b9acfc) |

| Voice Clone | Podcast Mode |
|:---:|:---:|
| ![Voice Clone](https://github.com/user-attachments/assets/1af92da4-8beb-4c42-871b-3cd7c46013f4) | ![Podcast Mode](https://github.com/user-attachments/assets/00e23608-f04c-45fa-ad49-54972c773118) |

---

## 🚀 Quick Start

**Requirements**: NVIDIA GPU (6 GB+ VRAM) · Python 3.10+ · ~15 GB disk space

```bash
git clone https://github.com/sammy995/Local-TTS-Studio.git
cd Local-TTS-Studio
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
python run_local.py
```

Open **http://localhost:8000** — that's it.

> First run downloads models automatically (~10 GB). Subsequent starts take ~30 s for model loading.

<details>
<summary><strong>📋 Detailed Installation</strong></summary>

### Hardware

| | Minimum | Recommended |
|---|---|---|
| **GPU** | GTX 1660 (6 GB VRAM) | RTX 3060+ (8 GB+ VRAM) |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 15 GB | 20 GB |

### Step-by-Step

```bash
# 1. Clone
git clone https://github.com/sammy995/Local-TTS-Studio.git
cd Local-TTS-Studio

# 2. Create environment
conda create -n local-tts python=3.12 -y
conda activate local-tts
pip install -r requirements.txt

# 3. Install ffmpeg (required for MP3/M4A export)
conda install -c conda-forge ffmpeg -y
# Alternative (Windows): winget install Gyan.FFmpeg

# 4. Launch
python run_local.py
```

### Optional: Pre-download Models

Skip the first-run download by pulling models ahead of time:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz       --local-dir models/Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base        --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

</details>

---

## 🏗️ Architecture

Hexagonal (ports & adapters) layout — core logic has zero framework dependencies:

```
Local-TTS-Studio/
├── core/                   # Pure domain logic (no I/O)
│   ├── tts_engine.py       #   TTS generation interface
│   ├── model_manager.py    #   Model loading & lifecycle
│   └── audio_pipeline.py   #   Mix, duck, loop, trim, fade, resample
├── services/               # Stateless orchestration
│   ├── tts_service.py      #   Single-voice generation
│   ├── podcast_service.py  #   Multi-speaker render pipeline
│   ├── podcast_models.py   #   Pydantic models for podcast scripts
│   └── music_service.py    #   Jamendo / Freesound / Openverse client
├── infra/                  # Side-effect adapters
│   └── storage.py          #   File I/O & output management
├── runtimes/               # Delivery mechanism
│   ├── local_api.py        #   FastAPI server & endpoints
│   └── config_loader.py    #   YAML config reader
├── simple-ui.html          # Single-file frontend (~3 400 lines)
├── config.yaml             # All tunables in one place
├── requirements.txt
└── run_local.py            # Entry point
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Single-file HTML frontend | Zero build step — open and go |
| Hexagonal backend | Core logic is testable without FastAPI |
| Speaker-stable deterministic seeds | Same speaker always gets the same voice timbre |
| Per-segment fault tolerance | One failed TTS segment can't crash the whole podcast |
| Music ducking in `audio_pipeline` | Keeps mixing logic out of the render loop |

---

## ⚙️ Configuration

All settings live in [`config.yaml`](config.yaml):

```yaml
# Model size — switch to 0.6B if running low on VRAM
models:
  default_size: "1.7B"    # or "0.6B"

# Music library API keys (optional — for Timeline Studio)
music_apis:
  jamendo:
    client_id: ""          # Free → https://devportal.jamendo.com
  freesound:
    token: ""              # Free → https://freesound.org/apiv2/apply
  openverse:
    token: ""              # Optional (anonymous access works)
```

> **Tip**: Copy `.env.example` → `.env` for secret management. The app reads both files.

---

## 📡 API Reference

All endpoints are served at `http://localhost:8000`.

### Core TTS

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/tts/generate` | Single-voice generation (custom, design, or clone) |
| `GET`  | `/api/v1/voices` | List available preset voices |
| `GET`  | `/api/v1/models/status` | Model load state & GPU memory |

### Podcast — v2 (Script Mode)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v2/podcast/render` | Render a multi-speaker script to audio |

### Podcast — v3 (Timeline Studio)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v3/podcast/render` | Render a timeline (speech + music tracks) |

### Music Library

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/api/v1/music/search` | Search royalty-free music (Jamendo, Freesound, Openverse) |
| `POST` | `/api/v1/music/download` | Download & cache a track locally |
| `GET`  | `/api/v1/music/assets` | List cached music assets |

<details>
<summary><strong>Example: Generate speech</strong></summary>

```bash
curl -X POST http://localhost:8000/api/v1/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world, this is Local TTS Studio.",
    "mode": "custom_voice",
    "speaker": "Serena",
    "language": "English"
  }'
```

</details>

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| **CUDA out of memory** | Set `default_size: "0.6B"` in `config.yaml`, or close other GPU programs |
| **MP3 / M4A export fails** | Install ffmpeg: `conda install -c conda-forge ffmpeg -y` |
| **First generation slow (~30 s)** | Normal — model loading. Subsequent runs: 8–12 s |
| **Flash Attention warning** | Safe to ignore (optional optimization) |
| **Music search returns no results** | Add API keys to `config.yaml` → `music_apis` section |
| **Voice clone sounds different each time** | Provide `ref_text` alongside `ref_audio` for ICL mode (more stable than x-vector) |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create a branch** — `git checkout -b feature/your-feature`
3. **Make changes** — follow the existing hexagonal structure
4. **Test** — make sure `python -c "import py_compile; py_compile.compile('runtimes/local_api.py')"` passes
5. **Submit a PR** with a clear description

### Areas We'd Love Help With

- [ ] Streaming audio output (chunked WAV)
- [ ] WebSocket progress events during render
- [ ] Additional TTS model backends (Bark, XTTS-v2)
- [ ] Docker image for one-command deployment
- [ ] Test suite (pytest) for core/ and services/

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

**Model License**: Qwen3-TTS models are subject to their [original license terms](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE). This application is a UI wrapper and does not claim ownership of underlying models.

---

## 🙏 Acknowledgements

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** — the 1.7B parameter TTS model that powers all generation
- **[Jamendo](https://www.jamendo.com)**, **[Freesound](https://freesound.org)**, **[Openverse](https://openverse.org)** — royalty-free music APIs
- **[FastAPI](https://fastapi.tiangolo.com)** — the async Python web framework

**Resources**: [Qwen3-TTS Paper](https://arxiv.org/abs/2601.15621) · [Models on HuggingFace](https://huggingface.co/collections/Qwen/qwen3-tts) · [Official Repo](https://github.com/QwenLM/Qwen3-TTS)
