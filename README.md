# Local TTS Studio v2

**Open-source, local-first speech studio.**

Generate production-ready speech and multi-speaker podcasts — entirely offline on your own GPU.

Think: a local, open-source alternative to ElevenLabs that you fully control.

Built to eliminate per-minute API costs and give you full control over your voice stack.

No APIs. No per-minute costs. No cloud lock-in.

⚡ 8–12s generation  
🎙️ Multi-speaker podcast compiler  
🔒 100% local & private  
💰 $0 marginal cost per minute

Powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 1.7B

---

## 🎙️ Podcast Mode (flagship)

**Script-to-audio compiler for deterministic multi-speaker production.**

Turn scripts into ready-to-publish audio in one click.

- Up to 10 speakers
- Mix preset, designed, and cloned voices
- Per-line timing & volume control
- Deterministic output (same script → identical audio)
- ~15–20s render for a 30s conversation on RTX 3060

**Perfect for:**  
Podcasts • Audiobooks • E-learning • Marketing videos

### Screenshots

**🎭 Custom Voice** – preset voice selection  
![Custom Voice interface](https://github.com/user-attachments/assets/00e23608-f04c-45fa-ad49-54972c773118)

**🎨 Voice Design** – describe-to-generate interface  
![Voice Design interface](https://github.com/user-attachments/assets/e0e5faa8-57d1-4f00-8257-128028b9acfc)

**🔊 Voice Clone** – upload and clone workflow  
![Voice Clone interface](https://github.com/user-attachments/assets/1af92da4-8beb-4c42-871b-3cd7c46013f4)

---

## Single Voice Generation

- 9 multilingual preset voices
- Voice design from text descriptions
- Voice cloning from short audio samples

---

## Quick Start

**Requirements**: NVIDIA GPU (6GB+ VRAM), Python 3.10+, 15GB disk space

```bash
git clone https://github.com/sammy995/Local-TTS-Studio.git
cd Local-TTS-Studio
pip install -r requirements.txt
conda install -c conda-forge ffmpeg -y
python run_local.py
```

Open: [http://localhost:8000](http://localhost:8000)

That's it.

_First run loads the model (~30s). After that, each generation takes ~8–12s._

<details>
<summary><b>📋 Full Installation Guide</b></summary>

### Hardware Requirements

**Minimum**: GTX 1660 (6GB VRAM), 16GB RAM  
**Recommended**: RTX 3060 (8GB+ VRAM), 32GB RAM

### Step-by-Step

**1. Clone repository**
```bash
git clone https://github.com/sammy995/Local-TTS-Studio.git
cd Local-TTS-Studio
```

**2. Create Python environment**
```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -r requirements.txt
```

**3. Install ffmpeg** (for M4A/MP3 support)
```bash
conda install -c conda-forge ffmpeg -y
# Or: winget install Gyan.FFmpeg (Windows)
```

**4. Run**
```bash
python run_local.py
```

Models download automatically on first use (~10GB, 5-10 minutes).

**Optional - Pre-download models:**
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

</details>

---

## Usage

### Podcast Mode

1. Add speakers
2. Write script
3. Click render

### Single Voice

1. Pick voice
2. Enter text
3. Generate

**Tips:**
- 500–1000ms pauses sound natural
- Adjust volume per line (0.8x–1.2x)
- Deterministic mode guarantees identical output

## Configuration

**Out of memory?** Edit `config.yaml`:

```yaml
models:
  default_size: "0.6B"  # Change from 1.7B → 0.6B
```

---

## Troubleshooting

**CUDA out of memory**: Edit `config.yaml` → `default_size: "0.6B"` or close GPU programs  
**M4A/MP3 files fail**: Install ffmpeg: `conda install -c conda-forge ffmpeg -y`  
**First generation slow**: Model loading (30s first time, 8–12s after)  
**Flash Attention warning**: Ignore it (optional optimization)

---

## Architecture

Simple layered architecture:

```
core/           # Generation logic only
services/       # Podcast compiler
infra/          # Storage + models
runtimes/       # FastAPI app
```

Models: Qwen3-TTS 1.7B (CustomVoice, VoiceDesign, Base)  
Inference: PyTorch + bfloat16 + CUDA

---

## Credits

Powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models from the Qwen Team.

**License**: Qwen3-TTS models are subject to their [original license terms](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE). This application is a UI wrapper and does not claim ownership of underlying models.

**Resources**: [Paper (arXiv)](https://arxiv.org/abs/2601.15621) • [Models (HuggingFace)](https://huggingface.co/collections/Qwen/qwen3-tts) • [Official Repo](https://github.com/QwenLM/Qwen3-TTS)
