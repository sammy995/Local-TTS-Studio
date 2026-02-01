# Local TTS Studio

Professional text-to-speech application with GPU acceleration. Powered by Qwen3-TTS 1.7B models with full support for Custom Voice, Voice Design, and Voice Cloning.

**⚡ Fast**: Local inference with GPU acceleration (8-12 seconds per generation)  
**🔒 Private**: 100% local processing, no cloud dependencies  

> **Powered by**: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) - State-of-the-art text-to-speech models from the Qwen Team  
> **License**: This application is provided as-is. Qwen3-TTS models are licensed under their respective terms. See [Credits](#credits) for details.  

## What This Does

This application lets you generate speech in three ways:

1. **Custom Voice**: Pick from 9 premium voices (male/female, various ages, different languages). Add instructions like "speak slowly" or "excited tone"
2. **Voice Design**: Describe what you want - "elderly woman with warm, caring voice" - and the AI creates it
3. **Voice Clone**: Upload any 3-second audio clip and clone that voice to speak any text

**Demo**: First generation takes 30-40 seconds (loading models). After that, each generation is ~8-12 seconds on RTX 3060.

## UI Screenshots

### Custom Voice
![Main TTS interface showing text input box, voice selection controls, and generate speech button](https://github.com/user-attachments/assets/00e23608-f04c-45fa-ad49-54972c773118)

### Voice Design
![Generated audio playback panel with waveform preview and download options](https://github.com/user-attachments/assets/e0e5faa8-57d1-4f00-8257-128028b9acfc)

### Voice Clone
![Settings or configuration screen with model parameters and runtime controls](https://github.com/user-attachments/assets/1af92da4-8beb-4c42-871b-3cd7c46013f4)

## Requirements

**Hardware (Minimum)**:
- NVIDIA GPU: GTX 1660 or better (6GB VRAM minimum, 8GB recommended)
- RAM: 16GB (32GB recommended)
- Storage: 15GB free space (models are ~10GB)

**Software**:
- Windows 10/11 or Linux (Ubuntu 20.04+)
- Python 3.10, 3.11, or 3.12
- CUDA-capable GPU with drivers installed

## Installation

### Step 1: Get the code
```bash
# Download or clone this project
cd path/to/TTS-opensource-app
```

### Step 2: Install Python dependencies
```bash
# Using conda (recommended)
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -r requirements.txt

# Or using venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux
pip install -r requirements.txt
```

### Step 3: Install ffmpeg (for M4A/MP3 support)
```bash
# Via conda (easiest)
conda install -c conda-forge ffmpeg -y

# Or Windows with winget
winget install Gyan.FFmpeg

# Or download from https://ffmpeg.org
```

### Step 4: Download models (~10GB, one-time)
**Option A - Automatic (recommended)**: Models download on first use

**Option B - Manual pre-download**:
```bash
# Install HuggingFace CLI
pip install -U "huggingface_hub[cli]"

# Download all models (takes 5-10 minutes)
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir models/Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/Qwen3-TTS-12Hz-1.7B-Base
```

### Step 5: Run the application
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# You'll see:
# INFO: Uvicorn running on http://0.0.0.0:8000
```

Access the application at `http://localhost:8000`

## How to Use

### Custom Voice (Fastest)
Example: "Hello world" spoken by a young female Chinese speaker

1. Type or paste your text
2. Select speaker: `xiaorui_enthusiastic_chinese_female_young`
3. Language: `Auto` (or pick specific language)
4. (Optional) Add instructions: "speak with excitement"
5. Click **Generate**
6. Audio generates in ~8-12 seconds

**Available speakers**: 9 voices covering male/female, young/middle-aged/old, Chinese/English/Multilingual

### Voice Design (Most Flexible)
Example: Create a voice from description

1. Type your text: "Welcome to our podcast"
2. Describe the voice: "middle-aged male broadcaster with authoritative professional tone"
3. Click **Generate**
4. AI creates a voice matching your description (~12-15 seconds)

**Tips**: Be specific - include gender, age, tone, emotion, style

### Voice Clone (Most Personal)
Example: Clone your own voice or anyone else's

1. Upload audio file (WAV/MP3/M4A, 3+ seconds)
2. If you have transcript, enter it. If not, enable **X-Vector Only** mode
3. Type what you want the cloned voice to say
4. Click **Generate** (~15-20 seconds first time, ~8-10 seconds after)

**X-Vector mode**: Faster, no transcript needed. Good for most use cases.

## Configuration

`config.yaml` controls behavior:

```yaml
models:
  default_size: "1.7B"  # Use "0.6B" if you have 6GB VRAM

generation:
  default_params:
    temperature: 0.9      # Lower = more consistent, Higher = more varied
    top_k: 50            # Sampling diversity
    top_p: 1.0           # Nucleus sampling
    max_new_tokens: 2048 # Max output length
```

**If out of memory**: Change `default_size` to `"0.6B"` or reduce `max_new_tokens` to `1024`

## 📁 Project Structure

```
TTS-opensource-app/
├── backend/
│   ├── main.py              # FastAPI server with SSE progress
│   ├── model_manager.py     # Model loading and inference
│   ├── audio_processor.py   # Audio I/O and format conversion
│   └── config_loader.py     # Configuration management
├── models/                   # Local model storage (auto-created)
│   ├── Qwen3-TTS-Tokenizer-12Hz/
│   ├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
│   ├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
│   └── Qwen3-TTS-12Hz-1.7B-Base/
├── outputs/                  # Generated audio files
├── temp_audio/              # Temporary audio processing
├── voice_prompts/           # Saved voice prompts (.pt files)
├── simple-ui.html           # Professional single-page UI
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Troubleshooting

### "Model loading failed" or "CUDA out of memory"
**Solution**: You need more VRAM
```yaml
# Edit config.yaml
models:
  default_size: "0.6B"  # Change from 1.7B to 0.6B
```
Or close other GPU programs (browsers with hardware acceleration, games, etc.)

### "Unable to convert M4A audio file"
**Solution**: Install ffmpeg
```bash
conda install -c conda-forge ffmpeg -y
# Restart the backend after installing
```

### Progress bar stuck at 0% or not updating
**Problem**: Browser can't connect to progress stream

**Check**:
1. Open browser console (F12) → Look for errors
2. Backend still running? Check terminal for "Uvicorn running"
3. Try refreshing page
4. Check firewall isn't blocking localhost:8000

### "Generation failed" with M4A file in Voice Clone
This happens if ffmpeg isn't installed. Install it:
```bash
conda install -c conda-forge ffmpeg -y
```
Then restart backend and try again with your M4A file.

### First generation is slow (30+ seconds)
**This is normal**: First time loads model into GPU memory. Second generation is much faster (8-12 seconds).

### Flash Attention warning appears
**Ignore it**: The app works fine without flash-attn. It's optional performance optimization.

If you want to install it anyway:
```bash
pip install flash-attn --no-build-isolation
# Warning: This takes 10+ minutes to compile
```

## Technical Details

### Architecture
- **Backend**: FastAPI server
- **Models**: Qwen3-TTS 1.7B (CustomVoice, VoiceDesign, Base for cloning)
- **Inference**: PyTorch with bfloat16 precision on CUDA

### What Happens When You Generate

1. Request sent to FastAPI backend
2. Model loads into GPU (if not already loaded)
3. Text gets processed and tokenized
4. Model generates audio codes
5. Codes get decoded to waveform
6. Waveform returned as WAV file

### File Structure
```
TTS-opensource-app/
├── backend/
│   ├── main.py              # FastAPI server, endpoints, SSE progress
│   ├── model_manager.py     # Model loading, inference, GPU handling
│   ├── audio_processor.py   # Audio I/O, format conversion, ffmpeg
│   └── config_loader.py     # YAML config, device detection
├── models/                   # Downloaded models (10GB)
├── outputs/                  # Your generated audio files
├── temp_audio/              # Temporary processing (auto-cleaned)
├── simple-ui.html           # The UI you see in browser
└── config.yaml              # Settings
```

### API Examples

**Custom Voice**:
```bash
curl -X POST http://localhost:8000/api/tts/custom-voice \
  -F "text=Hello world" \
  -F "speaker=xiaorui_enthusiastic_chinese_female_young" \
  -F "language=Auto" \
  > output.wav
```

### Performance Benchmarks
On RTX 3060 12GB:
- Model loading: 2-4 seconds (first time)
- CustomVoice: 8-12 seconds
- VoiceDesign: 12-15 seconds  
- VoiceClone: 8-10 seconds (X-Vector), 15-20 seconds (full)

On GTX 1660 6GB (0.6B models):
- Similar times but ~15-20% slower

## Credits

**Powered by Qwen3-TTS**: This application uses [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) models developed by the Qwen Team at Alibaba Cloud.

**License & Attribution**:
- Qwen3-TTS models are subject to their original license terms as specified in the [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)
- Model weights and code from Qwen3-TTS are used in accordance with their license
- This application is a user interface wrapper and does not claim ownership of the underlying models
- Please review the [Qwen3-TTS License](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE) for complete terms

**Technology Stack**:
- **Models**: Qwen3-TTS 1.7B (CustomVoice, VoiceDesign, Base)
- **Backend**: FastAPI, PyTorch, Transformers

### Resources
- [Qwen3-TTS Paper (arXiv)](https://arxiv.org/abs/2601.15621)
- [Model Hub (HuggingFace)](https://huggingface.co/collections/Qwen/qwen3-tts)
- [Official Documentation](https://github.com/QwenLM/Qwen3-TTS)

---

**Questions? Issues?** Check the [Troubleshooting](#troubleshooting) section above.
