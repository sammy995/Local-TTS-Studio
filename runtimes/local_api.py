"""
Local API Runtime - Thin FastAPI Wrapper
Composition only: parse request → call service → return response
No generation logic, no storage logic, just HTTP handling
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import uvicorn
import os
import json
import asyncio
import logging
import uuid
import torch

# Import from layers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_manager import ModelManager
from core.tts_engine import TTSEngine
from services.tts_service import TTSService
from services.podcast_service import PodcastService
from services.podcast_models import PodcastProject, Speaker as PodcastSpeaker, Segment as PodcastSegment, MusicTrack, AudioAsset, AssetSource
from services.music_service import MusicService
from infra.storage import LocalStorage
from core.audio_pipeline import load_audio_from_bytes

# Runtime config
from .config_loader import load_config, get_device_config, create_directories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== INITIALIZATION (Composition) =====

# Load configuration
config = load_config()

# Create directories
create_directories(config)

# Get device configuration
device_config = get_device_config()

# Initialize storage (LocalStorage for local runtime)
storage = LocalStorage(
    output_dir=Path("./outputs"),
    temp_dir=Path("./temp_audio"),
    prompts_dir=Path("./voice_prompts")
)

# Initialize model manager
model_manager = ModelManager(
    model_base_path=Path(config['models']['cache_dir']),
    device=device_config['device_map'],
    dtype=device_config['dtype'],
    use_flash_attn=device_config['use_flash_attn']
)

# Initialize engine
engine = TTSEngine(model_manager=model_manager)

# Initialize services
service = TTSService(engine=engine)
podcast_service = PodcastService(engine=engine)

# Initialize music service (v3)
music_config = config.get('music_apis', {})
music_service = MusicService(config=music_config)

# ===== FASTAPI APP =====

app = FastAPI(
    title="Qwen3-TTS API",
    description="Local Text-to-Speech API with multiple modes",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Segment-Timings", "X-Total-Duration-Ms"],
)

# Progress tracking (runtime-specific)
progress_queues = {}

def emit_progress(session_id: str, stage: str, progress: float, message: str):
    """Emit progress update to session queue"""
    if session_id in progress_queues:
        progress_queues[session_id].put_nowait({
            "stage": stage,
            "progress": progress,
            "message": message
        })

async def progress_stream(session_id: str):
    """SSE stream generator for progress updates"""
    queue = asyncio.Queue()
    progress_queues[session_id] = queue
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {json.dumps(data)}\n\n"
                
                if data.get("progress", 0) >= 100:
                    break
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"
    finally:
        if session_id in progress_queues:
            del progress_queues[session_id]

# ===== PYDANTIC MODELS =====

class SystemInfo(BaseModel):
    status: str
    models_loaded: List[str]
    gpu_available: bool
    device: str

# ===== ROUTES =====

@app.get("/")
async def root():
    """Serve the simple UI HTML file"""
    ui_path = Path(__file__).parent.parent / "simple-ui.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {
        "status": "running",
        "message": "Qwen3-TTS API Server",
        "version": "1.0.0"
    }

@app.get("/favicon.svg")
async def favicon():
    """Serve favicon"""
    favicon_path = Path(__file__).parent.parent / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/api/health")
async def health_check():
    """Health check"""
    return {"status": "healthy"}

@app.get("/api/system/info", response_model=SystemInfo)
async def get_system_info():
    """Get system information"""
    return {
        "status": "ready",
        "models_loaded": model_manager.get_loaded_models(),
        "gpu_available": device_config['gpu_available'],
        "device": device_config['device_map']
    }

@app.get("/api/models/available")
async def get_available_models():
    """List available models"""
    return {
        "custom_voice": config["models"]["available_models"]["custom_voice"],
        "voice_design": config["models"]["available_models"]["voice_design"],
        "base_clone": config["models"]["available_models"]["base_clone"]
    }

@app.post("/api/models/load/{model_type}")
async def load_model(model_type: str, model_size: str = "1.7B"):
    """Load a specific model"""
    try:
        model_manager.load_model(model_type, model_size)
        return {"status": "success", "message": f"Model {model_type} loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/speakers")
async def get_speakers():
    """Get list of available speakers"""
    return config.get("speakers", {}).get("custom_voice", [])

@app.get("/api/languages")
async def get_languages():
    """Get list of supported languages"""
    return config.get("languages", [])

@app.get("/api/progress/{session_id}")
async def stream_progress(session_id: str):
    """SSE endpoint for progress updates"""
    return StreamingResponse(
        progress_stream(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# ===== TTS GENERATION ENDPOINTS =====

@app.post("/api/tts/custom-voice")
async def generate_custom_voice(
    text: str = Form(...),
    speaker: str = Form(...),
    language: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(1.0),
    max_new_tokens: int = Form(2048),
    session_id: Optional[str] = Form(None)
):
    """Generate speech using CustomVoice model"""
    try:
        if session_id:
            emit_progress(session_id, "loading", 10, "Loading CustomVoice model...")
        
        # Ensure model loaded
        model_manager.ensure_model_loaded("custom_voice")
        
        if session_id:
            emit_progress(session_id, "loading", 30, "Model loaded successfully")
        
        # Convert "Auto" to None
        lang = None if language in [None, "Auto", ""] else language
        
        if session_id:
            emit_progress(session_id, "generating", 40, "Generating audio...")
        
        # Create heartbeat task
        async def heartbeat():
            await asyncio.sleep(1)
            if session_id:
                emit_progress(session_id, "generating", 60, "Synthesizing speech...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 80, "Finalizing audio...")
        
        heartbeat_task = asyncio.create_task(heartbeat())
        
        # Generate (run in executor to not block)
        loop = asyncio.get_event_loop()
        file_path = await loop.run_in_executor(
            None,
            lambda: service.generate_speech(
                text=text,
                mode="custom_voice",
                storage=storage,
                speaker=speaker,
                language=lang,
                instruct=instruct,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )
        )
        
        heartbeat_task.cancel()
        
        if session_id:
            emit_progress(session_id, "complete", 100, "Complete!")
        
        # Return audio file
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=Path(file_path).name
        )
        
    except Exception as e:
        if session_id:
            emit_progress(session_id, "error", 0, f"Error: {str(e)}")
        logger.error(f"Error in custom_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts/voice-design")
async def generate_voice_design(
    text: str = Form(...),
    language: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(1.0),
    max_new_tokens: int = Form(2048),
    session_id: Optional[str] = Form(None)
):
    """Generate speech using VoiceDesign model"""
    try:
        if session_id:
            emit_progress(session_id, "loading", 10, "Loading VoiceDesign model...")
        
        model_manager.ensure_model_loaded("voice_design")
        
        if session_id:
            emit_progress(session_id, "loading", 30, "Model loaded successfully")
        
        lang = None if language in [None, "Auto", ""] else language
        
        if session_id:
            emit_progress(session_id, "generating", 40, "Generating audio...")
        
        async def heartbeat():
            await asyncio.sleep(1)
            if session_id:
                emit_progress(session_id, "generating", 60, "Designing voice...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 80, "Synthesizing...")
        
        heartbeat_task = asyncio.create_task(heartbeat())
        
        loop = asyncio.get_event_loop()
        file_path = await loop.run_in_executor(
            None,
            lambda: service.generate_speech(
                text=text,
                mode="voice_design",
                storage=storage,
                language=lang,
                instruct=instruct,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )
        )
        
        heartbeat_task.cancel()
        
        if session_id:
            emit_progress(session_id, "complete", 100, "Complete!")
        
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=Path(file_path).name
        )
        
    except Exception as e:
        if session_id:
            emit_progress(session_id, "error", 0, f"Error: {str(e)}")
        logger.error(f"Error in voice_design: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts/voice-clone")
async def generate_voice_clone(
    text: str = Form(...),
    language: Optional[str] = Form(None),
    ref_text: Optional[str] = Form(None),
    x_vector_only: bool = Form(False),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(1.0),
    max_new_tokens: int = Form(2048),
    ref_audio: Optional[UploadFile] = File(None),
    ref_audio_url: Optional[str] = Form(None),
    ref_audio_base64: Optional[str] = Form(None),
    voice_prompt_file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
    """Generate speech using Voice Clone model"""
    try:
        if session_id:
            emit_progress(session_id, "loading", 10, "Loading Voice Clone model...")
        
        model_manager.ensure_model_loaded("base_clone")
        
        if session_id:
            emit_progress(session_id, "loading", 25, "Processing reference audio...")
        
        # Process reference audio
        ref_audio_data = await process_ref_audio_input(
            file=ref_audio,
            url=ref_audio_url,
            base64_data=ref_audio_base64,
            prompt_file=voice_prompt_file
        )
        
        if session_id:
            emit_progress(session_id, "generating", 40, "Cloning voice...")
        
        async def heartbeat():
            await asyncio.sleep(1)
            if session_id:
                emit_progress(session_id, "generating", 60, "Extracting voice features...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 80, "Synthesizing...")
        
        heartbeat_task = asyncio.create_task(heartbeat())
        
        lang = None if language in [None, "Auto", ""] else language
        
        loop = asyncio.get_event_loop()
        file_path = await loop.run_in_executor(
            None,
            lambda: service.generate_speech(
                text=text,
                mode="voice_clone",
                storage=storage,
                language=lang,
                ref_audio=ref_audio_data,
                ref_text=ref_text,
                x_vector_only=x_vector_only,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )
        )
        
        heartbeat_task.cancel()
        
        if session_id:
            emit_progress(session_id, "complete", 100, "Complete!")
        
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=Path(file_path).name
        )
        
    except Exception as e:
        if session_id:
            emit_progress(session_id, "error", 0, f"Error: {str(e)}")
        logger.error(f"Error in voice_clone: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== VOICE PROMPT ENDPOINT =====

@app.post("/api/voice-prompt/save")
async def save_voice_prompt(
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(...),
    x_vector_only: bool = Form(False),
    prompt_name: str = Form("voice_prompt")
):
    """Create and save a reusable voice prompt"""
    try:
        model_manager.ensure_model_loaded("base_clone")
        
        # Process reference audio
        temp_path = storage.get_temp_path(f"{uuid.uuid4()}_{ref_audio.filename}")
        temp_path.write_bytes(await ref_audio.read())
        # Convert to WAV if needed (TTS engine requires WAV)
        temp_path = _ensure_wav(temp_path)
        
        # Create voice prompt
        prompt_path = service.create_voice_prompt(
            name=prompt_name,
            ref_audio=str(temp_path),
            ref_text=ref_text,
            storage=storage,
            x_vector_only=x_vector_only
        )
        
        # Clean up temp file
        temp_path.unlink()
        
        return {
            "status": "success",
            "message": "Voice prompt saved",
            "path": prompt_path
        }
        
    except Exception as e:
        logger.error(f"Error saving voice prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== AUDIO DOWNLOAD ENDPOINT =====

@app.get("/api/audio/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio file"""
    file_path = Path(f"./outputs/{filename}")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )

# ===== PODCAST HELPERS =====

def _ensure_wav(file_path: Path) -> Path:
    """
    Ensure the audio file is in WAV format.
    If it's MP3, M4A, OGG, etc. — convert to WAV via pydub.
    Returns the (possibly new) WAV path.
    """
    suffix = file_path.suffix.lower()
    if suffix == '.wav':
        return file_path
    
    try:
        from pydub import AudioSegment as PydubSegment
        audio_seg = PydubSegment.from_file(str(file_path))
        wav_path = file_path.with_suffix('.wav')
        audio_seg.export(str(wav_path), format='wav')
        # Remove original non-wav temp file
        try:
            file_path.unlink()
        except Exception:
            pass
        logger.info(f"Converted {suffix} → .wav: {wav_path}")
        return wav_path
    except Exception as e:
        logger.warning(f"Could not convert {suffix} to WAV ({e}), passing as-is")
        return file_path


def _resolve_clone_speakers(speakers_list, storage_obj):
    """
    For clone-mode speakers that carry base64 reference audio,
    decode the audio into a temp file and replace the config with
    a local 'ref_audio_path' the engine can consume.
    
    Returns list of temp file paths to clean up after rendering.
    """
    import base64 as b64mod
    temp_paths = []
    
    for speaker_dict in speakers_list:
        if speaker_dict.get('mode') != 'clone':
            continue
        cfg = speaker_dict.get('config', {})
        b64_data = cfg.get('ref_audio_base64')
        if not b64_data:
            continue
        
        # Detect format from data URI prefix: "data:audio/mpeg;base64,..."
        ext = '.wav'
        raw_b64 = b64_data
        if ',' in raw_b64:
            header = raw_b64.split(',', 1)[0].lower()
            raw_b64 = raw_b64.split(',', 1)[1]
            if 'audio/mpeg' in header or 'audio/mp3' in header:
                ext = '.mp3'
            elif 'audio/mp4' in header or 'audio/m4a' in header:
                ext = '.m4a'
            elif 'audio/ogg' in header:
                ext = '.ogg'
            elif 'audio/flac' in header:
                ext = '.flac'
        
        # Also check the original filename stored in config
        orig_name = (cfg.get('ref_audio_name') or '').lower()
        if orig_name.endswith('.mp3'):
            ext = '.mp3'
        elif orig_name.endswith('.m4a'):
            ext = '.m4a'
        elif orig_name.endswith('.ogg'):
            ext = '.ogg'
        elif orig_name.endswith('.flac'):
            ext = '.flac'
        
        audio_bytes = b64mod.b64decode(raw_b64)
        temp_path = storage_obj.get_temp_path(f"{uuid.uuid4()}_clone_ref{ext}")
        temp_path.write_bytes(audio_bytes)
        
        # Convert to WAV if needed (TTS engine requires WAV)
        temp_path = _ensure_wav(temp_path)
        temp_paths.append(temp_path)
        
        # Replace config: remove base64 blob, set file path
        cfg['ref_audio_path'] = str(temp_path)
        cfg.pop('ref_audio_base64', None)
        cfg.pop('ref_audio_name', None)
        
        logger.info(f"Clone speaker '{speaker_dict.get('name', speaker_dict['id'])}': saved ref audio → {temp_path}")
    
    return temp_paths

# ===== PODCAST ENDPOINT =====

@app.post("/api/podcast/render")
async def render_podcast(request: Request):
    """
    Render podcast from structured project
    
    Content production compiler: script → deterministic audio artifact
    
    Synchronous for local runtime (no job queue)
    Runtime loops segments for progress control
    """
    try:
        # Parse JSON directly from request body
        project_dict = await request.json()
        
        # Resolve clone speaker ref audio (base64 → temp file)
        clone_temp_files = _resolve_clone_speakers(project_dict.get('speakers', []), storage)
        
        try:
            # Build dataclasses
            speakers = [
                PodcastSpeaker(
                    id=s['id'],
                    name=s['name'],
                    mode=s['mode'],
                    config=s['config'],
                    style_instruction=s.get('style_instruction')
                )
                for s in project_dict['speakers']
            ]
            
            segments = [
                PodcastSegment(
                    id=seg['id'],
                    order=seg['order'],
                    speaker_id=seg['speaker_id'],
                    text=seg['text'],
                    pause_after_ms=seg.get('pause_after_ms', 500),
                    volume=seg.get('volume', 1.0),
                    emotion=seg.get('emotion')
                )
                for seg in project_dict['segments']
            ]
            
            project = PodcastProject(
                id=project_dict['id'],
                title=project_dict['title'],
                speakers=speakers,
                segments=segments,
                output_format=project_dict.get('output_format', 'wav'),
                target_sample_rate=project_dict.get('target_sample_rate', 44100),
                deterministic=project_dict.get('deterministic', True)
            )
            
            # Validate
            podcast_service.validate_project(project)
            
            # Load required models based on speaker modes
            required_models = set()
            for speaker in project.speakers:
                if speaker.mode == "custom":
                    required_models.add("custom_voice")
                elif speaker.mode == "design":
                    required_models.add("voice_design")
                elif speaker.mode == "clone":
                    required_models.add("base_clone")
            
            logger.info(f"Loading models: {required_models}")
            for model_name in required_models:
                model_manager.ensure_model_loaded(model_name)
            
            # Precompute prompts
            prompt_cache = podcast_service._precompute_speaker_prompts(project.speakers)
            
            # Build speaker map
            speaker_map = {s.id: s for s in project.speakers}
            
            # Sort segments
            sorted_segments = sorted(project.segments, key=lambda s: s.order)
            total_segments = len(sorted_segments)
            
            logger.info(f"Rendering podcast: {project.title} ({total_segments} segments, {len(project.speakers)} speakers)")
            
            # Render segments with progress (runtime controls progress)
            from core.audio_pipeline import generate_silence
            import numpy as np
            
            arrays = []
            sample_rate = None
            failed_segments = []
            
            for i, segment in enumerate(sorted_segments):
                speaker = speaker_map[segment.speaker_id]
                
                logger.info(f"Rendering segment {i+1}/{total_segments}: {speaker.name} - '{segment.text[:50]}...'")
                
                try:
                    # Render segment (includes retry logic in podcast_service)
                    audio_array, sr = podcast_service.render_segment(
                        segment=segment,
                        speaker=speaker,
                        prompt_cache=prompt_cache,
                        deterministic=project.deterministic
                    )
                    
                    if sample_rate is None:
                        sample_rate = sr
                    
                    arrays.append(audio_array)
                    
                except Exception as seg_error:
                    # Segment failed even after retries — log and insert brief silence placeholder
                    logger.error(f"Segment {i+1} ({speaker.name}) failed permanently: {seg_error}")
                    failed_segments.append((i+1, speaker.name, str(seg_error)))
                    
                    # Use known sample rate or default
                    sr = sample_rate or 24000
                    if sample_rate is None:
                        sample_rate = sr
                    
                    # Insert 200ms silence placeholder (not the full pause — avoids long gaps)
                    placeholder = generate_silence(duration_ms=200, sample_rate=sr)
                    arrays.append(placeholder)
                
                # Add pause between segments
                if segment.pause_after_ms > 0 and sample_rate is not None:
                    silence = generate_silence(
                        duration_ms=segment.pause_after_ms,
                        sample_rate=sample_rate,
                        reference_array=arrays[-1]
                    )
                    arrays.append(silence)
            
            if failed_segments:
                logger.warning(f"{len(failed_segments)}/{total_segments} segments failed: {failed_segments}")
            
            # Memory-safe concatenation
            total_length = sum(len(arr) for arr in arrays)
            final_array = np.zeros(total_length, dtype=arrays[0].dtype)
            
            offset = 0
            for arr in arrays:
                final_array[offset:offset + len(arr)] = arr
                offset += len(arr)
            
            logger.info(f"Concatenated {total_segments} segments: {total_length} samples")
            
            # Encode
            from core.audio_pipeline import to_wav_bytes, to_mp3_bytes
            if sample_rate is None:
                raise RuntimeError("No audio generated - sample_rate is None")
            
            fmt = project.output_format or 'mp3'
            if fmt == 'mp3':
                audio_bytes = to_mp3_bytes(final_array, sample_rate, project.target_sample_rate)
                media_type = 'audio/mpeg'
            else:
                audio_bytes = to_wav_bytes(final_array, sample_rate, project.target_sample_rate)
                media_type = 'audio/wav'
            
            # Save
            filename = f"podcast_{project.id}.{fmt}"
            file_path = storage.save_audio(audio_bytes, filename)
            
            logger.info(f"Podcast rendered: {file_path}")
            
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=Path(file_path).name
            )
        finally:
            # Clean up clone ref audio temp files
            for tmp in clone_temp_files:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
    
    except Exception as e:
        logger.error(f"Error rendering podcast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== v3: MUSIC LIBRARY ENDPOINTS =====

@app.get("/api/assets")
async def list_assets():
    """List all assets in the local library"""
    try:
        assets = storage.list_assets()
        return {"assets": assets, "count": len(assets)}
    except Exception as e:
        logger.error(f"Error listing assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/assets/upload")
async def upload_asset(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Upload a local audio file to the asset library"""
    try:
        audio_bytes = await file.read()
        
        # Determine format from filename
        ext = Path(file.filename).suffix.lstrip('.') if file.filename else 'mp3'
        
        # Try to get duration
        duration_ms = 0
        try:
            from core.audio_pipeline import load_audio_from_bytes
            audio_array, sr = load_audio_from_bytes(audio_bytes)
            duration_ms = int(len(audio_array) / sr * 1000)
        except Exception:
            pass
        
        metadata = {
            'title': title or (Path(file.filename).stem if file.filename else 'Uploaded'),
            'source': 'local',
            'source_url': '',
            'duration_ms': duration_ms,
            'license': 'local',
            'attribution': '',
            'artist': artist or '',
            'tags': [t.strip() for t in (tags or '').split(',')] if tags else [],
            'format': ext,
        }
        
        result = storage.save_asset(audio_bytes, metadata)
        return {"asset": result}
        
    except Exception as e:
        logger.error(f"Error uploading asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/assets/{asset_id}")
async def delete_asset(asset_id: str):
    """Delete an asset from the library"""
    try:
        deleted = storage.delete_asset(asset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Asset not found: {asset_id}")
        return {"deleted": True, "asset_id": asset_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/music/search")
async def search_music(
    q: str,
    source: str = "jamendo",
    limit: int = 20,
    instrumental: bool = True
):
    """
    Search for music from external sources (Jamendo, Openverse, Freesound)
    
    Proxied through backend to avoid CORS issues.
    """
    try:
        results = await music_service.search(
            query=q,
            source=source,
            limit=limit,
            instrumental_only=instrumental
        )
        return {
            "results": [r.to_dict() for r in results],
            "count": len(results),
            "source": source,
            "query": q
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error searching music: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/music/import")
async def import_music(request: Request):
    """
    Import a track from external source into local asset library
    
    Body: {
        "url": "https://...",
        "title": "Track Name",
        "artist": "Artist",
        "source": "jamendo",
        "source_url": "https://...",
        "license": "CC BY 4.0",
        "duration_ms": 180000,
        "tags": ["ambient", "electronic"]
    }
    """
    try:
        data = await request.json()
        url = data.get('url')
        
        if not url:
            raise HTTPException(status_code=400, detail="Missing 'url' field")
        
        # Download the audio
        audio_bytes = await music_service.download_track(url)
        
        # Determine format
        ext = 'mp3'
        if '.ogg' in url:
            ext = 'ogg'
        elif '.wav' in url:
            ext = 'wav'
        elif '.flac' in url:
            ext = 'flac'
        
        # Try to get actual duration from audio
        duration_ms = data.get('duration_ms', 0)
        try:
            from core.audio_pipeline import load_audio_from_bytes
            audio_array, sr = load_audio_from_bytes(audio_bytes)
            duration_ms = int(len(audio_array) / sr * 1000)
        except Exception:
            pass
        
        metadata = {
            'title': data.get('title', 'Imported'),
            'source': data.get('source', 'unknown'),
            'source_url': data.get('source_url', url),
            'duration_ms': duration_ms,
            'license': data.get('license', 'unknown'),
            'attribution': data.get('attribution', ''),
            'artist': data.get('artist', 'Unknown'),
            'tags': data.get('tags', []),
            'format': ext,
        }
        
        result = storage.save_asset(audio_bytes, metadata)
        return {"asset": result}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing music: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== v3: TIMELINE RENDER ENDPOINT =====

@app.post("/api/podcast/render/v3")
async def render_podcast_v3(request: Request):
    """
    v3 Timeline render: speech + music tracks with ducking
    
    Accepts the full v3 project format with music_tracks.
    Uses absolute timeline positioning instead of sequential concatenation.
    """
    try:
        from core.audio_pipeline import (
            generate_silence, to_wav_bytes, mix_timeline,
            loop_to_length, trim_audio, apply_fade, resample,
            load_audio_from_bytes as load_audio
        )

        project_dict = await request.json()
        
        # Resolve clone speaker ref audio (base64 → temp file)
        clone_temp_files = _resolve_clone_speakers(project_dict.get('speakers', []), storage)
        
        try:
            # Build speakers and segments (same as v2)
            speakers = [
                PodcastSpeaker(
                    id=s['id'],
                    name=s['name'],
                    mode=s['mode'],
                    config=s['config'],
                    style_instruction=s.get('style_instruction')
                )
                for s in project_dict['speakers']
            ]
            
            segments = [
                PodcastSegment(
                    id=seg['id'],
                    order=seg['order'],
                    speaker_id=seg.get('speaker_id', ''),
                    text=seg.get('text', ''),
                    kind=seg.get('kind', 'speech'),
                    start_ms=seg.get('start_ms'),
                    pause_after_ms=seg.get('pause_after_ms', 500),
                    volume=seg.get('volume', 1.0),
                    emotion=seg.get('emotion')
                )
                for seg in project_dict.get('segments', [])
            ]
            
            # Build music tracks
            music_tracks = [
                MusicTrack(
                    id=mt['id'],
                    asset_id=mt['asset_id'],
                    start_ms=mt.get('start_ms', 0),
                    end_ms=mt.get('end_ms'),
                    volume=mt.get('volume', 0.3),
                    duck_under_speech=mt.get('duck_under_speech', True),
                    duck_level=mt.get('duck_level', 0.2),
                    duck_attack_ms=mt.get('duck_attack_ms', 200),
                    duck_release_ms=mt.get('duck_release_ms', 400),
                    fade_in_ms=mt.get('fade_in_ms', 500),
                    fade_out_ms=mt.get('fade_out_ms', 500),
                    loop=mt.get('loop', True)
                )
                for mt in project_dict.get('music_tracks', [])
            ]
            
            deterministic = project_dict.get('deterministic', True)
            target_sr = project_dict.get('target_sample_rate', 44100)
            
            # Load required TTS models
            speech_segments = [s for s in segments if s.kind == 'speech']
            required_models = set()
            for speaker in speakers:
                if speaker.mode == "custom":
                    required_models.add("custom_voice")
                elif speaker.mode == "design":
                    required_models.add("voice_design")
                elif speaker.mode == "clone":
                    required_models.add("base_clone")
            
            for model_name in required_models:
                model_manager.ensure_model_loaded(model_name)
        
            logger.info(f"v3 Render: {len(speech_segments)} speech segments, {len(music_tracks)} music tracks")
            
            import numpy as np
            
            # === Phase 1: Render all speech segments ===
            speaker_map = {s.id: s for s in speakers}
            prompt_cache = podcast_service._precompute_speaker_prompts(speakers)
            sorted_speech = sorted(speech_segments, key=lambda s: s.order)
            
            # Render each speech segment and track positions
            speech_results = []  # List of (start_ms, audio_array, duration_ms)
            current_ms = 0  # Running position for auto-sequential
            engine_sr = None
            failed_segments_v3 = []
            
            for i, segment in enumerate(sorted_speech):
                speaker = speaker_map.get(segment.speaker_id)
                if not speaker:
                    raise ValueError(f"Speaker not found: {segment.speaker_id}")
                
                logger.info(f"Rendering speech {i+1}/{len(sorted_speech)}: {speaker.name}")
                
                try:
                    audio_array, sr = podcast_service.render_segment(
                        segment=segment,
                        speaker=speaker,
                        prompt_cache=prompt_cache,
                        deterministic=deterministic
                    )
                except Exception as seg_error:
                    # Segment failed even after retries — insert tiny placeholder
                    logger.error(f"v3 Segment {i+1} ({speaker.name}) failed permanently: {seg_error}")
                    failed_segments_v3.append((i+1, speaker.name, str(seg_error)))
                    sr = engine_sr or 24000
                    # 200ms silence placeholder instead of dead air
                    audio_array = np.zeros(int(0.2 * sr), dtype=np.float32)
                
                if engine_sr is None:
                    engine_sr = sr
                
                # Determine timeline position
                if segment.start_ms is not None:
                    pos_ms = segment.start_ms
                else:
                    pos_ms = current_ms  # Auto-sequential fallback
                
                duration_ms = int(len(audio_array) / sr * 1000)
                speech_results.append((pos_ms, audio_array, duration_ms))
                
                # Advance auto position
                current_ms = pos_ms + duration_ms + segment.pause_after_ms
            
            if failed_segments_v3:
                logger.warning(f"v3: {len(failed_segments_v3)}/{len(sorted_speech)} segments failed: {failed_segments_v3}")
            
            if engine_sr is None:
                engine_sr = 24000  # Default engine sample rate
            
            # === Phase 2: Calculate total timeline duration ===
            max_speech_end = 0
            for pos_ms, audio, dur_ms in speech_results:
                end = pos_ms + dur_ms
                if end > max_speech_end:
                    max_speech_end = end
            
            max_music_end = 0
            for mt in music_tracks:
                if mt.end_ms and mt.end_ms > max_music_end:
                    max_music_end = mt.end_ms
            
            total_duration_ms = max(max_speech_end, max_music_end) + 1000  # 1s padding
            total_samples = int(total_duration_ms / 1000.0 * engine_sr)
            
            # === Phase 3: Build speech placements ===
            speech_placements = []
            for pos_ms, audio, dur_ms in speech_results:
                offset_samples = int(pos_ms / 1000.0 * engine_sr)
                speech_placements.append((offset_samples, audio))
            
            # === Phase 4: Build music placements ===
            music_placements = []
            for mt in music_tracks:
                try:
                    asset_audio_bytes = storage.load_asset_audio(mt.asset_id)
                    asset_audio, asset_sr = load_audio(asset_audio_bytes)
                    
                    # Resample to match engine
                    if asset_sr != engine_sr:
                        asset_audio = resample(asset_audio, asset_sr, engine_sr)
                    
                    # Calculate duration
                    start_samples = int(mt.start_ms / 1000.0 * engine_sr)
                    if mt.end_ms is not None:
                        end_samples = int(mt.end_ms / 1000.0 * engine_sr)
                        target_len = end_samples - start_samples
                    else:
                        target_len = total_samples - start_samples
                    
                    # Loop or trim to fit
                    if mt.loop and len(asset_audio) < target_len:
                        asset_audio = loop_to_length(asset_audio, target_len)
                    else:
                        asset_audio = asset_audio[:target_len]
                    
                    music_placements.append({
                        'audio': asset_audio,
                        'offset_samples': start_samples,
                        'volume': mt.volume,
                        'duck_under_speech': mt.duck_under_speech,
                        'duck_level': mt.duck_level,
                        'duck_attack_samples': int(mt.duck_attack_ms / 1000.0 * engine_sr),
                        'duck_release_samples': int(mt.duck_release_ms / 1000.0 * engine_sr),
                        'fade_in_samples': int(mt.fade_in_ms / 1000.0 * engine_sr),
                        'fade_out_samples': int(mt.fade_out_ms / 1000.0 * engine_sr),
                    })
                    
                    logger.info(f"Music track {mt.id}: asset={mt.asset_id}, {len(asset_audio)} samples")
                    
                except FileNotFoundError:
                    logger.warning(f"Asset not found for music track {mt.id}: {mt.asset_id}")
                    continue
            
            # === Phase 5: Mix timeline ===
            logger.info(f"Mixing timeline: {len(speech_placements)} speech + {len(music_placements)} music, {total_samples} total samples")
            
            final_audio = mix_timeline(
                speech_placements=speech_placements,
                music_placements=music_placements,
                total_samples=total_samples,
                sample_rate=engine_sr
            )
            
            # === Phase 6: Encode and save ===
            fmt = project_dict.get('output_format', 'mp3')
            if fmt == 'mp3':
                from core.audio_pipeline import to_mp3_bytes
                audio_bytes = to_mp3_bytes(final_audio, engine_sr, target_sr)
            else:
                audio_bytes = to_wav_bytes(final_audio, engine_sr, target_sr)
            
            project_id = project_dict.get('id', uuid.uuid4().hex[:8])
            filename = f"podcast_{project_id}.{fmt}"
            file_path = storage.save_audio(audio_bytes, filename)
            
            # Build segment timing metadata for timeline UI
            segment_timings = []
            for pos_ms, audio, dur_ms in speech_results:
                segment_timings.append({
                    'start_ms': pos_ms,
                    'duration_ms': dur_ms,
                    'end_ms': pos_ms + dur_ms
                })
            
            logger.info(f"v3 Podcast rendered: {file_path} ({total_duration_ms/1000:.1f}s)")
            
            return FileResponse(
                path=file_path,
                media_type='audio/mpeg' if fmt == 'mp3' else 'audio/wav',
                filename=Path(file_path).name,
                headers={
                    'X-Segment-Timings': json.dumps(segment_timings),
                    'X-Total-Duration-Ms': str(total_duration_ms)
                }
            )
        finally:
            # Clean up clone ref audio temp files
            for tmp in clone_temp_files:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
    
    except Exception as e:
        logger.error(f"Error in v3 render: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ===== HELPER FUNCTIONS =====

async def process_ref_audio_input(
    file: Optional[UploadFile] = None,
    url: Optional[str] = None,
    base64_data: Optional[str] = None,
    prompt_file: Optional[UploadFile] = None
):
    """Process reference audio from various sources (runtime-specific)"""
    # Priority: prompt_file > file > url > base64
    
    if prompt_file:
        # Load voice prompt directly
        temp_path = storage.get_temp_path(f"{uuid.uuid4()}.pt")
        temp_path.write_bytes(await prompt_file.read())
        prompt = torch.load(temp_path)
        temp_path.unlink()
        return prompt
    
    if file:
        # Save uploaded file
        temp_path = storage.get_temp_path(f"{uuid.uuid4()}_{file.filename}")
        temp_path.write_bytes(await file.read())
        # Convert to WAV if needed (TTS engine requires WAV)
        temp_path = _ensure_wav(temp_path)
        return str(temp_path)
    
    if url:
        # Download from URL
        import requests
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Detect extension from URL
            from urllib.parse import urlparse
            url_path = urlparse(url).path.lower()
            ext = '.wav'
            for e in ['.mp3', '.m4a', '.ogg', '.flac', '.wav']:
                if url_path.endswith(e):
                    ext = e
                    break
            temp_path = storage.get_temp_path(f"{uuid.uuid4()}_from_url{ext}")
            temp_path.write_bytes(response.content)
            temp_path = _ensure_wav(temp_path)
            return str(temp_path)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")
    
    if base64_data:
        # Decode base64 audio
        import base64
        try:
            ext = '.wav'
            if "," in base64_data:
                header = base64_data.split(",")[0].lower()
                base64_data = base64_data.split(",")[1]
                if 'audio/mpeg' in header or 'audio/mp3' in header:
                    ext = '.mp3'
                elif 'audio/mp4' in header or 'audio/m4a' in header:
                    ext = '.m4a'
                elif 'audio/ogg' in header:
                    ext = '.ogg'
            
            audio_bytes = base64.b64decode(base64_data)
            temp_path = storage.get_temp_path(f"{uuid.uuid4()}_from_base64{ext}")
            temp_path.write_bytes(audio_bytes)
            temp_path = _ensure_wav(temp_path)
            return str(temp_path)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error decoding base64 audio: {str(e)}")
    
    raise HTTPException(status_code=400, detail="No audio input provided")

# ===== SERVER STARTUP =====

def start_server(host: Optional[str] = None, port: Optional[int] = None):
    """Start the FastAPI server"""
    server_host: str = host or config['app']['host']
    server_port: int = port or config['app']['port']
    
    logger.info(f"Starting server on {server_host}:{server_port}")
    logger.info(f"Device: {device_config['device_map']}, GPU: {device_config['gpu_available']}")
    
    uvicorn.run(
        app,
        host=server_host,
        port=server_port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
