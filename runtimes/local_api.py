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
from services.podcast_models import PodcastProject, Speaker as PodcastSpeaker, Segment as PodcastSegment
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
                required_models.add("voice_clone")
        
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
        
        for i, segment in enumerate(sorted_segments):
            speaker = speaker_map[segment.speaker_id]
            
            logger.info(f"Rendering segment {i+1}/{total_segments}: {speaker.name} - '{segment.text[:50]}...'")
            
            # Render segment
            audio_array, sr = podcast_service.render_segment(
                segment=segment,
                speaker=speaker,
                prompt_cache=prompt_cache,
                deterministic=project.deterministic
            )
            
            if sample_rate is None:
                sample_rate = sr
            
            arrays.append(audio_array)
            
            # Add silence
            if segment.pause_after_ms > 0:
                silence = generate_silence(
                    duration_ms=segment.pause_after_ms,
                    sample_rate=sr,
                    reference_array=audio_array
                )
                arrays.append(silence)
        
        # Memory-safe concatenation
        total_length = sum(len(arr) for arr in arrays)
        final_array = np.zeros(total_length, dtype=arrays[0].dtype)
        
        offset = 0
        for arr in arrays:
            final_array[offset:offset + len(arr)] = arr
            offset += len(arr)
        
        logger.info(f"Concatenated {total_segments} segments: {total_length} samples")
        
        # Encode
        from core.audio_pipeline import to_wav_bytes
        if sample_rate is None:
            raise RuntimeError("No audio generated - sample_rate is None")
        
        audio_bytes = to_wav_bytes(final_array, sample_rate, project.target_sample_rate)
        
        # Save
        filename = f"podcast_{project.id}.{project.output_format}"
        file_path = storage.save_audio(audio_bytes, filename)
        
        logger.info(f"Podcast saved: {file_path}")
        
        logger.info(f"Podcast rendered: {file_path}")
        
        return FileResponse(
            path=file_path,
            media_type="audio/wav",
            filename=Path(file_path).name
        )
        
    except Exception as e:
        logger.error(f"Error rendering podcast: {str(e)}")
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
        return str(temp_path)
    
    if url:
        # Download from URL
        import requests
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            temp_path = storage.get_temp_path(f"{uuid.uuid4()}_from_url.wav")
            temp_path.write_bytes(response.content)
            return str(temp_path)
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")
    
    if base64_data:
        # Decode base64 audio
        import base64
        try:
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            
            audio_bytes = base64.b64decode(base64_data)
            temp_path = storage.get_temp_path(f"{uuid.uuid4()}_from_base64.wav")
            temp_path.write_bytes(audio_bytes)
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
