"""
Qwen3-TTS Backend API Server
FastAPI-based backend for local TTS application
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Union
import uvicorn
import os
import json
import asyncio
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .model_manager import ModelManager
    from .audio_processor import AudioProcessor
    from .config_loader import load_config
except ImportError:
    from model_manager import ModelManager
    from audio_processor import AudioProcessor
    from config_loader import load_config

# Load configuration
config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-TTS API",
    description="Local Text-to-Speech API with multiple modes",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Progress event queue
progress_queues = {}

# Initialize managers
model_manager = ModelManager(config)
audio_processor = AudioProcessor(config)

# Pydantic models for request validation
class CustomVoiceRequest(BaseModel):
    text: Union[str, List[str]]
    speaker: str
    language: Optional[str] = None
    instruct: Optional[str] = None
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.05

class VoiceDesignRequest(BaseModel):
    text: Union[str, List[str]]
    language: Optional[str] = None
    instruct: str  # Voice description (required)
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 2048

class VoiceCloneRequest(BaseModel):
    text: Union[str, List[str]]
    language: Optional[str] = None
    ref_text: Optional[str] = None
    x_vector_only: bool = False
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 2048

class SystemInfo(BaseModel):
    status: str
    models_loaded: List[str]
    gpu_available: bool
    device: str

# Progress tracking helpers
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
                # Wait for progress update with timeout
                data = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield f"data: {json.dumps(data)}\n\n"
                
                # If progress is 100%, end the stream
                if data.get("progress", 0) >= 100:
                    break
            except asyncio.TimeoutError:
                # Send keepalive
                yield f": keepalive\n\n"
    finally:
        # Clean up queue
        if session_id in progress_queues:
            del progress_queues[session_id]

@app.get("/api/progress/{session_id}")
async def stream_progress(session_id: str):
    """SSE endpoint for real-time progress updates"""
    return StreamingResponse(
        progress_stream(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# Health check endpoint
@app.get("/")
async def root():
    # Serve the simple UI HTML file
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
    # Serve the favicon file
    favicon_path = Path(__file__).parent.parent / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

# System info endpoint
@app.get("/api/system/info", response_model=SystemInfo)
async def get_system_info():
    """Get system and model information"""
    info = model_manager.get_system_info()
    return info

# Model management endpoints
@app.get("/api/models/available")
async def get_available_models():
    """List all available models"""
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

# Speaker and language info endpoints
@app.get("/api/speakers")
async def get_speakers():
    """Get list of available speakers"""
    return config.get("speakers", {}).get("custom_voice", [])

@app.get("/api/languages")
async def get_languages():
    """Get list of supported languages"""
    return config.get("languages", [])

# CustomVoice TTS endpoint
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
        # Emit progress: Loading model
        if session_id:
            emit_progress(session_id, "loading", 10, "Loading CustomVoice model...")
        
        # Ensure model is loaded
        model_manager.ensure_model_loaded("custom_voice")
        
        if session_id:
            emit_progress(session_id, "loading", 30, "Model loaded successfully")
        
        # Convert "Auto" to None for language
        language = None if language in [None, "Auto", ""] else language
        
        if session_id:
            emit_progress(session_id, "generating", 40, "Preparing generation...")
        
        # Create async task to emit heartbeat progress during generation
        async def heartbeat():
            await asyncio.sleep(1)
            if session_id:
                emit_progress(session_id, "generating", 50, "Generating audio codes...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 60, "Processing text and speaker...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 70, "Synthesizing speech...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 80, "Finalizing audio waveform...")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat())
        
        # Run generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            lambda: model_manager.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )
        )
        
        # Cancel heartbeat if still running
        heartbeat_task.cancel()
        
        if session_id:
            emit_progress(session_id, "processing", 90, "Processing audio...")
        
        # Process and return audio
        response = audio_processor.prepare_response(audio_data)
        
        if session_id:
            emit_progress(session_id, "complete", 100, "Audio generation complete!")
        
        return response
        
    except Exception as e:
        if session_id:
            emit_progress(session_id, "error", 0, f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# VoiceDesign TTS endpoint
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
        
        # Convert "Auto" to None for language
        language = None if language in [None, "Auto", ""] else language
        
        if session_id:
            emit_progress(session_id, "generating", 40, "Preparing generation...")
        
        # Create async task to emit heartbeat progress during generation
        async def heartbeat():
            await asyncio.sleep(1)
            if session_id:
                emit_progress(session_id, "generating", 50, "Generating audio codes...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 60, "Processing text and instructions...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 70, "Synthesizing speech...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 80, "Finalizing audio waveform...")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat())
        
        # Run generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            lambda: model_manager.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )
        )
        
        # Cancel heartbeat if still running
        heartbeat_task.cancel()
        
        if session_id:
            emit_progress(session_id, "processing", 90, "Processing audio...")
        
        response = audio_processor.prepare_response(audio_data)
        
        if session_id:
            emit_progress(session_id, "complete", 100, "Audio generation complete!")
        
        return response
        
    except Exception as e:
        if session_id:
            emit_progress(session_id, "error", 0, f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Voice Clone TTS endpoint
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
            emit_progress(session_id, "loading", 25, "Model loaded successfully")
            emit_progress(session_id, "processing", 35, "Processing reference audio...")
        
        # Handle audio input
        ref_audio_data = await audio_processor.process_audio_input(
            file=ref_audio,
            url=ref_audio_url,
            base64_data=ref_audio_base64,
            prompt_file=voice_prompt_file
        )
        
        if session_id:
            emit_progress(session_id, "generating", 45, "Preparing voice cloning...")
        
        # Create async task to emit heartbeat progress during generation
        async def heartbeat():
            await asyncio.sleep(1)
            if session_id:
                emit_progress(session_id, "generating", 55, "Analyzing reference audio...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 65, "Extracting voice features...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 75, "Cloning voice characteristics...")
            await asyncio.sleep(2)
            if session_id:
                emit_progress(session_id, "generating", 85, "Synthesizing with cloned voice...")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat())
        
        # Convert "Auto" to None for auto-detection
        lang = None if language in [None, "Auto", ""] else language
        
        # Run generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            lambda: model_manager.generate_voice_clone(
                text=text,
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
        
        # Cancel heartbeat if still running
        heartbeat_task.cancel()
        
        if session_id:
            emit_progress(session_id, "processing", 90, "Processing audio...")
        
        response = audio_processor.prepare_response(audio_data)
        
        if session_id:
            emit_progress(session_id, "complete", 100, "Voice cloning complete!")
        
        return response
        
    except Exception as e:
        if session_id:
            emit_progress(session_id, "error", 0, f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Save voice prompt endpoint
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
        
        # Process audio
        ref_audio_data = await audio_processor.process_audio_input(file=ref_audio)
        
        # Create prompt
        prompt_path = model_manager.create_voice_prompt(
            ref_audio=ref_audio_data,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
            save_path=f"./voice_prompts/{prompt_name}.pt"
        )
        
        return {
            "status": "success",
            "message": "Voice prompt saved",
            "path": prompt_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Download audio endpoint
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

# Serve frontend static files
if os.path.exists("../frontend/build"):
    app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="frontend")

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
