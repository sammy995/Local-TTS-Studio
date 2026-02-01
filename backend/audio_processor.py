"""
Audio Processor
Handles audio input/output processing, format conversion, and file management
"""

import io
import base64
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Union, Any, Dict, List
import numpy as np
import soundfile as sf
import librosa
import requests
from fastapi import UploadFile, HTTPException
from fastapi.responses import Response
import logging

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub not available, MP3/FLAC export will be limited")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles all audio processing operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config['audio']['sample_rate']
        self.output_dir = Path("./outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path("./temp_audio")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_to_wav(self, audio_path: str) -> str:
        """
        Convert any audio format to WAV using pydub (which uses ffmpeg if available)
        or librosa as fallback
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Path to converted WAV file (same path if already WAV and valid)
        """
        audio_path = Path(audio_path)
        
        # If it's already a WAV, try to load it
        if audio_path.suffix.lower() == '.wav':
            try:
                sf.read(audio_path)
                return str(audio_path)
            except:
                pass  # Fall through to conversion
        
        # Create output path
        wav_path = audio_path.parent / f"{audio_path.stem}_converted.wav"
        
        try:
            # Try pydub first (works with more formats via ffmpeg)
            if PYDUB_AVAILABLE:
                try:
                    logger.info(f"Converting {audio_path.suffix} to WAV using pydub...")
                    audio = AudioSegment.from_file(str(audio_path))
                    audio = audio.set_channels(1)  # Mono
                    audio.export(str(wav_path), format="wav")
                    logger.info(f"Converted successfully to {wav_path}")
                    return str(wav_path)
                except Exception as e:
                    logger.warning(f"pydub conversion failed: {e}, trying librosa...")
            
            # Fall back to librosa
            logger.info(f"Converting {audio_path.suffix} to WAV using librosa...")
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
            sf.write(str(wav_path), y, sr)
            logger.info(f"Converted successfully to {wav_path}")
            return str(wav_path)
            
        except Exception as e:
            error_msg = (
                f"Unable to convert {audio_path.suffix} audio file to WAV. "
                f"FFmpeg is required for M4A/MP3/FLAC support.\n\n"
                f"To install FFmpeg:\n"
                f"  • Via conda: conda install -c conda-forge ffmpeg\n"
                f"  • Via winget: winget install --id Gyan.FFmpeg\n"
                f"  • Or download from: https://ffmpeg.org/download.html\n\n"
                f"Alternatively, please convert your audio to WAV format first.\n"
                f"Error details: {str(e)}"
            )
            raise HTTPException(status_code=400, detail=error_msg)
    
    async def process_audio_input(
        self,
        file: Optional[UploadFile] = None,
        url: Optional[str] = None,
        base64_data: Optional[str] = None,
        prompt_file: Optional[UploadFile] = None
    ) -> Union[str, Any]:
        """
        Process audio input from various sources
        
        Args:
            file: Uploaded audio file
            url: URL to audio file
            base64_data: Base64 encoded audio
            prompt_file: Pre-saved voice prompt file (.pt)
            
        Returns:
            Path to processed audio file or loaded prompt object
        """
        # Priority: prompt_file > file > url > base64
        
        if prompt_file:
            # Load voice prompt directly
            import torch
            temp_path = self.temp_dir / f"{uuid.uuid4()}.pt"
            with open(temp_path, "wb") as f:
                f.write(await prompt_file.read())
            return torch.load(temp_path)
        
        if file:
            # Save uploaded file
            temp_path = self.temp_dir / f"{uuid.uuid4()}_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(await file.read())
            
            # Convert to WAV if needed
            wav_path = self.convert_to_wav(str(temp_path))
            return wav_path
        
        if url:
            # Download from URL
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                temp_path = self.temp_dir / f"{uuid.uuid4()}_from_url.wav"
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                # Convert to WAV if needed
                wav_path = self.convert_to_wav(str(temp_path))
                return wav_path
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")
        
        if base64_data:
            # Decode base64 audio
            try:
                # Remove data URL prefix if present
                if "," in base64_data:
                    base64_data = base64_data.split(",")[1]
                
                audio_bytes = base64.b64decode(base64_data)
                temp_path = self.temp_dir / f"{uuid.uuid4()}_from_base64.wav"
                
                with open(temp_path, "wb") as f:
                    f.write(audio_bytes)
                return str(temp_path)
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error decoding base64 audio: {str(e)}")
        
        raise HTTPException(status_code=400, detail="No audio input provided")
    
    def save_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        filename: Optional[str] = None,
        format: str = "wav"
    ) -> str:
        """
        Save audio data to file
        
        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate in Hz
            filename: Output filename (auto-generated if None)
            format: Output format ('wav', 'mp3', 'flac')
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"tts_output_{uuid.uuid4()}.{format}"
        
        output_path = self.output_dir / filename
        
        # Ensure audio is in correct format (float32, [-1, 1])
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        if format == "wav":
            sf.write(output_path, audio_data, sample_rate)
        
        elif format in ["mp3", "flac"] and PYDUB_AVAILABLE:
            # Convert via WAV first
            temp_wav = self.temp_dir / f"temp_{uuid.uuid4()}.wav"
            sf.write(temp_wav, audio_data, sample_rate)
            
            # Convert to target format
            audio_segment = AudioSegment.from_wav(str(temp_wav))
            audio_segment.export(output_path, format=format)
            
            # Clean up temp file
            temp_wav.unlink()
        
        else:
            # Fallback to WAV
            logger.warning(f"Format {format} not available, saving as WAV")
            output_path = output_path.with_suffix(".wav")
            sf.write(output_path, audio_data, sample_rate)
        
        logger.info(f"Audio saved to {output_path}")
        return str(output_path)
    
    def prepare_response(
        self,
        audio_data: tuple,
        format: str = "wav",
        return_file: bool = False
    ) -> Union[Response, Dict[str, Any]]:
        """
        Prepare audio response for API
        
        Args:
            audio_data: Tuple of (wavs, sample_rate) from model
            format: Output format
            return_file: If True, return file path instead of streaming
            
        Returns:
            FastAPI Response or dict with file info
        """
        wavs, sample_rate = audio_data
        
        # Handle batch output (list of arrays)
        if isinstance(wavs, list):
            if len(wavs) == 1:
                audio_array = wavs[0]
            else:
                # For batch, concatenate or return multiple files
                # For now, concatenate with silence
                silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
                audio_array = np.concatenate([
                    item for wav in wavs for item in (wav, silence)
                ])[:-len(silence)]  # Remove trailing silence
        else:
            audio_array = wavs
        
        if return_file:
            # Save to file and return path
            file_path = self.save_audio(audio_array, sample_rate, format=format)
            return {
                "status": "success",
                "file_path": file_path,
                "filename": Path(file_path).name,
                "sample_rate": sample_rate,
                "duration": len(audio_array) / sample_rate
            }
        else:
            # Stream audio directly
            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format='WAV')
            buffer.seek(0)
            
            return Response(
                content=buffer.read(),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="tts_output.wav"'
                }
            )
    
    def convert_format(self, input_path: str, output_format: str) -> str:
        """Convert audio file to different format"""
        input_path = Path(input_path)
        output_path = input_path.with_suffix(f".{output_format}")
        
        if output_format == "wav" or not PYDUB_AVAILABLE:
            # Load and save with soundfile
            audio, sr = sf.read(input_path)
            sf.write(output_path, audio, sr)
        else:
            # Use pydub for MP3/FLAC
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format=output_format)
        
        return str(output_path)
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary audio files"""
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in self.temp_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    logger.info(f"Cleaned up temp file: {file_path}")
