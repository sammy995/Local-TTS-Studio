"""
Audio Pipeline - Pure Data Transformations
All functions are side-effect free: arrays in → arrays/bytes out
No file I/O, no FastAPI imports, no environment awareness
"""

import io
from typing import Optional
import numpy as np
import soundfile as sf
import librosa
from typing import Tuple, List


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range
    
    Args:
        audio: Audio waveform as numpy array
        
    Returns:
        Normalized audio array
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio = audio / max_val
    
    return audio


def to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: int = 44100
) -> bytes:
    """
    Convert audio array to WAV bytes with optional resampling
    
    Args:
        audio: Audio waveform as numpy array
        sample_rate: Source sample rate
        target_sr: Target sample rate (default 44.1kHz)
        
    Returns:
        WAV audio as bytes
    """
    # Resample if needed
    if sample_rate != target_sr:
        audio = resample(audio, sample_rate, target_sr)
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Convert to bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio, target_sr, format='WAV')
    buffer.seek(0)
    
    return buffer.read()


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to different sample rate
    
    Args:
        audio: Audio waveform
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def concat_audio(
    audio_arrays: List[np.ndarray],
    sample_rate: int,
    silence_duration: float = 0.5
) -> np.ndarray:
    """
    Concatenate multiple audio arrays with silence between them
    
    Args:
        audio_arrays: List of audio waveforms
        sample_rate: Sample rate for silence generation
        silence_duration: Duration of silence in seconds
        
    Returns:
        Concatenated audio array
    """
    if not audio_arrays:
        return np.array([], dtype=np.float32)
    
    if len(audio_arrays) == 1:
        return audio_arrays[0]
    
    # Create silence
    silence = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)
    
    # Interleave audio with silence
    result = []
    for audio in audio_arrays:
        result.append(audio)
        result.append(silence)
    
    # Remove trailing silence
    result = result[:-1]
    
    return np.concatenate(result)


def generate_silence(
    duration_ms: int,
    sample_rate: int,
    reference_array: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate silence matching target dtype and device
    
    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        reference_array: Optional array to match dtype/device (future: GPU support)
        
    Returns:
        Silence array of zeros
    """
    num_samples = int((duration_ms / 1000.0) * sample_rate)
    
    if reference_array is not None:
        # Match dtype (and device if GPU tensor later)
        dtype = reference_array.dtype
    else:
        # Default to float32
        dtype = np.float32
    
    return np.zeros(num_samples, dtype=dtype)


def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes
    
    Args:
        audio_bytes: Audio data as bytes
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    buffer = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buffer)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    return audio.astype(np.float32), sr
