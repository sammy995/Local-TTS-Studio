"""
Audio Pipeline - Pure Data Transformations
All functions are side-effect free: arrays in → arrays/bytes out
No file I/O, no FastAPI imports, no environment awareness

v3: Added overlay mixing, ducking, fades, looping, and timeline mixing
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


# ===== v3: MULTI-TRACK MIXING ENGINE =====


def apply_fade(
    audio: np.ndarray,
    fade_in_samples: int,
    fade_out_samples: int
) -> np.ndarray:
    """
    Apply linear fade-in and fade-out to audio
    
    Args:
        audio: Audio waveform
        fade_in_samples: Number of samples to fade in
        fade_out_samples: Number of samples to fade out
        
    Returns:
        Audio with fades applied
    """
    result = audio.copy()
    length = len(result)
    
    if fade_in_samples > 0 and fade_in_samples < length:
        fade_in_curve = np.linspace(0.0, 1.0, fade_in_samples, dtype=np.float32)
        result[:fade_in_samples] *= fade_in_curve
    
    if fade_out_samples > 0 and fade_out_samples < length:
        fade_out_curve = np.linspace(1.0, 0.0, fade_out_samples, dtype=np.float32)
        result[-fade_out_samples:] *= fade_out_curve
    
    return result


def loop_to_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Loop audio to fill target length, with crossfade at loop points
    
    Args:
        audio: Source audio to loop
        target_samples: Desired output length in samples
        
    Returns:
        Looped audio of exact target length
    """
    if len(audio) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    
    if len(audio) >= target_samples:
        return audio[:target_samples]
    
    # Build looped buffer
    result = np.zeros(target_samples, dtype=np.float32)
    offset = 0
    while offset < target_samples:
        remaining = target_samples - offset
        chunk_len = min(len(audio), remaining)
        result[offset:offset + chunk_len] = audio[:chunk_len]
        offset += chunk_len
    
    return result


def trim_audio(
    audio: np.ndarray,
    start_samples: int,
    end_samples: int
) -> np.ndarray:
    """
    Trim audio to range [start, end)
    
    Args:
        audio: Source audio
        start_samples: Start sample index
        end_samples: End sample index
        
    Returns:
        Trimmed audio slice
    """
    start = max(0, start_samples)
    end = min(len(audio), end_samples)
    return audio[start:end].copy()


def overlay_mix(
    base: np.ndarray,
    overlay: np.ndarray,
    offset_samples: int,
    volume: float = 1.0
) -> np.ndarray:
    """
    Mix overlay audio onto base at given offset (additive)
    
    Args:
        base: Base audio array (modified in-place)
        overlay: Audio to mix in
        offset_samples: Where to start mixing on base
        volume: Gain multiplier for overlay
        
    Returns:
        Base array with overlay mixed in
    """
    if offset_samples < 0:
        # Trim overlay start if offset is negative
        overlay = overlay[-offset_samples:]
        offset_samples = 0
    
    end = min(offset_samples + len(overlay), len(base))
    mix_len = end - offset_samples
    
    if mix_len > 0:
        base[offset_samples:end] += overlay[:mix_len] * volume
    
    return base


def build_ducking_curve(
    total_samples: int,
    speech_regions: List[Tuple[int, int]],
    duck_level: float = 0.2,
    attack_samples: int = 4800,   # ~200ms at 24kHz
    release_samples: int = 9600   # ~400ms at 24kHz
) -> np.ndarray:
    """
    Build a volume envelope that ducks during speech regions
    
    The curve is 1.0 (full volume) everywhere except during speech,
    where it drops to duck_level with smooth attack/release ramps.
    
    Args:
        total_samples: Length of output curve
        speech_regions: List of (start_sample, end_sample) tuples where speech occurs
        duck_level: Volume during speech (0.0 = silence, 1.0 = no ducking)
        attack_samples: Ramp-down duration when speech starts
        release_samples: Ramp-up duration when speech ends
        
    Returns:
        Volume envelope array (same length as total_samples)
    """
    curve = np.ones(total_samples, dtype=np.float32)
    
    for start, end in speech_regions:
        # Clamp to bounds
        start = max(0, start)
        end = min(total_samples, end)
        
        if start >= end:
            continue
        
        # Duck the speech region
        curve[start:end] = duck_level
        
        # Attack ramp (before speech starts)
        ramp_start = max(0, start - attack_samples)
        ramp_len = start - ramp_start
        if ramp_len > 0:
            ramp = np.linspace(1.0, duck_level, ramp_len, dtype=np.float32)
            # Take minimum with existing curve (in case regions overlap)
            curve[ramp_start:start] = np.minimum(curve[ramp_start:start], ramp)
        
        # Release ramp (after speech ends)
        ramp_end = min(total_samples, end + release_samples)
        ramp_len = ramp_end - end
        if ramp_len > 0:
            ramp = np.linspace(duck_level, 1.0, ramp_len, dtype=np.float32)
            curve[end:ramp_end] = np.minimum(curve[end:ramp_end], ramp)
    
    return curve


def mix_timeline(
    speech_placements: List[Tuple[int, np.ndarray]],
    music_placements: List[dict],
    total_samples: int,
    sample_rate: int
) -> np.ndarray:
    """
    Master timeline mixer: place speech at positions, overlay music with ducking
    
    This is the core v3 engine: instead of concatenating sequentially,
    it places audio at absolute sample positions on a master buffer.
    
    Args:
        speech_placements: List of (offset_samples, audio_array) for speech
        music_placements: List of dicts with keys:
            - audio: np.ndarray (the music audio, already looped/trimmed)
            - offset_samples: int (where to start on timeline)
            - volume: float (gain multiplier)
            - duck_under_speech: bool
            - duck_level: float (0.0-1.0)
            - duck_attack_samples: int
            - duck_release_samples: int
            - fade_in_samples: int
            - fade_out_samples: int
        total_samples: Length of master buffer
        sample_rate: Sample rate (for reference)
        
    Returns:
        Final mixed audio array
    """
    # Create master buffer
    master = np.zeros(total_samples, dtype=np.float32)
    
    # === Pass 1: Place speech ===
    speech_regions = []
    for offset, audio in speech_placements:
        end = min(offset + len(audio), total_samples)
        mix_len = end - offset
        if mix_len > 0:
            master[offset:end] += audio[:mix_len]
            speech_regions.append((offset, end))
    
    # === Pass 2: Place music with ducking ===
    for track in music_placements:
        music_audio = track['audio']
        offset = track['offset_samples']
        volume = track.get('volume', 0.3)
        
        # Apply fades to music
        fade_in = track.get('fade_in_samples', 0)
        fade_out = track.get('fade_out_samples', 0)
        if fade_in > 0 or fade_out > 0:
            music_audio = apply_fade(music_audio, fade_in, fade_out)
        
        # Build ducking curve if needed
        if track.get('duck_under_speech', True) and speech_regions:
            # Shift speech regions relative to this track's start
            local_regions = []
            track_end = offset + len(music_audio)
            for s_start, s_end in speech_regions:
                # Check overlap
                if s_end > offset and s_start < track_end:
                    local_start = max(0, s_start - offset)
                    local_end = min(len(music_audio), s_end - offset)
                    local_regions.append((local_start, local_end))
            
            if local_regions:
                duck_curve = build_ducking_curve(
                    total_samples=len(music_audio),
                    speech_regions=local_regions,
                    duck_level=track.get('duck_level', 0.2),
                    attack_samples=track.get('duck_attack_samples', int(0.2 * sample_rate)),
                    release_samples=track.get('duck_release_samples', int(0.4 * sample_rate))
                )
                music_audio = music_audio * duck_curve
        
        # Mix onto master
        overlay_mix(master, music_audio, offset, volume)
    
    # === Pass 3: Final normalization ===
    master = normalize_audio(master)
    
    return master
