"""
Podcast Data Models - Content Production Infrastructure
Immutable dataclasses for script-to-audio compilation

v3: Multi-track timeline with music beds, ducking, and asset management
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal
from enum import Enum


# ===== v3: ENUMS =====

class SegmentKind(str, Enum):
    """Type of content in a timeline segment"""
    SPEECH = "speech"
    MUSIC = "music"
    SFX = "sfx"


class AssetSource(str, Enum):
    """Where an audio asset originated"""
    LOCAL = "local"        # User uploaded
    JAMENDO = "jamendo"    # Jamendo API
    OPENVERSE = "openverse"  # Openverse API
    FREESOUND = "freesound"  # Freesound API


# ===== v3: AUDIO ASSETS =====

@dataclass(frozen=True)
class AudioAsset:
    """
    Reusable audio asset (music track, sound effect, etc.)
    
    Assets live in the library and can be referenced by multiple
    MusicTrack or SFX segments across projects.
    
    Attributes:
        id: Unique asset identifier
        title: Display name
        source: Where the asset came from
        source_url: Original URL (for attribution/re-download)
        local_path: Path to locally cached file
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate of the audio
        license: License type (e.g., "CC BY 4.0", "CC0")
        attribution: Required attribution text
        artist: Artist/creator name
        tags: Searchable tags
    """
    id: str
    title: str
    source: AssetSource = AssetSource.LOCAL
    source_url: Optional[str] = None
    local_path: Optional[str] = None
    duration_ms: int = 0
    sample_rate: int = 44100
    license: str = "unknown"
    attribution: Optional[str] = None
    artist: Optional[str] = None
    tags: List[str] = field(default_factory=list)


# ===== v3: MUSIC TRACKS =====

@dataclass(frozen=True)
class MusicTrack:
    """
    A music/SFX placement on the timeline
    
    Defines when and how an audio asset plays relative to the
    master timeline, including volume, ducking, and fades.
    
    Attributes:
        id: Unique track identifier
        asset_id: Reference to AudioAsset.id
        start_ms: Absolute start position on timeline
        end_ms: End position (asset trimmed/looped to fit)
        volume: Gain multiplier (1.0 = normal)
        duck_under_speech: Auto-reduce volume during speech
        duck_level: Volume multiplier during speech (0.0-1.0)
        duck_attack_ms: Fade-down time when speech starts
        duck_release_ms: Fade-up time when speech ends
        fade_in_ms: Fade in at start
        fade_out_ms: Fade out at end
        loop: Loop asset to fill duration
    """
    id: str
    asset_id: str
    start_ms: int = 0
    end_ms: Optional[int] = None  # None = asset's natural duration
    volume: float = 0.3  # Background music default lower
    duck_under_speech: bool = True
    duck_level: float = 0.2  # 20% volume during speech
    duck_attack_ms: int = 200  # 200ms fade-down
    duck_release_ms: int = 400  # 400ms fade-up
    fade_in_ms: int = 500
    fade_out_ms: int = 500
    loop: bool = True  # Loop by default for background music


# ===== CORE MODELS =====

@dataclass(frozen=True)
class Speaker:
    """
    Voice persona definition - reusable across segments
    
    Attributes:
        id: Unique speaker identifier
        name: Human-readable speaker name
        mode: Voice generation mode
        config: Mode-specific configuration (NOT validated by service)
        style_instruction: Optional style guidance
        
    Config Keys (documented, not enforced):
        mode='custom': {'voice_id': str}
        mode='design': {'description': str}  # e.g., "young female, cheerful"
        mode='clone': {'ref_audio_path': str, 'ref_text': str|None}  # reference audio file path
    """
    id: str
    name: str
    mode: Literal["custom", "design", "clone"]
    config: Dict[str, Any]
    style_instruction: Optional[str] = None


@dataclass(frozen=True)
class Segment:
    """
    Atomic content unit on the timeline
    
    v3: Segments now have absolute timeline positions (start_ms)
    and can be speech, music, or sfx.
    
    Attributes:
        id: Unique segment identifier
        order: Sort order (allow gaps: 10, 20, 30 for easy insertion)
        speaker_id: Reference to Speaker.id (speech only)
        text: Text to synthesize (speech only)
        kind: Type of segment (speech/music/sfx)
        start_ms: Absolute position on timeline (None = auto-sequential)
        duration_ms: Actual duration after render (set by engine)
        asset_id: Reference to AudioAsset.id (music/sfx only)
        pause_after_ms: Silence duration after segment (speech, sequential mode)
        volume: Gain multiplier (1.0 = normal, 0.5 = half, 2.0 = double)
        emotion: Optional emotion hint (model-dependent)
        
    Note: Segments are fault-tolerance units - keep atomic for retries
    """
    id: str
    order: int
    speaker_id: str = ""
    text: str = ""
    kind: str = "speech"  # speech | music | sfx
    start_ms: Optional[int] = None  # None = auto-sequential (v2 compat)
    duration_ms: Optional[int] = None  # Filled after render
    asset_id: Optional[str] = None  # For music/sfx segments
    pause_after_ms: int = 500  # Default 0.5s pause
    volume: float = 1.0
    emotion: Optional[str] = None


@dataclass(frozen=True)
class PodcastProject:
    """
    Top-level podcast compilation definition
    
    v3: Now includes music_tracks and assets for multi-track timeline
    
    Attributes:
        id: Unique project identifier
        title: Project name
        speakers: List of voice personas
        segments: List of content segments (speech/music/sfx)
        music_tracks: Background music track placements
        assets: Audio assets referenced by music_tracks
        output_format: Audio format ('wav' or 'mp3')
        target_sample_rate: Output sample rate (Hz)
        deterministic: Compiler mode (True) vs Creative mode (False)
        
    Compiler mode (deterministic=True):
        - Greedy decoding (temp=0)
        - Stable hashing for seed
        - Identical output bytes
        
    Creative mode (deterministic=False):
        - Stochastic sampling (temp=0.9)
        - Variety across renders
    """
    id: str
    title: str
    speakers: List[Speaker]
    segments: List[Segment]
    music_tracks: List[MusicTrack] = field(default_factory=list)
    assets: List[AudioAsset] = field(default_factory=list)
    output_format: Literal["wav", "mp3"] = "wav"
    target_sample_rate: int = 44100
    deterministic: bool = True  # Default to compiler mode


@dataclass(frozen=True)
class RenderJob:
    """
    Async render job (for cloud runtime)
    
    Attributes:
        id: Unique job identifier
        project_id: Reference to PodcastProject.id
        status: Current job state
        progress: Completion percentage (0-100)
        output_url: URL/path to final audio (when done)
        error: Error message (when failed)
        created_at: Job creation timestamp
        completed_at: Job completion timestamp
        
    Local runtime: Skips jobs, renders synchronously
    Cloud runtime: Creates job → enqueues worker → updates status
    """
    id: str
    project_id: str
    status: Literal["pending", "running", "done", "failed"]
    progress: float = 0.0
    output_url: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
