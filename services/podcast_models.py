"""
Podcast Data Models - Content Production Infrastructure
Immutable dataclasses for script-to-audio compilation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal


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
        mode='clone': {'prompt_id': str}     # pre-saved voice prompt
    """
    id: str
    name: str
    mode: Literal["custom", "design", "clone"]
    config: Dict[str, Any]
    style_instruction: Optional[str] = None


@dataclass(frozen=True)
class Segment:
    """
    Atomic speech unit - smallest renderable piece
    
    Attributes:
        id: Unique segment identifier
        order: Sort order (allow gaps: 10, 20, 30 for easy insertion)
        speaker_id: Reference to Speaker.id
        text: Text to synthesize
        pause_after_ms: Silence duration after segment
        volume: Gain multiplier (1.0 = normal, 0.5 = half, 2.0 = double)
        emotion: Optional emotion hint (model-dependent)
        
    Note: Segments are fault-tolerance units - keep atomic for retries
    """
    id: str
    order: int
    speaker_id: str
    text: str
    pause_after_ms: int = 500  # Default 0.5s pause
    volume: float = 1.0
    emotion: Optional[str] = None


@dataclass(frozen=True)
class PodcastProject:
    """
    Top-level podcast compilation definition
    
    Attributes:
        id: Unique project identifier
        title: Project name
        speakers: List of voice personas
        segments: List of speech segments
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
