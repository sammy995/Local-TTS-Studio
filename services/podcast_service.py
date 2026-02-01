"""
Podcast Service - Content Production Compiler
Deterministic script-to-audio pipeline with segment-level atomicity
"""

import hashlib
import numpy as np
from typing import Dict, Tuple, Any, Optional
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.podcast_models import PodcastProject, Speaker, Segment
from core.tts_engine import TTSEngine
from core.audio_pipeline import to_wav_bytes, concat_audio, generate_silence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PodcastService:
    """
    Stateless podcast compilation service
    
    Design principles:
    - Segment-level atomicity (fault tolerance)
    - Speaker prompt caching (performance)
    - Deterministic by default (compiler mode)
    - Memory-safe concatenation (preallocate)
    - Storage injected per-call (environment agnostic)
    """
    
    MAX_SPEAKERS = 10
    MAX_SEGMENT_CHARS = 5000
    MAX_DURATION_MINUTES = 60
    
    def __init__(self, engine: TTSEngine):
        """
        Initialize PodcastService
        
        Args:
            engine: TTSEngine instance (no storage, no pipeline - injected per-call)
        """
        self.engine = engine
    
    def validate_project(self, project: PodcastProject) -> None:
        """
        Structural validation only (cheap + deterministic)
        
        Rules:
        - Max 10 speakers
        - Max 5000 chars per segment
        - Max 60 min estimated duration
        - Speaker references valid
        - Segment order uniqueness
        
        Does NOT validate:
        - Text language/chars (engine decides)
        - Model availability (engine decides)
        - Config keys (engine interprets)
        
        Args:
            project: PodcastProject to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Speaker limits
        if len(project.speakers) > self.MAX_SPEAKERS:
            raise ValueError(f"Too many speakers: {len(project.speakers)} (max {self.MAX_SPEAKERS})")
        
        if len(project.speakers) == 0:
            raise ValueError("Project must have at least one speaker")
        
        # Build speaker ID map
        speaker_ids = {s.id for s in project.speakers}
        
        # Segment validation
        if len(project.segments) == 0:
            raise ValueError("Project must have at least one segment")
        
        segment_orders = set()
        total_chars = 0
        
        for segment in project.segments:
            # Text length
            if len(segment.text) > self.MAX_SEGMENT_CHARS:
                raise ValueError(f"Segment {segment.id} exceeds {self.MAX_SEGMENT_CHARS} chars")
            
            if not segment.text.strip():
                raise ValueError(f"Segment {segment.id} has empty text")
            
            # Speaker reference
            if segment.speaker_id not in speaker_ids:
                raise ValueError(f"Segment {segment.id} references unknown speaker: {segment.speaker_id}")
            
            # Order uniqueness (allow gaps)
            if segment.order in segment_orders:
                raise ValueError(f"Duplicate segment order: {segment.order}")
            segment_orders.add(segment.order)
            
            total_chars += len(segment.text)
        
        # Rough duration estimate (150 words/min, 5 chars/word = 750 chars/min)
        estimated_minutes = total_chars / 750
        if estimated_minutes > self.MAX_DURATION_MINUTES:
            raise ValueError(f"Estimated duration {estimated_minutes:.1f}min exceeds max {self.MAX_DURATION_MINUTES}min")
        
        logger.info(f"Project validation passed: {len(project.speakers)} speakers, {len(project.segments)} segments, ~{estimated_minutes:.1f}min")
    
    def _precompute_speaker_prompts(
        self,
        speakers: list[Speaker]
    ) -> Dict[str, Tuple[Any, int]]:
        """
        Precompute voice prompts for all speakers
        
        Huge optimization: If same speaker used 50 times, only generate prompt once
        
        Args:
            speakers: List of speakers to precompute
            
        Returns:
            Dict mapping speaker_id to (prompt_tensor, sample_rate)
        """
        prompt_cache = {}
        
        for speaker in speakers:
            if speaker.mode == "clone" and "prompt_id" in speaker.config:
                # Load pre-saved prompt (future: from storage)
                # For now, skip - handled by engine at generation time
                logger.info(f"Speaker {speaker.id} uses saved prompt: {speaker.config['prompt_id']}")
                prompt_cache[speaker.id] = (None, None)  # Placeholder
            else:
                # No pre-computation needed for custom/design modes
                prompt_cache[speaker.id] = (None, None)
        
        return prompt_cache
    
    def render_segment(
        self,
        segment: Segment,
        speaker: Speaker,
        prompt_cache: Dict[str, Tuple[Any, int]],
        deterministic: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Atomic segment rendering (fault-tolerance unit)
        
        Args:
            segment: Segment to render
            speaker: Speaker definition
            prompt_cache: Precomputed prompts
            deterministic: Use greedy decoding (True) or sampling (False)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Stable seed for deterministic mode
        if deterministic:
            # SHA256 hash of segment ID → stable across machines/versions
            seed_bytes = hashlib.sha256(segment.id.encode()).digest()[:8]
            seed = int.from_bytes(seed_bytes, byteorder='big')
            logger.debug(f"Deterministic seed for {segment.id}: {seed}")
        else:
            seed = None
        
        # Build generation params
        gen_params = {}
        
        if deterministic:
            # Greedy decoding (strictly deterministic)
            gen_params['do_sample'] = False
            gen_params['top_k'] = 1
        else:
            # Creative mode (stochastic sampling)
            gen_params['do_sample'] = True
            gen_params['temperature'] = 0.9
            gen_params['top_p'] = 0.95
            gen_params['top_k'] = 50
        
        # Add seed if deterministic
        if seed is not None:
            gen_params['seed'] = seed
        
        # Generate based on speaker mode
        if speaker.mode == "custom":
            voice_id = speaker.config.get('voice_id')
            if not voice_id:
                raise ValueError(f"Speaker {speaker.id} (custom mode) missing 'voice_id' in config")
            
            audio_array, sample_rate = self.engine.generate_custom_voice(
                text=segment.text,
                speaker=voice_id,
                language=None,  # Auto-detect
                instruct=speaker.style_instruction,
                **gen_params
            )
        
        elif speaker.mode == "design":
            description = speaker.config.get('description')
            if not description:
                raise ValueError(f"Speaker {speaker.id} (design mode) missing 'description' in config")
            
            audio_array, sample_rate = self.engine.generate_voice_design(
                text=segment.text,
                language=None,
                instruct=description,
                **gen_params
            )
        
        elif speaker.mode == "clone":
            prompt_id = speaker.config.get('prompt_id')
            if not prompt_id:
                raise ValueError(f"Speaker {speaker.id} (clone mode) missing 'prompt_id' in config")
            
            # Load prompt from cache or storage (future implementation)
            prompt_tensor, _ = prompt_cache.get(speaker.id, (None, None))
            
            audio_array, sample_rate = self.engine.generate_voice_clone(
                text=segment.text,
                language=None,
                ref_audio=prompt_id,  # For now, pass ID (engine handles loading)
                ref_text=None,
                **gen_params
            )
        
        else:
            raise ValueError(f"Unknown speaker mode: {speaker.mode}")
        
        # Apply volume (gain multiplier)
        if segment.volume != 1.0:
            audio_array = audio_array * segment.volume
        
        logger.info(f"Rendered segment {segment.id}: {len(audio_array)} samples @ {sample_rate}Hz")
        
        return audio_array, sample_rate
    
    def render_podcast(
        self,
        project: PodcastProject,
        storage,  # Storage protocol
        pipeline  # AudioPipeline (injected for flexibility)
    ) -> str:
        """
        Convenience wrapper - renders entire podcast in one call
        
        Memory-safe: Preallocates final array, concat once
        
        Args:
            project: PodcastProject to render
            storage: Storage implementation
            pipeline: AudioPipeline for transformations
            
        Returns:
            Audio file ID/path
        """
        # Validate
        self.validate_project(project)
        
        # Precompute prompts
        prompt_cache = self._precompute_speaker_prompts(project.speakers)
        
        # Build speaker map
        speaker_map = {s.id: s for s in project.speakers}
        
        # Sort segments by order
        sorted_segments = sorted(project.segments, key=lambda s: s.order)
        
        # Render all segments
        arrays = []
        sample_rate = None
        
        for segment in sorted_segments:
            speaker = speaker_map[segment.speaker_id]
            
            audio_array, sr = self.render_segment(
                segment=segment,
                speaker=speaker,
                prompt_cache=prompt_cache,
                deterministic=project.deterministic
            )
            
            if sample_rate is None:
                sample_rate = sr
            
            arrays.append(audio_array)
            
            # Add silence gap
            if segment.pause_after_ms > 0:
                silence = generate_silence(
                    duration_ms=segment.pause_after_ms,
                    sample_rate=sr,
                    reference_array=audio_array
                )
                arrays.append(silence)
        
        # Memory-safe concatenation
        # Calculate total length
        total_length = sum(len(arr) for arr in arrays)
        
        if sample_rate is None:
            raise RuntimeError("No audio generated - sample_rate is None")
        
        # Preallocate
        final_array = np.zeros(total_length, dtype=arrays[0].dtype)
        
        # Copy segments
        offset = 0
        for arr in arrays:
            final_array[offset:offset + len(arr)] = arr
            offset += len(arr)
        
        logger.info(f"Concatenated {len(sorted_segments)} segments: {total_length} samples ({total_length/sample_rate:.1f}s)")
        
        # Encode to bytes
        audio_bytes = to_wav_bytes(final_array, sample_rate, project.target_sample_rate)
        
        # Save to storage
        filename = f"podcast_{project.id}.{project.output_format}"
        file_path = storage.save_audio(audio_bytes, filename)
        
        logger.info(f"Podcast rendered: {file_path}")
        
        return file_path
