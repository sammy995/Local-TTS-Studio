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
            if speaker.mode == "clone" and speaker.config.get('ref_audio_path'):
                logger.info(f"Speaker {speaker.id} uses clone ref audio: {speaker.config['ref_audio_path']}")
                prompt_cache[speaker.id] = (None, None)
            else:
                # No pre-computation needed for custom/design modes
                prompt_cache[speaker.id] = (None, None)
        
        return prompt_cache
    
    # Minimum expected audio length: ~50ms per character (very fast speech floor)
    MIN_MS_PER_CHAR = 50
    # Max retries for short/failed audio
    MAX_SEGMENT_RETRIES = 2

    def render_segment(
        self,
        segment: Segment,
        speaker: Speaker,
        prompt_cache: Dict[str, Tuple[Any, int]],
        deterministic: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Atomic segment rendering with retry on short/failed output
        
        Args:
            segment: Segment to render
            speaker: Speaker definition
            prompt_cache: Precomputed prompts
            deterministic: Use greedy decoding (True) or sampling (False)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Speaker-stable seed: same speaker always gets same voice characteristics
        # We hash speaker.id (not segment.id) so tone stays consistent across all
        # segments for that speaker. The segment text itself provides the variation.
        if deterministic:
            seed_bytes = hashlib.sha256(speaker.id.encode()).digest()[:8]
            seed = int.from_bytes(seed_bytes, byteorder='big')
            logger.debug(f"Deterministic speaker seed for {speaker.id}: {seed}")
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
        
        # Resolve style instruction: segment emotion overrides speaker default
        effective_style = segment.emotion or speaker.style_instruction
        
        # Attempt generation with retries for short/failed audio
        last_error = None
        for attempt in range(1, self.MAX_SEGMENT_RETRIES + 1):
            try:
                audio_array, sample_rate = self._generate_for_speaker(
                    segment=segment,
                    speaker=speaker,
                    effective_style=effective_style,
                    gen_params=gen_params
                )
                
                # Short-audio detection: TTS sometimes produces near-empty output
                expected_min_ms = len(segment.text) * self.MIN_MS_PER_CHAR
                actual_ms = len(audio_array) / sample_rate * 1000
                
                if actual_ms < min(expected_min_ms, 200):  # Floor of 200ms
                    logger.warning(
                        f"Segment {segment.id} attempt {attempt}: suspiciously short audio "
                        f"({actual_ms:.0f}ms for {len(segment.text)} chars, expected ≥{expected_min_ms:.0f}ms)"
                    )
                    if attempt < self.MAX_SEGMENT_RETRIES:
                        # Retry with sampling enabled to break out of degenerate greedy path
                        gen_params['do_sample'] = True
                        gen_params['temperature'] = 0.7
                        gen_params.pop('top_k', None)
                        continue
                
                # Apply volume (gain multiplier)
                if segment.volume != 1.0:
                    audio_array = audio_array * segment.volume
                
                logger.info(f"Rendered segment {segment.id}: {len(audio_array)} samples @ {sample_rate}Hz ({actual_ms:.0f}ms)")
                return audio_array, sample_rate
                
            except Exception as e:
                last_error = e
                logger.error(f"Segment {segment.id} attempt {attempt} failed: {e}")
                if attempt < self.MAX_SEGMENT_RETRIES:
                    # Retry with relaxed params
                    gen_params['do_sample'] = True
                    gen_params['temperature'] = 0.7
                    gen_params.pop('top_k', None)
                    continue
        
        # All retries exhausted — raise with context
        raise RuntimeError(
            f"Segment {segment.id} (speaker: {speaker.name}) failed after "
            f"{self.MAX_SEGMENT_RETRIES} attempts: {last_error}"
        )
    
    def _generate_for_speaker(
        self,
        segment: Segment,
        speaker: Speaker,
        effective_style: Optional[str],
        gen_params: dict
    ) -> Tuple[np.ndarray, int]:
        """Core TTS dispatch by speaker mode (no retry logic here)"""
        
        if speaker.mode == "custom":
            voice_id = speaker.config.get('voice_id')
            if not voice_id:
                raise ValueError(f"Speaker {speaker.id} (custom mode) missing 'voice_id' in config")
            
            return self.engine.generate_custom_voice(
                text=segment.text,
                speaker=voice_id,
                language=None,  # Auto-detect
                instruct=effective_style,
                **gen_params
            )
        
        elif speaker.mode == "design":
            description = speaker.config.get('description')
            if not description:
                raise ValueError(f"Speaker {speaker.id} (design mode) missing 'description' in config")
            
            # For design mode, the voice description IS the instruct;
            # append emotional style if present
            full_instruct = description
            if effective_style and effective_style != description:
                full_instruct = f"{description}, {effective_style}"
            
            return self.engine.generate_voice_design(
                text=segment.text,
                language=None,
                instruct=full_instruct,
                **gen_params
            )
        
        elif speaker.mode == "clone":
            ref_audio_path = speaker.config.get('ref_audio_path')
            if not ref_audio_path:
                raise ValueError(f"Speaker {speaker.id} (clone mode) missing reference audio")
            
            ref_text = speaker.config.get('ref_text') or None
            
            # If no ref_text provided, use x_vector_only mode (voice fingerprint only)
            # ICL mode (x_vector_only=False) requires ref_text transcription
            x_vector_only = ref_text is None
            
            return self.engine.generate_voice_clone(
                text=segment.text,
                language=None,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                x_vector_only=x_vector_only,
                **gen_params
            )
        
        else:
            raise ValueError(f"Unknown speaker mode: {speaker.mode}")
    
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
