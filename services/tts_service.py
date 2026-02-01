"""
TTS Service - Business Orchestration Layer
Stateless pure functions: validate → generate → save → return metadata
Storage passed per-call, no environment awareness
"""

from typing import List, Dict, Any, Optional, Union, Protocol
from pathlib import Path
import uuid
import logging

# Import from core (compute)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.tts_engine import TTSEngine
from core.audio_pipeline import to_wav_bytes, concat_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Storage protocol (structural typing)
class Storage(Protocol):
    """Storage interface - any object with these methods works"""
    
    def save_audio(self, audio_bytes: bytes, filename: str) -> str:
        """Save audio bytes, return path or URL"""
        ...
    
    def save_prompt(self, tensor: Any, name: str) -> str:
        """Save prompt tensor, return path or URL"""
        ...
    
    def load_prompt(self, name: str) -> Any:
        """Load prompt tensor"""
        ...


class TTSService:
    """Stateless TTS orchestration service"""
    
    MAX_TEXT_LENGTH = 5000
    
    def __init__(self, engine: TTSEngine):
        """
        Initialize TTS Service
        
        Args:
            engine: TTSEngine instance (no storage stored)
        """
        self.engine = engine
    
    def generate_speech(
        self,
        text: str,
        mode: str,
        storage: Storage,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[Any] = None,
        ref_text: Optional[str] = None,
        x_vector_only: bool = False,
        target_sample_rate: int = 44100,
        **generation_params
    ) -> str:
        """
        Generate speech and save to storage
        
        Args:
            text: Text to synthesize
            mode: Generation mode ('custom_voice', 'voice_design', 'voice_clone')
            storage: Storage implementation (passed per call)
            speaker: Speaker name (for custom_voice)
            language: Language code
            instruct: Instructions (for voice_design or custom_voice)
            ref_audio: Reference audio (for voice_clone)
            ref_text: Reference text (for voice_clone)
            x_vector_only: X-vector only mode (for voice_clone)
            target_sample_rate: Target sample rate for output
            **generation_params: Additional parameters
            
        Returns:
            Audio file ID/path
        """
        # Validation
        self._validate_text(text)
        self._validate_mode(mode, speaker, instruct, ref_audio)
        
        # Generate (pure compute)
        if mode == "custom_voice":
            if not speaker:
                raise ValueError("speaker is required for custom_voice mode")
            audio_array, sample_rate = self.engine.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                **generation_params
            )
        elif mode == "voice_design":
            if not instruct:
                raise ValueError("instruct is required for voice_design mode")
            audio_array, sample_rate = self.engine.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                **generation_params
            )
        elif mode == "voice_clone":
            audio_array, sample_rate = self.engine.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only=x_vector_only,
                **generation_params
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Transform to bytes
        audio_bytes = to_wav_bytes(audio_array, sample_rate, target_sample_rate)
        
        # Save to storage
        filename = f"tts_{uuid.uuid4()}.wav"
        file_path = storage.save_audio(audio_bytes, filename)
        
        logger.info(f"Generated speech: {len(text)} chars → {file_path}")
        
        return file_path
    
    def generate_batch(
        self,
        requests: List[Dict[str, Any]],
        storage: Storage
    ) -> List[str]:
        """
        Generate batch of speech files
        
        Args:
            requests: List of generation request dicts
            storage: Storage implementation
            
        Returns:
            List of audio file IDs/paths
        """
        results = []
        
        for i, request in enumerate(requests):
            try:
                file_path = self.generate_speech(storage=storage, **request)
                results.append(file_path)
                logger.info(f"Batch {i+1}/{len(requests)}: {file_path}")
            except Exception as e:
                logger.error(f"Batch {i+1}/{len(requests)} failed: {str(e)}")
                results.append(None)
        
        return results
    
    def create_voice_prompt(
        self,
        name: str,
        ref_audio: Any,
        ref_text: str,
        storage: Storage,
        x_vector_only: bool = False
    ) -> str:
        """
        Create and save reusable voice prompt
        
        Args:
            name: Prompt name
            ref_audio: Reference audio
            ref_text: Reference transcript
            storage: Storage implementation
            x_vector_only: X-vector only mode
            
        Returns:
            Prompt ID/path
        """
        # Generate prompt tensor
        prompt_tensor = self.engine.create_voice_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only
        )
        
        # Save to storage
        prompt_path = storage.save_prompt(prompt_tensor, name)
        
        logger.info(f"Created voice prompt: {name} → {prompt_path}")
        
        return prompt_path
    
    def _validate_text(self, text: str):
        """Validate text input"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValueError(f"Text exceeds maximum length of {self.MAX_TEXT_LENGTH} characters")
    
    def _validate_mode(
        self,
        mode: str,
        speaker: Optional[str],
        instruct: Optional[str],
        ref_audio: Optional[Any]
    ):
        """Validate mode and required parameters"""
        valid_modes = ["custom_voice", "voice_design", "voice_clone"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of: {valid_modes}")
        
        if mode == "custom_voice" and not speaker:
            raise ValueError("speaker is required for custom_voice mode")
        
        if mode == "voice_design" and not instruct:
            raise ValueError("instruct is required for voice_design mode")
        
        if mode == "voice_clone" and not ref_audio:
            raise ValueError("ref_audio is required for voice_clone mode")
