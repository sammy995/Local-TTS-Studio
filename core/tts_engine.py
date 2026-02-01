"""
TTS Engine - Pure Compute Orchestration
Composes ModelManager, returns (array, sample_rate) tuples
No file I/O, no FastAPI, no storage, no config reading
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
from .model_manager import ModelManager


class TTSEngine:
    """Pure compute engine for text-to-speech generation"""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize TTS Engine
        
        Args:
            model_manager: ModelManager instance for inference
        """
        self.model_manager = model_manager
    
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: str,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        **generation_params
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using CustomVoice model
        
        Args:
            text: Text to synthesize (string or list of strings)
            speaker: Speaker name/ID
            language: Language code (optional, None for auto)
            instruct: Additional instructions (optional)
            **generation_params: Additional generation parameters
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        wavs, sample_rate = self.model_manager.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            **generation_params
        )
        
        # Handle batch output - concatenate if multiple arrays
        audio_array = self._merge_audio_list(wavs)
        
        return audio_array, sample_rate
    
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        language: Optional[str],
        instruct: str,
        **generation_params
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using VoiceDesign model
        
        Args:
            text: Text to synthesize
            language: Language code (None for auto)
            instruct: Voice design instructions (e.g., "young female, cheerful")
            **generation_params: Additional generation parameters
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        wavs, sample_rate = self.model_manager.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            **generation_params
        )
        
        audio_array = self._merge_audio_list(wavs)
        
        return audio_array, sample_rate
    
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Optional[str],
        ref_audio: Union[str, Any],
        ref_text: Optional[str] = None,
        x_vector_only: bool = False,
        **generation_params
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using Voice Clone model
        
        Args:
            text: Text to synthesize
            language: Language code (None for auto)
            ref_audio: Reference audio (path or prompt tensor)
            ref_text: Reference audio transcript (optional)
            x_vector_only: Use x-vector only mode
            **generation_params: Additional generation parameters
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        wavs, sample_rate = self.model_manager.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
            **generation_params
        )
        
        audio_array = self._merge_audio_list(wavs)
        
        return audio_array, sample_rate
    
    def create_voice_prompt(
        self,
        ref_audio: Union[str, Any],
        ref_text: str,
        x_vector_only: bool = False
    ) -> Any:
        """
        Create a reusable voice prompt tensor
        
        Args:
            ref_audio: Reference audio path
            ref_text: Reference audio transcript
            x_vector_only: Use x-vector only mode
            
        Returns:
            Prompt tensor
        """
        return self.model_manager.create_voice_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only
        )
    
    def _merge_audio_list(self, wavs: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Merge list of audio arrays into single array
        
        Args:
            wavs: Single array or list of arrays
            
        Returns:
            Single concatenated audio array
        """
        if isinstance(wavs, np.ndarray):
            return wavs
        
        if isinstance(wavs, list):
            if len(wavs) == 0:
                return np.array([], dtype=np.float32)
            if len(wavs) == 1:
                return wavs[0]
            
            # Concatenate directly (no silence - pure compute)
            return np.concatenate(wavs)
        
        # Fallback for unknown types
        return np.array(wavs, dtype=np.float32)
