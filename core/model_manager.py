"""
Model Manager - Pure Model Loading and Inference
No hardcoded paths, no config loading, no file I/O side effects
All parameters injected via constructor
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import logging
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages Qwen3-TTS model loading and inference"""
    
    def __init__(
        self,
        model_base_path: Path,
        device: str,
        dtype: torch.dtype,
        use_flash_attn: bool = False
    ):
        """
        Initialize ModelManager with all parameters
        
        Args:
            model_base_path: Base directory for local models
            device: Device to load models on ('cuda', 'cpu', etc.)
            dtype: Torch dtype for model weights
            use_flash_attn: Whether to use flash attention
        """
        self.model_base_path = model_base_path
        self.device = device
        self.dtype = dtype
        self.use_flash_attn = use_flash_attn
        
        self.models: Dict[str, Optional[Qwen3TTSModel]] = {
            "custom_voice": None,
            "voice_design": None,
            "base_clone": None
        }
        self.current_model_sizes: Dict[str, str] = {}
        
        logger.info(f"ModelManager initialized: device={device}, dtype={dtype}, flash_attn={use_flash_attn}")
    
    def get_model_path(self, model_type: str, model_size: str = "1.7B") -> str:
        """Get the model path - uses local path if available, otherwise HuggingFace"""
        model_names = {
            "custom_voice": f"Qwen3-TTS-12Hz-{model_size}-CustomVoice",
            "voice_design": f"Qwen3-TTS-12Hz-{model_size}-VoiceDesign",
            "base_clone": f"Qwen3-TTS-12Hz-{model_size}-Base"
        }
        
        model_name = model_names.get(model_type)
        if not model_name:
            raise ValueError(f"Invalid model type: {model_type}")
            
        # Check if local model exists
        local_path = self.model_base_path / model_name
        if local_path.exists() and (local_path / "config.json").exists():
            logger.info(f"Using local model: {local_path}")
            return str(local_path)
        
        # Fall back to HuggingFace
        hf_path = f"Qwen/{model_name}"
        logger.info(f"Local model not found, using HuggingFace: {hf_path}")
        return hf_path
    
    def load_model(self, model_type: str, model_size: str = "1.7B"):
        """
        Load a specific model type
        
        Args:
            model_type: One of 'custom_voice', 'voice_design', 'base_clone'
            model_size: Model size ('1.7B' or '0.6B')
        """
        if model_type not in self.models:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Check if already loaded with same size
        if (self.models[model_type] is not None and 
            self.current_model_sizes.get(model_type) == model_size):
            logger.info(f"Model {model_type} ({model_size}) already loaded")
            return
        
        # Unload existing model if different size
        if self.models[model_type] is not None:
            logger.info(f"Unloading previous {model_type} model")
            del self.models[model_type]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        model_path = self.get_model_path(model_type, model_size)
        logger.info(f"Loading model: {model_path}")
        
        try:
            attn_impl = "flash_attention_2" if self.use_flash_attn else None
            
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device,
                torch_dtype=self.dtype,
                attn_implementation=attn_impl,
                trust_remote_code=True
            )
            
            self.models[model_type] = model
            self.current_model_sizes[model_type] = model_size
            logger.info(f"Model {model_type} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {str(e)}")
            raise
    
    def ensure_model_loaded(self, model_type: str, model_size: str = "1.7B"):
        """Ensure a model is loaded, loading it if necessary"""
        if self.models[model_type] is None:
            self.load_model(model_type, model_size)
    
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: str,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        **generation_params
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech using CustomVoice model
        
        Returns:
            Tuple of (list of audio arrays, sample_rate)
        """
        if self.models["custom_voice"] is None:
            raise RuntimeError("CustomVoice model not loaded")
        
        model = self.models["custom_voice"]
        
        try:
            # Build kwargs to handle None values
            kwargs = {
                'text': text,
                'speaker': speaker,
                **generation_params
            }
            if language is not None:
                kwargs['language'] = language
            if instruct is not None:
                kwargs['instruct'] = instruct
                
            wavs, sample_rate = model.generate_custom_voice(**kwargs)
            return wavs, sample_rate
        except Exception as e:
            logger.error(f"Error generating custom voice: {str(e)}")
            raise
    
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        language: Optional[str],
        instruct: str,
        **generation_params
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech using VoiceDesign model
        
        Returns:
            Tuple of (list of audio arrays, sample_rate)
        """
        if self.models["voice_design"] is None:
            raise RuntimeError("VoiceDesign model not loaded")
        
        model = self.models["voice_design"]
        
        try:
            kwargs = {
                'text': text,
                'instruct': instruct,
                **generation_params
            }
            if language is not None:
                kwargs['language'] = language
                
            wavs, sample_rate = model.generate_voice_design(**kwargs)
            return wavs, sample_rate
        except Exception as e:
            logger.error(f"Error generating voice design: {str(e)}")
            raise
    
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Optional[str],
        ref_audio: Union[str, Any],
        ref_text: Optional[str] = None,
        x_vector_only: bool = False,
        **generation_params
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech using Voice Clone model
        
        Returns:
            Tuple of (list of audio arrays, sample_rate)
        """
        if self.models["base_clone"] is None:
            raise RuntimeError("Base/Clone model not loaded")
        
        model = self.models["base_clone"]
        
        try:
            kwargs = {
                'text': text,
                'ref_audio': ref_audio,
                'x_vector_only_mode': x_vector_only,
                **generation_params
            }
            if language is not None:
                kwargs['language'] = language
            if ref_text is not None:
                kwargs['ref_text'] = ref_text
                
            wavs, sample_rate = model.generate_voice_clone(**kwargs)
            return wavs, sample_rate
        except Exception as e:
            logger.error(f"Error generating voice clone: {str(e)}")
            raise
    
    def create_voice_prompt(
        self,
        ref_audio: Union[str, Any],
        ref_text: str,
        x_vector_only: bool = False
    ) -> Any:
        """
        Create a reusable voice prompt tensor
        
        Returns:
            Prompt tensor (caller handles saving)
        """
        if self.models["base_clone"] is None:
            raise RuntimeError("Base/Clone model not loaded")
        
        model = self.models["base_clone"]
        
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only
            )
            return prompt
            
        except Exception as e:
            logger.error(f"Error creating voice prompt: {str(e)}")
            raise
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return [
            name for name, model in self.models.items() 
            if model is not None
        ]
