"""
Model Manager
Handles loading, caching, and managing Qwen3-TTS models
"""

import torch
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
from qwen_tts import Qwen3TTSModel

# Handle both relative and absolute imports
try:
    from .config_loader import get_device_config
except ImportError:
    from config_loader import get_device_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages Qwen3-TTS model loading and inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device_config = get_device_config()
        self.models: Dict[str, Optional[Qwen3TTSModel]] = {
            "custom_voice": None,
            "voice_design": None,
            "base_clone": None
        }
        self.current_model_sizes: Dict[str, str] = {}
        
        # Local models directory (relative to project root)
        self.local_models_dir = Path(__file__).parent.parent / "models"
        
        logger.info(f"Device config: {self.device_config}")
        logger.info(f"Local models dir: {self.local_models_dir}")
    
    def get_model_path(self, model_type: str, model_size: str = "1.7B") -> str:
        """Get the model path - uses local path if available, otherwise HuggingFace"""
        model_names = {
            "custom_voice": f"Qwen3-TTS-12Hz-{model_size}-CustomVoice",
            "voice_design": f"Qwen3-TTS-12Hz-{model_size}-VoiceDesign",
            "base_clone": f"Qwen3-TTS-12Hz-{model_size}-Base"
        }
        
        model_name = model_names.get(model_type)
        if not model_name:
            return None
            
        # Check if local model exists
        local_path = self.local_models_dir / model_name
        if local_path.exists() and (local_path / "config.json").exists():
            logger.info(f"Using local model: {local_path}")
            return str(local_path)
        
        # Fall back to HuggingFace
        hf_path = f"Qwen/{model_name}"
        logger.info(f"Local model not found, using HuggingFace: {hf_path}")
        return hf_path
    
    def load_model(self, model_type: str, model_size: str = "1.7B", retry: bool = True):
        """
        Load a specific model type
        
        Args:
            model_type: One of 'custom_voice', 'voice_design', 'base_clone'
            model_size: Model size ('1.7B' or '0.6B')
            retry: Whether to retry on failure after clearing cache
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
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        model_path = self.get_model_path(model_type, model_size)
        logger.info(f"Loading model: {model_path}")
        
        try:
            # Load model with appropriate device settings
            attn_impl = "flash_attention_2" if self.device_config['use_flash_attn'] else None
            
            is_local = model_path.startswith(str(self.local_models_dir)) or not model_path.startswith("Qwen/")
            if is_local:
                logger.info(f"Loading model from local directory...")
            else:
                logger.info(f"Downloading model from HuggingFace (this may take a few minutes)...")
            
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=self.device_config['device_map'],
                torch_dtype=self.device_config['dtype'],
                attn_implementation=attn_impl,
                trust_remote_code=True
            )
            
            self.models[model_type] = model
            self.current_model_sizes[model_type] = model_size
            logger.info(f"Model {model_type} loaded successfully")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading model {model_type}: {error_msg}")
            
            # If it's a cache/download issue and we haven't retried yet, try clearing cache
            if retry and ("preprocessor_config.json" in error_msg or "feature extractor" in error_msg.lower()):
                logger.warning(f"Detected corrupted cache. Clearing and retrying...")
                
                # Clear the specific model cache
                import shutil
                cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_path.replace('/', '--')}"
                if cache_path.exists():
                    logger.info(f"Removing corrupted cache at {cache_path}")
                    shutil.rmtree(cache_path, ignore_errors=True)
                
                # Retry once
                logger.info("Retrying model download...")
                return self.load_model(model_type, model_size, retry=False)
            
            raise
    
    def ensure_model_loaded(self, model_type: str, model_size: str = None):
        """Ensure a model is loaded, loading it if necessary"""
        if model_size is None:
            model_size = self.config['models']['default_size']
        
        if self.models[model_type] is None:
            self.load_model(model_type, model_size)
    
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        **generation_params
    ) -> tuple:
        """Generate speech using CustomVoice model"""
        if self.models["custom_voice"] is None:
            raise RuntimeError("CustomVoice model not loaded")
        
        model = self.models["custom_voice"]
        
        # Merge with default generation params
        params = {**self.config['generation']['default_params'], **generation_params}
        
        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language if language != "Auto" else None,
                instruct=instruct,
                **params
            )
            return wavs, sample_rate
        except Exception as e:
            logger.error(f"Error generating custom voice: {str(e)}")
            raise
    
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        language: str,
        instruct: str,
        **generation_params
    ) -> tuple:
        """Generate speech using VoiceDesign model"""
        if self.models["voice_design"] is None:
            raise RuntimeError("VoiceDesign model not loaded")
        
        model = self.models["voice_design"]
        params = {**self.config['generation']['default_params'], **generation_params}
        
        try:
            wavs, sample_rate = model.generate_voice_design(
                text=text,
                language=language if language != "Auto" else None,
                instruct=instruct,
                **params
            )
            return wavs, sample_rate
        except Exception as e:
            logger.error(f"Error generating voice design: {str(e)}")
            raise
    
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: str,
        ref_audio: Union[str, Any],
        ref_text: Optional[str] = None,
        x_vector_only: bool = False,
        **generation_params
    ) -> tuple:
        """Generate speech using Voice Clone model"""
        if self.models["base_clone"] is None:
            raise RuntimeError("Base/Clone model not loaded")
        
        model = self.models["base_clone"]
        params = {**self.config['generation']['default_params'], **generation_params}
        
        try:
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=language if language != "Auto" else None,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
                **params
            )
            return wavs, sample_rate
        except Exception as e:
            import traceback
            logger.error(f"Error generating voice clone: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def create_voice_prompt(
        self,
        ref_audio: Union[str, Any],
        ref_text: str,
        x_vector_only: bool = False,
        save_path: Optional[str] = None
    ) -> str:
        """Create a reusable voice prompt"""
        if self.models["base_clone"] is None:
            raise RuntimeError("Base/Clone model not loaded")
        
        model = self.models["base_clone"]
        
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only
            )
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(prompt, save_path)
                logger.info(f"Voice prompt saved to {save_path}")
                return str(save_path)
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error creating voice prompt: {str(e)}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system and model status information"""
        loaded_models = [
            name for name, model in self.models.items() 
            if model is not None
        ]
        
        return {
            "status": "ready",
            "models_loaded": loaded_models,
            "gpu_available": self.device_config['gpu_available'],
            "device": self.device_config['device_map']
        }
