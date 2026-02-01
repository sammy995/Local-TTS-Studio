"""
Configuration Loader (Runtime Concern)
Loads and merges YAML + environment variables
Directory creation removed - handled by runtime
"""

import yaml
import os
import torch
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file and environment variables
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing merged configuration
    """
    # Load YAML config
    config_file = Path(config_path)
    if not config_file.exists():
        # Use relative path from project root
        config_file = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if os.getenv('APP_HOST'):
        config['app']['host'] = os.getenv('APP_HOST')
    if os.getenv('APP_PORT'):
        port_str = os.getenv('APP_PORT')
        if port_str:
            config['app']['port'] = int(port_str)
    if os.getenv('MODEL_CACHE_DIR'):
        config['models']['cache_dir'] = os.getenv('MODEL_CACHE_DIR')
    if os.getenv('DEFAULT_MODEL_SIZE'):
        config['models']['default_size'] = os.getenv('DEFAULT_MODEL_SIZE')
    
    return config


def get_device_config() -> Dict[str, Any]:
    """
    Determine optimal device configuration (GPU/CPU)
    
    Returns:
        Dictionary with device settings
    """
    use_gpu = os.getenv('USE_GPU', 'auto')
    
    if use_gpu == 'auto':
        gpu_available = torch.cuda.is_available()
    else:
        gpu_available = use_gpu.lower() == 'true' and torch.cuda.is_available()
    
    if gpu_available:
        device_map = os.getenv('DEVICE_MAP', 'cuda:0')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device_map = 'cpu'
        dtype = torch.float32
    
    # Check if flash-attn is available
    flash_attn_available = False
    if gpu_available:
        try:
            import flash_attn
            flash_attn_available = True
        except ImportError:
            flash_attn_available = False
    
    return {
        'gpu_available': gpu_available,
        'device_map': device_map,
        'dtype': dtype,
        'use_flash_attn': flash_attn_available
    }


def create_directories(config: Dict[str, Any]):
    """
    Create necessary directories (runtime initialization)
    
    Args:
        config: Configuration dictionary
    """
    # Model cache directory
    model_cache_dir = Path(config['models']['cache_dir'])
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Output directories (for LocalStorage)
    outputs_dir = Path("./outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = Path("./temp_audio")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    voice_prompts_dir = Path("./voice_prompts")
    voice_prompts_dir.mkdir(parents=True, exist_ok=True)
