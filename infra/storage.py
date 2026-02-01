"""
Storage Infrastructure - Polymorphic I/O Adapters
Protocol-based (structural typing) for flexibility
LocalStorage for filesystem, CloudStorage for S3 (future)
"""

import torch
from pathlib import Path
from typing import Protocol, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Storage(Protocol):
    """
    Storage protocol - duck typing for storage implementations
    Any class with these methods can be used as Storage
    """
    
    def save_audio(self, audio_bytes: bytes, filename: str) -> str:
        """
        Save audio bytes to storage
        
        Args:
            audio_bytes: WAV audio data as bytes
            filename: Filename to save as
            
        Returns:
            Path or URL to saved file
        """
        ...
    
    def save_prompt(self, tensor: Any, name: str) -> str:
        """
        Save voice prompt tensor to storage
        
        Args:
            tensor: Prompt tensor from model
            name: Prompt name
            
        Returns:
            Path or URL to saved prompt
        """
        ...
    
    def load_prompt(self, name: str) -> Any:
        """
        Load voice prompt tensor from storage
        
        Args:
            name: Prompt name
            
        Returns:
            Prompt tensor
        """
        ...


class LocalStorage:
    """Local filesystem storage implementation"""
    
    def __init__(
        self,
        output_dir: Path,
        temp_dir: Path,
        prompts_dir: Path
    ):
        """
        Initialize LocalStorage
        
        Args:
            output_dir: Directory for audio outputs
            temp_dir: Directory for temporary files
            prompts_dir: Directory for voice prompts
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.prompts_dir = Path(prompts_dir)
        
        # Ensure directories exist (runtime responsibility moved here)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LocalStorage initialized: output={output_dir}, temp={temp_dir}, prompts={prompts_dir}")
    
    def save_audio(self, audio_bytes: bytes, filename: str) -> str:
        """
        Save audio bytes to local filesystem
        
        Args:
            audio_bytes: WAV audio data
            filename: Output filename
            
        Returns:
            Absolute path to saved file
        """
        file_path = self.output_dir / filename
        file_path.write_bytes(audio_bytes)
        
        logger.info(f"Saved audio: {file_path} ({len(audio_bytes)} bytes)")
        
        return str(file_path)
    
    def save_prompt(self, tensor: Any, name: str) -> str:
        """
        Save voice prompt tensor to local filesystem
        
        Args:
            tensor: Prompt tensor
            name: Prompt name (without extension)
            
        Returns:
            Absolute path to saved prompt
        """
        # Ensure .pt extension
        if not name.endswith('.pt'):
            name = f"{name}.pt"
        
        file_path = self.prompts_dir / name
        torch.save(tensor, file_path)
        
        logger.info(f"Saved prompt: {file_path}")
        
        return str(file_path)
    
    def load_prompt(self, name: str) -> Any:
        """
        Load voice prompt tensor from local filesystem
        
        Args:
            name: Prompt name (with or without .pt extension)
            
        Returns:
            Prompt tensor
        """
        # Ensure .pt extension
        if not name.endswith('.pt'):
            name = f"{name}.pt"
        
        file_path = self.prompts_dir / name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt not found: {file_path}")
        
        tensor = torch.load(file_path)
        
        logger.info(f"Loaded prompt: {file_path}")
        
        return tensor
    
    def get_temp_path(self, filename: str) -> Path:
        """
        Get path for temporary file
        
        Args:
            filename: Temporary filename
            
        Returns:
            Path in temp directory
        """
        return self.temp_dir / filename
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up old temporary files
        
        Args:
            max_age_hours: Maximum age in hours before deletion
        """
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        deleted_count = 0
        for file_path in self.temp_dir.glob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} temp files older than {max_age_hours}h")


# Future: CloudStorage for S3
class CloudStorage:
    """
    Cloud storage implementation (S3, Azure Blob, etc.)
    
    Future implementation will match Storage protocol:
    - save_audio() → upload to S3, return URL
    - save_prompt() → upload to S3, return URL
    - load_prompt() → download from S3, return tensor
    """
    
    def __init__(self, bucket_name: str, region: str):
        raise NotImplementedError("CloudStorage not yet implemented")
