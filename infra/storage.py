"""
Storage Infrastructure - Polymorphic I/O Adapters
Protocol-based (structural typing) for flexibility
LocalStorage for filesystem, CloudStorage for S3 (future)

v3: Added asset library management for music/SFX files
"""

import json
import torch
from pathlib import Path
from typing import Protocol, Any, List, Dict, Optional
import logging
import uuid
import time

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
        prompts_dir: Path,
        assets_dir: Optional[Path] = None
    ):
        """
        Initialize LocalStorage
        
        Args:
            output_dir: Directory for audio outputs
            temp_dir: Directory for temporary files
            prompts_dir: Directory for voice prompts
            assets_dir: Directory for music/SFX assets (v3)
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.prompts_dir = Path(prompts_dir)
        self.assets_dir = Path(assets_dir) if assets_dir else self.output_dir.parent / "assets"
        
        # Ensure directories exist (runtime responsibility moved here)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        (self.assets_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.assets_dir / "metadata").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LocalStorage initialized: output={output_dir}, temp={temp_dir}, prompts={prompts_dir}, assets={self.assets_dir}")
    
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
    
    # ===== v3: ASSET LIBRARY MANAGEMENT =====
    
    def save_asset(
        self,
        audio_bytes: bytes,
        metadata: Dict[str, Any],
        asset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save audio asset + metadata to library
        
        Args:
            audio_bytes: Audio file bytes
            metadata: Asset metadata dict (title, source, license, etc.)
            asset_id: Optional ID (auto-generated if None)
            
        Returns:
            Complete metadata dict with id and local_path
        """
        if asset_id is None:
            asset_id = f"asset_{uuid.uuid4().hex[:12]}"
        
        # Determine file extension from metadata or default to mp3
        ext = metadata.get('format', 'mp3')
        audio_filename = f"{asset_id}.{ext}"
        
        # Save audio file
        audio_path = self.assets_dir / "audio" / audio_filename
        audio_path.write_bytes(audio_bytes)
        
        # Build complete metadata
        metadata['id'] = asset_id
        metadata['local_path'] = str(audio_path)
        metadata['file_size_bytes'] = len(audio_bytes)
        metadata['saved_at'] = time.time()
        
        # Save metadata JSON
        meta_path = self.assets_dir / "metadata" / f"{asset_id}.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        
        logger.info(f"Saved asset: {asset_id} ({metadata.get('title', 'Untitled')}, {len(audio_bytes)} bytes)")
        
        return metadata
    
    def load_asset_audio(self, asset_id: str) -> bytes:
        """
        Load asset audio bytes from library
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Audio file bytes
        """
        # Find audio file (could be any extension)
        audio_dir = self.assets_dir / "audio"
        matches = list(audio_dir.glob(f"{asset_id}.*"))
        
        if not matches:
            raise FileNotFoundError(f"Asset audio not found: {asset_id}")
        
        return matches[0].read_bytes()
    
    def load_asset_metadata(self, asset_id: str) -> Dict[str, Any]:
        """
        Load asset metadata from library
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Metadata dict
        """
        meta_path = self.assets_dir / "metadata" / f"{asset_id}.json"
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Asset metadata not found: {asset_id}")
        
        return json.loads(meta_path.read_text())
    
    def list_assets(self) -> List[Dict[str, Any]]:
        """
        List all assets in library
        
        Returns:
            List of metadata dicts
        """
        assets = []
        meta_dir = self.assets_dir / "metadata"
        
        for meta_file in sorted(meta_dir.glob("*.json")):
            try:
                metadata = json.loads(meta_file.read_text())
                assets.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load asset metadata {meta_file}: {e}")
        
        return assets
    
    def delete_asset(self, asset_id: str) -> bool:
        """
        Delete asset from library (audio + metadata)
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            True if deleted, False if not found
        """
        deleted = False
        
        # Delete audio file
        audio_dir = self.assets_dir / "audio"
        for f in audio_dir.glob(f"{asset_id}.*"):
            f.unlink()
            deleted = True
        
        # Delete metadata
        meta_path = self.assets_dir / "metadata" / f"{asset_id}.json"
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        
        if deleted:
            logger.info(f"Deleted asset: {asset_id}")
        
        return deleted


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
