"""
Music Service - Unified Search Across Free Music APIs
Proxies search/import from Jamendo, Openverse, and Freesound

All methods are stateless. Storage is injected per-call.
"""

import logging
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MusicSearchResult:
    """
    Unified search result across all music sources
    
    Normalized from Jamendo/Openverse/Freesound into one shape
    so the UI doesn't care which source returned it.
    """
    id: str
    title: str
    artist: str
    duration_ms: int
    preview_url: str          # Direct URL to stream/preview
    download_url: Optional[str]  # Direct URL to download (if available)
    license: str
    source: str               # jamendo | openverse | freesound
    source_url: str            # Link to original page
    tags: List[str]
    artwork_url: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class MusicService:
    """
    Unified music search and import service
    
    Supports:
    - Jamendo: 500K+ CC-licensed music tracks
    - Openverse: Aggregated CC audio (Jamendo + others)
    - Freesound: 600K+ sounds (SFX-heavy, some music)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with API keys from config.yaml
        
        Args:
            config: Dict with keys:
                jamendo.client_id: str
                openverse.token: str (optional, anonymous works)
                freesound.token: str
        """
        self.jamendo_client_id = config.get('jamendo', {}).get('client_id', '')
        self.openverse_token = config.get('openverse', {}).get('token', '')
        self.freesound_token = config.get('freesound', {}).get('token', '')
        
        logger.info(f"MusicService initialized: jamendo={'✓' if self.jamendo_client_id else '✗'}, "
                     f"openverse={'✓' if self.openverse_token else 'anon'}, "
                     f"freesound={'✓' if self.freesound_token else '✗'}")
    
    async def search(
        self,
        query: str,
        source: str = "jamendo",
        limit: int = 20,
        instrumental_only: bool = True
    ) -> List[MusicSearchResult]:
        """
        Search for music across sources
        
        Args:
            query: Search query
            source: Which source to search (jamendo/openverse/freesound)
            limit: Max results
            instrumental_only: Filter for instrumental tracks (Jamendo only)
            
        Returns:
            List of normalized MusicSearchResult
        """
        if source == "jamendo":
            return await self._search_jamendo(query, limit, instrumental_only)
        elif source == "openverse":
            return await self._search_openverse(query, limit)
        elif source == "freesound":
            return await self._search_freesound(query, limit)
        else:
            raise ValueError(f"Unknown music source: {source}")
    
    async def _search_jamendo(
        self,
        query: str,
        limit: int = 20,
        instrumental_only: bool = True
    ) -> List[MusicSearchResult]:
        """Search Jamendo API for music tracks"""
        if not self.jamendo_client_id:
            raise ValueError(
                "Jamendo API key not configured. "
                "Get a free client_id at https://devportal.jamendo.com "
                "and add it to config.yaml → music_apis.jamendo.client_id"
            )
        
        params = {
            'client_id': self.jamendo_client_id,
            'format': 'json',
            'search': query,
            'limit': min(limit, 200),
            'include': 'musicinfo',
            'audioformat': 'mp32',
        }
        
        if instrumental_only:
            params['vocalinstrumental'] = 'instrumental'
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get('https://api.jamendo.com/v3.0/tracks', params=params)
                resp.raise_for_status()
                data = resp.json()
            
            results = []
            for track in data.get('results', []):
                results.append(MusicSearchResult(
                    id=f"jamendo_{track['id']}",
                    title=track.get('name', 'Untitled'),
                    artist=track.get('artist_name', 'Unknown'),
                    duration_ms=int(track.get('duration', 0)) * 1000,
                    preview_url=track.get('audio', ''),
                    download_url=track.get('audiodownload', '') if track.get('audiodownload_allowed') else None,
                    license=track.get('license_ccurl', 'CC'),
                    source='jamendo',
                    source_url=track.get('shareurl', ''),
                    tags=track.get('musicinfo', {}).get('tags', {}).get('genres', []),
                    artwork_url=track.get('image', None)
                ))
            
            logger.info(f"Jamendo search '{query}': {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Jamendo search error: {e}")
            return []
    
    async def _search_openverse(
        self,
        query: str,
        limit: int = 20
    ) -> List[MusicSearchResult]:
        """Search Openverse API for CC-licensed audio"""
        headers = {}
        if self.openverse_token:
            headers['Authorization'] = f'Bearer {self.openverse_token}'
        
        params = {
            'q': query,
            'page_size': min(limit, 50),
            'category': 'music',
        }
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    'https://api.openverse.org/v1/audio/',
                    params=params,
                    headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
            
            results = []
            for item in data.get('results', []):
                duration_ms = 0
                if item.get('duration'):
                    duration_ms = int(item['duration'])
                
                results.append(MusicSearchResult(
                    id=f"openverse_{item['id']}",
                    title=item.get('title', 'Untitled'),
                    artist=item.get('creator', 'Unknown'),
                    duration_ms=duration_ms,
                    preview_url=item.get('url', ''),
                    download_url=item.get('url', ''),
                    license=item.get('license', 'unknown'),
                    source='openverse',
                    source_url=item.get('foreign_landing_url', ''),
                    tags=[t.get('name', '') for t in item.get('tags', []) if isinstance(t, dict)],
                    artwork_url=item.get('thumbnail', None)
                ))
            
            logger.info(f"Openverse search '{query}': {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Openverse search error: {e}")
            return []
    
    async def _search_freesound(
        self,
        query: str,
        limit: int = 20
    ) -> List[MusicSearchResult]:
        """Search Freesound API for audio (previews free, originals need OAuth)"""
        if not self.freesound_token:
            raise ValueError(
                "Freesound API token not configured. "
                "Get a free token at https://freesound.org/apiv2/apply "
                "and add it to config.yaml → music_apis.freesound.token"
            )
        
        params = {
            'query': query,
            'page_size': min(limit, 150),
            'fields': 'id,name,username,duration,previews,license,tags,description,url',
            'token': self.freesound_token,
        }
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    'https://freesound.org/apiv2/search/',
                    params=params
                )
                resp.raise_for_status()
                data = resp.json()
            
            results = []
            for sound in data.get('results', []):
                previews = sound.get('previews', {})
                preview_url = previews.get('preview-hq-mp3', previews.get('preview-lq-mp3', ''))
                
                results.append(MusicSearchResult(
                    id=f"freesound_{sound['id']}",
                    title=sound.get('name', 'Untitled'),
                    artist=sound.get('username', 'Unknown'),
                    duration_ms=int(float(sound.get('duration', 0)) * 1000),
                    preview_url=preview_url,
                    download_url=None,  # Originals require OAuth2
                    license=sound.get('license', 'unknown'),
                    source='freesound',
                    source_url=sound.get('url', ''),
                    tags=sound.get('tags', []),
                ))
            
            logger.info(f"Freesound search '{query}': {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Freesound search error: {e}")
            return []
    
    async def download_track(
        self,
        url: str,
        timeout: float = 60.0
    ) -> bytes:
        """
        Download audio from URL
        
        Args:
            url: Direct URL to audio file
            timeout: Download timeout in seconds
            
        Returns:
            Audio file bytes
        """
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            
            content_length = len(resp.content)
            logger.info(f"Downloaded {content_length} bytes from {url[:80]}...")
            
            return resp.content
