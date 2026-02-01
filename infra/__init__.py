"""Infrastructure package initialization"""

from .storage import Storage, LocalStorage, CloudStorage

__all__ = ['Storage', 'LocalStorage', 'CloudStorage']
