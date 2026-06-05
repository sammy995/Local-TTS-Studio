"""TTS provider abstraction.

A lightweight, dependency-free layer that lets the app support multiple local
TTS engines (Qwen3-TTS, Piper, Kokoro, ...) behind one interface. Engines differ
in what they can do — Qwen can clone and design voices, Piper only has presets —
so each provider declares its ``capabilities`` and the registry refuses
unsupported modes with a clear error instead of crashing mid-generation.

This module imports nothing heavy on purpose, so the registry/dispatch logic is
unit-testable without torch or any model installed. Concrete providers (which do
import torch/onnx/etc.) are registered by the runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

# Generation modes
MODE_CUSTOM_VOICE = "custom_voice"
MODE_VOICE_DESIGN = "voice_design"
MODE_VOICE_CLONE = "voice_clone"
ALL_MODES: Tuple[str, ...] = (MODE_CUSTOM_VOICE, MODE_VOICE_DESIGN, MODE_VOICE_CLONE)


class ProviderError(Exception):
    """Base class for provider-related errors."""


class ProviderNotFoundError(ProviderError):
    """Raised when a provider id is not registered."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is registered but its backend isn't installed/loaded."""


class UnsupportedModeError(ProviderError):
    """Raised when a provider is asked for a generation mode it doesn't support."""


@dataclass(frozen=True)
class ProviderInfo:
    """Static, serializable metadata describing a TTS engine."""

    id: str
    name: str
    capabilities: Tuple[str, ...]
    description: str = ""
    requires: Tuple[str, ...] = ()      # pip extras / system deps needed to enable it
    available: bool = True              # is the backend importable/loaded right now?


@runtime_checkable
class TTSProvider(Protocol):
    """Structural interface a concrete engine adapter implements.

    Adapters only need to implement the methods for modes they advertise in
    ``info.capabilities``; the registry guards the rest.
    """

    info: ProviderInfo

    def generate_custom_voice(self, text, speaker, language=None, instruct=None, **params) -> Tuple[Any, int]: ...

    def generate_voice_design(self, text, language, instruct, **params) -> Tuple[Any, int]: ...

    def generate_voice_clone(self, text, language, ref_audio, ref_text=None, x_vector_only=False, **params) -> Tuple[Any, int]: ...


@dataclass
class _Entry:
    info: ProviderInfo
    provider: Optional[Any]


class TTSProviderRegistry:
    """Holds the set of known TTS engines and the current default selection."""

    def __init__(self) -> None:
        self._entries: Dict[str, _Entry] = {}
        self.default_id: Optional[str] = None

    def register(self, info: ProviderInfo, provider: Optional[Any], default: bool = False) -> None:
        """Register a provider. ``provider`` may be ``None`` when the backend
        isn't installed — it will then be listed but raise on use."""
        self._entries[info.id] = _Entry(info=info, provider=provider)
        if default:
            self.default_id = info.id

    def list(self) -> List[ProviderInfo]:
        """All known providers (available or not), in registration order."""
        return [entry.info for entry in self._entries.values()]

    def info(self, provider_id: str) -> ProviderInfo:
        entry = self._entries.get(provider_id)
        if entry is None:
            raise ProviderNotFoundError(f"No TTS provider registered with id '{provider_id}'")
        return entry.info

    def get(self, provider_id: str) -> Any:
        """Return the live provider instance, or raise if missing/unavailable."""
        entry = self._entries.get(provider_id)
        if entry is None:
            raise ProviderNotFoundError(f"No TTS provider registered with id '{provider_id}'")
        if entry.provider is None:
            raise ProviderUnavailableError(
                f"TTS provider '{provider_id}' is not available. "
                f"Install its requirements ({', '.join(entry.info.requires) or 'see docs'}) to enable it."
            )
        return entry.provider

    def get_default(self) -> Any:
        if self.default_id is None:
            raise ProviderNotFoundError("No default TTS provider has been set")
        return self.get(self.default_id)

    def supports(self, provider_id: str, mode: str) -> bool:
        return mode in self.info(provider_id).capabilities

    def ensure_supported(self, provider_id: str, mode: str) -> None:
        if not self.supports(provider_id, mode):
            raise UnsupportedModeError(
                f"TTS provider '{provider_id}' does not support mode '{mode}'"
            )


def dispatch_generate(registry: TTSProviderRegistry, provider_id: str, mode: str, **kwargs) -> Tuple[Any, int]:
    """Validate the mode, then route to the right method on the selected provider.

    Capability is checked *before* the provider is invoked, so an unsupported
    request fails fast and never touches the model.
    """
    registry.ensure_supported(provider_id, mode)
    provider = registry.get(provider_id)
    methods = {
        MODE_CUSTOM_VOICE: provider.generate_custom_voice,
        MODE_VOICE_DESIGN: provider.generate_voice_design,
        MODE_VOICE_CLONE: provider.generate_voice_clone,
    }
    return methods[mode](**kwargs)
