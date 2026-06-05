"""Services package initialization.

Lazy on purpose (PEP 562): importing a light submodule such as
``services.podcast_models`` or ``services.script_generation_service`` must not
drag in torch via ``tts_service``/``podcast_service``. The package-level names
below still resolve on demand.
"""
import importlib

_EXPORTS = {
    "TTSService": "tts_service",
    "Storage": "tts_service",
    "PodcastService": "podcast_service",
    "Speaker": "podcast_models",
    "Segment": "podcast_models",
    "PodcastProject": "podcast_models",
    "RenderJob": "podcast_models",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)
