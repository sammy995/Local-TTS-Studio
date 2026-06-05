"""Infrastructure package initialization.

Lazy on purpose: importing a lightweight submodule (e.g. ``infra.tts_provider``)
must not pull in heavy dependencies like torch that ``infra.storage`` needs.
``from infra import LocalStorage`` still works via PEP 562.
"""

__all__ = ["Storage", "LocalStorage", "CloudStorage"]


def __getattr__(name):
    if name in __all__:
        from . import storage
        return getattr(storage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
