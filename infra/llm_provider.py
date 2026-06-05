"""LLM provider abstraction for podcast script generation.

Local-first: ``OllamaProvider`` talks to a local Ollama server. The same
``LLMProvider`` interface leaves room for cloud providers (OpenAI/Anthropic)
to be added later behind user-supplied keys, without touching callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


class LLMError(Exception):
    """Base class for LLM provider errors."""


class LLMUnavailableError(LLMError):
    """Raised when the LLM backend can't be reached or used."""


@runtime_checkable
class LLMProvider(Protocol):
    """Any object that can turn a prompt into text."""

    def complete(self, prompt: str, system: Optional[str] = None, **opts: Any) -> str: ...


@dataclass
class OllamaProvider:
    """Local LLM via an Ollama server (https://ollama.com).

    Requires ``ollama serve`` to be running with ``model`` pulled. This is an
    I/O boundary — only its error path is unit-tested.
    """

    model: str = "llama3.2"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0

    def complete(self, prompt: str, system: Optional[str] = None, **opts: Any) -> str:
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {"model": self.model, "messages": messages, "stream": False}
        if opts:
            payload["options"] = opts

        try:
            resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.base_url}. Is 'ollama serve' running "
                f"and the model '{self.model}' pulled? ({exc})"
            ) from exc

        data = resp.json()
        return (data.get("message") or {}).get("content", "")
