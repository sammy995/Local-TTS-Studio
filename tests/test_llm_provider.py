"""Tests for the LLM provider boundary.

We can't reach a real Ollama server here, but we can verify the error path: a
clear ``LLMUnavailableError`` when the server is unreachable.
"""
import pytest

from infra.llm_provider import OllamaProvider, LLMUnavailableError


def test_ollama_provider_raises_when_server_unreachable():
    # Port 1 has nothing listening -> connection refused.
    provider = OllamaProvider(base_url="http://127.0.0.1:1", timeout=2.0)
    with pytest.raises(LLMUnavailableError):
        provider.complete("hello", system="be brief")
