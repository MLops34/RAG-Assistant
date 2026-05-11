"""Provider-agnostic OpenAI-compatible LLM configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMProviderConfig:
    """Resolved credentials and base URL for OpenAI-compatible APIs."""

    api_key: str
    base_url: Optional[str] = None
    provider: str = "openai"


def get_llm_provider_config() -> LLMProviderConfig:
    """
    Resolve provider config from environment variables.

    Supported setups:
    - OpenAI:
        OPENAI_API_KEY=...
    - Groq:
        GROQ_API_KEY=...
    - OpenRouter:
        OPENROUTER_API_KEY=...
        OPENAI_BASE_URL=https://openrouter.ai/api/v1   (recommended)
    """
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if groq_key:
        return LLMProviderConfig(
            api_key=groq_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
            provider="groq",
        )
    if openai_key:
        return LLMProviderConfig(
            api_key=openai_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
            provider="openai",
        )
    if openrouter_key:
        return LLMProviderConfig(
            api_key=openrouter_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            provider="openrouter",
        )
    raise ValueError(
        "No LLM API key found. Set GROQ_API_KEY (Groq), OPENAI_API_KEY (OpenAI), or "
        "OPENROUTER_API_KEY (OpenRouter)."
    )
