"""Minimal client for any OpenAI-compatible chat completions API.

Mira Network, Groq, Gemini and OpenRouter all speak the same /chat/completions
protocol, so one client covers them. Configure with environment variables (or .env):

    PINCH_LLM_PROVIDER = mira | groq | gemini | openrouter   (sets base URL + default model)
    PINCH_LLM_API_KEY  = your key
    PINCH_LLM_MODEL    = optional, overrides the provider's default model
    PINCH_LLM_BASE_URL = optional, for any other OpenAI-compatible endpoint
"""
import os
from dataclasses import dataclass

import httpx

PROVIDERS = {
    # name: (base URL, default model)
    "mira": ("https://apis.mira.network/v1", "llama-3.3-70b-instruct"),
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free"),
}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMClient:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0
    transport: httpx.BaseTransport | None = None  # injected in tests

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        """Build a client from the environment, or None if no API key is configured."""
        key = os.environ.get("PINCH_LLM_API_KEY")
        if not key:
            return None
        provider = os.environ.get("PINCH_LLM_PROVIDER", "mira").lower()
        if provider not in PROVIDERS and not os.environ.get("PINCH_LLM_BASE_URL"):
            raise LLMError(f"unknown provider {provider!r}; pick one of {sorted(PROVIDERS)} "
                           "or set PINCH_LLM_BASE_URL")
        base, model = PROVIDERS.get(provider, ("", ""))
        return cls(base_url=os.environ.get("PINCH_LLM_BASE_URL", base).rstrip("/"),
                   api_key=key,
                   model=os.environ.get("PINCH_LLM_MODEL", model))

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1200) -> str:
        body = {"model": self.model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
                r = http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            raise LLMError(f"request to {self.base_url} failed: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"{self.base_url} returned {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"unexpected response shape: {r.text[:300]}") from e
