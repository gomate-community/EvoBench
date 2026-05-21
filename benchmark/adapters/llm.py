from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from benchmark.core.config import settings


class LLMAdapter(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        raw = await self.complete(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        return self._parse_json_object(raw)

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return {}
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}


class MockLLMAdapter(LLMAdapter):
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "这是一个模拟模型输出。请在生产环境替换为真实 LLM 适配器。"


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model: str
    api_key: str
    api_base: str
    timeout_seconds: float = 45.0
    default_temperature: float = 0.2
    default_max_tokens: int = 2048
    request_retries: int = 1

    @property
    def chat_completions_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/chat/completions"


class OpenAICompatibleLLMAdapter(LLMAdapter):
    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.default_temperature if temperature is None else temperature,
            "max_tokens": self.config.default_max_tokens if max_tokens is None else max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(
            timeout=self.config.timeout_seconds,
            connect=min(10.0, self.config.timeout_seconds),
        )

        last_error: Exception | None = None
        for _ in range(max(1, self.config.request_retries + 1)):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.config.chat_completions_url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    return self._extract_text(data)
            except Exception as exc:  # pragma: no cover - runtime connectivity
                last_error = exc
        raise RuntimeError(f"LLM request failed for model={self.config.model}") from last_error

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(chunk for chunk in chunks if chunk)
        return str(content)


def build_llm_adapter(
    provider: str | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout_seconds: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    request_retries: int | None = None,
) -> LLMAdapter:
    resolved_provider = (provider or settings.llm_provider).lower()
    if resolved_provider == "mock":
        return MockLLMAdapter()
    if resolved_provider in {"openai", "openai_compatible", "local_openai"}:
        return OpenAICompatibleLLMAdapter(
            OpenAICompatibleConfig(
                model=model or settings.llm_model,
                api_key=api_key or settings.llm_api_key,
                api_base=api_base or settings.llm_api_base,
                timeout_seconds=timeout_seconds or settings.llm_timeout_seconds,
                default_temperature=settings.llm_temperature if temperature is None else temperature,
                default_max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
                request_retries=settings.llm_request_retries if request_retries is None else request_retries,
            )
        )
    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")
