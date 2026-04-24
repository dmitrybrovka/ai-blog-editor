from __future__ import annotations

from typing import Any
from typing import Optional

import httpx


def _ollama_hint(base_url: str) -> str:
    return (
        "Ollama is not reachable.\n"
        f"- Expected URL: {base_url}\n"
        "- Start Ollama and ensure it listens on that address.\n"
        "- Install models, e.g.: `ollama pull qwen2.5:7b-instruct` and `ollama pull nomic-embed-text`.\n"
    )


class OllamaClient:
    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def embeddings(self, model: str, prompt: str) -> list[float]:
        try:
            resp = self._client.post("/api/embeddings", json={"model": model, "prompt": prompt})
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(_ollama_hint(str(self._client.base_url))) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama embeddings error: {e.response.text}") from e
        data = resp.json()
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise RuntimeError(f"Unexpected embeddings response: {data!r}")
        return [float(x) for x in emb]

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_ctx: Optional[int] = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "top_p": top_p},
        }
        if num_ctx is not None:
            payload["options"]["num_ctx"] = num_ctx
        try:
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.RequestError as e:
            raise RuntimeError(_ollama_hint(str(self._client.base_url))) from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama chat error: {e.response.text}") from e
        data = resp.json()
        msg = (data.get("message") or {}).get("content")
        if not isinstance(msg, str):
            raise RuntimeError(f"Unexpected chat response: {data!r}")
        return msg

