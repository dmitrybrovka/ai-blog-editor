from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiblog.config import AppConfig
from aiblog.ollama_client import OllamaClient
from aiblog.prompts import build_prompts
from aiblog.store import VectorStore


@dataclass(frozen=True)
class RagContext:
    text: str


def _format_context(hits: list[dict], *, max_chars: int = 8000) -> str:
    parts: list[str] = []
    total = 0
    for h in hits:
        meta = h.get("meta") or {}
        title = meta.get("title") or meta.get("source_name") or "source"
        snippet = (h.get("text") or "").strip()
        block = f"[{title}]\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


class RagWriter:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.ollama = OllamaClient(cfg.ollama.base_url, timeout_s=cfg.ollama.request_timeout_s)
        self.store = VectorStore(
            persist_dir=Path(cfg.rag.persist_dir),
            collection=cfg.rag.collection,
            ollama=self.ollama,
            embed_model=cfg.ollama.embed_model,
        )
        self.prompts = build_prompts(cfg.style)

    def retrieve(self, query: str) -> RagContext:
        hits = self.store.query(query, top_k=self.cfg.rag.top_k)
        return RagContext(text=_format_context(hits))

    def outline(self, *, topic: str, notes: str) -> str:
        ctx = self.retrieve(topic + "\n" + notes).text
        prompt = self.prompts.outline.format(topic=topic, notes=notes or "-", context=ctx or "-")
        return self.ollama.chat(
            self.cfg.ollama.chat_model,
            messages=[
                {"role": "system", "content": self.prompts.system},
                {"role": "user", "content": prompt},
            ],
        )

    def draft(self, *, topic: str, notes: str) -> str:
        ctx = self.retrieve(topic + "\n" + notes).text
        prompt = self.prompts.draft.format(topic=topic, notes=notes or "-", context=ctx or "-")
        return self.ollama.chat(
            self.cfg.ollama.chat_model,
            messages=[
                {"role": "system", "content": self.prompts.system},
                {"role": "user", "content": prompt},
            ],
        )

    def rewrite(self, *, input_text: str, topic_hint: str = "") -> str:
        ctx = self.retrieve(topic_hint or input_text[:400]).text
        prompt = self.prompts.rewrite.format(input_text=input_text, context=ctx or "-")
        return self.ollama.chat(
            self.cfg.ollama.chat_model,
            messages=[
                {"role": "system", "content": self.prompts.system},
                {"role": "user", "content": prompt},
            ],
        )

    def headline(self, *, topic: str) -> str:
        ctx = self.retrieve(topic).text
        prompt = self.prompts.headline.format(topic=topic, context=ctx or "-")
        return self.ollama.chat(
            self.cfg.ollama.chat_model,
            messages=[
                {"role": "system", "content": self.prompts.system},
                {"role": "user", "content": prompt},
            ],
        )

