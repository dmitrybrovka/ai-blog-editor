from __future__ import annotations

import re
from dataclasses import dataclass

from markdown_it import MarkdownIt

from aiblog.obsidian import normalize_obsidian_markdown


_md = MarkdownIt("commonmark")


def md_to_text(md: str) -> str:
    # Normalize Obsidian syntax then render tokens and collect plain text.
    md = normalize_obsidian_markdown(md)
    tokens = _md.parse(md)
    parts: list[str] = []
    for t in tokens:
        if t.type == "inline" and t.children:
            for c in t.children:
                if c.type == "text":
                    parts.append(c.content)
                elif c.type == "code_inline":
                    parts.append(c.content)
        elif t.type == "code_block":
            parts.append(t.content)
        elif t.type == "fence":
            parts.append(t.content)
    text = "\n".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


@dataclass(frozen=True)
class Chunk:
    text: str
    idx: int
    start: int
    end: int


def chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[Chunk]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be < chunk_chars")

    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(text):
        start = i
        end = min(len(text), i + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, idx=idx, start=start, end=end))
            idx += 1
        if end >= len(text):
            break
        i = max(0, end - overlap_chars)
    return chunks

