from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import frontmatter


_WIKILINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")
_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
_OBSIDIAN_COMMENT_RE = re.compile(r"%%[\s\S]*?%%")


def normalize_obsidian_markdown(md: str) -> str:
    """
    Turn common Obsidian constructs into plain Markdown-ish text before parsing:
    - embeds: ![[file]] -> (removed)
    - wikilinks: [[Page]] -> Page, [[Page|Alias]] -> Alias
    - comments: %% ... %% -> (removed)
    - callouts/quotes: strip leading '>' to keep content searchable
    """
    md = _OBSIDIAN_COMMENT_RE.sub("", md)
    md = _EMBED_RE.sub("", md)

    def _wikilink_sub(m: re.Match[str]) -> str:
        page = (m.group(1) or "").strip()
        alias = (m.group(2) or "").strip()
        return alias or page

    md = _WIKILINK_RE.sub(_wikilink_sub, md)

    # Keep quoted/callout content but remove quoting markers for indexing.
    md = re.sub(r"(?m)^\s*>\s?", "", md)

    # Common callout header line: [!NOTE] etc. Keep label as plain text.
    md = re.sub(r"(?mi)^\s*\[\!([A-Z0-9_-]+)\]\s*", r"\1: ", md)

    return md.strip()


def _coerce_tags(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        # allow "tag1, tag2" or "tag1 tag2"
        raw = re.split(r"[,\s]+", v.strip())
        return [t.lstrip("#") for t in raw if t.strip()]
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip().lstrip("#"))
        return out
    return []


def _coerce_str(v: Any) -> Optional[str]:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _pick_title(meta: Dict[str, Any], fallback: str) -> str:
    for key in ("title", "name", "heading"):
        t = _coerce_str(meta.get(key))
        if t:
            return t
    return fallback


@dataclass(frozen=True)
class ObsidianDoc:
    path: Path
    title: str
    tags: List[str]
    status: Optional[str]
    date: Optional[str]
    summary: Optional[str]
    content_md: str
    metadata: Dict[str, Any]


def parse_obsidian_markdown(path: Path) -> ObsidianDoc:
    raw = path.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    meta: Dict[str, Any] = dict(post.metadata or {})

    fallback_title = path.stem
    title = _pick_title(meta, fallback=fallback_title)
    tags = _coerce_tags(meta.get("tags"))

    status = _coerce_str(meta.get("status"))
    date = _coerce_str(meta.get("date") or meta.get("created") or meta.get("published"))
    summary = _coerce_str(meta.get("summary") or meta.get("description"))

    md = (post.content or "").strip()

    meta.update(
        {
            "title": title,
            "tags": tags,
            "status": status,
            "date": date,
            "summary": summary,
            "source_path": str(path),
            "source_name": path.name,
        }
    )

    return ObsidianDoc(
        path=path,
        title=title,
        tags=tags,
        status=status,
        date=date,
        summary=summary,
        content_md=md,
        metadata=meta,
    )

