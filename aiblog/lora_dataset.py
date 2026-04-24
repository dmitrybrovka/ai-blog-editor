from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from aiblog.obsidian import ObsidianDoc, parse_obsidian_markdown
from aiblog.redact import default_redaction_rules, redact_text


_HEADING_RE = re.compile(r"(?m)^(#{1,4})\s+(.+?)\s*$")


@dataclass(frozen=True)
class LoraExample:
    instruction: str
    input: str
    output: str
    meta: Dict[str, Any]


def _extract_headings(md: str, *, limit: int = 12) -> List[str]:
    out: List[str] = []
    for _, title in _HEADING_RE.findall(md):
        t = title.strip()
        if t and t.lower() not in {"draft", "черновик"}:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _build_instruction(title: str, *, language: str = "ru") -> str:
    if language == "en":
        return f"Write a blog post in the author's style on the topic: {title}"
    return f"Напиши пост в стиле автора на тему: {title}"


def _build_input(
    *,
    title: str,
    tags: Sequence[str],
    summary: Optional[str],
    headings: Sequence[str],
    language: str = "ru",
) -> str:
    if language == "en":
        lines = [f"Topic: {title}"]
        if tags:
            lines.append("Tags: " + ", ".join(tags))
        if summary:
            lines.append("Notes: " + summary)
        if headings:
            lines.append("Suggested structure:")
            lines.extend([f"- {h}" for h in headings])
        return "\n".join(lines).strip()

    lines = [f"Тема: {title}"]
    if tags:
        lines.append("Теги: " + ", ".join(tags))
    if summary:
        lines.append("Заметки: " + summary)
    if headings:
        lines.append("Желаемая структура:")
        lines.extend([f"- {h}" for h in headings])
    return "\n".join(lines).strip()


def post_to_example(
    doc: ObsidianDoc,
    *,
    source_relpath: str,
    language: str = "ru",
    redact: bool = True,
) -> Optional[LoraExample]:
    # Skip empty notes.
    body = (doc.content_md or "").strip()
    if len(body) < 200:
        return None

    # Skip very short/utility notes.
    title = doc.title.strip() if doc.title else path.stem
    tags = list(doc.tags or [])
    summary = doc.summary
    headings = _extract_headings(body)

    instruction = _build_instruction(title, language=language)
    in_text = _build_input(
        title=title,
        tags=tags,
        summary=summary,
        headings=headings,
        language=language,
    )

    out_text = body

    # Avoid leaking absolute paths into a dataset that will be uploaded to cloud GPUs.
    meta: Dict[str, Any] = dict(doc.metadata)
    meta.pop("source_path", None)
    meta.update({"source_relpath": source_relpath})

    if redact:
        rules = default_redaction_rules()
        instruction, _ = redact_text(instruction, rules)
        in_text, _ = redact_text(in_text, rules)
        out_text, _ = redact_text(out_text, rules)

    return LoraExample(
        instruction=instruction,
        input=in_text,
        output=out_text,
        meta=meta,
    )


def iter_post_paths(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".markdown"}:
            yield p


def build_dataset(
    posts_root: Path,
    *,
    language: str = "ru",
    include_drafts: bool = False,
    redact: bool = True,
    limit: Optional[int] = None,
) -> Tuple[List[LoraExample], Dict[str, Any]]:
    examples: List[LoraExample] = []
    skipped = {"too_short": 0, "draft": 0, "other": 0}

    for p in sorted(iter_post_paths(posts_root)):
        try:
            doc = parse_obsidian_markdown(p)
        except Exception:
            skipped["other"] += 1
            continue

        if not include_drafts and (doc.status or "").lower() == "draft":
            skipped["draft"] += 1
            continue

        try:
            rel = str(p.relative_to(posts_root))
        except ValueError:
            rel = p.name

        ex = post_to_example(doc, source_relpath=rel, language=language, redact=redact)
        if ex is None:
            skipped["too_short"] += 1
            continue

        examples.append(ex)
        if limit is not None and len(examples) >= limit:
            break

    stats = {
        "posts_root": str(posts_root),
        "examples": len(examples),
        "skipped": skipped,
        "language": language,
        "include_drafts": include_drafts,
        "redact": redact,
    }
    return examples, stats


def write_jsonl(examples: Sequence[LoraExample], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            obj = {"instruction": ex.instruction, "input": ex.input, "output": ex.output, "meta": ex.meta}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

