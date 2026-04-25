from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.status import Status

from aiblog.config import load_config, save_default_config
from aiblog.lora_dataset import build_dataset, write_jsonl
from aiblog.rag import RagWriter
from aiblog.store import build_chunks, load_post


app = typer.Typer(add_completion=False, no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(dataset_app, name="dataset")

console = Console()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _status(message: str) -> Status:
    # Rich auto-disables animations when not a TTY.
    # Use an ASCII-ish spinner to keep the output simple.
    return console.status(message, spinner="line")


_FILENAME_SAFE_RE = re.compile(r"[^\w-]+", flags=re.UNICODE)
_FILENAME_DASH_RE = re.compile(r"[-_]{2,}")


def _slugify(text: str, *, max_len: int = 48) -> str:
    s = text.strip().lower().replace("ё", "е")
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"\s+", "-", s)
    s = _FILENAME_SAFE_RE.sub("", s)
    s = _FILENAME_DASH_RE.sub("-", s).strip("-_")
    if not s:
        return "post"
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "post"


def _timestamp() -> str:
    # Include seconds to reduce collision probability.
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_out_path(cfg_out_dir: str, *, kind: str, topic: Optional[str] = None, stem: Optional[str] = None) -> Path:
    base = Path(cfg_out_dir)
    ts = _timestamp()
    counter = 0

    if stem:
        slug = _slugify(stem)
        candidate = base / f"{kind}-{slug}-{ts}.md"
    elif topic:
        slug = _slugify(topic)
        candidate = base / f"{kind}-{slug}-{ts}.md"
    else:
        candidate = base / f"{kind}-{ts}.md"

    # Ensure we don't overwrite an existing file if called multiple times quickly.
    out = candidate
    while out.exists():
        counter += 1
        out = out.with_name(f"{candidate.stem}-{counter}{candidate.suffix}")
    return out


@config_app.command("init")
def config_init(path: Path = typer.Option(Path("config.yaml"), "--path")) -> None:
    """Create `config.yaml` with defaults."""
    if path.exists():
        console.print(f"[yellow]Config already exists:[/] {path}")
        raise typer.Exit(code=1)
    save_default_config(path)
    console.print(f"[green]Created:[/] {path}")


@app.command()
def ingest(
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    posts_dir: Optional[Path] = typer.Option(None, "--posts-dir", help="Overrides config posts_dir"),
    reset: bool = typer.Option(False, "--reset", help="Drop and rebuild the vector store"),
) -> None:
    """Ingest your posts from `data/posts/` and build/update the local index."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    if reset:
        writer.store.reset()
        console.print("[yellow]Vector store reset[/]")

    root = posts_dir or Path(cfg.posts_dir)
    if not root.exists():
        console.print(f"[red]Posts dir not found:[/] {root}")
        raise typer.Exit(code=2)

    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".txt", ".markdown"}]
    if not paths:
        console.print(f"[red]No posts found in:[/] {root}")
        raise typer.Exit(code=2)

    total_chunks = 0
    for p in sorted(paths):
        doc = load_post(p, posts_root=root)
        doc = build_chunks(doc, chunk_chars=cfg.rag.chunk_chars, overlap_chars=cfg.rag.chunk_overlap_chars)
        n = writer.store.upsert_doc(doc)
        total_chunks += n
        console.print(f"[green]Indexed[/] {p} ({n} chunks)")

    console.print(f"[bold green]Done[/]. Total chunks upserted: {total_chunks}")


@app.command()
def outline(
    topic: str,
    notes: str = typer.Option("", "--notes"),
    out: Optional[Path] = typer.Option(None, "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """Generate an outline using RAG."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    with _status(f"Generating outline with {cfg.ollama.chat_model} (RAG + Ollama)..."):
        text = writer.outline(topic=topic, notes=notes)
    out = out or _default_out_path(cfg.out_dir, kind="outline", topic=topic)
    _write_text(out, text)
    console.print(f"[green]Wrote:[/] {out}")


@app.command()
def draft(
    topic: str,
    notes: str = typer.Option("", "--notes"),
    out: Optional[Path] = typer.Option(None, "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """Generate a draft post using RAG."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    with _status(f"Generating post with {cfg.ollama.chat_model} (RAG + Ollama)..."):
        text = writer.draft(topic=topic, notes=notes)
    out = out or _default_out_path(cfg.out_dir, kind="post", topic=topic)
    _write_text(out, text)
    console.print(f"[green]Wrote:[/] {out}")


@app.command()
def rewrite(
    in_path: Path = typer.Option(..., "--in"),
    out: Optional[Path] = typer.Option(None, "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    topic_hint: str = typer.Option("", "--topic-hint"),
) -> None:
    """Rewrite text in your style."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    with _status(f"Rewriting with {cfg.ollama.chat_model} (RAG + Ollama)..."):
        text = writer.rewrite(input_text=_read_text(in_path), topic_hint=topic_hint)
    out = out or _default_out_path(cfg.out_dir, kind="rewrite", stem=in_path.stem)
    _write_text(out, text)
    console.print(f"[green]Wrote:[/] {out}")


@app.command()
def headline(
    topic: str,
    out: Optional[Path] = typer.Option(None, "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """Generate headline variants."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    with _status(f"Generating headlines with {cfg.ollama.chat_model} (RAG + Ollama)..."):
        text = writer.headline(topic=topic)
    out = out or _default_out_path(cfg.out_dir, kind="headlines", topic=topic)
    _write_text(out, text)
    console.print(f"[green]Wrote:[/] {out}")


@dataset_app.command("lora")
def dataset_lora(
    out: Optional[Path] = typer.Option(None, "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    posts_dir: Optional[Path] = typer.Option(None, "--posts-dir", help="Overrides config posts_dir"),
    include_drafts: bool = typer.Option(False, "--include-drafts"),
    redact: bool = typer.Option(True, "--redact/--no-redact", help="Redact obvious PII-like strings"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max number of examples"),
) -> None:
    """Build an instruction-tuning JSONL dataset for LoRA from your Markdown posts."""
    cfg = load_config(config)
    root = posts_dir or Path(cfg.posts_dir)
    if not root.exists():
        console.print(f"[red]Posts dir not found:[/] {root}")
        raise typer.Exit(code=2)

    examples, stats = build_dataset(
        root,
        language=cfg.style.language,
        include_drafts=include_drafts,
        redact=redact,
        limit=limit,
    )
    if not examples:
        console.print("[red]No training examples produced (posts may be too short).[/]")
        console.print_json(data=stats)
        raise typer.Exit(code=2)

    out = out or (Path(cfg.out_dir) / "lora_dataset.jsonl")
    write_jsonl(examples, out)
    console.print(f"[green]Wrote:[/] {out}")
    console.print_json(data=stats)

