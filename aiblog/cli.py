from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from aiblog.config import load_config, save_default_config
from aiblog.rag import RagWriter
from aiblog.store import build_chunks, load_post


app = typer.Typer(add_completion=False, no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name="config")

console = Console()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


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
    out: Path = typer.Option(Path("out/outline.md"), "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """Generate an outline using RAG."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    text = writer.outline(topic=topic, notes=notes)
    _write_text(out, text)
    console.print(f"[green]Wrote:[/] {out}")


@app.command()
def draft(
    topic: str,
    notes: str = typer.Option("", "--notes"),
    out: Path = typer.Option(Path("out/draft.md"), "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """Generate a draft post using RAG."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    text = writer.draft(topic=topic, notes=notes)
    _write_text(out, text)
    console.print(f"[green]Wrote:[/] {out}")


@app.command()
def rewrite(
    in_path: Path = typer.Option(..., "--in"),
    out: Path = typer.Option(Path("out/rewrite.md"), "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
    topic_hint: str = typer.Option("", "--topic-hint"),
) -> None:
    """Rewrite text in your style."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    text = writer.rewrite(input_text=_read_text(in_path), topic_hint=topic_hint)
    _write_text(out, text)
    console.print(f"[green]Wrote:[/] {out}")


@app.command()
def headline(
    topic: str,
    out: Path = typer.Option(Path("out/headlines.md"), "--out"),
    config: Path = typer.Option(Path("config.yaml"), "--config"),
) -> None:
    """Generate headline variants."""
    cfg = load_config(config)
    writer = RagWriter(cfg)
    text = writer.headline(topic=topic)
    _write_text(out, text)
    console.print(f"[green]Wrote:[/] {out}")

