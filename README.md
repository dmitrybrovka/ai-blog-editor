# ai-blog-editor
Local-first blog writing assistant: **RAG on your old posts + optional LoRA fine-tuning**.

## Quickstart (CPU-only)
### 1) Install Ollama
- Install from [Ollama](https://ollama.com) and start it.
- Pull a chat model and an embeddings model:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

### 2) Install this project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 3) Put your posts
Put your existing posts as `.md`/`.txt` into `data/posts/`.

#### Use `.env` to point to your Obsidian vault
Create `.env` (or copy from `.env.example`) and set:

```bash
AIBLOG_POSTS_DIR="/absolute/path/to/your/posts"
AIBLOG_OUT_DIR="/absolute/path/to/where/to/store/drafts"
```

This overrides `posts_dir` from `config.yaml`.
`--out` CLI flags always override both `.env` and `config.yaml`.

#### Obsidian-friendly Markdown format
We support Obsidian notes with YAML frontmatter. Example:

```md
---
title: "Как я пишу технические посты"
date: "2026-04-01"
tags: [writing, blog, ai]
status: published # or draft
summary: "Короткое описание для себя."
---

Текст поста.

Ссылки на заметки Obsidian: [[My_note|человеческое имя]].

Callouts тоже ок:
> [!NOTE]
> Это превратится в обычный текст для индексации.

%% Это комментарий Obsidian, он будет удалён при индексации %%
```

What we do for indexing / dataset:
- remove `%% comments %%`
- remove embeds like `![[file]]`
- convert wikilinks `[[Page]]` → `Page`, `[[Page|Alias]]` → `Alias`
- keep callout/blockquote content, but strip the leading `>` markers

### 4) Build the local index (RAG)

```bash
aiblog config init
aiblog ingest
```

### 5) Generate drafts

```bash
aiblog outline "Тема поста" --notes "Опорные тезисы/факты, если есть"
aiblog draft "Тема поста" --notes "Опорные тезисы/факты, если есть"
aiblog rewrite --in "out/draft.md" --out "out/draft_rewrite.md"
aiblog headline "Тема поста"
```

## LoRA dataset (for fine-tuning)
Build JSONL for instruction-tuning from your posts:

```bash
aiblog dataset lora --out out/lora_dataset.jsonl
```

By default it:
- skips `status: draft` notes
- redacts obvious PII-like strings (emails/phones/urls/paths). Use `--no-redact` to disable.

## QLoRA training (cloud) and local usage
See `docs/lora-qlora-training.md`.

## Files / directories
- `config.yaml`: models, RAG settings, style rules
- `data/posts/`: your source posts (private)
- `.aiblog/chroma/`: local vector store (generated)
- `out/`: generated outputs (you can keep it in gitignored state)
