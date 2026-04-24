from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional

import chromadb

from aiblog.ollama_client import OllamaClient
from aiblog.obsidian import parse_obsidian_markdown
from aiblog.text_utils import Chunk, chunk_text, md_to_text


@dataclass(frozen=True)
class IngestedDoc:
    doc_id: str
    source_path: str
    title: Optional[str]
    text: str
    chunks: list[Chunk]
    metadata: dict[str, Any]


def _is_scalar_metadata_value(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def sanitize_chroma_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """
    Chroma metadata restrictions (practical):
    - values: str/int/float/bool/None or a non-empty list of a single scalar type
    - empty lists are invalid
    - dicts/objects are invalid
    """
    out: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            out[k] = None
            continue

        # Normalize common non-scalar types
        if isinstance(v, Path):
            v = str(v)

        if _is_scalar_metadata_value(v):
            out[k] = v
            continue

        if isinstance(v, list):
            if len(v) == 0:
                continue
            # ensure homogeneous scalar list
            first = v[0]
            if not _is_scalar_metadata_value(first):
                continue
            first_t = type(first)
            ok = True
            for item in v:
                if type(item) is not first_t or not _is_scalar_metadata_value(item):
                    ok = False
                    break
            if ok:
                out[k] = v
            continue

        # Drop unsupported metadata values (dicts, sets, tuples, etc.)
        continue

    return out


def load_post(path: Path, *, posts_root: Optional[Path] = None) -> IngestedDoc:
    doc_id = str(path)
    if posts_root is not None:
        try:
            doc_id = str(path.relative_to(posts_root))
        except ValueError:
            doc_id = str(path)

    if path.suffix.lower() in {".md", ".markdown"}:
        doc = parse_obsidian_markdown(path)
        title = doc.title
        text = md_to_text(doc.content_md)
        meta: dict[str, Any] = dict(doc.metadata)
    else:
        raw = path.read_text(encoding="utf-8")
        title = path.stem
        text = raw.strip()
        meta = {"source_path": str(path), "source_name": path.name, "title": title}
    return IngestedDoc(
        doc_id=doc_id,
        source_path=str(path),
        title=title,
        text=text,
        chunks=[],
        metadata=meta,
    )


def build_chunks(doc: IngestedDoc, *, chunk_chars: int, overlap_chars: int) -> IngestedDoc:
    base = doc.title + "\n\n" + doc.text if doc.title else doc.text
    chunks = chunk_text(base, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    return IngestedDoc(
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        title=doc.title,
        text=doc.text,
        chunks=chunks,
        metadata=doc.metadata,
    )


class VectorStore:
    def __init__(
        self,
        *,
        persist_dir: Path,
        collection: str,
        ollama: OllamaClient,
        embed_model: str,
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        self._ollama = ollama
        self._embed_model = embed_model

    def reset(self) -> None:
        name = self._col.name
        self._client.delete_collection(name)
        self._col = self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def upsert_doc(self, doc: IngestedDoc) -> int:
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []
        embs: list[list[float]] = []

        for ch in doc.chunks:
            chunk_id = f"{doc.doc_id}::chunk::{ch.idx}"
            ids.append(chunk_id)
            docs.append(ch.text)
            m = dict(doc.metadata)
            m.update({"chunk_idx": ch.idx, "chunk_start": ch.start, "chunk_end": ch.end, "title": doc.title})
            m = sanitize_chroma_metadata(m)
            metas.append(m)
            embs.append(self._ollama.embeddings(self._embed_model, ch.text))

        if not ids:
            return 0
        self._col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        return len(ids)

    def query(self, query_text: str, *, top_k: int) -> list[dict[str, Any]]:
        qemb = self._ollama.embeddings(self._embed_model, query_text)
        res = self._col.query(query_embeddings=[qemb], n_results=top_k, include=["documents", "metadatas", "distances"])
        out: list[dict[str, Any]] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"text": doc, "meta": meta, "distance": dist})
        return out

