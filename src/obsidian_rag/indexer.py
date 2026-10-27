"""Build the vector index from a folder of Markdown files (Obsidian vault or eval corpus).

Flow: read .md files -> chunk -> embed locally (HuggingFace) -> persist to Chroma.

Each (embedding model, chunking) combination gets its own Chroma collection, so
experiments with different settings never overwrite each other. The collection
records a fingerprint of the source files; `get_or_build_index` reuses it only
while that fingerprint still matches.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import Config

log = logging.getLogger(__name__)

SOURCE_EXTS = (".md",)


@dataclass(frozen=True)
class IndexSpec:
    """Everything that determines the contents of an index, apart from the source data."""

    embedding_model: str
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_config(cls, config: Config) -> IndexSpec:
        return cls(config.embedding_model, config.chunk_size, config.chunk_overlap)

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def _source_files(source_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file()
        and p.suffix in SOURCE_EXTS
        and not any(part.startswith(".") for part in p.relative_to(source_dir).parts)
    )


def source_fingerprint(source_dir: Path) -> str:
    """Hash of every source file's relative path and content."""
    digest = hashlib.sha256()
    for path in _source_files(source_dir):
        digest.update(path.relative_to(source_dir).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def load_documents(source_dir: Path) -> list[Document]:
    """One Document per file, identified by its path relative to `source_dir`.

    `doc_id` is the stable identifier used for citations and eval relevance labels.
    It is the only metadata, so chunk embeddings don't depend on machine-specific
    absolute paths or timestamps.
    """
    documents = []
    for path in _source_files(source_dir):
        doc_id = path.relative_to(source_dir).as_posix()
        documents.append(
            Document(
                text=path.read_text(encoding="utf-8"),
                id_=doc_id,
                metadata={"doc_id": doc_id},
            )
        )
    return documents


def make_embed_model(name: str) -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=name)


def collection_name(prefix: str, spec: IndexSpec) -> str:
    return f"{prefix}-{spec.key()}"


def _client(chroma_path: Path) -> chromadb.ClientAPI:
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_path))


def build_index(
    source_dir: Path,
    spec: IndexSpec,
    chroma_path: Path,
    prefix: str,
    embed_model: HuggingFaceEmbedding | None = None,
) -> VectorStoreIndex:
    """Read `source_dir` and (re)build its collection from scratch."""
    embed_model = embed_model or make_embed_model(spec.embedding_model)
    client = _client(chroma_path)
    name = collection_name(prefix, spec)
    if name in {c.name for c in client.list_collections()}:
        client.delete_collection(name)

    documents = load_documents(source_dir)
    log.info("loaded %d document(s) from %s", len(documents), source_dir)

    fingerprint = source_fingerprint(source_dir)
    collection = client.create_collection(
        name, metadata={"source_fingerprint": fingerprint, **asdict(spec)}
    )
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=collection)
        ),
        embed_model=embed_model,
        transformations=[
            SentenceSplitter(chunk_size=spec.chunk_size, chunk_overlap=spec.chunk_overlap)
        ],
        show_progress=True,
    )
    log.info("embedded %d chunk(s) into collection %s", collection.count(), name)
    return index


def load_index(
    spec: IndexSpec,
    chroma_path: Path,
    prefix: str,
    embed_model: HuggingFaceEmbedding | None = None,
) -> VectorStoreIndex:
    """Load an existing collection for `spec`. Raises if it was never built."""
    client = _client(chroma_path)
    name = collection_name(prefix, spec)
    if name not in {c.name for c in client.list_collections()}:
        raise RuntimeError(
            f"No index for these settings ({name}). Run `python scripts/index.py` first."
        )
    return VectorStoreIndex.from_vector_store(
        ChromaVectorStore(chroma_collection=client.get_collection(name)),
        embed_model=embed_model or make_embed_model(spec.embedding_model),
    )


def get_or_build_index(
    source_dir: Path,
    spec: IndexSpec,
    chroma_path: Path,
    prefix: str,
    *,
    rebuild: bool = False,
) -> tuple[VectorStoreIndex, bool]:
    """Reuse the collection if its source fingerprint matches, else rebuild.

    Returns (index, built) where `built` says whether a rebuild happened.
    """
    embed_model = make_embed_model(spec.embedding_model)
    if not rebuild:
        client = _client(chroma_path)
        name = collection_name(prefix, spec)
        if name in {c.name for c in client.list_collections()}:
            stored = client.get_collection(name).metadata or {}
            if stored.get("source_fingerprint") == source_fingerprint(source_dir):
                return load_index(spec, chroma_path, prefix, embed_model), False
    return build_index(source_dir, spec, chroma_path, prefix, embed_model), True
