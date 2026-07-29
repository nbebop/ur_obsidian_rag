"""Build the vector index from an Obsidian vault.

Flow: read .md notes -> chunk -> embed locally (HuggingFace) -> persist to Chroma.
"""

from __future__ import annotations

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import Config, load_config


def configure_embeddings(config: Config) -> None:
    """Point LlamaIndex at the local HuggingFace embedding model + splitter.

    Shared by indexing and querying so both use identical chunking/embeddings.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.embedding_model)
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def _chroma_collection(config: Config, *, reset: bool):
    client = chromadb.PersistentClient(path=str(config.chroma_path))
    if reset:
        try:
            client.delete_collection(config.collection_name)
        except Exception:
            pass  # Collection didn't exist yet -- fine.
    return client.get_or_create_collection(config.collection_name)


def build_index(config: Config | None = None) -> VectorStoreIndex:
    """Read the vault and (re)build the persisted Chroma index. Full rebuild."""
    config = config or load_config(require_token=False)
    configure_embeddings(config)

    print(f"Reading notes from: {config.vault_path}")
    documents = SimpleDirectoryReader(
        input_dir=str(config.vault_path),
        recursive=True,
        required_exts=[".md"],
        filename_as_id=True,
    ).load_data()
    print(f"  loaded {len(documents)} note(s)")

    config.storage_dir.mkdir(parents=True, exist_ok=True)
    collection = _chroma_collection(config, reset=True)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Embedding with: {config.embedding_model} (first run downloads the model)")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"  {collection.count()} chunk(s) embedded")
    print(f"Index persisted to: {config.chroma_path}")
    return index


def load_index(config: Config) -> VectorStoreIndex:
    """Load an existing persisted index (used by the query engine)."""
    configure_embeddings(config)
    collection = _chroma_collection(config, reset=False)
    if collection.count() == 0:
        raise RuntimeError(
            "The index is empty. Run `python scripts/index.py` first to build it."
        )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)
