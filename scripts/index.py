"""CLI entry point to (re)build the vector index from your Obsidian vault.

Usage:  python scripts/index.py
"""

import sys
from pathlib import Path

# Put src/ on the path so `obsidian_rag` imports without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obsidian_rag.config import load_config  # noqa: E402
from obsidian_rag.indexer import build_index  # noqa: E402


def main() -> None:
    # Token not needed for indexing (embeddings run locally).
    config = load_config(require_token=False)
    build_index(config)
    print("Done. Start the chat UI with:  python src/obsidian_rag/app.py")


if __name__ == "__main__":
    main()
