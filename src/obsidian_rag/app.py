"""Gradio chat UI over the Obsidian vault RAG engine.

Run:  python -m obsidian_rag.app   (from the src/ dir, or with src on PYTHONPATH)
      python src/obsidian_rag/app.py
Then open http://localhost:7860
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

# Allow running this file directly (python src/obsidian_rag/app.py).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from obsidian_rag.query import RagEngine  # type: ignore
else:
    from .query import RagEngine


def _format_sources(sources) -> str:
    if not sources:
        return ""
    lines = ["\n\n---", "**Sources:**"]
    for s in sources:
        lines.append(f"- `{s.note}` (score {s.score}) — {s.snippet}")
    return "\n".join(lines)


def build_ui() -> gr.ChatInterface:
    engine = RagEngine()  # Loads models + index once at startup.

    def respond(message: str, history):
        result = engine.answer(message)
        return result.text + _format_sources(result.sources)

    return gr.ChatInterface(
        fn=respond,
        title="🗒️ Obsidian RAG",
        description=(
            "Ask questions about your Obsidian vault. Answers cite the notes they "
            "came from. Change models in `config.yaml`."
        ),
    )


def main() -> None:
    build_ui().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
