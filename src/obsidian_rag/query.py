"""Query engine: retrieve relevant notes and generate an answer with an HF LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from .config import Config, load_config
from .indexer import IndexSpec, load_index

# Keep the model grounded in retrieved notes rather than its own knowledge.
QA_PROMPT = PromptTemplate(
    "You are a helpful assistant answering questions about the user's personal "
    "Obsidian notes. Use ONLY the context below. If the answer is not in the "
    "context, say you don't know based on the notes.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)


@dataclass
class Source:
    note: str
    path: Path  # absolute path in the vault (derived at query time, not indexed)
    score: float
    snippet: str


@dataclass
class Answer:
    text: str
    sources: list[Source]


class RagEngine:
    """Holds a loaded index + query engine so models load once, not per question."""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        llm = HuggingFaceInferenceAPI(
            model_name=self.config.llm_model,
            provider=self.config.llm_provider,
            token=self.config.hf_token,
        )
        index = load_index(
            IndexSpec.from_config(self.config),
            self.config.chroma_path,
            self.config.collection_name,
        )
        self.query_engine = index.as_query_engine(
            llm=llm,
            similarity_top_k=self.config.top_k,
            text_qa_template=QA_PROMPT,
        )

    def answer(self, question: str) -> Answer:
        response = self.query_engine.query(question)
        sources: list[Source] = []
        for node in response.source_nodes:
            name = node.node.metadata.get("doc_id") or node.node.node_id
            text = node.node.get_content().strip().replace("\n", " ")
            sources.append(
                Source(
                    note=name,
                    path=self.config.vault_path / name,
                    score=round(node.score or 0.0, 3),
                    snippet=text[:200] + ("..." if len(text) > 200 else ""),
                )
            )
        return Answer(text=str(response), sources=sources)
