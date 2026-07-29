# ur_obsidian_rag

Chat with your **Obsidian vault**. This is a small, local Retrieval-Augmented
Generation (RAG) app: it indexes your Markdown notes, retrieves the most relevant
chunks for a question, and asks a HuggingFace model to answer using only those notes —
with the source notes cited in every answer.

The design goal is **easy model swapping**: the answer model and the embedding model
are plain strings in `config.yaml`. HuggingFace is used throughout.

## How it works

```
Obsidian vault (.md)
      │  read + chunk (LlamaIndex SimpleDirectoryReader + SentenceSplitter)
      ▼
  embed locally (HuggingFace sentence-transformers)
      │
      ▼
  ChromaDB  ── persisted to ./storage ──►  (built once by scripts/index.py)
      │
      ▼  a question comes in
  retrieve top-k relevant chunks
      │
      ▼
  HuggingFace Inference API LLM  →  answer + cited source notes
      │
      ▼
  Gradio chat UI (localhost:7860)
```

Your notes are embedded **on your machine**; only the retrieved chunks + your question
are sent to the HuggingFace Inference API to generate the final answer.

## Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.10+ |
| RAG framework | [LlamaIndex](https://docs.llamaindex.ai/) |
| Note loading | `SimpleDirectoryReader` (recursive `.md`) |
| Embeddings (local) | HuggingFace `sentence-transformers` — default `BAAI/bge-small-en-v1.5` |
| Answer model | HuggingFace Inference API — default `mistralai/Mistral-7B-Instruct-v0.3` (ungated) |
| Vector store | [ChromaDB](https://www.trychroma.com/) (persisted to `./storage`) |
| UI | [Gradio](https://www.gradio.app/) chat |
| Config / secrets | `config.yaml` + `.env` (`python-dotenv`) |

## Setup

Requires **Python 3.10+**.

```powershell
# 1. Create and activate a virtual environment (isolates deps + pins the version)
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# source .venv/bin/activate           # macOS/Linux

python --version                      # confirm 3.10+

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your HuggingFace token (free: https://huggingface.co/settings/tokens)
#    The token needs the "Make calls to Inference Providers" permission
#    (fine-grained tokens: tick that box; a classic "Read" token also works).
copy .env.example .env                # then edit .env and paste your token

# 4. Point the app at your vault: edit `vault_path` in config.yaml
```

## Usage

```powershell
# Build (or rebuild) the index from your vault
python scripts/index.py

# Launch the chat UI, then open http://localhost:7860
python src/obsidian_rag/app.py
```

> Run commands from the project root with the venv activated.

## Changing the model

Everything lives in **`config.yaml`**:

- **Swap the answer model** — set `llm_model` to any text-generation model served by the
  HuggingFace Inference API (e.g. `mistralai/Mistral-7B-Instruct-v0.3`). Just restart the
  app — **no re-indexing needed**.
- **Swap the embedding model** — set `embedding_model` to any HuggingFace
  sentence-transformers model (e.g. `sentence-transformers/all-MiniLM-L6-v2`). Because
  the stored vectors change, **re-run `python scripts/index.py`** afterward.

- **Change the inference provider** — `llm_provider` selects which HuggingFace inference
  provider serves the answer model. `auto` (default) routes to whichever provider hosts
  the model; you can pin a specific one (`hf-inference`, `together`, `novita`,
  `fireworks-ai`, `nebius`, ...).

Other tunables in `config.yaml`: `chunk_size`, `chunk_overlap`, `top_k` (how many note
chunks are retrieved per question), `storage_dir`, `collection_name`.

## Project layout

```
config.yaml            # vault path + models + tunables (edit this)
.env.example           # copy to .env, add HF token
requirements.txt
scripts/index.py       # build/rebuild the index
src/obsidian_rag/
  config.py            # load + validate config.yaml and .env
  indexer.py           # vault -> chunk -> embed -> Chroma
  query.py             # retrieve + generate (RagEngine)
  app.py               # Gradio chat UI
storage/               # persisted Chroma index (auto-created, gitignored)
```

## Notes & limitations

- The index is a **full rebuild** each time `scripts/index.py` runs. For large vaults
  this is the simple, correct baseline; incremental re-indexing is a natural next step.
- First run downloads the embedding model (a few hundred MB) and caches it locally.
- Some HF Inference API models are gated (e.g. Llama) — accept the model's terms on its
  HuggingFace page once, or pick an ungated model like `mistralai/Mistral-7B-Instruct-v0.3`.
