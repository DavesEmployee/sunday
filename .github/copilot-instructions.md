# Copilot / AI Agent Instructions for this repository

This repository is minimal and contains a small example script that demonstrates how an agent uses a local LLM provider and Langfuse for tracing. Keep instructions concise and focused on making an AI coding agent productive quickly.

Key files
- `pydantic_connection_test.py`: primary example script. Shows how `pydantic_ai.Agent` is constructed, how the `OpenAIProvider` is pointed to a local LM Studio instance at `http://localhost:1234/v1`, and how `langfuse` is used to record an observation.

Big picture
- Single-purpose demo script that wires together three components: `pydantic_ai` (agent/model abstraction), a provider configured for a local LM Studio (`OpenAIProvider`), and `langfuse` for telemetry. There is no larger service or package structure.
- Data flow: script loads `.env`, constructs a provider & model, builds an `Agent` with a `system_prompt`, executes `agent.run_sync(...)`, and writes a Langfuse observation with `generation.update(output=...)`.

Run / dev workflow
- Recommended Python version: use the same major version as the developer environment (terminal shows Python 3.13). Use a virtual environment.
- Run the example directly:

```powershell
python pydantic_connection_test.py
```

-- The author ran it with `uv run .\pydantic_connection_test.py`; prefer `uv run` for local development. Install `uv` via `pipx install uv` or `pip install uv` if you don't have it. If `uv` is not available, fallback to `python`.

Project-specific conventions
- Environment variables: `.env` + `python-dotenv` is used (`load_dotenv()` in the script). Do not hardcode secret API keys — prefer `.env`.
- Local LM Studio usage: the provider is configured with `base_url="http://localhost:1234/v1"` and `api_key="not-needed"` (LM Studio ignores the key). When updating providers, follow this pattern.
- Model naming: models are passed by name to `OpenAIChatModel(model_name=...)`. Example in repo: `llama-3.1-8b-instruct`.
- Langfuse: the code uses `langfuse.start_as_current_observation(as_type="agent", name="test-pydantic-agent")` and then `generation.update(output=...)`. Maintain this pattern when adding new traces.

Common code patterns to preserve
- Use `Agent(..., system_prompt="...")` for system-level role instruction.
- Synchronous call pattern: `agent.run_sync("...")` — scripts expect sync runs rather than async.
- Wrap traces in the Langfuse context manager, then call `generation.update(output=...)` with the agent output string.

Dependencies / integrations
- External packages referenced in the code: `pydantic_ai`, `langfuse`, `python-dotenv` and their transitive deps. Pin dependencies in `requirements.txt` if adding one.
- External services: local LM Studio at `http://localhost:1234/v1`; Langfuse backend (configured via `get_client()` environment or defaults).

What to change when extending this repo
- If adding tests or additional modules, follow the simple, single-responsibility pattern: each script/module should construct providers/models and perform a single scenario run.
- Prefer explicit examples (copy the live pattern in `pydantic_connection_test.py`) for new integrations.

Examples (copy-paste friendly)
- Minimal provider change to point at a different local server:

```py
provider = OpenAIProvider(base_url="http://localhost:1234/v1", api_key="not-needed")
model = OpenAIChatModel(model_name="llama-3.1-8b-instruct", provider=provider)
```

Notes / limits
- This repo is intentionally small; do not invent large project structure without asking the maintainer. Document any new top-level modules in README.md.
- There are no tests or CI workflows present — add CI and pinned deps when you add more code.

If anything here is unclear or you need more examples (tests, CI, packaging), say what you'd like and I'll expand this guidance.

VLM, RAG, Langfuse & Production notes
- Vision-language (VLM) intent: this project will ingest lawn photos, run lightweight preprocessing (resize, normalize), extract visual features, and pass them to the agent as either embeddings or inline context. Keep vision code small and explicit — see `src/agent.py` for the scaffold.
- RAG integration: this repo favors a modular retriever + embedder pattern. For prototyping prefer `chromadb` or `faiss` with `sentence-transformers` embeddings; keep retriever code isolated (e.g., `build_retriever()`), and call it from the agent when answering domain-specific queries about lawn care.
- Langfuse tracing: store prompts, decisions, tool calls and retrieval metadata in Langfuse observations. Use `langfuse.start_as_current_observation(as_type="agent", name=...)` and then call `generation.update(output=..., metadata={...})` with retrieval ids / tool results.
- Tool calling & multiagent: prefer explicit tool interfaces (functions or small classes) that the agent calls. Add a `tools/` folder for any real tools (e.g., `image_processor`, `retriever`, `db_writer`). Multiagent orchestration should be explicit: each agent has a narrow responsibility and communicates via clearly typed messages.
- Containerization & production: include `requirements.txt` and a simple `Dockerfile` for local testing. Keep secrets out of images — use runtime env vars and mounts for `.env`.

- Packaging: this repo now uses `pyproject.toml` as the canonical dependency manifest. When updating dependencies, edit `pyproject.toml` and prefer `pip install .` in Docker images for reproducible builds. Keep `requirements.txt` as a convenience for quick venv installs if you like, but prefer `pyproject.toml` for builds.

Docker & scale tips
- Use the included `Dockerfile` to build a lightweight image for demoing locally. For production, move to multi-stage builds, pin dependency versions, and prefer a non-root user in the image.
- Scale considerations: separate the retriever (vector DB) as its own service (e.g., Chroma/FAISS/Weaviate) and run the model provider (LM Studio) separately. Langfuse can be run remotely; send traces asynchronously to avoid blocking requests.

Files to look at
- `pydantic_connection_test.py` — minimal run example and canonical usage patterns to copy.
- `src/agent.py` — scaffold for building the provider, model, Langfuse tracing wrapper, image preprocessing and RAG hooks.
- `requirements.txt`, `Dockerfile`, `DEPLOYMENT.md` — build and run references.

Next steps
- If you want, I can scaffold a `tools/` folder with an `image_processor.py` and a small `retriever.py` that uses `chromadb` and `sentence-transformers` for embeddings. Tell me whether to target CPU-only packages (`faiss-cpu` / `sentence-transformers`) or GPU-enabled stacks.

