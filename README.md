# Quickstart Instructions

Note: This is a weekend project and work in progress. Only `scripts/rag_chatbot.py` has been tested end-to-end.
Vision features are currently blocked by a PyTorch Windows DLL issue:
https://github.com/pytorch/pytorch/issues/166628

TODO (still untested):
- Multi-agent flow (`scripts/multiagent_chatbot.py`)
- API server (`src/api_server.py`)
- Vision plant ID pipeline

1. Start LM Studio (OpenAI-compatible server on `http://localhost:1234/v1`).
2. Build the index if needed:
   `uv run -- python scripts/build_product_index.py --reset`
3. Run the chatbot:
   `uv run -- python scripts/rag_chatbot.py`

## Multi-agent demo

`uv run -- python scripts/multiagent_chatbot.py`

## API Mode (for deployment demos)

Start a minimal HTTP API:

`uv run -- python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000`

POST a message:

```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What is a good lawn care product for a beginner?\"}"
```

## Docker

```
docker compose up --build
```

For API mode in Docker:

```
docker compose run --rm app uv run -- python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

## Env Vars

- `OPENAI_BASE_URL` (default: `http://localhost:1234/v1`)
- `OPENAI_API_KEY` (default: `not-needed`)
- `MODEL_NAME` (default: `qwen3-vl-30b-a3b-instruct`)
- `ALGOLIA_APP_ID`, `ALGOLIA_API_KEY`, `ALGOLIA_CT_PRODUCTS_INDEX`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`
- `LANGFUSE_PROMPT_NAME` (default: `rag-chatbot`)

## Langfuse Setup (Cloud)

1. Create a Langfuse Cloud account at https://cloud.langfuse.com/
2. Create a project.
3. Copy the public and secret keys from the project settings.
4. Add them to `.env` (see `.env.example`).

## RAG Eval

1. Create or update `data/eval/retrieval_cases.json` with entries like:
   ```
   [
     {"query": "beginner lawn fertilizer", "expected_slug": "spring-starter-kit"}
   ]
   ```
2. Run the eval harness:
   `uv run -- python scripts/eval_retrieval.py --eval-file data/eval/retrieval_cases.json`
