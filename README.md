# Quickstart Instructions

1. Start LM Studio (OpenAI-compatible server on `http://localhost:1234/v1`).
2. Build the index if needed:
   `uv run -- python build_product_index.py --reset`
3. Run the chatbot:
   `uv run -- python rag_chatbot.py`

## Multi-agent demo

`uv run -- python multiagent_chatbot.py`

## API Mode (for deployment demos)

Start a minimal HTTP API:

`uv run -- python -m uvicorn api_server:app --host 0.0.0.0 --port 8000`

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
docker compose run --rm app uv run -- python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## Env Vars

- `OPENAI_BASE_URL` (default: `http://localhost:1234/v1`)
- `OPENAI_API_KEY` (default: `not-needed`)
- `MODEL_NAME` (default: `qwen3-vl-30b-a3b-instruct`)
- `ALGOLIA_APP_ID`, `ALGOLIA_API_KEY`, `ALGOLIA_CT_PRODUCTS_INDEX`
