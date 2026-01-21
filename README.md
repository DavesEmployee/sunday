# Quickstart Instructions

Example session:

`uv run -- python scripts/rag_chatbot.py`

**USER:** hi what is a good lawn care product for a beginner trying to spruce up their lawn?

**ASSISTANT:**  
Based on your needs as a beginner, I recommend the **Lawn Vitality Liquid Lawn Fertilizer**. It's ideal for beginners because:

- It’s easy to apply with a hose-on system (no spreading or measuring required)
- Contains natural ingredients like soy protein hydrolysate and molasses that support soil health
- Has moderate nitrogen content, which is less risky for new lawn care users
- Works well on all grass types and supports sustainable growth

It's also designed for regular maintenance, so you can use it consistently without worrying about over-fertilizing.

Would you like me to check the current price or help with application timing based on your location?

**USER:** yes please, find the price and give me application timing given that im in Boston Ma

**ASSISTANT:**  
Here’s the information you requested for your lawn care in Boston, MA:

**Product:** Lawn Vitality Liquid Lawn Fertilizer  
**Price:** $30.00

**Application Timing Recommendation:**  
Based on the current weather forecast for Boston:
- The next few days show a mix of cold temperatures and precipitation (rain expected Jan 22–26).
- For best results, apply the fertilizer **early morning or evening** when it’s cool and your lawn is well-hydrated.
- Avoid applying during rain or extreme cold. The ideal window would be **after January 27**, once conditions stabilize with temperatures above freezing (currently forecasted at -5°C to -10°C) and no precipitation.

Since you're in a colder climate, it's best to wait until spring (March–April) for optimal results when grass begins active growth. If your lawn is still green now, applying the fertilizer during a warm spell could help maintain health—but avoid application if frost or snow is expected.

Would you like me to check historical weather trends for Boston to better plan next year’s timing?

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
