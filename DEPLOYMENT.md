# Build and Run (local Docker)

Build the image (image will install dependencies from `pyproject.toml`):

```bash
docker build -t sunday-agent:latest .
```

Run the container (mount `.env` to provide secrets/config).

The image installs project dependencies from `pyproject.toml` and also installs `uv` so `uv run` is available in-container.

```bash
docker run --rm -v ${PWD}/.env:/app/.env -p 8000:8000 sunday-agent:latest
```

Local dev: prefer `uv run` to execute the example without Docker. Install `uv` via `pipx` or `pip` and run:

```bash
# install (recommended via pipx)
pipx install uv

# run the example
uv run pydantic_connection_test.py
```

Notes
- LM Studio is external to this image; run LM Studio locally (default `http://localhost:1234/v1`) or point `OPENAI_BASE_URL` to your provider.
- For production, replace local vector DB with an external service (Chroma/Weaviate) and run as separate services.
