FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# install the `uv` runner in the container (host can still use system `uv`)
RUN pip install --no-cache-dir uv

# Install the project (reads dependencies from pyproject.toml)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system .

COPY . /app

RUN chmod +x /app/scripts/docker_entrypoint.sh

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
CMD ["uv", "run", "rag_chatbot.py"]
