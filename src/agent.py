from pathlib import Path
from PIL import Image
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from langfuse import get_client
import os

from dotenv import load_dotenv
load_dotenv()


def build_provider(base_url: str | None = None, api_key: str | None = None) -> OpenAIProvider:
    """Build an OpenAIProvider using env vars when not provided.

    Env keys:
    - `OPENAI_BASE_URL` (default: http://localhost:1234/v1)
    - `OPENAI_API_KEY` (default: not-needed)
    """
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    api_key = api_key or os.getenv("OPENAI_API_KEY", "not-needed")
    return OpenAIProvider(base_url=base_url, api_key=api_key)


def build_model(provider: OpenAIProvider, model_name: str | None = None) -> OpenAIChatModel:
    """Build an OpenAIChatModel; default model comes from `MODEL_NAME` env var.

    Env keys:
    - `MODEL_NAME` (default: "OPENAI_BASE_URL", "http://localhost:1234/v1"))
    """
    model_name = model_name or os.getenv("MODEL_NAME", "qwen3-vl-30b-a3b-instruct")
    return OpenAIChatModel(model_name=model_name, provider=provider)


def build_agent(model: OpenAIChatModel, system_prompt: str = "You are a concise and accurate assistant.") -> Agent:
    return Agent(model=model, system_prompt=system_prompt)


def load_image(path: str, target_size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Simple image preload + resize used by the VLM pipeline."""
    p = Path(path)
    img = Image.open(p).convert("RGB")
    img = img.resize(target_size)
    return img


def build_langfuse_client():
    return get_client()


def build_retriever_stub():
    """Placeholder: implement a retriever (Chroma/FAISS) and embedder (sentence-transformers).

    Keep retriever code isolated and return a simple `query(text, top_k)` function.
    """
    def query(text: str, top_k: int = 5):
        return []

    return query
