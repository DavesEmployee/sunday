from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    openai_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str = "not-needed"
    model_name: str = "qwen3-vl-30b-a3b-instruct"
    request_timeout_s: int = 20
    model_retries: int = 3
    model_retry_backoff_s: float = 1.5
    algolia_app_id: str = "H8BCLEKQVV"
    algolia_api_key: str = "5e2db58799b436ab448f2f3b6dc0696f"
    algolia_ct_products_index: str = "prod_ct_products"
    langfuse_prompt_name: str = "rag-chatbot"
    plant_id_model: str = "juppy44/plant-identification-2m-vit-b"
    plant_id_confidence_threshold: float = 0.55

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
