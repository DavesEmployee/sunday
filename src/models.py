from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class AgentConfig(BaseModel):
    model_name: str = "qwen3-vl-30b-a3b-instruct"
    openai_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str = "not-needed"
    system_prompt: str | None = None
