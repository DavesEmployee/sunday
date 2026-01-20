from src.models import AgentConfig
from src.base_agent import BaseAgent
from src.agent import build_langfuse_client


langfuse = build_langfuse_client()

# Try to fetch a chat prompt from Langfuse (optional). This should contain
# system/user messages that drive the agent. See the repo's Copilot instructions
# for the Langfuse tracing pattern.
chat_prompt = None
try:
    chat_prompt = langfuse.get_prompt("test-pydantic-agent", type="chat")
except Exception:
    chat_prompt = None

system_content: str | None = None
user_content: str | None = None
if chat_prompt and getattr(chat_prompt, "prompt", None):
    system_content = next((m.get("content") for m in chat_prompt.prompt if m.get("role") == "system"), None)
    user_content = next((m.get("content") for m in chat_prompt.prompt if m.get("role") == "user"), None)


config = AgentConfig(
    model_name=(chat_prompt.metadata.get("model") if chat_prompt and getattr(chat_prompt, "metadata", None) else None) or None,
    openai_base_url=None,
    openai_api_key=None,
    system_prompt=system_content,
)

agent = BaseAgent(config)
user_message = user_content or "Explain RAG in one paragraph."

result = agent.run(user_message)
print(result)
