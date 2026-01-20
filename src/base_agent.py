from typing import Callable
from langfuse import get_client

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .models import AgentConfig


class BaseAgent:
    """A small, testable base agent wrapper that centralizes provider/model creation
    and Langfuse tracing. Keep orchestration and tool-calling in subclasses or
    composition for DRY, testable code.
    """

    def __init__(self, config: AgentConfig, retriever: Callable | None = None) -> None:
        self.config = config
        self.retriever = retriever

        provider = OpenAIProvider(base_url=config.openai_base_url, api_key=config.openai_api_key)
        model = OpenAIChatModel(model_name=config.model_name, provider=provider)

        system_prompt = config.system_prompt or "You are a concise and accurate assistant."
        self.agent: PydanticAgent = PydanticAgent(model=model, system_prompt=system_prompt)

        self.langfuse = get_client()

    def run(self, user_message: str) -> str:
        """Run the agent synchronously, recording the generation to Langfuse.

        This method is intentionally small so subclassing is straightforward.
        """
        with self.langfuse.start_as_current_observation(as_type="agent", name="base-agent-run") as gen:
            output = self.agent.run_sync(user_message)
            gen.update(output=output)
            return output
