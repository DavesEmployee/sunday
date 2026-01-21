from __future__ import annotations

import contextvars
import re
import time
import uuid

from langfuse import propagate_attributes
from pydantic_ai import Agent, RunContext

from .agent import build_langfuse_client, build_model, build_provider
from .settings import AppSettings
from .tools import (
    geocode_core,
    identify_plant_core,
    lookup_price_core,
    safety_guidance,
    search_products_core,
    weather_forecast_core,
)


settings = AppSettings()
langfuse = build_langfuse_client()
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id",
    default=None,
)


def _truncate_text(text: object, max_len: int = 350) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _format_hit(hit: dict[str, object]) -> dict[str, object]:
    return {
        "slug": hit.get("slug"),
        "product_name": hit.get("product_name"),
        "description": _truncate_text(hit.get("description")),
        "sections": hit.get("sections"),
        "product_url": hit.get("product_url") or hit.get("url"),
    }


def _search_products(query: str, top_k: int = 3) -> list[dict[str, object]]:
    hits = search_products_core(query, top_k=top_k)
    return [_format_hit(h) for h in hits]


def _lookup_price(product_url: str) -> dict[str, object]:
    return lookup_price_core(product_url)


def _geocode(city: str, state: str) -> dict[str, object]:
    return geocode_core(city, state)


def _weather_forecast(latitude: float, longitude: float) -> dict[str, object]:
    return weather_forecast_core(latitude, longitude)


def _run_with_retries(agent: Agent, prompt: str, message_history: list[object] | None) -> object:
    last_exc: Exception | None = None
    for attempt in range(1, settings.model_retries + 1):
        try:
            return agent.run_sync(prompt, message_history=message_history)
        except Exception as exc:
            last_exc = exc
            if attempt < settings.model_retries:
                time.sleep(settings.model_retry_backoff_s * attempt)
    if last_exc:
        raise last_exc
    return agent.run_sync(prompt, message_history=message_history)


class MultiAgentOrchestrator:
    def __init__(self) -> None:
        provider = build_provider(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
        model = build_model(provider, model_name=settings.model_name)

        self.product_agent = Agent(
            model=model,
            system_prompt=(
                "You recommend a single best product from Sunday based on the catalog results. "
                "Use the search_products tool and only recommend from the results. "
                "Return a compact JSON object with keys: product_name, product_url, reason. "
                "The reason should be one sentence."
            ),
        )
        self.weather_agent = Agent(
            model=model,
            system_prompt=(
                "You summarize weather data for lawn care timing. "
                "Use geocode_location and get_weather_forecast tools."
            ),
        )
        self.price_agent = Agent(
            model=model,
            system_prompt=(
                "Given a product_url, call the lookup_price tool and return the price details."
            ),
        )
        self.safety_agent = Agent(
            model=model,
            system_prompt=(
                "You provide short safety guidance when chemicals or pesticides are involved."
            ),
        )
        self.vision_agent = Agent(
            model=model,
            system_prompt=(
                "You identify plants from image URLs. Call identify_plant_image and "
                "return the label and confidence. If confidence is low, say so."
            ),
        )
        self.planner_agent = Agent(
            model=model,
            system_prompt=(
                "You are an orchestration agent. Decide which specialist agents to call "
                "(product, weather, price, safety, vision) based on the user question. "
                "If the user asks for a product AND price, call product_agent first, then "
                "call price_agent with the returned product_url, and include the price in "
                "the same response. "
                "If the user provides an image URL, call vision_agent to identify the plant, "
                "then call product_agent to recommend a relevant product. "
                "Use the tools to call agents, then synthesize a final response."
            ),
        )

        @self.product_agent.tool
        def search_products(ctx: RunContext, query: str) -> list[dict[str, object]]:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="retriever",
                name="search_products",
                input=query,
                metadata={"session_id": session_id},
            ) as span:
                hits = _search_products(query)
                span.update(output={"hits": [h.get("slug") for h in hits]})
            return hits

        @self.price_agent.tool
        def lookup_price(ctx: RunContext, product_url: str) -> dict[str, object]:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="tool",
                name="lookup_price",
                input=product_url,
                metadata={"session_id": session_id},
            ) as span:
                payload = _lookup_price(product_url)
                span.update(output=payload)
            return payload

        @self.weather_agent.tool
        def geocode_location(ctx: RunContext, city: str, state: str) -> dict[str, object]:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="tool",
                name="geocode_location",
                input={"city": city, "state": state},
                metadata={"session_id": session_id},
            ) as span:
                payload = _geocode(city, state)
                span.update(output=payload)
            return payload

        @self.weather_agent.tool
        def get_weather_forecast(ctx: RunContext, latitude: float, longitude: float) -> dict[str, object]:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="tool",
                name="get_weather_forecast",
                input={"lat": latitude, "lon": longitude},
                metadata={"session_id": session_id},
            ) as span:
                payload = _weather_forecast(latitude, longitude)
                span.update(output={"days": len(payload.get("time", []))})
            return payload

        @self.safety_agent.tool
        def safety_check(ctx: RunContext, query: str) -> dict[str, object]:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="guardrail",
                name="safety_check",
                input=query,
                metadata={"session_id": session_id},
            ) as span:
                guidance = safety_guidance(query)
                payload = {"safety_note": guidance}
                span.update(output=payload)
            return payload

        @self.vision_agent.tool
        def identify_plant_image(ctx: RunContext, image_url: str) -> dict[str, object]:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="embedding",
                name="identify_plant_image",
                input=image_url,
                metadata={"session_id": session_id},
            ) as span:
                try:
                    result = identify_plant_core(image_url)
                    if result is None:
                        payload = {"error": "model_unavailable"}
                        span.update(output=payload)
                        return payload
                    payload = {
                        "label": result.label,
                        "score": result.score,
                        "low_confidence": result.score < settings.plant_id_confidence_threshold,
                    }
                    span.update(output=payload)
                    return payload
                except Exception as exc:
                    span.update(output={"error": str(exc)})
                    return {"error": "request_failed"}

        @self.planner_agent.tool
        def call_product_agent(ctx: RunContext, query: str) -> str:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="agent",
                name="product_agent",
                input=query,
                metadata={"session_id": session_id},
            ) as span:
                result = _run_with_retries(self.product_agent, query, None)
                span.update(output=str(result.output))
            return str(result.output)

        @self.planner_agent.tool
        def call_weather_agent(ctx: RunContext, query: str) -> str:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="agent",
                name="weather_agent",
                input=query,
                metadata={"session_id": session_id},
            ) as span:
                result = _run_with_retries(self.weather_agent, query, None)
                span.update(output=str(result.output))
            return str(result.output)

        @self.planner_agent.tool
        def call_price_agent(ctx: RunContext, query: str) -> str:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="agent",
                name="price_agent",
                input=query,
                metadata={"session_id": session_id},
            ) as span:
                result = _run_with_retries(self.price_agent, query, None)
                span.update(output=str(result.output))
            return str(result.output)

        @self.planner_agent.tool
        def call_safety_agent(ctx: RunContext, query: str) -> str:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="agent",
                name="safety_agent",
                input=query,
                metadata={"session_id": session_id},
            ) as span:
                result = _run_with_retries(self.safety_agent, query, None)
                span.update(output=str(result.output))
            return str(result.output)

        @self.planner_agent.tool
        def call_vision_agent(ctx: RunContext, query: str) -> str:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="agent",
                name="vision_agent",
                input=query,
                metadata={"session_id": session_id},
            ) as span:
                result = _run_with_retries(self.vision_agent, query, None)
                span.update(output=str(result.output))
            return str(result.output)

    def run(self, message: str, message_history: list[object] | None) -> object:
        return _run_with_retries(self.planner_agent, message, message_history)


def run_multiagent_session() -> None:
    orchestrator = MultiAgentOrchestrator()
    message_history = None
    session_id = str(uuid.uuid4())
    session_id_var.set(session_id)
    with propagate_attributes(session_id=session_id):
        while True:
            user_message = input("You: ").strip()
            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                break
            with langfuse.start_as_current_observation(
                as_type="agent",
                name="multiagent_orchestrator",
                input=user_message,
                metadata={"session_id": session_id},
            ) as span:
                result = orchestrator.run(user_message, message_history)
                span.update(output=str(result.output))
            history = result.all_messages()
            message_history = history[-40:] if history else None
            print(f"\nAssistant:\n{result.output}\n")
