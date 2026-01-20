from __future__ import annotations

from typing import Dict, List, Optional
import contextvars
import re
import time

import requests
from langfuse import propagate_attributes
from pydantic_ai import Agent, RunContext

from .agent import build_langfuse_client, build_model, build_provider
from .retriever import HybridRetriever
from .settings import AppSettings


settings = AppSettings()
langfuse = build_langfuse_client()
retriever = HybridRetriever()
session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id",
    default=None,
)


def _truncate_text(text: object, max_len: int = 350) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _format_hit(hit: Dict[str, object]) -> Dict[str, object]:
    return {
        "slug": hit.get("slug"),
        "product_name": hit.get("product_name"),
        "description": _truncate_text(hit.get("description")),
        "sections": hit.get("sections"),
        "product_url": hit.get("product_url") or hit.get("url"),
    }


def _algolia_headers() -> Dict[str, str]:
    return {
        "X-Algolia-API-Key": settings.algolia_api_key,
        "X-Algolia-Application-Id": settings.algolia_app_id,
    }


def _algolia_index() -> str:
    return settings.algolia_ct_products_index


def _slug_from_url(value: str) -> str:
    return value.strip().rstrip("/").split("/")[-1]


def _search_products(query: str, top_k: int = 3) -> List[Dict[str, object]]:
    hits = retriever.query(query, top_k=top_k)
    return [_format_hit(h) for h in hits]


def _lookup_price(product_url: str) -> Dict[str, object]:
    slug = _slug_from_url(product_url)
    app_id = _algolia_headers()["X-Algolia-Application-Id"]
    index = _algolia_index()
    url = f"https://{app_id}-dsn.algolia.net/1/indexes/{index}/query"
    query_params = f"query={slug}&hitsPerPage=5"
    resp = requests.post(
        url,
        headers=_algolia_headers(),
        json={"params": query_params},
        timeout=settings.request_timeout_s,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits") or []
    if not hits:
        return {"product_url": product_url, "error": "price_not_found"}
    hit = next((h for h in hits if h.get("slug") == slug), hits[0])
    unit_price = hit.get("unitPrice")
    full_price = hit.get("fullPrice")
    is_discounted = bool(hit.get("isDiscounted"))
    if unit_price is None:
        return {"product_url": product_url, "error": "price_not_found"}
    payload = {
        "product_url": product_url,
        "price": float(unit_price) / 100.0,
        "currency": "USD",
        "is_discounted": is_discounted,
    }
    if full_price and full_price != unit_price:
        payload["full_price"] = float(full_price) / 100.0
    return payload


def _geocode(city: str, state: str) -> Dict[str, object]:
    state_map = {
        "alabama": "AL",
        "alaska": "AK",
        "arizona": "AZ",
        "arkansas": "AR",
        "california": "CA",
        "colorado": "CO",
        "connecticut": "CT",
        "delaware": "DE",
        "florida": "FL",
        "georgia": "GA",
        "hawaii": "HI",
        "idaho": "ID",
        "illinois": "IL",
        "indiana": "IN",
        "iowa": "IA",
        "kansas": "KS",
        "kentucky": "KY",
        "louisiana": "LA",
        "maine": "ME",
        "maryland": "MD",
        "massachusetts": "MA",
        "michigan": "MI",
        "minnesota": "MN",
        "mississippi": "MS",
        "missouri": "MO",
        "montana": "MT",
        "nebraska": "NE",
        "nevada": "NV",
        "new hampshire": "NH",
        "new jersey": "NJ",
        "new mexico": "NM",
        "new york": "NY",
        "north carolina": "NC",
        "north dakota": "ND",
        "ohio": "OH",
        "oklahoma": "OK",
        "oregon": "OR",
        "pennsylvania": "PA",
        "rhode island": "RI",
        "south carolina": "SC",
        "south dakota": "SD",
        "tennessee": "TN",
        "texas": "TX",
        "utah": "UT",
        "vermont": "VT",
        "virginia": "VA",
        "washington": "WA",
        "west virginia": "WV",
        "wisconsin": "WI",
        "wyoming": "WY",
        "district of columbia": "DC",
    }
    state_clean = state.strip().lower()
    state_abbr = state_map.get(state_clean, state.strip().upper())
    url = "https://geocoding-api.open-meteo.com/v1/search"
    resp = requests.get(
        url,
        params={
            "name": city.strip(),
            "count": 1,
            "language": "en",
            "country": "US",
            "admin1": state_abbr,
        },
        timeout=settings.request_timeout_s,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        return {"query": f"{city}, {state}", "error": "not_found"}
    hit = results[0]
    return {
        "query": f"{city}, {state}",
        "name": hit.get("name"),
        "country": hit.get("country"),
        "latitude": hit.get("latitude"),
        "longitude": hit.get("longitude"),
    }


def _forecast_url(lat: float, lon: float) -> str:
    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&forecast_days=7&timezone=auto"
    )


def _weather_forecast(latitude: float, longitude: float) -> Dict[str, object]:
    resp = requests.get(_forecast_url(latitude, longitude), timeout=settings.request_timeout_s)
    resp.raise_for_status()
    return resp.json().get("daily") or {}


def _safety_guidance(text: str) -> Optional[str]:
    keywords = [
        "pesticide",
        "herbicide",
        "weed killer",
        "insecticide",
        "fungicide",
        "chemical",
        "spray",
        "glyphosate",
    ]
    lowered = text.lower()
    if any(k in lowered for k in keywords):
        return (
            "Safety note: follow the label, wear gloves/eye protection, "
            "keep kids and pets off treated areas until dry, and avoid drift on windy days."
        )
    return None


def _run_with_retries(agent: Agent, prompt: str, message_history: Optional[List[object]]) -> object:
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
        self.planner_agent = Agent(
            model=model,
            system_prompt=(
                "You are an orchestration agent. Decide which specialist agents to call "
                "(product, weather, price, safety) based on the user question. "
                "If the user asks for a product AND price, call product_agent first, then "
                "call price_agent with the returned product_url, and include the price in "
                "the same response. Use the tools to call agents, then synthesize a final response."
            ),
        )

        @self.product_agent.tool
        def search_products(ctx: RunContext, query: str) -> List[Dict[str, object]]:
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
        def lookup_price(ctx: RunContext, product_url: str) -> Dict[str, object]:
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
        def geocode_location(ctx: RunContext, city: str, state: str) -> Dict[str, object]:
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
        def get_weather_forecast(ctx: RunContext, latitude: float, longitude: float) -> Dict[str, object]:
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
        def safety_check(ctx: RunContext, query: str) -> Dict[str, object]:
            session_id = session_id_var.get()
            with langfuse.start_as_current_observation(
                as_type="guardrail",
                name="safety_check",
                input=query,
                metadata={"session_id": session_id},
            ) as span:
                guidance = _safety_guidance(query)
                payload = {"safety_note": guidance}
                span.update(output=payload)
            return payload

        @self.planner_agent.tool
        def call_product_agent(ctx: RunContext, query: str) -> str:
            """Call product agent to return product_name, product_url, reason as JSON."""
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
            """Call weather agent to summarize forecast for timing guidance."""
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
            """Call price agent with a product_url to fetch pricing."""
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
            """Call safety agent for chemical/pesticide guidance."""
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

    def run(self, message: str, message_history: Optional[List[object]]) -> object:
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
