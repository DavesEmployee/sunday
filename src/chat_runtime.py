from __future__ import annotations

import contextvars
import re
import time
import uuid
from datetime import date, timedelta

import requests
from langfuse import propagate_attributes
from pydantic_ai import Agent, RunContext

from .agent import build_langfuse_client, build_model, build_provider
from .settings import AppSettings
from .tools import (
    geocode_core,
    identify_plant_core,
    lookup_price_core,
    search_products_core,
    weather_forecast_core,
)


settings = AppSettings()
langfuse = build_langfuse_client()
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id",
    default=None,
)


def _truncate_text(text: object, max_len: int = 450) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _trim_sections(sections: object, max_items: int = 3) -> dict[str, str]:
    if not isinstance(sections, dict):
        return {}
    trimmed: dict[str, str] = {}
    for key in list(sections.keys())[:max_items]:
        val = sections.get(key)
        if isinstance(val, str) and val.strip():
            trimmed[key] = _truncate_text(val, max_len=450) or ""
    return trimmed


def _format_hit(hit: dict[str, object]) -> dict[str, object]:
    return {
        "slug": hit.get("slug"),
        "product_name": hit.get("product_name"),
        "description": _truncate_text(hit.get("description")),
        "sections": _trim_sections(hit.get("sections")),
        "image_path": hit.get("image_path"),
        "image_url": hit.get("image_url"),
        "url": hit.get("url"),
        "product_url": hit.get("product_url"),
    }


def _extract_image_url(text: str) -> str | None:
    match = re.search(r"(https?://\\S+)", text)
    if not match:
        return None
    url = match.group(1).rstrip(").,]")
    lower = url.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return url
    return None


provider = build_provider(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
model = build_model(provider, model_name=settings.model_name)


def _load_system_prompt() -> str:
    try:
        prompt = langfuse.get_prompt(settings.langfuse_prompt_name, type="chat")
    except Exception as exc:
        raise RuntimeError("Failed to load system prompt from Langfuse.") from exc
    if prompt and getattr(prompt, "prompt", None):
        for msg in prompt.prompt:
            if msg.get("role") == "system" and msg.get("content"):
                return str(msg.get("content"))
    if prompt and getattr(prompt, "text", None):
        return str(prompt.text)
    raise RuntimeError("Langfuse prompt is missing a system message.")


agent: Agent = Agent(model=model, system_prompt=_load_system_prompt())


@agent.tool
def search_products(ctx: RunContext, query: str) -> list[dict[str, object]]:
    """Search product catalog by query."""
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="search_products",
        input=query,
        metadata={"session_id": session_id},
    ) as span:
        hits = search_products_core(query, top_k=3)
        span.update(output={"hits": [h.get("slug") for h in hits]})
    return [_format_hit(h) for h in hits]


@agent.tool
def lookup_price(ctx: RunContext, product_url: str) -> dict[str, object]:
    """Look up a product's current price from the catalog index."""
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="tool",
        name="lookup_price",
        input=product_url,
        metadata={"session_id": session_id},
    ) as span:
        try:
            price_info = lookup_price_core(product_url)
            span.update(output=price_info)
            return price_info
        except Exception as exc:
            span.update(output={"error": str(exc)})
            return {"product_url": product_url, "error": "request_failed"}


@agent.tool
def identify_plant_image(ctx: RunContext, image_url: str) -> dict[str, object]:
    """Identify a plant from an image URL using the plant identification model."""
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
            payload = {"label": result.label, "score": result.score}
            span.update(output=payload)
            return payload
        except Exception as exc:
            span.update(output={"error": str(exc)})
            return {"error": "request_failed"}


@agent.tool
def get_today(ctx: RunContext) -> dict[str, str]:
    """Get today's date in ISO format."""
    today = date.today().isoformat()
    return {"today": today}


@agent.tool
def geocode_location(ctx: RunContext, city: str, state: str) -> dict[str, object]:
    """Look up latitude/longitude for a city and state."""
    query = f"{city}, {state}"
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="tool",
        name="geocode_location",
        input=query,
        metadata={"session_id": session_id},
    ) as span:
        try:
            payload = geocode_core(city, state)
            span.update(output=payload)
            return payload
        except Exception as exc:
            span.update(output={"error": str(exc)})
            return {"query": query, "error": "request_failed"}


@agent.tool
def get_weather_forecast(ctx: RunContext, latitude: float, longitude: float) -> dict[str, object]:
    """Get 7-day weather forecast for a location."""
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="tool",
        name="get_weather_forecast",
        input={"lat": latitude, "lon": longitude},
        metadata={"session_id": session_id},
    ) as span:
        try:
            daily = weather_forecast_core(latitude, longitude)
            payload = {"latitude": latitude, "longitude": longitude, "daily": daily}
            span.update(output={"days": len(daily.get("time", []))})
            return payload
        except Exception as exc:
            span.update(output={"error": str(exc)})
            return {"latitude": latitude, "longitude": longitude, "error": "request_failed"}


def _month_windows(start: date, months: int = 3) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    year = start.year
    month = start.month
    for _ in range(months):
        month_start = date(year, month, 1)
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        month_end = next_month - timedelta(days=1)
        windows.append((month_start, month_end))
        month = next_month.month
        year = next_month.year
    return windows


@agent.tool
def get_weather_historical_trend(ctx: RunContext, latitude: float, longitude: float) -> dict[str, object]:
    """Get 3-year historical averages for the next 3 months."""
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="tool",
        name="get_weather_historical_trend",
        input={"lat": latitude, "lon": longitude},
        metadata={"session_id": session_id},
    ) as span:
        try:
            today = date.today()
            windows = _month_windows(today, months=3)
            summaries = []
            for month_start, month_end in windows:
                temps_max: list[float] = []
                temps_min: list[float] = []
                precips: list[float] = []
                for years_back in range(1, 4):
                    start = date(month_start.year - years_back, month_start.month, month_start.day)
                    end = date(month_end.year - years_back, month_end.month, month_end.day)
                    url = (
                        "https://archive-api.open-meteo.com/v1/archive"
                        f"?latitude={latitude}&longitude={longitude}"
                        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
                        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
                        "&timezone=auto"
                    )
                    resp = requests.get(url, timeout=settings.request_timeout_s)
                    resp.raise_for_status()
                    daily = resp.json().get("daily") or {}
                    temps_max.extend(daily.get("temperature_2m_max") or [])
                    temps_min.extend(daily.get("temperature_2m_min") or [])
                    precips.extend(daily.get("precipitation_sum") or [])
                def _avg(values: list[float]) -> float | None:
                    return round(sum(values) / len(values), 2) if values else None

                summaries.append(
                    {
                        "month": month_start.strftime("%Y-%m"),
                        "avg_temp_max": _avg(temps_max),
                        "avg_temp_min": _avg(temps_min),
                        "avg_precipitation_sum": _avg(precips),
                    }
                )
            payload = {"latitude": latitude, "longitude": longitude, "months": summaries}
            span.update(output={"months": len(summaries)})
            return payload
        except Exception as exc:
            span.update(output={"error": str(exc)})
            return {"latitude": latitude, "longitude": longitude, "error": "request_failed"}


def _check_inference_health() -> None:
    url = settings.openai_base_url.rstrip("/") + "/models"
    headers = {}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=settings.request_timeout_s)
        resp.raise_for_status()
    except Exception:
        print(f"Warning: unable to reach model endpoint at {url}")


def _run_with_retries(user_message: str, message_history: list[object] | None) -> object:
    last_exc: Exception | None = None
    for attempt in range(1, settings.model_retries + 1):
        try:
            return agent.run_sync(user_message, message_history=message_history)
        except Exception as exc:
            last_exc = exc
            if attempt < settings.model_retries:
                time.sleep(settings.model_retry_backoff_s * attempt)
    if last_exc:
        raise last_exc
    return agent.run_sync(user_message, message_history=message_history)


def run_cli() -> None:
    message_history = None
    session_id = str(uuid.uuid4())
    session_id_var.set(session_id)
    _check_inference_health()
    with propagate_attributes(session_id=session_id):
        while True:
            user_message = input("You: ").strip()
            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                break
            image_url = _extract_image_url(user_message)
            if image_url:
                plant = identify_plant_image(None, image_url)
                plant_label = plant.get("label")
                if isinstance(plant_label, str):
                    user_message = (
                        f"{user_message}\n\n"
                        f"Identified plant: {plant_label}. "
                        "Recommend the best Sunday product for this plant."
                    )
            with langfuse.start_as_current_observation(
                as_type="agent",
                name="rag_chatbot",
                input=user_message,
                metadata={"session_id": session_id},
            ) as gen:
                result = _run_with_retries(user_message, message_history)
                gen.update(output=str(result.output))
            history = result.all_messages()
            message_history = history[-40:] if history else None
            print(f"\nAssistant:\n{result.output}\n")
