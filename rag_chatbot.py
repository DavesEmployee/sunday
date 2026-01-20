#!/usr/bin/env python3
"""
RAG chatbot using Pydantic AI + hybrid search over product JSONs.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import contextvars
import re
import time
import uuid
from datetime import date, timedelta
from urllib.parse import urlparse

import requests
from langfuse import propagate_attributes

from pydantic_ai import Agent, RunContext

from src.agent import build_langfuse_client, build_model, build_provider
from src.retriever import HybridRetriever
from src.settings import AppSettings


SYSTEM_PROMPT = (
    "You are a lawn care shopping assistant. "
    "Decide when to use the search tool based on the user's question. "
    "If you recommend a specific product, call the search tool first and only "
    "recommend from its results (no outside brands). "
    "Choose a single best option and explain why it fits the user's situation. "
    "Only mention features that are relevant to the context. "
    "Ask a brief follow-up question when it would help narrow the recommendation. "
    "Include image_path only if it helps the user."
)


settings = AppSettings()
retriever = HybridRetriever()
langfuse = build_langfuse_client()
session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id",
    default=None,
)


def _truncate_text(text: object, max_len: int = 450) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _trim_sections(sections: object, max_items: int = 3) -> Dict[str, str]:
    if not isinstance(sections, dict):
        return {}
    trimmed: Dict[str, str] = {}
    for key in list(sections.keys())[:max_items]:
        val = sections.get(key)
        if isinstance(val, str) and val.strip():
            trimmed[key] = _truncate_text(val, max_len=450) or ""
    return trimmed


def _format_hit(hit: Dict[str, object]) -> Dict[str, object]:
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


def _slug_from_url(value: str) -> str:
    if value.startswith("http"):
        return urlparse(value).path.rstrip("/").split("/")[-1]
    return value.strip().strip("/")


def _algolia_headers() -> Dict[str, str]:
    return {
        "X-Algolia-API-Key": settings.algolia_api_key,
        "X-Algolia-Application-Id": settings.algolia_app_id,
    }


def _algolia_index() -> str:
    return settings.algolia_ct_products_index


provider = build_provider(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
model = build_model(provider, model_name=settings.model_name)
agent: Agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)


@agent.tool
def search_products(ctx: RunContext, query: str) -> List[Dict[str, object]]:
    """Search product catalog by query."""
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="retriever",
        name="search_products",
        input=query,
        metadata={"session_id": session_id},
    ) as span:
        hits = retriever.query(query, top_k=3)
        span.update(output={"hits": [h.get("slug") for h in hits]})
    return [_format_hit(h) for h in hits]


@agent.tool
def lookup_price(ctx: RunContext, product_url: str) -> Dict[str, object]:
    """Look up a product's current price from the catalog index."""
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="tool",
        name="lookup_price",
        input=product_url,
        metadata={"session_id": session_id},
    ) as span:
        try:
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
                span.update(output={"error": "price_not_found"})
                return {"product_url": product_url, "error": "price_not_found"}
            hit = next((h for h in hits if h.get("slug") == slug), hits[0])
            unit_price = hit.get("unitPrice")
            full_price = hit.get("fullPrice")
            is_discounted = bool(hit.get("isDiscounted"))
            if unit_price is None:
                span.update(output={"error": "price_not_found"})
                return {"product_url": product_url, "error": "price_not_found"}
            price_info = {
                "product_url": product_url,
                "price": float(unit_price) / 100.0,
                "currency": "USD",
                "is_discounted": is_discounted,
            }
            if full_price and full_price != unit_price:
                price_info["full_price"] = float(full_price) / 100.0
            span.update(output=price_info)
            return price_info
        except Exception as exc:
            span.update(output={"error": str(exc)})
            return {"product_url": product_url, "error": "request_failed"}


@agent.tool
def get_today(ctx: RunContext) -> Dict[str, str]:
    """Get today's date in ISO format."""
    today = date.today().isoformat()
    return {"today": today}


@agent.tool
def geocode_location(ctx: RunContext, city: str, state: str) -> Dict[str, object]:
    """Look up latitude/longitude for a city and state."""
    query = f"{city}, {state}"
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
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="tool",
        name="geocode_location",
        input=query,
        metadata={"session_id": session_id},
    ) as span:
        try:
            params = {
                "name": city.strip(),
                "count": 1,
                "language": "en",
                "country": "US",
                "admin1": state_abbr,
            }
            resp = requests.get(url, params=params, timeout=settings.request_timeout_s)
            resp.raise_for_status()
            results = resp.json().get("results") or []
            if not results:
                resp = requests.get(
                    url,
                    params={"name": query, "count": 1, "language": "en"},
                    timeout=settings.request_timeout_s,
                )
                resp.raise_for_status()
                results = resp.json().get("results") or []
            if not results:
                span.update(output={"error": "not_found"})
                return {"query": query, "error": "not_found"}
            hit = results[0]
            payload = {
                "query": query,
                "name": hit.get("name"),
                "country": hit.get("country"),
                "latitude": hit.get("latitude"),
                "longitude": hit.get("longitude"),
            }
            span.update(output=payload)
            return payload
        except Exception as exc:
            span.update(output={"error": str(exc)})
            return {"query": query, "error": "request_failed"}


def _forecast_url(lat: float, lon: float) -> str:
    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&forecast_days=7&timezone=auto"
    )


def _archive_url(lat: float, lon: float, start: date, end: date) -> str:
    return (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        "&timezone=auto"
    )


@agent.tool
def get_weather_forecast(ctx: RunContext, latitude: float, longitude: float) -> Dict[str, object]:
    """Get 7-day weather forecast for a location."""
    session_id = session_id_var.get()
    with langfuse.start_as_current_observation(
        as_type="tool",
        name="get_weather_forecast",
        input={"lat": latitude, "lon": longitude},
        metadata={"session_id": session_id},
    ) as span:
        try:
            resp = requests.get(_forecast_url(latitude, longitude), timeout=settings.request_timeout_s)
            resp.raise_for_status()
            daily = resp.json().get("daily") or {}
            payload = {"latitude": latitude, "longitude": longitude, "daily": daily}
            span.update(output={"days": len(daily.get("time", []))})
            return payload
        except Exception as exc:
            span.update(output={"error": str(exc)})
            return {"latitude": latitude, "longitude": longitude, "error": "request_failed"}


def _month_windows(start: date, months: int = 3) -> List[tuple[date, date]]:
    windows: List[tuple[date, date]] = []
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
def get_weather_historical_trend(ctx: RunContext, latitude: float, longitude: float) -> Dict[str, object]:
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
                temps_max: List[float] = []
                temps_min: List[float] = []
                precips: List[float] = []
                for years_back in range(1, 4):
                    start = date(month_start.year - years_back, month_start.month, month_start.day)
                    end = date(month_end.year - years_back, month_end.month, month_end.day)
                    resp = requests.get(
                        _archive_url(latitude, longitude, start, end),
                        timeout=settings.request_timeout_s,
                    )
                    resp.raise_for_status()
                    daily = resp.json().get("daily") or {}
                    temps_max.extend(daily.get("temperature_2m_max") or [])
                    temps_min.extend(daily.get("temperature_2m_min") or [])
                    precips.extend(daily.get("precipitation_sum") or [])
                def _avg(values: List[float]) -> Optional[float]:
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


def _run_with_retries(user_message: str, message_history: Optional[List[object]]) -> object:
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


def main() -> None:
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


if __name__ == "__main__":
    main()
