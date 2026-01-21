from __future__ import annotations

from dataclasses import dataclass

import requests

from .retriever import HybridRetriever
from .settings import AppSettings
from .vision import identify_plant


settings = AppSettings()
retriever = HybridRetriever()


@dataclass
class PlantResult:
    label: str
    score: float


def _algolia_headers() -> dict[str, str]:
    return {
        "X-Algolia-API-Key": settings.algolia_api_key,
        "X-Algolia-Application-Id": settings.algolia_app_id,
    }


def _algolia_index() -> str:
    return settings.algolia_ct_products_index


def slug_from_url(value: str) -> str:
    return value.strip().rstrip("/").split("/")[-1]


def search_products_core(query: str, top_k: int = 3) -> list[dict[str, object]]:
    return retriever.query(query, top_k=top_k)


def lookup_price_core(product_url: str) -> dict[str, object]:
    slug = slug_from_url(product_url)
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


def geocode_core(city: str, state: str) -> dict[str, object]:
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


def forecast_url(lat: float, lon: float) -> str:
    return (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&forecast_days=7&timezone=auto"
    )


def weather_forecast_core(latitude: float, longitude: float) -> dict[str, object]:
    resp = requests.get(forecast_url(latitude, longitude), timeout=settings.request_timeout_s)
    resp.raise_for_status()
    return resp.json().get("daily") or {}


def safety_guidance(text: str) -> str | None:
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


def identify_plant_core(image_url: str) -> PlantResult | None:
    result = identify_plant(image_url)
    if not result:
        return None
    return PlantResult(label=result.label, score=result.score)
