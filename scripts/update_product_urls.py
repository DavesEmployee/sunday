#!/usr/bin/env python3
"""
Update product JSON files with product_url based on shop/all links.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse


def _bootstrap_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def slug_from_url(url: str) -> str:
    return urlparse(url).path.rstrip("/").split("/")[-1]


def main() -> None:
    _bootstrap_repo_root()
    from scripts.scrape_sunday_shop_all import (
        collect_product_urls,
        collect_product_urls_playwright,
        load_html,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "url",
        nargs="?",
        default="https://www.getsunday.com/shop/all",
        help="Shop-all URL (default: https://www.getsunday.com/shop/all)",
    )
    ap.add_argument("--html-file", default=None, help="Path to saved HTML file to parse instead of fetching")
    ap.add_argument("--renderer", choices=["auto", "never", "always"], default="auto")
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="Show planned updates without writing")
    args = ap.parse_args()

    if args.renderer == "always" and not args.html_file:
        urls = collect_product_urls_playwright(args.url, timeout_s=args.timeout_s)
    else:
        html = load_html(args.url, renderer=args.renderer, timeout_s=args.timeout_s, html_file=args.html_file)
        urls = collect_product_urls(html, base_url=args.url)
    url_map = {slug_from_url(u): u for u in urls}

    json_dir = Path("data") / "json"
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing JSON directory: {json_dir}")

    updated = 0
    missing = 0
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if not isinstance(data, dict):
            continue
        slug = data.get("product_slug") or path.stem
        if not isinstance(slug, str):
            slug = path.stem
        product_url = url_map.get(slug)
        if not product_url:
            missing += 1
            continue
        if data.get("product_url") == product_url:
            continue
        data["product_url"] = product_url
        updated += 1
        if args.dry_run:
            print(f"{path.name} -> {product_url}")
        else:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Updated {updated} files. Missing matches: {missing}.")


if __name__ == "__main__":
    main()
