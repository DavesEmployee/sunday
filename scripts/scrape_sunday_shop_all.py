#!/usr/bin/env python3
"""
scrape_sunday_shop_all.py

Fetch product URLs from https://www.getsunday.com/shop/all (or a saved HTML),
then run the product scraper for each URL.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import re
from typing import Iterable, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_html_requests(url: str, timeout_s: int) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout_s)
    resp.raise_for_status()
    return resp.text


def fetch_html_playwright(url: str, timeout_ms: int) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright is required to render this page, but it isn't installed.\n"
            "Install with:\n  uv pip install playwright\n  playwright install chromium\n"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        try:
            btn = page.locator("button", has_text="View all")
            if btn.count() > 0:
                btn.first.click(timeout=2000)
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(500)
        except Exception:
            pass
        html = page.content()
        browser.close()
        return html


def load_html(url: str, renderer: str, timeout_s: int, html_file: Optional[str]) -> str:
    if html_file:
        return Path(html_file).read_text(encoding="utf-8", errors="ignore")
    if renderer == "always":
        return fetch_html_playwright(url, timeout_ms=timeout_s * 1000)
    return fetch_html_requests(url, timeout_s=timeout_s)


def iter_shop_links(html: str, base_url: str) -> Iterable[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        if abs_url.startswith("https://www.getsunday.com/shop/"):
            yield abs_url


def is_product_url(url: str) -> bool:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if len(parts) < 3:
        return False
    if parts[0] != "shop":
        return False
    if parts[1] in {"all", "category"}:
        return False
    return True


def collect_product_urls(html: str, base_url: str) -> List[str]:
    urls = {u.split("#", 1)[0] for u in iter_shop_links(html, base_url)}
    products = sorted(u for u in urls if is_product_url(u))
    return products


def collect_product_urls_playwright(url: str, timeout_s: int) -> List[str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright is required to render this page, but it isn't installed.\n"
            "Install with:\n  uv pip install playwright\n  playwright install chromium\n"
        ) from e

    urls: Set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout_s * 1000)
        try:
            page.wait_for_selector('a[href^=\"/shop/\"]', timeout=10000)
        except Exception:
            pass
        page.wait_for_timeout(500)
        try:
            page.wait_for_selector('button[aria-label^="Page "]', timeout=10000)
        except Exception:
            pass

        def page_numbers() -> Set[int]:
            nums: Set[int] = set()
            btns = page.locator('button[aria-label^="Page "]')
            for i in range(btns.count()):
                aria = btns.nth(i).get_attribute("aria-label") or ""
                m = re.search(r"Page (\d+)", aria)
                if m:
                    nums.add(int(m.group(1)))
            return nums

        page1_html = page.content()
        page1_urls = collect_product_urls(page1_html, base_url=url)
        urls.update(page1_urls)

        total_results = None
        m = re.search(r"View all\s+(\d+)\s+results", page1_html)
        if m:
            total_results = int(m.group(1))

        per_page = len(page1_urls) or 1
        if total_results:
            max_page = (total_results + per_page - 1) // per_page
        else:
            nums = page_numbers()
            max_page = max(nums) if nums else 1

        for n in range(2, max_page + 1):
            page_url = f"{url}?prod_ct_products%5Bpage%5D={n}"
            page.goto(page_url, wait_until="networkidle", timeout=timeout_s * 1000)
            try:
                page.wait_for_selector('a[href^=\"/shop/\"]', timeout=10000)
            except Exception:
                pass
            page.wait_for_timeout(500)
            urls.update(collect_product_urls(page.content(), base_url=url))

        browser.close()

    return sorted(urls)


def run_scraper(
    urls: List[str],
    renderer: str,
    timeout_s: int,
    limit: Optional[int],
    skip_existing: bool,
) -> None:
    count = 0
    for url in urls:
        if limit is not None and count >= limit:
            break
        if skip_existing:
            slug = urlparse(url).path.rstrip("/").split("/")[-1]
            out_path = Path("data") / "json" / f"{slug}.json"
            if out_path.exists():
                print(f"\n==> {url} (skip: {out_path} exists)")
                continue
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("scrape_sunday_product.py")),
            "--renderer",
            renderer,
            "--timeout-s",
            str(timeout_s),
            url,
        ]
        print(f"\n==> {url}")
        subprocess.run(cmd, check=False)
        count += 1


def main() -> None:
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
    ap.add_argument("--limit", type=int, default=None, help="Limit number of product URLs to scrape")
    ap.add_argument("--skip-existing", action="store_true", help="Skip URLs with existing JSON output")
    ap.add_argument("--dry-run", action="store_true", help="Only list product URLs, do not scrape")
    args = ap.parse_args()

    if args.renderer == "always" and not args.html_file:
        urls = collect_product_urls_playwright(args.url, timeout_s=args.timeout_s)
    else:
        html = load_html(args.url, renderer=args.renderer, timeout_s=args.timeout_s, html_file=args.html_file)
        urls = collect_product_urls(html, base_url=args.url)

    print(f"Found {len(urls)} product URLs.")
    if args.dry_run:
        for u in urls:
            print(u)
        return

    run_scraper(
        urls,
        renderer=args.renderer,
        timeout_s=args.timeout_s,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
