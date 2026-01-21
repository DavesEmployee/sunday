#!/usr/bin/env python3
"""
scrape_sunday_product.py

Scrape a single Sunday product page into JSON (data/json/<slug>.json) and
download the product image (data/images/<slug>.png).

Highlights:
- Dynamic heading-based section extraction with FAQ fallback for accordion text.
- Description trimming to avoid bleeding into the next section.
- Optional HTML input for offline debugging with --html-file.

Usage:
  python scripts/scrape_sunday_product.py "https://www.getsunday.com/shop/lawn-care/sunday-weed-pest-wand-sprayer"
  python scripts/scrape_sunday_product.py --renderer always "https://..."
  python scripts/scrape_sunday_product.py --html-file "page.html" "https://example.com/original-url"

Dependencies:
  uv pip install requests beautifulsoup4
Optional JS rendering:
  uv pip install playwright
  playwright install chromium
Optional PNG conversion:
  uv pip install pillow
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

HEADING_BLACKLIST_PATTERNS = [
    r"^terms?$",
    r"^sign in$",
    r"^cart$",
    r"^shop$",
    r"^the shed blog$",
    r"^get your plan$",
    r"^start your lawn analysis$",
    r"^learn more$",
    r"^add to cart$",
    r"^related products?$",
    r"^customer service$",
    r"^need help\??$",
    r"^customer reviews?$",
    r"^local guides$",
    r"^my account$",
    r"^privacy policy$",
    r"^terms of use$",
    r"^safety data sheets$",
    r"^do not sell my personal information$",
    r"^explore the shed$",
]

DESCRIPTION_SKIP_PHRASES = [
    "wondering what your lawn really needs",
    "let's create a custom plan",
    "we're committed to making sure sunday works for you",
    "related products",
]

DESCRIPTION_STOP_PHRASES = [
    "related products",
]

FAQ_STOP_MARKERS = [
    "documents",
    "trending items",
    "customer service",
    "need help",
    "customer reviews",
    "guarantee",
    "ingredients",
    "local guides",
    "my account",
    "privacy policy",
    "terms of use",
    "safety data sheets",
    "all rights reserved",
]


def clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def norm_text(s: str) -> str:
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_lower(s: str) -> str:
    return norm_text(s).lower()


def get_slug(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] if path else ""


def slugify_key(s: str) -> str:
    s = norm_text(s)
    s = re.sub(r"\s*:\s*$", "", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:80] or "section"


def fetch_html_requests(url: str, timeout_s: int) -> str:
    with requests.Session() as sess:
        r = sess.get(url, headers=HEADERS, timeout=timeout_s, allow_redirects=True)
        r.raise_for_status()
        return r.text


def fetch_html_playwright(url: str, timeout_ms: int) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise RuntimeError(
            "Playwright is required to render this page, but it isn't installed.\n"
            "Install with:\n  pip install playwright\n  playwright install chromium\n"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        # Try expanding accordions
        try:
            btns = page.locator("main button")
            count = btns.count()
            for i in range(min(count, 20)):
                txt = (btns.nth(i).inner_text(timeout=500) or "").strip().lower()
                if "faq" in txt or "helpful tips" in txt:
                    try:
                        btns.nth(i).click(timeout=800)
                    except Exception:
                        pass
            toggles = page.locator("main [aria-expanded='false']")
            c2 = toggles.count()
            for i in range(min(c2, 25)):
                try:
                    toggles.nth(i).click(timeout=800)
                except Exception:
                    pass
            page.wait_for_timeout(300)
        except Exception:
            pass

        html = page.content()
        browser.close()
        return html


def build_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(["style", "noscript"]):
        t.decompose()
    return soup


def parse_json_ld_product(soup: BeautifulSoup) -> dict[str, object]:
    out: dict[str, object] = {}
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if isinstance(it, dict) and it.get("@type") == "Product":
                for k in ("name", "description", "image", "sku"):
                    if k in it:
                        out[k] = it[k]
                offers = it.get("offers")
                if isinstance(offers, dict) and "price" in offers:
                    out["price"] = offers["price"]
                return out
    return out


def extract_product_name(soup: BeautifulSoup, jsonld: dict[str, object]) -> str | None:
    h1 = soup.find("h1")
    if h1:
        t = clean(h1.get_text(" ", strip=True))
        if t:
            return t
    if isinstance(jsonld.get("name"), str):
        return clean(str(jsonld["name"]))
    return None


def is_blacklisted_heading(text: str) -> bool:
    t = norm_lower(text)
    return any(re.match(pat, t) for pat in HEADING_BLACKLIST_PATTERNS)


def looks_like_heading(text: str) -> bool:
    t = norm_text(text)
    if not t or len(t) > 120:
        return False
    if t.endswith(":"):
        return True
    letters = re.sub(r"[^A-Za-z]+", "", t)
    if letters and letters.isupper() and len(t) <= 40:
        return True
    words = t.split()
    return 1 <= len(words) <= 10


def inline_heading_container(tag: Tag) -> Tag | None:
    for anc in tag.parents:
        if isinstance(anc, Tag) and anc.name == "div":
            if any(
                isinstance(ch, Tag) and ch.name in {"ul", "ol"}
                for ch in anc.find_all(recursive=False)
            ):
                return anc
    return tag.parent if isinstance(tag.parent, Tag) else None


def is_inline_heading_candidate(tag: Tag, txt: str) -> bool:
    if tag.find_parent("li"):
        return False
    if not tag.parent or tag.parent.name != "p":
        return False
    if not (txt.endswith(":") or (txt.isupper() and len(txt) <= 40)):
        return False
    for sib in tag.previous_siblings:
        if isinstance(sib, Tag):
            if norm_text(sib.get_text(" ", strip=True)):
                return False
        else:
            if str(sib).strip():
                return False
    return True


def heading_text(tag: Tag) -> str | None:
    if tag.name in {"h2", "h3", "h4"}:
        txt = norm_text(tag.get_text(" ", strip=True))
        if txt and not is_blacklisted_heading(txt):
            return txt
    if tag.name in {"button", "strong", "span"}:
        txt = norm_text(tag.get_text(" ", strip=True))
        if tag.name in {"strong", "span"}:
            if not is_inline_heading_candidate(tag, txt):
                return None
        if txt and not is_blacklisted_heading(txt) and looks_like_heading(txt):
            return txt
    return None


def iter_heading_tags_in_main(soup: BeautifulSoup) -> list[Tag]:
    root = soup.find("main") or soup.body or soup
    tags: list[Tag] = []
    h1 = soup.find("h1")
    after_h1 = False
    for t in root.find_all(["h1", "h2", "h3", "h4", "strong", "span", "button"]):
        if t == h1:
            after_h1 = True
            continue
        if h1 is not None and not after_h1:
            continue
        if t.name in {"strong", "span"}:
            txt = norm_text(t.get_text(" ", strip=True))
            if not is_inline_heading_candidate(t, txt):
                continue
        if heading_text(t):
            tags.append(t)
    return tags


def normalize_section_key(heading: str) -> str:
    h = norm_lower(heading)
    if "helpful tips" in h and ("advisor" in h or "adviser" in h):
        return "faq"
    if h in {"faq", "faqs", "frequently asked questions"}:
        return "faq"
    h_clean = re.sub(r"[^a-z0-9 ]+", "", h)
    if h_clean in {"whats included", "whats in the kit"}:
        return "whats_included"
    if h == "documents":
        return "documents"
    return slugify_key(heading)


def contains_heading_child(tag: Tag, heading_tags_set: set[Tag]) -> bool:
    for child in tag.find_all(["h2", "h3", "h4", "strong", "span", "button"]):
        if child in heading_tags_set and heading_text(child):
            return True
    return False


def strip_trailing_heading_line(text: str) -> str:
    lines = text.splitlines()
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i >= 0 and looks_like_heading(lines[i].strip()):
        lines = lines[:i]
    return clean("\n".join(lines))


def looks_like_catalog_text(text: str) -> bool:
    low = text.lower()
    if "related products" in low or "add to cart" in low:
        return True
    if re.search(r"\$\s*\d", text):
        return True
    return False


def maybe_shorten_description(desc: str) -> str:
    m = re.search(r"([^.?!]*where you need it[^.?!]*[.?!])", desc, re.I)
    if not m:
        return desc
    first = desc.split("\n\n")[0].strip()
    sentence = m.group(1).strip()
    if not first or sentence.lower() in first.lower():
        return desc
    return clean(f"{first}\n\n{sentence}")


def collect_section_content(start_heading_tag: Tag, heading_tags_set: set[Tag]) -> str:
    parts: list[str] = []
    start = start_heading_tag
    for el in start.next_elements:
        if el == start:
            continue
        if isinstance(el, Tag):
            if el in heading_tags_set and heading_text(el):
                break
            if el.name in {"footer", "nav", "header"}:
                break
            # broader: accept divs too for faq-ish structures, but keep it conservative:
            if el.name in {"p", "li"}:
                if contains_heading_child(el, heading_tags_set):
                    break
                txt = clean(el.get_text(" ", strip=True))
                if txt:
                    parts.append(txt)

    # De-dupe adjacent repeats
    dedup: list[str] = []
    for p in parts:
        if not dedup or dedup[-1] != p:
            dedup.append(p)

    return strip_trailing_heading_line(clean("\n".join(dedup)))


def find_add_to_cart_tag(soup: BeautifulSoup) -> Tag | None:
    root = soup.find("main") or soup.body or soup
    for el in root.find_all(["button", "a"]):
        if "add to cart" in norm_lower(el.get_text(" ", strip=True)):
            return el
    return None


def extract_description(soup: BeautifulSoup, jsonld: dict[str, object]) -> str | None:
    heading_tags = iter_heading_tags_in_main(soup)
    heading_set = set(heading_tags)

    def is_section_boundary(htxt: str) -> bool:
        h = norm_lower(htxt)
        if htxt.strip().endswith(":"):
            return True
        if h in {
            "product features",
            "ingredients",
            "instructions",
            "faq",
            "documents",
        }:
            return True
        if "helpful tips" in h:
            return True
        return False

    first_section_tag: Tag | None = None
    first_section_text: str | None = None
    for t in heading_tags:
        ht = heading_text(t)
        if ht and is_section_boundary(ht):
            first_section_tag = t
            first_section_text = ht
            break

    parts: list[str] = []
    if first_section_tag:
        headline_tag = first_section_tag.find_previous(["h2", "h3", "h4"])
        if headline_tag:
            htxt = clean(headline_tag.get_text(" ", strip=True))
            if htxt and htxt.lower() != (first_section_text or "").lower():
                parts.append(htxt)
            for el in headline_tag.next_elements:
                if el == headline_tag:
                    continue
                if isinstance(el, Tag):
                    if el == first_section_tag or (el in heading_set and heading_text(el)):
                        break
                    if any(
                        sp in norm_lower(el.get_text(" ", strip=True))
                        for sp in DESCRIPTION_STOP_PHRASES
                    ):
                        break
                    if el.name == "p":
                        if contains_heading_child(el, heading_set):
                            break
                        ptxt = clean(el.get_text(" ", strip=True))
                        if ptxt:
                            pl = ptxt.lower()
                            if any(sp in pl for sp in DESCRIPTION_SKIP_PHRASES):
                                continue
                            if "related products" in pl or "add to cart" in pl:
                                continue
                            if re.search(r"\$\s*\d", ptxt):
                                continue
                            parts.append(ptxt)
        else:
            anchor = find_add_to_cart_tag(soup)
            if anchor:
                for el in anchor.next_elements:
                    if isinstance(el, Tag):
                        if el == first_section_tag or (el in heading_set and heading_text(el)):
                            break
                        if el.name == "p":
                            if contains_heading_child(el, heading_set):
                                break
                            ptxt = clean(el.get_text(" ", strip=True))
                            if ptxt:
                                if "related products" in ptxt.lower() or "add to cart" in ptxt.lower():
                                    continue
                                if re.search(r"\$\s*\d", ptxt):
                                    continue
                                parts.append(ptxt)

    desc = strip_trailing_heading_line(clean("\n\n".join(parts)))
    if desc:
        if looks_like_catalog_text(desc):
            return None
        return maybe_shorten_description(desc)
    return None


def trim_faq(text: str | None) -> str | None:
    if not text:
        return None
    cut = None
    m = re.search(r"(?m)^\s*\$\d+(?:\.\d{2})?\s*$", text)
    if m:
        cut = m.start()
    m2 = re.search(r"(?im)^\s*add to cart\s*$", text)
    if m2:
        cut = m2.start() if cut is None else min(cut, m2.start())
    if cut is not None:
        text = text[:cut]
    return clean(text) or None


def faq_text_fallback(main_text: str) -> str | None:
    """
    Robust text fallback based on markers.
    Works on your uploaded HTML snippet.
    """
    txt = main_text
    low = txt.lower()

    # Find 'faq' and then 'helpful tips'
    idx_faq = low.find("\nfaq")
    if idx_faq == -1:
        idx_faq = low.find("faq\n")
    if idx_faq == -1:
        idx_faq = low.find(" faq ")
    if idx_faq == -1:
        return None

    sub = txt[idx_faq:]

    # Prefer starting at "Helpful Tips" if present
    m = re.search(r"(?i)helpful tips.*advis", sub)
    start = m.start() if m else 0

    block = sub[start:]

    # stop at footer markers
    stop_positions = []
    for marker in FAQ_STOP_MARKERS:
        p = block.lower().find("\n" + marker)
        if p != -1:
            stop_positions.append(p)
    if stop_positions:
        block = block[:min(stop_positions)]

    # remove any leading "FAQ" token if present
    block = re.sub(r"(?i)^\s*faq\s*", "", block).lstrip()
    block = trim_faq(block) or block
    return clean(block) or None


def build_sections(soup: BeautifulSoup) -> dict[str, str]:
    sections: dict[str, str] = {}
    used: set[str] = set()

    heading_tags = iter_heading_tags_in_main(soup)
    heading_set = set(heading_tags)

    for tag in heading_tags:
        ht = heading_text(tag)
        if not ht:
            continue
        if "faq" in used:
            break

        key = normalize_section_key(ht)

        if not ht.endswith(":") and key not in {"faq", "documents", "whats_included"}:
            if norm_lower(ht) not in {
                "ingredients",
                "instructions",
                "product features",
                "what's included",
                "whats included",
            }:
                continue

        base = key
        i = 2
        while key in used:
            key = f"{base}_{i}"
            i += 1

        content = collect_section_content(tag, heading_set)
        if key == "faq":
            content = trim_faq(content) or content

        if content:
            sections[key] = content
            used.add(key)

        if key == "faq":
            break

    # FAQ text fallback if still missing
    if "faq" not in sections:
        main = soup.find("main") or soup.body
        if main:
            main_text = clean(main.get_text("\n", strip=True))
            fb = faq_text_fallback("\n" + main_text)  # leading newline helps marker search
            if fb:
                sections["faq"] = fb

    return sections


def first_image_url(soup: BeautifulSoup, base_url: str, jsonld: dict[str, object]) -> str | None:
    imgs = jsonld.get("image")
    if isinstance(imgs, list) and imgs and isinstance(imgs[0], str) and imgs[0].startswith("http"):
        return imgs[0]
    if isinstance(imgs, str) and imgs.startswith("http"):
        return imgs
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        return urljoin(base_url, og["content"])
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not src:
            continue
        src = src.strip()
        if src.startswith("//"):
            src = "https:" + src
        if src.startswith("http://") or src.startswith("https://"):
            return src
        if src.startswith("/"):
            return urljoin(base_url, src)
    return None


def download_image_to_png(img_url: str, out_path: Path, timeout_s: int = 30) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(img_url, headers=HEADERS, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.content
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(data))
        im.save(out_path, format="PNG")
    except Exception:
        out_path.write_bytes(data)


def scrape(url: str, renderer: str, timeout_s: int, html_file: str | None) -> dict[str, object]:
    if html_file:
        html = Path(html_file).read_text(encoding="utf-8", errors="ignore")
    else:
        if renderer == "always":
            html = fetch_html_playwright(url, timeout_ms=timeout_s * 1000)
        else:
            html = fetch_html_requests(url, timeout_s=timeout_s)

    soup = build_soup(html)
    jsonld = parse_json_ld_product(soup)

    # Auto-fallback to Playwright if shell
    if not html_file and renderer == "auto":
        body_txt = (soup.body.get_text(" ", strip=True) if soup.body else "").lower()
        if len(body_txt) < 2000:
            try:
                html2 = fetch_html_playwright(url, timeout_ms=timeout_s * 1000)
                soup = build_soup(html2)
                jsonld = parse_json_ld_product(soup)
            except Exception:
                pass

    for t in soup.find_all("script"):
        t.decompose()

    product_name = extract_product_name(soup, jsonld)
    description = extract_description(soup, jsonld)
    sections = build_sections(soup)

    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}" if urlparse(url).netloc else ""
    img_url = first_image_url(soup, base_url=base, jsonld=jsonld) if base else None
    slug = get_slug(url) or (Path(html_file).stem if html_file else "")
    img_path = None
    if img_url and slug and not html_file:
        img_path = Path("data") / "images" / f"{slug}.png"
        try:
            download_image_to_png(img_url, img_path, timeout_s=timeout_s)
        except Exception:
            img_path = None

    return {
        "url": url,
        "product_slug": slug,
        "product_name": product_name,
        "description": description,
        "sections": sections,
        "image_url": img_url,
        "image_path": str(img_path) if img_path else None,
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Sunday product URL (or a placeholder if using --html-file)")
    ap.add_argument("--html-file", default=None, help="Path to saved HTML file to parse instead of fetching")
    ap.add_argument("--renderer", choices=["auto", "never", "always"], default="auto")
    ap.add_argument("--timeout-s", type=int, default=30)
    ap.add_argument("--out", default=None, help="Output JSON file path (optional)")
    args = ap.parse_args()

    data = scrape(args.url, renderer=args.renderer, timeout_s=args.timeout_s, html_file=args.html_file)
    out_path = Path(args.out) if args.out else Path("data") / "json" / f"{data['product_slug'] or 'unknown'}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    try:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(data, ensure_ascii=True, indent=2))
    print(f"\nWrote: {out_path}")
    if data.get("image_path"):
        print(f"Saved image: {data['image_path']}")


if __name__ == "__main__":
    main()
