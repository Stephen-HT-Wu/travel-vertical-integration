"""Local RAG-lite title index, built from a news site's sitemap.

The sitemap itself carries no titles (confirmed by inspecting
supertaste.tvbs.com.tw's sitemap — each <url> entry has only <loc> and
<lastmod>), so building a usable index means: walk the sitemap (index of
sub-sitemaps, or a flat urlset) to collect every article URL + lastmod,
keep the most recently updated `max_articles`, then fetch each of those
pages once to pull its real <title>. This is a deliberately lightweight
"RAG": no embeddings, no vector store — agents/planning_agent.py just hands
the whole cached title list to an LLM call and lets it pick the most
relevant 1-n URLs (see RagSelection in schemas.py).

This module is the only place in the codebase that makes bulk outbound HTTP
requests outside the Anthropic SDK (hence the direct `requests` dependency
in requirements.txt) — building an index over ~1000 pages via Claude's own
web_fetch tool would mean ~1000 Anthropic API calls, which is exactly the
cost/latency this feature exists to avoid.
"""
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree

import requests
from pydantic import BaseModel

from local_settings import LocalSettings

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_USER_AGENT = "travel-vertical-integration-demo/1.0 (+local RAG index builder)"

RAG_CACHE_PATH = Path(__file__).parent / "output" / "rag_cache" / "site_index.json"


class SiteArticle(BaseModel):
    url: str
    title: str
    lastmod: Optional[str] = None


class SiteIndex(BaseModel):
    source_sitemap: str
    built_at: str
    articles: List[SiteArticle] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(session: requests.Session, url: str, timeout: float) -> Optional[str]:
    try:
        resp = session.get(url, timeout=timeout, headers={"User-Agent": _USER_AGENT})
        if resp.status_code != 200:
            return None
        return resp.text
    except requests.RequestException:
        return None


def _parse_lastmod(value: Optional[str]) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _urlset_entries(xml_text: str) -> List[Tuple[str, Optional[str]]]:
    root = ElementTree.fromstring(xml_text)
    entries = []
    for url_el in root.findall(f"{_SITEMAP_NS}url"):
        loc_el = url_el.find(f"{_SITEMAP_NS}loc")
        if loc_el is None or not loc_el.text:
            continue
        lastmod_el = url_el.find(f"{_SITEMAP_NS}lastmod")
        entries.append((loc_el.text.strip(), lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None))
    return entries


def _walk_sitemap(
    index_url: str, session: requests.Session, timeout: float = 10.0, max_workers: int = 8
) -> List[Tuple[str, Optional[str]]]:
    """Returns a flat list of (article_url, lastmod) pairs, whether the root
    sitemap is a sitemapindex (walks every sub-sitemap concurrently) or
    already a plain urlset."""
    root_xml = _get(session, index_url, timeout)
    if root_xml is None:
        raise RuntimeError(f"無法讀取 sitemap：{index_url}")
    root = ElementTree.fromstring(root_xml)

    if root.tag == f"{_SITEMAP_NS}urlset":
        return _urlset_entries(root_xml)

    sub_sitemap_urls = [
        loc_el.text.strip()
        for sitemap_el in root.findall(f"{_SITEMAP_NS}sitemap")
        if (loc_el := sitemap_el.find(f"{_SITEMAP_NS}loc")) is not None and loc_el.text
    ]

    # Sites that split their sitemap index by content type (the common
    # WordPress/Yoast-style `{type}_sitemap_N.xml` convention) usually also
    # publish author/category/listing-page sub-sitemaps alongside the actual
    # article ones. Those listing pages' <lastmod> bumps every time any
    # article under them changes, so sorting the pooled set purely by
    # <lastmod> lets them crowd out real articles entirely (observed: the
    # top-1000-by-lastmod set was mostly author bio pages). Prefer
    # sub-sitemaps whose own URL names them as article sitemaps; fall back
    # to everything if none match, so a site with a different naming scheme
    # still gets results instead of an empty index.
    article_sitemap_urls = [u for u in sub_sitemap_urls if "article" in u.lower()]
    if article_sitemap_urls:
        sub_sitemap_urls = article_sitemap_urls

    entries: List[Tuple[str, Optional[str]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_get, session, url, timeout): url for url in sub_sitemap_urls}
        for future in as_completed(futures):
            xml_text = future.result()
            if xml_text is None:
                continue
            try:
                entries.extend(_urlset_entries(xml_text))
            except ElementTree.ParseError:
                continue
    return entries


def _fetch_title(url: str, session: requests.Session, timeout: float = 8.0) -> Optional[str]:
    page_html = _get(session, url, timeout)
    if page_html is None:
        return None
    match = _TITLE_RE.search(page_html)
    if not match:
        return None
    title = html.unescape(match.group(1)).strip()
    return title or None


def build_index(
    sitemap_index_url: str,
    max_articles: int = 1000,
    max_workers: int = 16,
    timeout: float = 10.0,
) -> SiteIndex:
    with requests.Session() as session:
        entries = _walk_sitemap(sitemap_index_url, session, timeout=timeout)
        entries.sort(key=lambda e: _parse_lastmod(e[1]), reverse=True)
        top = entries[:max_articles]

        articles: List[SiteArticle] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_fetch_title, url, session, timeout / 2): (url, lastmod)
                for url, lastmod in top
            }
            for future in as_completed(futures):
                url, lastmod = futures[future]
                title = future.result()
                if title:
                    articles.append(SiteArticle(url=url, title=title, lastmod=lastmod))

        skipped = len(top) - len(articles)
        if skipped:
            print(f"site_index: 略過 {skipped} 篇無法取得標題的頁面（共 {len(top)} 篇候選）")

    return SiteIndex(source_sitemap=sitemap_index_url, built_at=_now(), articles=articles)


def save_index(index: SiteIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(index.model_dump_json(indent=2), encoding="utf-8")


def load_index(path: Path) -> Optional[SiteIndex]:
    if not path.exists():
        return None
    try:
        return SiteIndex.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def is_stale(index: SiteIndex, ttl_hours: float) -> bool:
    built_at = _parse_lastmod(index.built_at)
    age_hours = (datetime.now(timezone.utc) - built_at).total_seconds() / 3600
    return age_hours > ttl_hours


def ensure_index(local_settings: LocalSettings, cache_path: Path = RAG_CACHE_PATH) -> Optional[SiteIndex]:
    """Lazy entrypoint: returns None if RAG is disabled (no rag_sitemap_url),
    otherwise a fresh-enough cached index — building and saving one on the
    spot if the cache is missing or stale. Intended to be called once at
    webapp startup, not per-request; see build_rag_index.py for pre-building
    ahead of time so the first request isn't slowed down by this."""
    if not local_settings.rag_sitemap_url:
        return None
    cached = load_index(cache_path)
    if cached is not None and not is_stale(cached, local_settings.rag_cache_ttl_hours):
        return cached
    start = time.monotonic()
    index = build_index(local_settings.rag_sitemap_url, max_articles=local_settings.rag_max_articles)
    save_index(index, cache_path)
    print(f"site_index: 建立索引完成，{len(index.articles)} 篇文章，耗時 {time.monotonic() - start:.1f}s")
    return index
