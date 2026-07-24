"""Crawl stage.

Responsibility: fetch raw JSON and write it to disk, faithfully. Nothing here
parses or interprets — that is the pipeline stage's job. Keeping the raw
payload means a parser bug costs a re-parse, not a re-crawl.

Politeness, in order of importance:
  1. robots.txt is checked once per host before the first request.
  2. A descriptive User-Agent identifies us to the operator.
  3. A configurable sleep throttles every request.
  4. Transient failures (429, 5xx, timeouts) retry with exponential backoff.
  5. Already-fetched pages are read from disk and never re-requested.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .config import Config, Topic

logger = logging.getLogger(__name__)

# Status codes worth retrying: rate limiting and server-side faults.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class CrawlResult:
    hits: list[dict[str, Any]]
    pages_fetched: int
    pages_from_cache: int


class CrawlError(Exception):
    """Raised when a page could not be retrieved after exhausting retries."""


def _robots_allows(base_url: str, user_agent: str) -> bool:
    """Check robots.txt for the source host.

    On any failure to retrieve or parse robots.txt we return True and log it:
    an unreachable robots.txt is not a disallow, but it is worth recording
    that we could not verify.
    """
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception as exc:  # network/parse failures are non-fatal
        logger.warning("Could not read %s (%s); proceeding.", robots_url, exc)
        return True

    allowed = parser.can_fetch(user_agent, base_url)
    logger.info("robots.txt at %s: fetching %s is %s.",
                robots_url, base_url, "allowed" if allowed else "DISALLOWED")
    return allowed


def _page_cache_path(cfg: Config, topic: Topic, page: int) -> Path:
    """Deterministic on-disk location for one topic's page of raw results."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic.name)
    return cfg.storage.raw_dir / safe / f"page_{page:03d}.json"


def _request_with_retry(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    max_retries: int,
    backoff_base: float,
) -> dict[str, Any]:
    """GET with exponential backoff on transient failures.

    Non-retryable HTTP errors (4xx other than 429) raise immediately —
    retrying a 404 just wastes the operator's bandwidth.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=15)
        except requests.RequestException as exc:
            last_error = exc
            logger.warning("Request error on attempt %d/%d: %s", attempt + 1, max_retries + 1, exc)
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise CrawlError(f"Non-JSON response from {response.url}") from exc

            if response.status_code not in RETRYABLE_STATUS:
                raise CrawlError(
                    f"HTTP {response.status_code} from {response.url} (not retryable)"
                )

            last_error = CrawlError(f"HTTP {response.status_code} from {response.url}")
            logger.warning(
                "Retryable HTTP %d on attempt %d/%d.",
                response.status_code, attempt + 1, max_retries + 1,
            )

        if attempt < max_retries:
            delay = backoff_base ** attempt
            logger.info("Backing off %.1fs before retry.", delay)
            time.sleep(delay)

    raise CrawlError(f"Exhausted {max_retries} retries for {url}") from last_error


def crawl_topic(
    cfg: Config,
    topic: Topic,
    session: requests.Session | None,
    offline: bool = False,
) -> CrawlResult:
    """Fetch up to max_items_per_topic hits for one topic, paginating as needed.

    In offline mode no network request is made; only pages already present in
    the raw cache are read. This is what makes the pipeline reproducible from
    a clean clone without re-crawling.
    """
    hits: list[dict[str, Any]] = []
    fetched = 0
    cached = 0
    page = 0

    per_page = cfg.source.hits_per_page
    max_pages = -(-cfg.source.max_items_per_topic // per_page)  # ceiling division

    while page < max_pages:
        cache_path = _page_cache_path(cfg, topic, page)

        if offline and not cache_path.is_file():
            logger.debug("Offline mode: no cached page %d for '%s'.", page, topic.name)
            break

        if cache_path.is_file():
            # Cache hit: no network request at all. This is what makes a
            # second run fast and keeps us off the operator's servers.
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cached += 1
                logger.debug("Cache hit: %s", cache_path)
            except (OSError, ValueError) as exc:
                logger.warning("Corrupt cache file %s (%s); re-fetching.", cache_path, exc)
                cache_path.unlink(missing_ok=True)
                continue
        else:
            if session is None:
                raise CrawlError("Network fetch required but no session available.")
            params = {
                "query": topic.query,
                "tags": cfg.source.tags,
                "page": page,
                "hitsPerPage": per_page,
            }
            logger.info("Fetching topic=%s page=%d", topic.name, page)
            payload = _request_with_retry(
                session,
                cfg.source.base_url,
                params,
                cfg.source.max_retries,
                cfg.source.backoff_base,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            fetched += 1
            # Throttle only after a real network request.
            time.sleep(cfg.source.rate_limit_seconds)

        page_hits = payload.get("hits", [])
        hits.extend(page_hits)

        # Stop early if the source has fewer pages than our cap.
        nb_pages = payload.get("nbPages")
        if not page_hits or (nb_pages is not None and page + 1 >= nb_pages):
            break
        page += 1

    hits = hits[: cfg.source.max_items_per_topic]
    logger.info(
        "Topic '%s': %d hit(s) across %d page(s) fetched + %d cached.",
        topic.name, len(hits), fetched, cached,
    )
    return CrawlResult(hits=hits, pages_fetched=fetched, pages_from_cache=cached)


def crawl(cfg: Config, offline: bool = False) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    """Crawl every configured topic.

    Returns (hits keyed by topic name, pages fetched, pages served from cache).
    A failure on one topic is logged and skipped rather than aborting the run —
    partial data is more useful than none.
    """
    cfg.storage.raw_dir.mkdir(parents=True, exist_ok=True)

    session: requests.Session | None = None
    if offline:
        logger.info("Offline mode: replaying cached payloads, no network requests.")
    else:
        # robots.txt is only consulted when we actually intend to fetch.
        if not _robots_allows(cfg.source.base_url, cfg.source.user_agent):
            raise CrawlError(
                f"robots.txt disallows fetching {cfg.source.base_url}. Refusing to crawl."
            )
        session = requests.Session()
        session.headers.update(
            {"User-Agent": cfg.source.user_agent, "Accept": "application/json"}
        )

    by_topic: dict[str, list[dict[str, Any]]] = {}
    total_fetched = 0
    total_cached = 0

    try:
        for topic in cfg.topics:
            try:
                result = crawl_topic(cfg, topic, session, offline=offline)
            except CrawlError as exc:
                logger.error("Topic '%s' failed: %s. Continuing.", topic.name, exc)
                by_topic[topic.name] = []
                continue
            by_topic[topic.name] = result.hits
            total_fetched += result.pages_fetched
            total_cached += result.pages_from_cache
    finally:
        if session is not None:
            session.close()

    logger.info(
        "Crawl complete: %d page(s) over the network, %d from cache.",
        total_fetched, total_cached,
    )
    return by_topic, total_fetched, total_cached
