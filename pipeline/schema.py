"""The record schema and the raw -> record normalisation step.

One dataclass defines the shape of everything downstream. Normalisation is a
pure function so it can be unit-tested without a network or a database.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Fields we require to be present and non-empty for a record to be usable.
REQUIRED_FIELDS = ("id", "topic", "text", "created_at")

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Record:
    """One public mention, normalised.

    `id` is the source-provided stable identifier (HN's objectID). Using the
    source's own ID rather than a content hash means the same story keeps the
    same key across runs even if its title is edited, which is what makes
    dedupe and incremental re-runs correct.
    """

    id: str
    topic: str
    source: str
    author: str | None
    title: str | None
    text: str
    url: str | None
    created_at: str          # ISO-8601 UTC, from the source
    fetched_at: str          # ISO-8601 UTC, when we retrieved it
    points: int | None
    num_comments: int | None
    sentiment: str | None = None
    sentiment_score: float | None = None
    category: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationError(Exception):
    """Raised when a raw hit cannot be turned into a usable Record."""


def clean_text(value: str | None) -> str:
    """Strip markup, unescape entities, and collapse whitespace.

    HN titles are plain text but comment/story text fields can contain HTML.
    Normalising here means two records that say the same thing in different
    encodings compare equal downstream.
    """
    if not value:
        return ""
    text = html.unescape(value)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _to_iso_utc(hit: dict[str, Any]) -> str | None:
    """Prefer the numeric timestamp; fall back to the string field.

    `created_at_i` is a Unix epoch and unambiguous. `created_at` is an
    ISO string that we only parse if the epoch is missing.
    """
    epoch = hit.get("created_at_i")
    if epoch is not None:
        try:
            return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
        except (ValueError, OSError, OverflowError):
            logger.debug("Unparseable created_at_i=%r, falling back", epoch)

    raw = hit.get("created_at")
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except ValueError:
            logger.debug("Unparseable created_at=%r", raw)
    return None


def _coerce_int(value: Any) -> int | None:
    """HN occasionally returns null for points/num_comments; don't crash on it."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalise_hit(hit: dict[str, Any], topic: str, source: str, fetched_at: str) -> Record:
    """Turn one raw HN Algolia hit into a Record.

    Raises ValidationError rather than returning None so the caller has to
    decide explicitly what to do with a bad row — we quarantine rather than
    silently drop, so bad data stays countable.
    """
    if not isinstance(hit, dict):
        raise ValidationError(f"Expected a dict hit, got {type(hit).__name__}")

    object_id = hit.get("objectID")
    if not object_id:
        raise ValidationError("Hit has no objectID; cannot assign a stable key.")

    title = clean_text(hit.get("title"))
    story_text = clean_text(hit.get("story_text") or hit.get("comment_text"))

    # Prefer the title as the classifiable text; fall back to body text for
    # Ask HN-style posts that have no title of their own.
    text = title or story_text
    if not text:
        raise ValidationError(f"Hit {object_id} has neither title nor text.")

    created_at = _to_iso_utc(hit)
    if created_at is None:
        raise ValidationError(f"Hit {object_id} has no parseable creation timestamp.")

    author = hit.get("author") or None
    url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"

    record = Record(
        id=str(object_id),
        topic=topic,
        source=source,
        author=author,
        title=title or None,
        text=text,
        url=url,
        created_at=created_at,
        fetched_at=fetched_at,
        points=_coerce_int(hit.get("points")),
        num_comments=_coerce_int(hit.get("num_comments")),
    )
    validate(record)
    return record


def validate(record: Record) -> None:
    """Assert the invariants the rest of the pipeline relies on."""
    for name in REQUIRED_FIELDS:
        value = getattr(record, name, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValidationError(f"Record {record.id!r} missing required field '{name}'")
