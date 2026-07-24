"""Unit tests for the pure logic: normalisation, dedupe, classification.

Deliberately no network and no fixtures beyond an in-memory SQLite database,
so the suite runs in under a second and can't fail because a site is down.
"""

from __future__ import annotations

import sqlite3

import pytest

from pipeline.classifier import Classifier
from pipeline.config import ClassificationConfig
from pipeline.schema import ValidationError, clean_text, normalise_hit
from pipeline.store import SCHEMA_SQL, count_rows, fetch_unclassified, upsert_records


@pytest.fixture
def cls_cfg() -> ClassificationConfig:
    return ClassificationConfig(
        positive_threshold=0.05,
        negative_threshold=-0.05,
        categories=["product_launch", "business_news", "criticism",
                    "technical_discussion", "other"],
    )


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA_SQL)
    return c


def make_hit(**overrides):
    base = {
        "objectID": "123",
        "title": "OpenAI launches ChatGPT Enterprise",
        "author": "pg",
        "url": "https://example.com/a",
        "created_at_i": 1_700_000_000,
        "points": 42,
        "num_comments": 7,
    }
    base.update(overrides)
    return base


# --- normalisation -------------------------------------------------------

def test_clean_text_strips_html_and_collapses_whitespace():
    assert clean_text("<p>hello   &amp;   world</p>") == "hello & world"


def test_clean_text_handles_none():
    assert clean_text(None) == ""


def test_normalise_produces_expected_fields():
    record = normalise_hit(make_hit(), "ChatGPT", "hackernews_algolia", "2026-01-01T00:00:00+00:00")
    assert record.id == "123"
    assert record.topic == "ChatGPT"
    assert record.created_at.startswith("2023-11-14")
    assert record.points == 42


def test_missing_object_id_is_rejected():
    with pytest.raises(ValidationError):
        normalise_hit(make_hit(objectID=None), "ChatGPT", "hn", "now")


def test_missing_title_falls_back_to_story_text():
    hit = make_hit(title=None, story_text="<i>Ask HN: is Claude any good?</i>")
    record = normalise_hit(hit, "Claude", "hn", "now")
    assert record.title is None
    assert "Ask HN" in record.text


def test_record_with_no_text_at_all_is_rejected():
    with pytest.raises(ValidationError):
        normalise_hit(make_hit(title=None, story_text=None), "Claude", "hn", "now")


def test_null_points_does_not_crash():
    record = normalise_hit(make_hit(points=None), "ChatGPT", "hn", "now")
    assert record.points is None


def test_url_falls_back_to_hn_permalink():
    record = normalise_hit(make_hit(url=None), "ChatGPT", "hn", "now")
    assert "news.ycombinator.com" in record.url


# --- idempotency ---------------------------------------------------------

def test_reinserting_same_records_is_a_noop(conn):
    records = [normalise_hit(make_hit(objectID=str(i)), "ChatGPT", "hn", "now")
               for i in range(5)]

    inserted, dupes = upsert_records(conn, records)
    assert (inserted, dupes) == (5, 0)
    assert count_rows(conn) == 5

    inserted, dupes = upsert_records(conn, records)
    assert (inserted, dupes) == (0, 5)
    assert count_rows(conn) == 5, "second insert must not duplicate rows"


def test_classified_rows_are_not_re_fetched(conn):
    records = [normalise_hit(make_hit(objectID="1"), "ChatGPT", "hn", "now")]
    upsert_records(conn, records)
    assert len(fetch_unclassified(conn)) == 1

    conn.execute("UPDATE mentions SET sentiment='neutral', category='other' WHERE id='1'")
    assert fetch_unclassified(conn) == [], "labelled rows must not be re-classified"


# --- classification ------------------------------------------------------

def test_sentiment_buckets(cls_cfg):
    c = Classifier(cls_cfg)
    assert c.sentiment("This is wonderful, I love it")[0] == "positive"
    assert c.sentiment("This is terrible and broken")[0] == "negative"
    assert c.sentiment("Version 4 released Tuesday")[0] == "neutral"


def test_empty_text_is_neutral_other(cls_cfg):
    c = Classifier(cls_cfg)
    label = c.classify("")
    assert label.sentiment == "neutral"
    assert label.category == "other"


def test_category_rules_are_ordered(cls_cfg):
    c = Classifier(cls_cfg)
    # Contains both a launch term and a technical term; launch is listed
    # first in CATEGORY_RULES so it must win.
    assert c.category("Announcing our new API")[0] == "product_launch"


def test_unmatched_text_falls_back_to_other(cls_cfg):
    c = Classifier(cls_cfg)
    assert c.category("zqx wibble frobnitz")[0] == "other"


def test_classification_is_deterministic(cls_cfg):
    c = Classifier(cls_cfg)
    text = "Gemini pricing raises concerns among developers"
    assert c.classify(text) == c.classify(text)


def test_word_boundaries_prevent_substring_matches(cls_cfg):
    c = Classifier(cls_cfg)
    # "bandwidth" contains "ban" but must not trigger the criticism rule.
    category, _ = c.category("Improving bandwidth utilisation")
    assert category != "criticism"
