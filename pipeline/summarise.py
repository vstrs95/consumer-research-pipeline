"""Summary stage.

Produces a Markdown report an analyst can read in about thirty seconds:
what we collected, how it broke down by category, and how sentiment differs
across topics. Aggregation is done in SQL rather than in Python because the
store is the source of truth and the queries double as documentation of
what the numbers mean.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _sentiment_table(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        """
        SELECT topic,
               COUNT(*) AS total,
               SUM(sentiment = 'positive') AS positive,
               SUM(sentiment = 'neutral')  AS neutral,
               SUM(sentiment = 'negative') AS negative,
               ROUND(AVG(sentiment_score), 3) AS mean_score
        FROM mentions
        GROUP BY topic
        ORDER BY mean_score DESC
        """
    ).fetchall()

    lines = [
        "| topic | records | positive | neutral | negative | mean score |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        total = r["total"] or 1
        pos_pct = 100 * (r["positive"] or 0) / total
        neg_pct = 100 * (r["negative"] or 0) / total
        lines.append(
            f"| {r['topic']} | {r['total']} | {r['positive']} ({pos_pct:.0f}%) "
            f"| {r['neutral']} | {r['negative']} ({neg_pct:.0f}%) | {r['mean_score']} |"
        )
    return "\n".join(lines)


def _category_table(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        "SELECT category, COUNT(*) AS n FROM mentions GROUP BY category ORDER BY n DESC"
    ).fetchall()
    total = sum(r["n"] for r in rows) or 1
    lines = ["| category | records | share |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['category']} | {r['n']} | {100 * r['n'] / total:.0f}% |")
    return "\n".join(lines)


def _top_items(conn: sqlite3.Connection, per_topic: int) -> str:
    """Highest-scoring stories per topic, as illustrative examples."""
    topics = [r["topic"] for r in conn.execute(
        "SELECT DISTINCT topic FROM mentions ORDER BY topic"
    ).fetchall()]

    blocks: list[str] = []
    for topic in topics:
        rows = conn.execute(
            """
            SELECT title, text, points, sentiment, category, url
            FROM mentions WHERE topic = ?
            ORDER BY COALESCE(points, 0) DESC, created_at DESC
            LIMIT ?
            """,
            (topic, per_topic),
        ).fetchall()
        if not rows:
            continue
        blocks.append(f"**{topic}**")
        blocks.append("")
        for r in rows:
            label = (r["title"] or r["text"])[:110]
            points = r["points"] if r["points"] is not None else "-"
            blocks.append(
                f"- {label} — *{r['sentiment']}* / `{r['category']}` ({points} points)"
            )
        blocks.append("")
    return "\n".join(blocks)


def _run_stats(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        """
        SELECT run_id, started_at, fetched, from_cache, normalised,
               quarantined, inserted, duplicates, classified
        FROM run_log WHERE finished_at IS NOT NULL
        ORDER BY started_at DESC LIMIT 5
        """
    ).fetchall()
    if not rows:
        return "_No completed runs recorded._"

    lines = [
        "| run started | pages fetched | pages cached | normalised | quarantined "
        "| inserted | duplicates | classified |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['started_at'][:19]} | {r['fetched']} | {r['from_cache']} "
            f"| {r['normalised']} | {r['quarantined']} | {r['inserted']} "
            f"| {r['duplicates']} | {r['classified']} |"
        )
    return "\n".join(lines)


def write_summary(conn: sqlite3.Connection, path: Path, top_per_topic: int = 3) -> None:
    total = conn.execute("SELECT COUNT(*) AS n FROM mentions").fetchone()["n"]
    topics = conn.execute("SELECT COUNT(DISTINCT topic) AS n FROM mentions").fetchone()["n"]
    quarantined = conn.execute("SELECT COUNT(*) AS n FROM quarantine").fetchone()["n"]
    unlabelled = conn.execute(
        "SELECT COUNT(*) AS n FROM mentions WHERE sentiment IS NULL OR category IS NULL"
    ).fetchone()["n"]

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    content = f"""# Consumer Research Summary

_Generated {generated} · source: Hacker News (Algolia Search API)_

**{total} unique records** across **{topics} topics**.
{quarantined} row(s) quarantined as unparseable. {unlabelled} record(s) unlabelled.

## Sentiment by topic

{_sentiment_table(conn)}

## Category distribution

{_category_table(conn)}

## Representative items

{_top_items(conn, top_per_topic)}
## Run history

Consecutive runs over unchanged data insert zero new rows — this table is the
evidence for the idempotency requirement.

{_run_stats(conn)}

---

_Sentiment is VADER compound score bucketed at ±0.05. Categories are ordered
keyword rules. Both are documented, with their known failure modes, in the
README and in `reports/evaluation.md`._
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Summary written to %s", path)
