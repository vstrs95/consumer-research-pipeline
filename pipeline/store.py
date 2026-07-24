"""SQLite persistence.

Idempotency lives here. `id` is the primary key and inserts use
INSERT OR IGNORE, so re-running the pipeline over the same data is a no-op
rather than a duplication. The insert count returned by `upsert_records` is
what the summary reports as "new this run".
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .schema import Record

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mentions (
    id              TEXT PRIMARY KEY,
    topic           TEXT NOT NULL,
    source          TEXT NOT NULL,
    author          TEXT,
    title           TEXT,
    text            TEXT NOT NULL,
    url             TEXT,
    created_at      TEXT NOT NULL,
    fetched_at      TEXT NOT NULL,
    points          INTEGER,
    num_comments    INTEGER,
    sentiment       TEXT,
    sentiment_score REAL,
    category        TEXT
);

CREATE INDEX IF NOT EXISTS idx_mentions_topic ON mentions(topic);
-- Partial index: the classifier queries for unlabelled rows on every run,
-- and this keeps that lookup cheap as the table grows.
CREATE INDEX IF NOT EXISTS idx_mentions_unlabelled
    ON mentions(id) WHERE sentiment IS NULL OR category IS NULL;

-- Rejected rows are kept, not dropped, so data loss stays countable.
-- The payload hash is unique so re-running does not re-quarantine the same
-- bad row over and over; this table has to be idempotent too.
CREATE TABLE IF NOT EXISTS quarantine (
    payload_hash TEXT PRIMARY KEY,
    raw_id       TEXT,
    topic        TEXT,
    reason       TEXT NOT NULL,
    payload      TEXT,
    recorded_at  TEXT NOT NULL
);

-- Per-run counters, so "run it twice and show the counts" is evidenced
-- by the database itself rather than by scrollback.
CREATE TABLE IF NOT EXISTS run_log (
    run_id        TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    fetched       INTEGER DEFAULT 0,
    from_cache    INTEGER DEFAULT 0,
    normalised    INTEGER DEFAULT 0,
    quarantined   INTEGER DEFAULT 0,
    inserted      INTEGER DEFAULT 0,
    duplicates    INTEGER DEFAULT 0,
    classified    INTEGER DEFAULT 0
);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO mentions
    (id, topic, source, author, title, text, url, created_at, fetched_at,
     points, num_comments, sentiment, sentiment_score, category)
VALUES
    (:id, :topic, :source, :author, :title, :text, :url, :created_at, :fetched_at,
     :points, :num_comments, :sentiment, :sentiment_score, :category)
"""


@contextmanager
def connect(db_path: Path):
    """Yield a connection with row access by name and foreign keys on."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    logger.debug("Schema ensured.")


def upsert_records(conn: sqlite3.Connection, records: Sequence[Record]) -> tuple[int, int]:
    """Insert records, ignoring ones whose id is already present.

    Returns (inserted, skipped_as_duplicate). We compare total_changes before
    and after rather than trusting rowcount, which is unreliable for
    executemany with OR IGNORE across sqlite3 versions.
    """
    if not records:
        return 0, 0

    before = conn.total_changes
    conn.executemany(INSERT_SQL, [r.as_dict() for r in records])
    inserted = conn.total_changes - before
    duplicates = len(records) - inserted

    logger.info(
        "Store: %d record(s) offered, %d inserted, %d already present.",
        len(records),
        inserted,
        duplicates,
    )
    return inserted, duplicates


def quarantine(
    conn: sqlite3.Connection,
    raw_id: str | None,
    topic: str,
    reason: str,
    payload: str,
    recorded_at: str,
) -> None:
    """Record a row we could not normalise, with the reason why.

    Keyed on a hash of (topic, payload) so repeated runs over the same bad
    input do not grow the table without bound.
    """
    digest = hashlib.sha256(f"{topic}|{payload}".encode("utf-8")).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO quarantine "
        "(payload_hash, raw_id, topic, reason, payload, recorded_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (digest, raw_id, topic, reason, payload[:2000], recorded_at),
    )


def fetch_unclassified(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Rows still missing a label.

    The classifier only ever touches these, which is what makes the classify
    stage incremental: a second run over unchanged data classifies nothing.
    """
    return conn.execute(
        "SELECT id, title, text FROM mentions "
        "WHERE sentiment IS NULL OR category IS NULL"
    ).fetchall()


def apply_labels(conn: sqlite3.Connection, labels: Iterable[tuple[str, float, str, str]]) -> int:
    """Write (sentiment, score, category) back against each row id."""
    payload = list(labels)
    if not payload:
        return 0
    conn.executemany(
        "UPDATE mentions SET sentiment = ?, sentiment_score = ?, category = ? WHERE id = ?",
        payload,
    )
    logger.info("Classifier: labelled %d record(s).", len(payload))
    return len(payload)


def count_rows(conn: sqlite3.Connection, table: str = "mentions") -> int:
    if table not in {"mentions", "quarantine", "run_log"}:
        raise ValueError(f"Refusing to count unknown table {table!r}")
    row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    return int(row["n"])


def start_run(conn: sqlite3.Connection, run_id: str, started_at: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO run_log (run_id, started_at) VALUES (?, ?)",
        (run_id, started_at),
    )


def finish_run(conn: sqlite3.Connection, run_id: str, finished_at: str, stats: dict[str, int]) -> None:
    conn.execute(
        """
        UPDATE run_log SET
            finished_at = :finished_at,
            fetched     = :fetched,
            from_cache  = :from_cache,
            normalised  = :normalised,
            quarantined = :quarantined,
            inserted    = :inserted,
            duplicates  = :duplicates,
            classified  = :classified
        WHERE run_id = :run_id
        """,
        {"run_id": run_id, "finished_at": finished_at, **stats},
    )


def all_records(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM mentions ORDER BY topic, created_at DESC").fetchall()
    return [dict(r) for r in rows]
