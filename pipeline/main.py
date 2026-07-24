"""Orchestrator: crawl -> transform -> classify -> report.

Each stage reads from and writes to disk, so any stage can be re-run on its
own. `python -m pipeline` runs all four in order.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone

from . import store
from .classifier import Classifier
from .config import Config, ConfigError, load_config
from .crawler import CrawlError, crawl
from .evaluate import export_gold_template, score_gold
from .schema import ValidationError, normalise_hit
from .summarise import write_summary

logger = logging.getLogger("pipeline")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(cfg: Config, skip_crawl: bool = False) -> int:
    """Execute the full pipeline. Returns a process exit code."""
    run_id = uuid.uuid4().hex[:12]
    started = _now()
    stats = {
        "fetched": 0, "from_cache": 0, "normalised": 0,
        "quarantined": 0, "inserted": 0, "duplicates": 0, "classified": 0,
    }

    logger.info("=" * 70)
    logger.info("Run %s starting at %s", run_id, started)
    logger.info("Topics: %s", ", ".join(t.name for t in cfg.topics))
    logger.info("=" * 70)

    with store.connect(cfg.storage.db_path) as conn:
        store.init_db(conn)
        store.start_run(conn, run_id, started)
        rows_before = store.count_rows(conn)
        logger.info("Store holds %d record(s) before this run.", rows_before)

        # ---- Stage 1: crawl -------------------------------------------------
        logger.info("[1/4] Crawl%s", " (offline)" if skip_crawl else "")
        try:
            hits_by_topic, fetched, cached = crawl(cfg, offline=skip_crawl)
            stats["fetched"] = fetched
            stats["from_cache"] = cached
        except CrawlError as exc:
            logger.error("Crawl aborted: %s", exc)
            return 1

        # ---- Stage 2: normalise + load --------------------------------------
        logger.info("[2/4] Transform and load")
        records = []
        for topic_name, hits in hits_by_topic.items():
            for hit in hits:
                try:
                    records.append(
                        normalise_hit(hit, topic_name, cfg.source.name, _now())
                    )
                except ValidationError as exc:
                    stats["quarantined"] += 1
                    store.quarantine(
                        conn,
                        raw_id=str(hit.get("objectID")) if isinstance(hit, dict) else None,
                        topic=topic_name,
                        reason=str(exc),
                        payload=json.dumps(hit, ensure_ascii=False)[:2000],
                        recorded_at=_now(),
                    )

        stats["normalised"] = len(records)
        # Dedupe within the batch before hitting the DB — the same story can
        # legitimately match two different topic queries.
        unique = {r.id: r for r in records}
        if len(unique) < len(records):
            logger.info(
                "In-batch dedupe: %d record(s) collapsed to %d unique id(s).",
                len(records), len(unique),
            )

        inserted, duplicates = store.upsert_records(conn, list(unique.values()))
        stats["inserted"] = inserted
        stats["duplicates"] = duplicates

        # ---- Stage 3: classify ----------------------------------------------
        logger.info("[3/4] Classify")
        pending = store.fetch_unclassified(conn)
        if pending:
            classifier = Classifier(cfg.classification)
            labels = []
            for row in pending:
                text = row["title"] or row["text"] or ""
                label = classifier.classify(text)
                labels.append((label.sentiment, label.score, label.category, row["id"]))
            stats["classified"] = store.apply_labels(conn, labels)
        else:
            logger.info("No unlabelled records — classify stage is a no-op.")

        # ---- Stage 4: report -------------------------------------------------
        logger.info("[4/4] Summarise and evaluate")
        store.finish_run(conn, run_id, _now(), stats)

        write_summary(conn, cfg.reporting.summary_path, cfg.reporting.top_items_per_topic)

        export_gold_template(
            conn,
            cfg.reporting.gold_labels_path,
            cfg.reporting.eval_sample_size,
            cfg.reporting.eval_seed,
        )
        scored = score_gold(cfg.reporting.gold_labels_path, cfg.reporting.evaluation_path)
        if not scored:
            logger.warning(
                "Evaluation pending: fill true_sentiment / true_category in %s, then re-run.",
                cfg.reporting.gold_labels_path,
            )

        rows_after = store.count_rows(conn)

    logger.info("=" * 70)
    logger.info("Run %s complete.", run_id)
    logger.info("  rows before ......... %d", rows_before)
    logger.info("  rows after .......... %d", rows_after)
    logger.info("  newly inserted ...... %d", stats["inserted"])
    logger.info("  duplicates ignored .. %d", stats["duplicates"])
    logger.info("  quarantined ......... %d", stats["quarantined"])
    logger.info("  newly classified .... %d", stats["classified"])
    logger.info("=" * 70)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Crawl, normalise, classify and summarise public mentions.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument(
        "--offline",
        "--crawl-skip",
        dest="offline",
        action="store_true",
        help="Replay cached raw payloads without making any network request.",
    )
    parser.add_argument("--log-level", default=None, help="Override the configured log level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(args.log_level or cfg.log_level)
    try:
        return run(cfg, skip_crawl=args.offline)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
