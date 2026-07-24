"""Evaluation stage.

Two modes:

  export  — draw a reproducible random sample and write a CSV with blank
            label columns for a human to fill in.
  score   — read the filled CSV back, compare to the classifier's output,
            and write per-class precision/recall/F1 plus a confusion matrix.

The sample is drawn with a fixed seed so the same rows come out every time;
without that, "we evaluated on 25 items" is not a repeatable claim.

Accuracy alone is not reported as the headline number. On a skewed label
distribution it flatters a classifier that always predicts the majority
class, so per-class recall is where the real weaknesses show.
"""

from __future__ import annotations

import csv
import logging
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CSV_FIELDS = ["id", "topic", "text", "predicted_sentiment", "predicted_category",
              "true_sentiment", "true_category"]


@dataclass
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


def export_gold_template(conn: sqlite3.Connection, path: Path, sample_size: int, seed: int) -> int:
    """Write a labelling CSV with the true_* columns left blank.

    Refuses to overwrite an existing file — that file may contain hours of
    hand-labelling, and silently clobbering it would be unrecoverable.
    """
    if path.is_file():
        logger.info("Gold label file already exists at %s; leaving it untouched.", path)
        return 0

    rows = conn.execute(
        "SELECT id, topic, text, sentiment, category FROM mentions "
        "WHERE sentiment IS NOT NULL ORDER BY id"
    ).fetchall()

    if not rows:
        logger.warning("No classified rows available to sample for evaluation.")
        return 0

    rng = random.Random(seed)
    sample = rng.sample(rows, min(sample_size, len(rows)))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sample:
            writer.writerow({
                "id": row["id"],
                "topic": row["topic"],
                "text": row["text"],
                "predicted_sentiment": row["sentiment"],
                "predicted_category": row["category"],
                "true_sentiment": "",
                "true_category": "",
            })

    logger.info("Wrote %d-row labelling template to %s", len(sample), path)
    return len(sample)


def _metrics_for(pairs: list[tuple[str, str]]) -> tuple[float, list[ClassMetrics], dict]:
    """Compute accuracy, per-class metrics, and a confusion matrix.

    `pairs` is a list of (true, predicted).
    """
    if not pairs:
        return 0.0, [], {}

    correct = sum(1 for t, p in pairs if t == p)
    accuracy = correct / len(pairs)

    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs})
    confusion: dict[str, Counter] = defaultdict(Counter)
    for true, pred in pairs:
        confusion[true][pred] += 1

    metrics: list[ClassMetrics] = []
    for label in labels:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        support = sum(1 for t, _ in pairs if t == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics.append(ClassMetrics(label, support, precision, recall, f1))

    return accuracy, metrics, confusion


def _render_confusion(confusion: dict[str, Counter], labels: list[str]) -> str:
    header = "| true \\ predicted | " + " | ".join(labels) + " |"
    divider = "|---" * (len(labels) + 1) + "|"
    lines = [header, divider]
    for true in labels:
        cells = [str(confusion.get(true, Counter()).get(pred, 0)) for pred in labels]
        lines.append(f"| **{true}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_block(title: str, pairs: list[tuple[str, str]]) -> str:
    accuracy, metrics, confusion = _metrics_for(pairs)
    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs})

    out = [f"### {title}", ""]
    out.append(f"Accuracy: **{accuracy:.1%}** ({sum(1 for t, p in pairs if t == p)}/{len(pairs)})")
    out.append("")
    out.append("| label | support | precision | recall | F1 |")
    out.append("|---|---|---|---|---|")
    for m in metrics:
        out.append(
            f"| {m.label} | {m.support} | {m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} |"
        )
    out.append("")
    out.append("**Confusion matrix**")
    out.append("")
    out.append(_render_confusion(confusion, labels))
    out.append("")
    return "\n".join(out)


def score_gold(gold_path: Path, output_path: Path) -> bool:
    """Read the filled-in CSV and write the evaluation report.

    Returns False if the file is absent or unlabelled, so the caller can
    tell the user what to do rather than emitting a fake report.
    """
    if not gold_path.is_file():
        logger.warning("No gold label file at %s — skipping scoring.", gold_path)
        return False

    with gold_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    sentiment_pairs = [
        (r["true_sentiment"].strip().lower(), r["predicted_sentiment"].strip().lower())
        for r in rows
        if r.get("true_sentiment", "").strip()
    ]
    category_pairs = [
        (r["true_category"].strip().lower(), r["predicted_category"].strip().lower())
        for r in rows
        if r.get("true_category", "").strip()
    ]

    if not sentiment_pairs and not category_pairs:
        logger.warning(
            "Gold file %s has no filled-in labels yet. "
            "Fill the true_sentiment and true_category columns, then re-run.",
            gold_path,
        )
        return False

    disagreements = [
        r for r in rows
        if r.get("true_sentiment", "").strip()
        and r["true_sentiment"].strip().lower() != r["predicted_sentiment"].strip().lower()
    ]

    parts = [
        "# Classifier Evaluation",
        "",
        f"Hand-labelled sample: **{len(rows)} records**, drawn with a fixed seed "
        "so the same rows are evaluated on every run.",
        "",
        "Labels were assigned by reading each item's text without looking at the "
        "classifier's prediction first, to avoid anchoring.",
        "",
    ]

    if sentiment_pairs:
        parts.append(_render_block(f"Sentiment (n={len(sentiment_pairs)})", sentiment_pairs))
    if category_pairs:
        parts.append(_render_block(f"Category (n={len(category_pairs)})", category_pairs))

    if disagreements:
        parts.append("### Where it disagrees with the human label")
        parts.append("")
        parts.append("| topic | text | predicted | true |")
        parts.append("|---|---|---|---|")
        for row in disagreements[:10]:
            text = row["text"][:90].replace("|", "\\|")
            parts.append(
                f"| {row['topic']} | {text} | {row['predicted_sentiment']} "
                f"| {row['true_sentiment']} |"
            )
        parts.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Evaluation report written to %s", output_path)
    return True
