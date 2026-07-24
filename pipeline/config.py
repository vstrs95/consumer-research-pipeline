"""Configuration loading.

The whole pipeline is driven by config.yaml. Nothing that a user might
reasonably want to change should be a literal anywhere else in the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Topic:
    """A subject we track.

    `name` is the label shown in output; `query` is the string sent to the
    search API. They differ when a brand name is ambiguous as a search term
    (e.g. "Claude" also matches Claude Shannon and Claude Monet).
    """

    name: str
    query: str


@dataclass(frozen=True)
class SourceConfig:
    name: str
    base_url: str
    user_agent: str
    rate_limit_seconds: float
    max_retries: int
    backoff_base: float
    hits_per_page: int
    max_items_per_topic: int
    tags: str


@dataclass(frozen=True)
class StorageConfig:
    raw_dir: Path
    db_path: Path


@dataclass(frozen=True)
class ClassificationConfig:
    positive_threshold: float
    negative_threshold: float
    categories: list[str]


@dataclass(frozen=True)
class ReportingConfig:
    summary_path: Path
    evaluation_path: Path
    gold_labels_path: Path
    eval_sample_size: int
    eval_seed: int
    top_items_per_topic: int


@dataclass(frozen=True)
class Config:
    source: SourceConfig
    topics: list[Topic]
    storage: StorageConfig
    classification: ClassificationConfig
    reporting: ReportingConfig
    log_level: str = "INFO"
    project_root: Path = field(default_factory=Path.cwd)


class ConfigError(Exception):
    """Raised when config.yaml is missing, malformed, or internally inconsistent."""


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read and validate config.yaml.

    Fails loudly on a bad config rather than silently falling back to
    defaults — a pipeline that runs with the wrong settings is worse than
    one that refuses to start.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path.resolve()}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level.")

    root = config_path.resolve().parent

    for section in ("source", "topics", "storage", "classification", "reporting"):
        if section not in raw:
            raise ConfigError(f"Missing required config section: '{section}'")

    src = raw["source"]
    source = SourceConfig(
        name=str(src["name"]),
        base_url=str(src["base_url"]),
        user_agent=str(src["user_agent"]),
        rate_limit_seconds=float(src.get("rate_limit_seconds", 1.0)),
        max_retries=int(src.get("max_retries", 3)),
        backoff_base=float(src.get("backoff_base", 2.0)),
        hits_per_page=int(src.get("hits_per_page", 50)),
        max_items_per_topic=int(src.get("max_items_per_topic", 100)),
        tags=str(src.get("tags", "story")),
    )

    topics_raw = raw["topics"]
    if not topics_raw:
        raise ConfigError("At least one topic must be configured.")
    topics: list[Topic] = []
    for entry in topics_raw:
        if isinstance(entry, str):
            topics.append(Topic(name=entry, query=entry))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                raise ConfigError(f"Topic entry missing 'name': {entry}")
            topics.append(Topic(name=str(name), query=str(entry.get("query", name))))
        else:
            raise ConfigError(f"Unrecognised topic entry: {entry!r}")

    names = [t.name for t in topics]
    if len(names) != len(set(names)):
        raise ConfigError(f"Duplicate topic names in config: {names}")

    sto = raw["storage"]
    storage = StorageConfig(
        raw_dir=root / str(sto.get("raw_dir", "data/raw")),
        db_path=root / str(sto.get("db_path", "data/mentions.db")),
    )

    cls = raw["classification"]
    thresholds = cls.get("sentiment_thresholds", {})
    categories = list(cls.get("categories", []))
    if "other" not in categories:
        # `other` is the fallback label; the classifier depends on it existing.
        categories.append("other")
    classification = ClassificationConfig(
        positive_threshold=float(thresholds.get("positive", 0.05)),
        negative_threshold=float(thresholds.get("negative", -0.05)),
        categories=categories,
    )

    rep = raw["reporting"]
    reporting = ReportingConfig(
        summary_path=root / str(rep.get("summary_path", "reports/summary.md")),
        evaluation_path=root / str(rep.get("evaluation_path", "reports/evaluation.md")),
        gold_labels_path=root / str(rep.get("gold_labels_path", "reports/gold_labels.csv")),
        eval_sample_size=int(rep.get("eval_sample_size", 25)),
        eval_seed=int(rep.get("eval_seed", 42)),
        top_items_per_topic=int(rep.get("top_items_per_topic", 3)),
    )

    log_level = str(raw.get("logging", {}).get("level", "INFO")).upper()

    return Config(
        source=source,
        topics=topics,
        storage=storage,
        classification=classification,
        reporting=reporting,
        log_level=log_level,
        project_root=root,
    )
