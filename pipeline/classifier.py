"""Classification stage.

Two labels per record.

Sentiment — VADER (Hutto & Gilbert, 2014), a lexicon-and-rules model. Chosen
over a trained classifier because it is deterministic, needs no training data,
and every score is traceable to specific words in the input. The +/-0.05
compound-score thresholds are the authors' own recommendation, not values we
tuned to make our numbers look better.

Category — ordered keyword rules over a five-label taxonomy. First match wins,
so the list is ordered most-specific first. This is deliberately simple and
deliberately transparent: for any label, we can point at the exact term that
produced it, which is not true of an embedding-based approach. Its weaknesses
are real and documented in the README and evaluation report.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import ClassificationConfig

logger = logging.getLogger(__name__)

# Ordered rules. The first pattern that matches assigns the category, so
# more specific categories must appear before broader ones. Patterns use
# word boundaries to avoid substring false positives ("ban" in "bandwidth").
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    (
        "product_launch",
        [
            r"\blaunch(?:e[sd]|ing)?\b", r"\brelease[sd]?\b", r"\breleasing\b",
            r"\bannounc(?:e[sd]?|ing|ement)\b", r"\bintroduc(?:e[sd]?|ing)\b",
            r"\bnow available\b", r"\bgeneral availability\b", r"\bunveil(?:s|ed)?\b",
            r"\bship(?:s|ped|ping)\b", r"\bnew (?:model|version|feature|app)\b",
            r"\bv?\d+\.\d+\b", r"\bbeta\b", r"\bpreview\b", r"\bupdate[sd]?\b",
        ],
    ),
    (
        "business_news",
        [
            r"\bfunding\b", r"\braise[sd]?\b", r"\bvaluation\b", r"\bipo\b",
            r"\bacqui(?:re[sd]?|sition)\b", r"\bmerger\b", r"\brevenue\b",
            r"\bearnings\b", r"\bprofit(?:s|able)?\b", r"\blayoffs?\b",
            r"\bhiring\b", r"\bceo\b", r"\bpartnership\b", r"\bdeal\b",
            r"\bmarket share\b", r"\bbillion\b", r"\bmillion\b", r"\bstartup\b",
            r"\bsubscription\b", r"\bpricing\b", r"\bprice hike\b",
        ],
    ),
    (
        "criticism",
        [
            r"\bproblem[s]?\b", r"\bissue[s]?\b", r"\bbug[s]?\b", r"\bbroken\b",
            r"\bfail(?:s|ed|ure|ing)?\b", r"\bworse\b", r"\bworst\b",
            r"\bdisappoint(?:ed|ing|ment)?\b", r"\bcomplain(?:t|ts|ing)?\b",
            r"\bconcern(?:s|ed)?\b", r"\bprivacy\b", r"\blawsuit\b", r"\bsued?\b",
            r"\bban(?:s|ned|ning)?\b", r"\bcontrovers(?:y|ial)\b", r"\bscandal\b",
            r"\bhallucinat(?:e[sd]?|ion|ions|ing)\b", r"\bmisinformation\b",
            r"\bwrong\b", r"\bunreliable\b", r"\boverrated\b", r"\bdeclin(?:e[sd]?|ing)\b",
        ],
    ),
    (
        "technical_discussion",
        [
            r"\bapi\b", r"\bbenchmark(?:s|ing)?\b", r"\bmodel\b", r"\btoken(?:s)?\b",
            r"\bcontext window\b", r"\blatency\b", r"\bperformance\b",
            r"\barchitecture\b", r"\bimplementation\b", r"\bopen[- ]source\b",
            r"\bfine[- ]tun(?:e[sd]?|ing)\b", r"\bprompt(?:s|ing)?\b",
            r"\bembedding[s]?\b", r"\binference\b", r"\btraining\b",
            r"\bhow (?:to|i|we)\b", r"\bbuilt?\b", r"\busing\b", r"\bcode\b",
            r"\bself[- ]host(?:ed|ing)?\b", r"\bgithub\b", r"\bcli\b",
        ],
    ),
]


@dataclass(frozen=True)
class Label:
    sentiment: str
    score: float
    category: str
    # The keyword that produced the category, kept for error analysis.
    matched_term: str | None


class Classifier:
    """Deterministic, explainable labelling."""

    def __init__(self, cfg: ClassificationConfig) -> None:
        self._cfg = cfg
        self._analyzer = SentimentIntensityAnalyzer()
        # Compile once; this runs across several hundred records.
        self._compiled: list[tuple[str, list[re.Pattern[str]]]] = [
            (category, [re.compile(p, re.IGNORECASE) for p in patterns])
            for category, patterns in CATEGORY_RULES
            if category in cfg.categories
        ]
        logger.debug("Classifier ready with %d category rule set(s).", len(self._compiled))

    def sentiment(self, text: str) -> tuple[str, float]:
        """Map VADER's compound score onto three buckets.

        The compound score is a normalised sum of lexicon valences in
        [-1, 1]. We use the authors' recommended +/-0.05 cutoffs.
        """
        if not text or not text.strip():
            return "neutral", 0.0
        score = float(self._analyzer.polarity_scores(text)["compound"])
        if score >= self._cfg.positive_threshold:
            return "positive", score
        if score <= self._cfg.negative_threshold:
            return "negative", score
        return "neutral", score

    def category(self, text: str) -> tuple[str, str | None]:
        """First matching rule wins; `other` if nothing matches."""
        if not text or not text.strip():
            return "other", None
        for name, patterns in self._compiled:
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    return name, match.group(0)
        return "other", None

    def classify(self, text: str) -> Label:
        sentiment, score = self.sentiment(text)
        category, term = self.category(text)
        return Label(sentiment=sentiment, score=score, category=category, matched_term=term)
