# Consumer Research Pipeline

Crawls public mentions of five AI assistant products from Hacker News, lands them
in SQLite, labels each record with sentiment and category, and emits a summary an
analyst can read in about thirty seconds.

**Time spent: ~4 hours.**

---

## Setup and run

```bash
git clone <repo-url> && cd consumer-research-pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pipeline
```

That single command runs all four stages. `make run` does the same.

Useful flags:

| Command | Effect |
|---|---|
| `python -m pipeline` | Full run: crawl, transform, classify, report |
| `python -m pipeline --offline` | Replay cached payloads, no network at all |
| `python -m pipeline --log-level DEBUG` | Verbose logging |
| `make test` | Unit tests |
| `make clean` | Delete the store and raw cache |

Outputs land in `data/mentions.db`, `reports/summary.md`, and `reports/evaluation.md`.

---

## Topics

| Topic | Search query | Why the query differs |
|---|---|---|
| ChatGPT | `ChatGPT` | Unambiguous |
| Claude | `Claude AI` | Bare "Claude" matches Claude Shannon and Claude Monet |
| Gemini | `Gemini AI` | Bare "Gemini" matches the Gemini protocol and the crypto exchange |
| Copilot | `GitHub Copilot` | Disambiguates from Microsoft 365 Copilot and autopilot usages |
| Perplexity | `Perplexity AI` | "Perplexity" is also a standard NLP evaluation metric |

Separating the display name from the search string is the main data-quality
decision in the crawl stage. Without it, roughly a fifth of the Gemini results
are about a text protocol rather than a product.

---

## Source and its terms

Hacker News via the [Algolia Search API](https://hn.algolia.com/api). No auth, no
paywall, no login. Chosen because it is the spec's recommended default, returns
clean JSON with real pagination, and publishes generous rate limits (~10,000
requests/hour) that this pipeline stays three orders of magnitude below.

`robots.txt` for `hn.algolia.com` is fetched and checked before the first request.
If it disallowed us, the crawl would refuse to start rather than proceed.

**Sampling limitation, stated plainly:** Hacker News is a technically sophisticated,
English-speaking, largely US-based audience. What this pipeline measures is
*developer sentiment* about AI products, not general-population sentiment. Any
analyst reading `reports/summary.md` should treat it as one segment's opinion.

---

## Architecture

```
config.yaml
    │
    ├─► crawler.py    → data/raw/<topic>/page_NNN.json   (raw payloads, cached)
    │
    ├─► schema.py     → Record dataclass                  (normalise + validate)
    │
    ├─► store.py      → data/mentions.db                  (SQLite, id primary key)
    │
    ├─► classifier.py → sentiment + category              (labels written back)
    │
    └─► summarise.py  → reports/summary.md
        evaluate.py   → reports/evaluation.md
```

Each stage reads from and writes to disk. The crawler never parses; the classifier
never fetches. That separation is what makes a parser fix cost a re-parse rather
than a re-crawl.

### Schema

`id, topic, source, author, title, text, url, created_at, fetched_at, points,
num_comments, sentiment, sentiment_score, category`

`id` is HN's `objectID` — the source's own stable identifier. Using it rather than
a content hash means a story keeps its key even if its title is edited later,
which is what makes dedupe correct across runs rather than merely consistent.

---

## Design decisions

**Idempotency has three layers, not one.** The database uses `INSERT OR IGNORE` on
a primary key, so duplicate rows are impossible. The crawler checks for a cached
page file before making any request, so a second run does no network I/O. The
classifier queries only for rows where `sentiment IS NULL`, so labelled records are
never re-scored. Any one of these alone would satisfy the letter of the requirement;
together they make the second run genuinely cheap rather than merely non-destructive.
The `quarantine` table is keyed on a payload hash for the same reason — an
append-only error log that grows on every run is its own kind of bug.

**Raw payloads are written before anything is parsed.** A parser bug then costs a
re-parse rather than another round of traffic to someone else's servers.

**Bad rows are quarantined, not dropped.** Rows that fail validation land in a
`quarantine` table with the reason and the original payload. Silent drops hide
bugs; a countable reject rate surfaces them. The committed dataset has ten such
rows, all missing either an `objectID` or any usable text.

**Per-run counters live in the database.** The `run_log` table records what each
run fetched, inserted, and skipped, so "run it twice and show the counts" is
evidenced by a queryable table rather than terminal scrollback. The last five runs
appear at the bottom of `reports/summary.md`.

**A failed topic does not abort the run.** If one topic's request fails after
exhausting retries, it is logged and the others continue. Partial data beats none.

---

## The classifier

### Sentiment — VADER

[VADER](https://github.com/cjhutto/vaderSentiment) (Hutto & Gilbert, 2014) is a
lexicon-and-rules model. Chosen over a trained classifier because it is
deterministic, needs no training data, runs in milliseconds, and every score
traces back to specific words in the input — for any label, the reason is
inspectable.

Bucketing uses the compound score at **±0.05**, which is the threshold the VADER
authors themselves recommend. It is not a value tuned to make these numbers look
better, which matters: a threshold chosen after seeing the evaluation set would
make the evaluation meaningless.

### Category — ordered keyword rules

Five labels: `product_launch`, `business_news`, `criticism`,
`technical_discussion`, `other`.

Rules are regex patterns with word boundaries, evaluated in the order listed in
`CATEGORY_RULES`. First match wins, so more specific categories precede broader
ones. `other` is the fallback when nothing matches.

Word boundaries matter more than they look: without `\b`, "ban" matches inside
"bandwidth" and quietly labels networking posts as criticism. There is a test for
exactly this.

This approach is deliberately simple and deliberately transparent. Its weaknesses
are real and documented below rather than hidden.

---

## Evaluation

25 records, sampled with a fixed seed (`eval_seed: 42`) so the same rows are
evaluated on every run. Labels were assigned by reading each item's text
**before** looking at the classifier's prediction, to avoid anchoring.

Full results, including per-class precision/recall/F1 and confusion matrices, are
in [`reports/evaluation.md`](reports/evaluation.md).

Accuracy is reported but is not the headline number. On a skewed label
distribution it flatters a classifier that always guesses the majority class —
per-class recall is where the real failures show.

### Where it fails, and why

**Every sentiment error is an under-call to `neutral` — the model never flips
polarity.** All 5 sentiment misses (25 − 20) land in the `neutral` column: 2 of
the 8 true `negative` rows (recall 0.75) and 3 of the 5 true `positive` rows
(recall 0.40) were predicted `neutral` (sentiment confusion matrix). Not one row
was predicted `negative` when true `positive`, or `positive` when true
`negative`. This concentration is also why `neutral`'s precision (0.71) is worse
than its recall (1.00): of the 17 rows predicted `neutral`, only 12 actually are
— the other 5 are every other class's errors, absorbed into the same bucket.

**The mechanism is a lexicon coverage gap, not a threshold problem.** VADER only
scores sentiment carried by specific words in its lexicon; a headline that reads
as positive or negative to a human but avoids that vocabulary nets to zero and
falls inside the ±0.05 neutral band. On the positive side: "Perplexity's new AI
tool aims to simplify patent research" and "Claude AI built me a React app to
compare maps side by side" are both product-positive headlines predicted
`neutral`. On the negative side: "OpenAI used Kenyan workers on less than $2 per
hour to make ChatGPT less toxic" and "Continue with LocalAI: An alternative to
GitHub's Copilot that runs locally" are both true `negative` but scored
`neutral` — nothing in either sentence is lexicon-negative even though a human
reads both as clearly critical. No amount of retuning ±0.05 fixes this; the
words that would carry the signal simply aren't in VADER's vocabulary.

**Category confusion concentrates in the two low-support classes, not the two
high-volume ones.** `technical_discussion` (n=11) is the strongest class — 9/11
correct, with 2 leaking to `other`. `other` (n=7) does well too, at 6/7. The two
weak spots are `criticism` (n=3, only 1 correct — one row misread as `other`, one
as `technical_discussion`) and `product_launch` (n=3, only 1 correct — split
across `other` and `technical_discussion`). At three examples per class, a single
misfire swings recall by 33 points, so this says "watch these classes as the eval
set grows" more than "the rule ordering is broken."

---

## Tradeoffs made under time pressure

**Titles only, not comment threads.** The crawl requests `tags=story`, so what gets
classified is the headline, not the discussion under it. Comments are where the
real consumer sentiment lives, but they would have multiplied the crawl size and
required a different unit of analysis. This is the biggest scope cut and the
first thing I would revisit.

**Keyword rules over a trained model.** The spec called for a transparent method
and ruled out training a model. Even without that constraint, rules are the right
call at this scale — a trained classifier would need labelled data I do not have,
and would be harder to defend line by line.

**No caching layer for classification.** It is fast enough locally that skipping
already-labelled rows is sufficient. An LLM-based classifier would need real
content-hash caching.

**No async or concurrency.** Five topics at one request per second is under thirty
seconds of wall time. Concurrency would add failure modes for no practical gain,
and would make the throttling harder to reason about.

---

## With more time

1. **Crawl comment threads**, not just story titles — the actual consumer voice.
2. **Handle negation and contrast** before sentiment scoring; splitting on "but"
   and scoring clauses separately would fix the largest observed error class.
3. **Expand the gold set to 100+ records** and label two annotators independently
   to get an agreement figure. 25 records gives error bars too wide to draw
   confident conclusions from.
4. **Track sentiment over time** — `created_at` is stored but unused. Sentiment
   trend per topic per month is the obvious next analyst question.
5. **Extend the quarantine review loop** into something that reports reject
   reasons by frequency, so schema drift at the source surfaces early.

---

## Repository layout

```
config.yaml              all tunable settings
pipeline/
  __main__.py            entrypoint for `python -m pipeline`
  main.py                orchestration
  config.py              config loading and validation
  crawler.py             polite fetching, pagination, retries, raw cache
  schema.py              Record dataclass, normalisation, validation
  store.py               SQLite, idempotent upserts, quarantine, run log
  classifier.py          VADER + keyword rules
  evaluate.py            gold-set export and scoring
  summarise.py           Markdown report generation
tests/test_pipeline.py   unit tests (no network required)
data/raw/                cached raw payloads (committed)
data/mentions.db         the store (committed)
reports/                 summary, evaluation, gold labels (committed)
```

---

## Proving idempotency

```
$ python -m pipeline
  rows before ......... 0
  rows after .......... 290
  newly inserted ...... 290
  duplicates ignored .. 0
  newly classified .... 290

$ python -m pipeline
  rows before ......... 290
  rows after .......... 290
  newly inserted ...... 0
  duplicates ignored .. 290
  newly classified .... 0
```

The second run makes no network requests, inserts no rows, and classifies nothing.
The same figures are recorded in the `run_log` table and rendered in
`reports/summary.md`.
