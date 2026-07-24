# Classifier Evaluation

Hand-labelled sample: **25 records**, drawn with a fixed seed so the same rows are evaluated on every run.

Labels were assigned by reading each item's text without looking at the classifier's prediction first, to avoid anchoring.

### Sentiment (n=25)

Accuracy: **80.0%** (20/25)

| label | support | precision | recall | F1 |
|---|---|---|---|---|
| negative | 8 | 1.00 | 0.75 | 0.86 |
| neutral | 12 | 0.71 | 1.00 | 0.83 |
| positive | 5 | 1.00 | 0.40 | 0.57 |

**Confusion matrix**

| true \ predicted | negative | neutral | positive |
|---|---|---|---|
| **negative** | 6 | 2 | 0 |
| **neutral** | 0 | 12 | 0 |
| **positive** | 0 | 3 | 2 |

### Category (n=25)

Accuracy: **68.0%** (17/25)

| label | support | precision | recall | F1 |
|---|---|---|---|---|
| business_news | 1 | 0.00 | 0.00 | 0.00 |
| criticism | 3 | 0.50 | 0.33 | 0.40 |
| other | 7 | 0.60 | 0.86 | 0.71 |
| product_launch | 3 | 0.50 | 0.33 | 0.40 |
| technical_discussion | 11 | 0.82 | 0.82 | 0.82 |

**Confusion matrix**

| true \ predicted | business_news | criticism | other | product_launch | technical_discussion |
|---|---|---|---|---|---|
| **business_news** | 0 | 0 | 0 | 1 | 0 |
| **criticism** | 0 | 1 | 1 | 0 | 1 |
| **other** | 0 | 1 | 6 | 0 | 0 |
| **product_launch** | 0 | 0 | 1 | 1 | 1 |
| **technical_discussion** | 0 | 0 | 2 | 0 | 9 |

### Where it disagrees with the human label

| topic | text | predicted | true |
|---|---|---|---|
| Perplexity | Perplexity's new AI tool aims to simplify patent research | neutral | positive |
| ChatGPT | OpenAI used Kenyan workers on less than $2 per hour to make ChatGPT less toxic | neutral | negative |
| Copilot | Continue with LocalAI: An alternative to GitHub's Copilot that runs locally | neutral | negative |
| Claude | Claude AI built me a React app to compare maps side by side | neutral | positive |
| Perplexity | Perplexity AI launching $50M venture fund to back early-stage startups | neutral | positive |
