# CineRankX

Movie Recommendation System — Mini Project (AY 2026-27)
Department of Artificial Intelligence and Machine Learning, Walchand College of Engineering, Sangli

> "With great data comes great recommendations."

## Overview

Existing recommender systems fix one strategy and rank by a single similarity score,
ignoring diversity and per-user variation. **CineRankX** predicts the best strategy per
user via ML classifiers and ranks candidates on six weighted objectives, for relevant,
diverse, explainable recommendations.

The system is built as three engines, in this order:

1. **Classical IR Engine** — TF-IDF vector space model over merged movie text
   (overview + genres + keywords), with cosine similarity search and a
   Trie for title autocomplete.
2. **Adaptive ML Strategy Selection Engine** — a classifier that predicts, per user,
   which recommendation strategy (content-based, collaborative filtering, or popularity
   baseline) fits them best, trained on backtested per-user hit-rates.
3. **Multi-Objective Hybrid Ranking Engine** — re-ranks candidates on six weighted
   objectives (relevance, diversity, novelty, coverage, popularity/quality, predicted
   rating), with weights and a genre-overlap relevance gate tunable via
   `eval_config.yaml`.

An **evaluation suite** (`src/evaluation/`) ties the three engines together, computing
every KPI in `eval_config.yaml` and rendering them as one results table.

## Datasets

- **Primary — [MovieLens 20M](https://grouplens.org/datasets/movielens/)**:
  ~20M ratings, ~27,000 movies (`rating.csv`, `movie.csv`, `link.csv`, `tag.csv`).
- **Secondary — [TMDB Movies Dataset 2023](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)**:
  ~930K movies with overview, keywords, genres (no cast/crew metadata).
- Joined via `link.csv` (`movieId → tmdbId`), with ~98% join rate between MovieLens and TMDB.

**Dataset location:** Downloaded files are stored in `newdata/` locally (gitignored — not committed to the repo).
See [`DATASET_MIGRATION.md`](./DATASET_MIGRATION.md) for dataset rationale and migration notes.

## Tech stack

Python, VS Code, Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Claude Code (build assistant).

## Project structure

```
newdata/                <- downloaded datasets (MovieLens 20M + TMDB 930K, gitignored)
data/processed/         <- generated artifacts: merged data, TF-IDF matrix, trained models (gitignored)
src/ingestion/          <- loading, joining, text cleaning (Module 1)
src/ir_engine/          <- TF-IDF + cosine similarity + Trie autocomplete (Module 2)
src/strategy_selector/  <- per-user strategy classifier (Module 3)
src/ranking_engine/     <- multi-objective hybrid ranking (Module 4)
src/evaluation/         <- metrics (search, recommendation, ML, system)
scripts/                <- per-module verification scripts (run with PYTHONPATH=.)
tests/                  <- pytest unit tests, one file per module
notebooks/              <- exploratory analysis
.gitignore              <- excludes datasets and processed artifacts
CLAUDE.md               <- project context/conventions for Claude Code
DATASET_MIGRATION.md    <- dataset rationale and migration notes
eval_config.yaml        <- exact KPI names + tunable ranking objective weights
requirements.txt
```

## Setup

1. **Create virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download datasets** (MovieLens 20M and TMDB 2023 930K) and place them under `newdata/`:
   - `newdata/movielens/` — `rating.csv`, `movie.csv`, `link.csv`, `tag.csv`
   - `newdata/tmdb/` — `TMDB_movie_dataset_v11.csv`

3. **Run the ingestion pipeline** to merge, clean, and cache data:
   ```bash
   python -m src.ingestion.build_dataset
   ```
   This generates `data/processed/movies_merged.csv` and the TF-IDF matrix (cached).

## Status

- [x] **Module 1 — Data Ingestion & Preprocessing**: loaders, join, NLTK cleaning
  pipeline. 6/6 unit tests passing. MovieLens 20M (~27K movies) → ~26.5K successfully joined
  to TMDB metadata (98.04% join rate); outputs cached to `data/processed/`.
- [x] **Module 2 — Classical IR Engine (TF-IDF + Trie)**: TF-IDF vector space model over
  `text_chunk_clean`, cosine-similarity search, movie-to-movie similarity, and Trie-based
  title autocomplete. 18/18 unit tests passing.
- [x] **Module 3 — Adaptive ML Strategy Selection Engine**: per-user profile features,
  backtest-driven strategy labeling, and a RandomForest classifier predicting
  content-based / collaborative / popularity per user. 41/41 unit tests passing.
  Partially effective — see [Module 3 results](#module-3-results) below.
- [x] **Module 4 — Multi-Objective Hybrid Ranking Engine**: blends IR and
  strategy-selected candidates, then re-ranks them on six weighted objectives with a
  genre-overlap relevance gate. 72/72 unit tests passing.
  See [Module 4 results](#module-4-results) below.
- [x] **Evaluation suite**: all 23 KPIs from `eval_config.yaml` across four groups
  (search, recommendation, machine learning, system), keyed by the exact metric-name
  strings so report and code cannot drift. 71/71 unit tests passing.
  See [Final KPI report](#final-kpi-report) below.

**Test suite:** 210/210 passing (`python -m pytest tests/ -q`) — 6 ingestion, 18 IR
engine, 43 strategy selector, 72 ranking engine, 71 evaluation.

## Running the engines

All verification scripts need the repo root on the import path:

```bash
# Module 2 — search, autocomplete, movie-to-movie similarity
PYTHONPATH=. python scripts/verify_ir_engine.py

# Module 3 — profile features, strategy labeling, classifier training + metrics
# Optional arg = number of users to sample (default 100)
PYTHONPATH=. python scripts/verify_strategy_selector.py 5000

# Module 4 — candidate blending, six-objective ranking, weight sensitivity,
# the genre-relevance gate, and the cold-start path
# Optional arg = number of ratings to load (default 1,000,000)
PYTHONPATH=. python scripts/verify_ranking_engine.py
```

The Module 3 script trains and saves the classifier to
`data/processed/strategy_classifier.pkl`. Module 4 loads the Module 2 and Module 3
artifacts from disk rather than refitting them, so run those two first.

Then generate the full KPI report:

```bash
# Every metric in eval_config.yaml, from real data, in one table
# Optional args = ratings to load (default 400,000), users to evaluate (default 300)
PYTHONPATH=. python scripts/generate_kpi_report.py 900000 1500
```

It refuses to finish if any KPI defined in `eval_config.yaml` is missing from the output.
Redirect it into `reports/` (gitignored) to keep a copy.

The SMOTE investigation behind Module 3's documented limitations can be reproduced with
`scripts/compare_smote.py`, `scripts/smote_seed_robustness.py` and
`scripts/smote_vs_class_weight.py`.

## Module 3 results

Measured on a 5,000-user sample (750,512 ratings), 5-split label averaging,
1,000-user held-out test set, **after** the popularity-source correction (see below):

| Metric | Value |
|---|---|
| Accuracy | 0.6600 |
| Majority-class baseline | 0.5160 |
| **Lift over baseline** | **+0.1440** |
| ROC-AUC | 0.6841 |
| F1 (weighted) | 0.6513 |

Per-class:

| Strategy | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `collaborative` | 516 | 0.6796 | 0.7112 | 0.6951 |
| `popularity` | 453 | 0.6454 | 0.6468 | **0.6461** |
| `content_based` | 31 | 0.0000 | 0.0000 | **0.0000** |

Label distribution: `collaborative` 51.6%, `popularity` 45.3%, `content_based` 3.1%.

### The popularity-source correction

Both Module 3's backtest and Module 4's candidate generator originally ranked popularity
by TMDB's `popularity` column. TMDB popularity is a live, recency-weighted metric while
MovieLens 20M's ratings stop in 2015, so the two disagree almost completely — the TMDB
top-10 was touched by only **39%** of users and covered **0.39%** of ratings, against
**87%** and **2.87%** for the most-rated top-10. Both now rank by observed rating counts.

| Metric | Before | After |
|---|---|---|
| Accuracy | 0.7960 | 0.6600 |
| Majority-class baseline | 0.8410 | 0.5160 |
| **Lift over baseline** | **−0.0450** | **+0.1440** |
| ROC-AUC | 0.6421 | 0.6841 |
| `popularity` F1 | ~0.1103 | **0.6461** |
| `content_based` F1 | 0.1212 | **0.0000** |

Read honestly: raw accuracy *fell*, but the old 0.796 was mostly majority-class mass on
an 84%-imbalanced problem. The balance is now 52/45/3, so the baseline itself drops to
0.516 and the classifier clears it by **+0.144** — the first time in this project it has
beaten "always guess the majority." ROC-AUC confirms the improvement independently.

`popularity` improved 5.9× because the backtest previously scored it against a top-10
most users never touched, so `pop_score` was 0.0 for the majority and the label was
assigned largely by a tie-break rather than by winning on merit.

**Disclosed regression:** `content_based` fell to F1 0.0000 at 3.1% prevalence — the
classifier now predicts it essentially never. Partly honest reallocation (users labeled
`content_based` only because `pop_score` was artificially 0 now correctly resolve to
`popularity`), but the class is now below learnability at this data scale. Net effect:
two of three classes are genuinely predicted where one was before.

This remains **partially effective personalization**, not a solved classification
problem. Module 4 therefore treats the predicted strategy as a weighted prior rather than
a hard routing decision. Full write-up, including the four remedies tried and rejected,
is in [`DATASET_MIGRATION.md`](./DATASET_MIGRATION.md#known-limitations--module-3-adaptive-ml-strategy-selection-engine).

SMOTE oversampling was investigated as a fourth remedy and **ships disabled**
(`fit(..., use_smote=False)`). Across 10 seeds it lifted `content_based` F1 by +0.038,
but ROC-AUC stayed flat (0.605–0.627) across SMOTE *and* every `class_weight` variant —
rebalancing shifts the decision threshold without improving the model's probability
ranking. Since Module 4 consumes the classifier's `confidence` rather than its hard
label, a threshold shift buys nothing there while costing accuracy in 10/10 seeds.

## Module 4 results

The ranking engine takes a per-user candidate pool and re-ranks it on six objectives.

**Candidate blending.** Three sources contribute to every pool — IR/content similarity
(Module 2), user-user collaborative filtering, and popularity. Module 3's predicted
strategy does not route between them; it shifts how much each one contributes, weighted
by classifier confidence, degrading toward a `collaborative`-heavy prior when confidence
is low.

**Greedy selection.** Two of the six objectives (diversity, coverage) are defined
relative to what has already been picked, so the engine selects sequentially — rescoring
every remaining candidate against the current list at each step — rather than scoring
once and sorting.

**Genre-overlap relevance gate.** TF-IDF matches on shared vocabulary, which is not the
same as shared tone: the neighbours of *Toy Story* include *Silent Night, Deadly Night 5:
The Toy Maker* and *Dollman vs. Demonic Toys*, which match on the word "toy" alone. The
gate scales a candidate's IR relevance by its Jaccard genre overlap with the seed it was
retrieved from:

```
gate = floor + (1 - floor) * jaccard(genres(candidate), genres(seed))
```

Tunable via `ranking_objectives.genre_relevance_gate.floor` in `eval_config.yaml`
(default **0.25**). `1.0` disables the gate; `0.0` fully suppresses zero-overlap
candidates. Movies with no genre metadata are never penalised — an unknown genre is not
evidence of a mismatch. Only IR candidates are gated; collaborative and popularity
candidates have no seed movie to compare against, so the cold-start path (where no seeds
exist) is provably unaffected.

Effect on the *Toy Story* pool — candidates sharing **no** genre with the seed, in the
top 10:

| | Gate disabled | Gate enabled (floor 0.25) |
|---|---|---|
| Zero-overlap titles in top 10 | 3 | 1 |
| Rank of *Silent Night, Deadly Night 5* | #2 | #7 |
| Genuine *Toy Story* entries | #1, #5 | #1, #2, #3 |

The one surviving mismatch at #7 is expected behaviour, not a gate failure: its
`coverage` score is 1.000 because horror/sci-fi are genres nothing above it covers, so
the coverage objective actively rewards it. Its relevance was cut from 0.735 to 0.226.

**Weight sensitivity.** Raising diversity 0.15 → 0.60 and coverage 0.10 → 0.40 lifts
mean diversity 0.952 → 0.972 and genre spread 13 → 16 distinct genres, paid for with
mean relevance 0.814 → 0.657 — the trade-off is visible and moves in the expected
direction. Each of the six objectives, weighted alone, produces a visibly different
top-3.

## Final KPI report

Produced by `scripts/generate_kpi_report.py` on a 900,000-rating sample (6,034 users),
1,500 users evaluated, 20 search queries, K = [5, 10, 20]. All 23 metrics come from
`eval_config.yaml` by their exact names.

### 1. Search — Module 2, Classical IR Engine

| Metric | Value |
|---|---|
| Precision@10 | 0.4100 |
| Recall@10 | 0.0101 |
| MRR | 0.4396 |
| NDCG | 0.4123 |
| Search Latency | 7.94 ms (p95 8.38 ms) |

There is no hand-labeled relevance set, so relevance uses the documented proxy: a result
is relevant if it shares **at least one genre or keyword** with the query. That marks on
average **1,812 of 26,743** movies relevant per query, so `Recall@10` is bounded above by
10/1,812 and measures catalog breadth rather than retrieval quality. `Precision@10` is
correspondingly generous — one shared genre is a low bar.

### 2. Recommendation — Module 4, Multi-Objective Hybrid Ranking

| Metric | K=5 | K=10 | K=20 |
|---|---|---|---|
| Precision@K | 0.1944 | 0.1529 | 0.1104 |
| Recall@K | 0.1130 | 0.1732 | 0.2381 |
| F1 | 0.1167 | 0.1308 | 0.1226 |

| Metric | Value |
|---|---|
| MAP | 0.1174 |
| Coverage | 0.0713 |
| Diversity | 0.8322 |
| Novelty | 0.1715 |
| Cold-Start Accuracy | 0.6061 |

**Cold-start:** CineRankX 0.6061 vs. popularity baseline 0.5859 → lift **+0.0202**.

Two caveats, both recorded in `DATASET_MIGRATION.md`:

- MovieLens 20M only admits users with ≥20 ratings, so it contains **no natural
  cold-start users**. The cohort is *simulated* — real users whose profiles are truncated
  to 5 ratings, scored on their untouched held-out items. Other users keep full
  histories so the collaborative neighbourhood stays realistic. The report labels which
  mode produced the figure.
- Since the popularity-source correction, the baseline's top-20 is a **strict subset of
  Module 4's own candidate pool**. The comparison is therefore an *ablation* of the
  ranking layer — "does the six-objective ranker order a shared pool better than plain
  popularity rank?" — not a test against an independent baseline.

### 3. Machine Learning — Module 3, Strategy Selection Classifier

| Metric | Value |
|---|---|
| Accuracy | 0.6333 (majority baseline 0.4833) |
| Precision | 0.6160 |
| Recall | 0.6333 |
| F1 | 0.6245 |
| ROC-AUC | 0.6432 |

Confusion Matrix (rows = true, columns = predicted):

| | collaborative | content_based | popularity |
|---|---|---|---|
| **collaborative** | 91 | 3 | 46 |
| **content_based** | 10 | 0 | 5 |
| **popularity** | 42 | 4 | 99 |

*(This run uses 1,500 users; the 5,000-user figures in [Module 3 results](#module-3-results)
are the headline numbers for that engine.)*

### 4. System — end-to-end instrumentation

| Metric | Value |
|---|---|
| Response Time | 22.51 ms/call |
| Memory Usage | 457.46 MB (peak RSS) |
| CPU Usage | 90.1% of one core (7.5% of 12 cores) |
| Throughput | 46.2 ops/sec |

| Operation | ms/call | p95 ms | ops/sec |
|---|---|---|---|
| IR search (Module 2) | 11.094 | 15.107 | 90.1 |
| IR similar_to (Module 2) | 12.433 | 17.765 | 80.4 |
| Strategy predict (Module 3) | 48.485 | 69.554 | 20.6 |
| Rank top-10 (Module 4) | 18.038 | 19.267 | 55.4 |

### Known trade-off

The popularity-source correction improved accuracy metrics (Precision@10 0.1389 → 0.1529,
MAP 0.1124 → 0.1174, cold-start lift −0.0303 → +0.0202) while **Coverage fell 0.0964 →
0.0713** and **Novelty fell 0.2440 → 0.1715**. The popularity source now proposes
genuinely widely-rated films, which are by definition less novel and recur across users.
Better hit-rates, less discovery. The six objective weights in `eval_config.yaml` are the
lever for rebalancing this.

## Evaluation

KPIs tracked across search, recommendation, ML classifier, and system performance are
defined in [`eval_config.yaml`](./eval_config.yaml) — see that file for the exact metric
names used throughout the codebase and report.

## References

[1] Y. Zerikat & M. Zerikat, "Movie Recommendation System Based on Machine Learning Using Profiling," IJAIA, vol. 16, no. 1, Jan. 2025.
[2] A. M. Sarhan et al., "Integrating Machine Learning and Sentiment Analysis in Movie Recommendation Systems," Journal of Electrical Systems and Information Technology, vol. 11, no. 53, 2024.
[3] B. B. Sinha, R. Sinha & V. Priye, "Beyond Classical Approaches: Redefining the Landscape of High-Accurate Movie Recommendation using QNN," The Journal of Supercomputing, vol. 81, art. 347, 2025.
[4] M. M. Sultan, "Network-Based Movie Quality Prediction and Recommendation System Using Hybrid Machine Learning Techniques," Scientific Research Journal of Engineering and Computer Sciences, vol. 5, no. 1, pp. 1–6, 2025.
[5] M. A. Ahmad & S. Singh, "Movie Recommendation System," IJFMR, vol. 7, no. 3, 2025.
