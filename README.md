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
- [ ] Evaluation suite (see `eval_config.yaml` for exact metric names)

**Test suite:** 137/137 passing (`python -m pytest tests/ -q`).

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

The SMOTE investigation behind Module 3's documented limitations can be reproduced with
`scripts/compare_smote.py`, `scripts/smote_seed_robustness.py` and
`scripts/smote_vs_class_weight.py`.

## Module 3 results

Measured on a 5,000-user sample (750,512 ratings), 5-split label averaging,
1,000-user held-out test set:

| Metric | Value |
|---|---|
| Accuracy | 0.7960 |
| Majority-class baseline | 0.8410 |
| Lift over baseline | −0.0450 |
| ROC-AUC | 0.6421 |
| F1 (weighted) | 0.7657 |

Per-class precision against each class's own base rate in the test set:

| Strategy | Support | Base rate | Precision | Ratio |
|---|---|---|---|---|
| `collaborative` | 841 | 84.1% | 0.8540 | 1.02× |
| `content_based` | 70 | 7.0% | 0.2069 | **2.96×** |
| `popularity` | 89 | 8.9% | 0.1132 | 1.27× |

Read honestly: overall accuracy sits *below* the majority-class baseline, partly by
design (`class_weight='balanced'` trades majority accuracy for minority recall). The
model is nonetheless learning real signal — ROC-AUC 0.64 is well above chance, and
`content_based` predictions land ~3× more often than its base rate would give, though
with low recall (0.086). `collaborative`'s high raw F1 is mostly majority-class mass,
not discrimination.

This is **partially effective personalization**, not a solved classification problem.
Module 4 therefore treats the predicted strategy as a weighted prior rather than a hard
routing decision. Full limitation write-up, including the four remedies tried and
rejected, is in [`DATASET_MIGRATION.md`](./DATASET_MIGRATION.md#known-limitations--module-3-adaptive-ml-strategy-selection-engine).

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
mean diversity 0.959 → 0.977 and genre spread 15 → 16 distinct genres, paid for with
mean relevance 0.548 → 0.460 — the trade-off is visible and moves in the expected
direction. Each of the six objectives, weighted alone, produces a visibly different
top-3.

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
