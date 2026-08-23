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
   rating), with weights tunable via `eval_config.yaml`.

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
  pipeline. All unit tests passing. MovieLens 20M (~27K movies) → ~26.5K successfully joined
  to TMDB metadata (98.04% join rate); outputs cached to `data/processed/`.
- [ ] Module 2 — Classical IR Engine (TF-IDF + Trie)
- [ ] Module 3 — Adaptive ML Strategy Selection Engine
- [ ] Module 4 — Multi-Objective Hybrid Ranking Engine
- [ ] Evaluation suite (see `eval_config.yaml` for exact metric names)

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
