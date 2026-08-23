# CineRankX — Project Context for Claude Code

## What this project is
A mini-project (AY 2026-27, Dept. of AI & ML) building a movie recommendation system,
CineRankX, with three engines built in this order:

1. **Classical IR engine** — TF-IDF vector space model + a Trie for prefix/autocomplete
   search over movie titles.
   - Per-movie text chunk = overview + genres + keywords merged together. No cast/crew —
     the TMDB dataset in use has no cast/crew data (see Datasets below).
   - Cleaned with NLTK: lowercasing, stop-word removal, stemming.
   - Vectorized with TF-IDF; similarity between movies = cosine similarity of their vectors.
2. **Adaptive ML Strategy Selection Engine** — a classifier that predicts, per user, which
   recommendation strategy fits them best, based on that user's rating history/profile
   features.
3. **Multi-Objective Hybrid Ranking Engine** — re-ranks candidates on six weighted
   objectives (relevance, diversity, novelty, coverage, popularity/quality, predicted
   rating), combining the IR engine's output with the selected strategy's output.

Problem statement it's solving: existing systems fix one strategy and rank by a single
similarity score, ignoring diversity and per-user variation. CineRankX predicts the best
strategy per user and ranks on six objectives instead of one.

## Datasets (already downloaded into newdata/)
- **Primary — MovieLens 20M** (`newdata/movielens/`): ~20M ratings, ~27,000 movies.
  Files: `rating.csv`, `movie.csv`, `link.csv`, `tag.csv` (note: singular filenames,
  unlike ml-latest-small's `ratings.csv` etc.). `rating.csv`/`tag.csv` timestamps are
  datetime strings, not epoch ints. Also ships `genome_scores.csv`/`genome_tags.csv`
  (per-movie tag-relevance vectors) — not loaded by this pipeline, left unused.
- **Secondary — TMDB Movies Dataset 2023** (`newdata/tmdb/`): single file
  `TMDB_movie_dataset_v11.csv`, ~1.48M movies (overview, genres, keywords as plain
  comma-separated strings, not JSON). No cast/crew data or credits file at all —
  a deliberate scope decision, not a placeholder (see DATASET_MIGRATION.md).
- **Join key**: `link.csv` maps MovieLens `movieId` -> `tmdbId`, which joins to TMDB's
  `id` column. Join rate ~98% (vs. ~36% with the original ml-latest-small + TMDB 5000
  pairing) since this TMDB dataset is near-full TMDB coverage rather than a fixed slice.
- Save the cleaned/merged dataframe to `data/processed/` — don't recompute the join or
  the TF-IDF matrix on every run.

## Tech stack (fixed — don't substitute)
Python, VS Code, Pandas, NumPy, Scikit-learn, NLTK, Matplotlib. No deep learning /
embedding models unless explicitly asked — this stays explainable and rule-based
(TF-IDF + classical ML classifiers only).

## Project structure
```
newdata/              <- downloaded datasets (MovieLens 20M + TMDB 930K), do not modify
data/processed/       <- cleaned/merged/cached artifacts (TF-IDF matrix, merged df, etc.)
src/ingestion/        <- loading + joining MovieLens/TMDB
src/ir_engine/         <- TF-IDF pipeline, cosine similarity, Trie autocomplete
src/strategy_selector/ <- ML classifier for per-user strategy prediction
src/ranking_engine/    <- multi-objective hybrid ranking
src/evaluation/        <- metrics computation (see eval_config.yaml for exact metric names)
notebooks/
tests/
```

## Evaluation metrics (must match these exact names — see eval_config.yaml)
- Search: Precision@10, Recall@10, MRR, NDCG, search latency.
- Recommendation: Precision@K, Recall@K, F1, MAP, Coverage, Diversity, Novelty,
  Cold-Start Accuracy.
- ML classifier (strategy selector): Accuracy, Precision, Recall, F1, ROC-AUC,
  confusion matrix.
- System: response time, memory usage, CPU usage, throughput.

## Conventions
- One module at a time — don't implement multiple engines in a single change.
- Each module gets pytest tests in tests/ before moving to the next module.
- Commit after each module passes its tests.
- Ranking objective weights (for the six-objective ranking engine) must be exposed as
  tunable config, not hardcoded.
- Keep files under ~200 lines where reasonable; small testable functions over large ones.
