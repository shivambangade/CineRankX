# Chapter 11. Module-by-Module Codebase Walkthrough

The repository holds about 11,000 lines of Python in 89 files (counted by `docs/_build/extract_code.py`). This chapter walks through them package by package. Appendix D lists every class and function automatically extracted from the source, so nothing is missing even where this chapter summarises.

## 11.1 Directory structure

```
CineRankX/
  app.py                       Streamlit entry point: routing, caching, recommend()
  eval_config.yaml             exact KPI names + all tunable settings
  requirements.txt  CLAUDE.md  README.md  DATASET_MIGRATION.md
  .streamlit/config.toml       theme colours, upload limit
  src/
    ingestion/                 Module 1: load, join, clean
    ir_engine/                 Module 2: TF-IDF search + Trie
    strategy_selector/         Module 3: features, labels, classifiers, utility selection, stats
    ranking_engine/            Module 4: candidates, six objectives, ranker, EASE
    evaluation/                23-KPI suite (search, recommendation, ML, system)
    benchmark/                 frozen split, frozen baseline, CineRankX strategy, tuning
    storage/                   SQLite: ratings, user stats, recommendation runs
    ui/                        Streamlit screens, theme, icons, navigation
  scripts/                     verification, training, benchmark, experiment scripts
  tests/                       12 pytest files (272 test cases)
  data/processed/              generated artifacts (git-ignored)
  newdata/                     raw datasets (git-ignored)
  reports/                     generated reports (git-ignored)
```

## 11.2 `eval_config.yaml`: the single configuration file

Blocks: `search.metrics`, `recommendation.k_values: [5, 10, 20]` plus `metrics`, `machine_learning.metrics`, `system.metrics`, then `ranking_objectives` (genre gate floor 0.25 and the six `weights`), `strategy_selector` (`holdout_fraction`, `label_margin`, `label_top_n`, `inner_splits`, `inner_fraction`, `min_history`, `max_history`, `cv_folds`, `content_options`) and `benchmark` (seed, split fractions, cohort sizes, `ks`, `headline_k`). Loaders merge the block over safe defaults, so an older config still loads. *Design decision:* anything tunable lives here, never hard-coded in logic (a convention in `CLAUDE.md`).

## 11.3 Package `src/ingestion` (Module 1)

| File | Purpose and logic |
|---|---|
| `loaders.py` | `load_movielens(base_dir)` reads `rating.csv, movie.csv, link.csv, tag.csv` into a dict of DataFrames; `load_tmdb` reads `TMDB_movie_dataset_v11.csv`. Default dirs `newdata/movielens`, `newdata/tmdb` |
| `joiner.py` | `join_movielens_tmdb(ml, tmdb)` returns `(merged_frame, report)`. Steps in 5.3. `_split_names` turns `"Action, Adventure"` into a list. The report dict (`movielens_movies`, `movielens_with_tmdb_id`, `tmdb_movies`, `joined_rows`) is printed by `build_dataset` |
| `text_cleaner.py` | `clean_text` (lowercase, regex letters, NLTK stop-words, Porter stem); `_ensure_nltk_data` downloads stop-words once |
| `build_dataset.py` | `build()`: load, join, clean `text_chunk` -> `text_chunk_clean`, write `movies_merged.csv` and `ratings_clean.csv`, print the join report. Run as `python -m src.ingestion.build_dataset` |

## 11.4 Package `src/ir_engine` (Module 2)

* `tfidf_vectorizer.py::TFIDFEngine`: `fit(df)` builds the TF-IDF matrix over `text_chunk_clean`; `save/load` pickle the vectorizer, write the sparse matrix with `save_npz`, and store movie ids/titles; `search(query, limit)` transforms the query and computes `cosine_similarity` against all rows, `np.argsort(...)[::-1]` for the top-k, keeping only positive similarities; `similar_to(movie_id, limit)` uses the movie's row as the query.
* `trie.py::Trie`: nodes keyed by character; `insert(title, movie_id)` lower-cases titles and stores `(movie_id, title)` at the final node; `search(prefix, limit)` descends to the prefix node then runs a depth-first `collect_movies` until `limit` results. Complexity: O(length of prefix + results).
* `ir_engine.py::IREngine`: a façade that owns a `TFIDFEngine` and a `Trie`. **Design quirk:** the Trie is rebuilt in `fit` and cannot be persisted, so `load()` restores only TF-IDF; autocomplete after `load` needs a re-fit. The ranking engine uses only `similar_to` and the TF-IDF matrix.

## 11.5 Package `src/ranking_engine` (Module 4)

### `config.py`
`OBJECTIVES = ("relevance", "diversity", "novelty", "coverage", "popularity_quality", "predicted_rating")` is the canonical order. `RankingWeights` validates (all six present, none unknown, none negative, sum > 0) and **normalises to sum 1**; `.items()` returns canonical order; `.with_overrides(**kw)` returns a new object (no mutation, used for sensitivity experiments); `.from_config()` reads the YAML. `load_genre_gate_floor` reads the gate floor (default 0.25).

### `objectives.py`
`CatalogStats(movies_df, ratings_df)` precomputes per-item values once: `_build_genres` (genre sets; understands stringified lists, comma strings and pipe strings), `_build_popularity_quality` (Bayesian weighted rating, min-max), `_build_novelty` (self-information of observed engagement). Accessors `genres()`, `popularity_quality()`, `novelty()` return neutral values for unknown ids. Module functions `diversity_score(movie_id, selected_ids, similarity)` and `coverage_score(movie_id, covered_genres, stats)` implement objectives 2 and 4.

### `predictor.py`
`BaselinePredictor(damping=10)`: `fit(ratings_df)` (global mean, damped item bias, then damped user bias on the item-adjusted residual), `predict(user, movie)` (clipped to [0.5, 5]), `predict_normalized` (rescaled to [0, 1]).

### `candidates.py`
* `strategy_source_weights(strategy, confidence)`: `_FALLBACK_PRIOR = {collaborative 0.60, content_based 0.20, popularity 0.20}`; `strength = clip((confidence - 1/3) / (2/3), 0, 1)`; result `(1 - strength) x prior + strength x one_hot(strategy)`. Unknown/missing strategy returns the prior. Weights always sum to 1 (tested).
* `genre_overlap_gate`, `_normalize` (min-max per source), `CandidateGenerator`: constructor builds `_popular` (200 most-rated) and the CF structures; `_content_candidates`, `_collaborative_candidates`, and `generate(user_id, strategy, confidence, pool_size=200)` which excludes already-rated movies, blends sources (sum of weight x normalised score), records which sources proposed each movie in a `sources` string such as `"content_based+collaborative"` and returns `[movieId, relevance, sources]` sorted by relevance. `candidates_for_seed` serves "similar to this one movie" lookups through the same gate.

### `ranker.py::MultiObjectiveRanker`
Constructor builds `CatalogStats`, a `BaselinePredictor` (or accepts one), a `CandidateGenerator` (skippable with `build_generator=False`, used by the benchmark), and a title lookup. `_similarity_lookup(movie_ids)` builds pairwise similarity for the pool (TF-IDF cosine, genre-Jaccard fallback). **`rank(user_id, k, strategy, confidence, candidates, weights, pool_size)`** is the heart of the project:

1. obtain candidates (given, or generated);
2. precompute the four item/user-only objectives per candidate (`fixed`);
3. loop: for every remaining candidate compute `diversity_score` against the selected list and `coverage_score` against the covered genres, form `total = sum(weight x score)`, pick the maximum, remove it from `remaining`, add to `selected`, update `covered_genres`;
4. return a DataFrame `[rank, movieId, title, score, sources, relevance, diversity, novelty, coverage, popularity_quality, predicted_rating]`.

Complexity O(K x P) objective evaluations for pool size P (200) and list length K. `recommend()` is a thin wrapper that returns just movie ids.

### `ease.py::EaseModel`
`fit(train, n_items=6000, l2=500)` (see 6.A.6), `user_matrix(profiles, users)` (binary sparse rows), `blended_scores(user_rows, beta)`, `save/load` (`data/processed/ease_model.npz`).

## 11.6 Package `src/strategy_selector` (Module 3)

| File | Role |
|---|---|
| `profile_features.py` | `extract_profile_features(ratings_df, movies_df)`: vectorised user feature table (section 8.4). `_catalog_tables` caches per-catalog genre maps and popularity percentiles (a speed fix: rebuilding them per call dominated inference time). `_genre_features` computes `num_genres, genre_diversity, top_genre_share, genre_entropy` and 19 `affinity_*` columns; `_time_features` the span and recency features |
| `strategy_labeler.py` | The *backtest*. `_BacktestContext` builds shared structures once per batch: popularity top-N, a TF-IDF matrix, a user-item matrix with dense user-user cosine similarity (bounded to 5,000 users x 2,000 movies). `_content_based_hit_rate`, `_collaborative_hit_rate`, `_popularity_hit_rate`; `select_best_strategy(cb, cf, pop)` (tie goes to popularity); `label_users_with_best_strategy` (the original multi-split labeller, kept for tests and legacy scripts) |
| `holdout_labeler.py` | `split_profile_holdout(ratings, fraction, seed)` (per user, at least one rating each side); `label_holdout(profile, holdout, movies, top_n, content_model)` uses the same hit-rate functions but builds everything from the profile only; `clean_labels(labels, margin)` is the (now optional) ambiguity filter |
| `dataset.py` | `sample_user_ids` (random users from `user_stats`), `load_ratings` (SQL `WHERE userId IN (...)`, restricted to catalog movies), `cache_path`, `build_labelled_frame` (sample, split, label, build features, cache CSV and context pickle) |
| `strategy_context.py` | `LiteContext`: a compact reference population (popularity set, TF-IDF row-normalised matrix, sparse CF matrix, item means, item surprisal). `inner_scores(ctx, user, ratings)` runs the inner backtest (3 seeded splits, 25% pseudo-holdout). `LiteContext.engine_scores` computes similarity strengths. `build_features(ratings, movies, ctx)` = profile features + head shares + engine scores + bias/liked share/surprisal + inner scores, **the single function used at training and at serving** |
| `classifier.py` | `StrategyClassifier`: original Random Forest `fit` with optional SMOTE (train fold only), `fit_full` (refit a chosen estimator on all users), `predict` (single pass, argmax + max probability), `save/load` (pickle model and `(LabelEncoder, feature_names)`; sets `n_jobs=1` on load for cheap single-user inference) |
| `formulations.py` | `winner`, `margin`, five formulation classes sharing `fit(X, scores)` / `predict_proba`, `decision_quality` |
| `model_selection.py` | `CANDIDATES` registry, `TUNING` search spaces, `cross_validate`, `compare_candidates`, `evaluate_holdout` |
| `estimators.py` | `make(family, **params)`: importable factory (so pickled models reference a real module path instead of a script) |
| `utility.py` | `UtilitySelector` (per-strategy regressors, margin `tau` against the fallback) and `choose_tau` |
| `content_model.py` | `ContentModel` with options `vectorizer`, `genre_weight`, `profile`, `quality_alpha` |
| `stats.py` | `bootstrap_ci`, `paired_bootstrap`, `mcnemar_p`, `proportion_ci` |
| `config.py` | `load_selector_config` merging the YAML block over defaults |

### Walk-through: how one user's label is created (`label_holdout`)
```python
ctx = _BacktestContext(profile, movies_df, top_n=top_n)        # structures from PROFILE data only
hold_by_user = holdout.groupby("userId")["movieId"].apply(lambda s: s.values)
for user_id, train in profile.groupby("userId"):
    test_ids = hold_by_user.get(user_id, np.array([]))
    cb  = content_model.hit_rate(...) or _content_based_hit_rate(ctx, train, test_ids)
    cf  = _collaborative_hit_rate(ctx, user_id, test_ids)
    pop = _popularity_hit_rate(ctx, test_ids)
    rows.append({... "best_strategy": select_best_strategy(cb, cf, pop), ...})
```
In words: build the strategies from visible data, recommend, and count how many of the recommended movies the user later rated (hits divided by holdout size).

### Walk-through: serving one prediction (`app.py::recommend`)
```python
ratings_of_user = ratings[ratings["userId"] == user_id]
profile = build_features(ratings_of_user, movies, context, cfg["inner_splits"], cfg["inner_fraction"])
prediction = classifier.predict(profile).iloc[0]       # predicted_strategy, confidence
ranked = ranker.rank(user_id, k=k, strategy=strategy, confidence=confidence, weights=weights)
```
If the classifier or context artifact is missing the strategy stays `None` and the default prior blend is used, so the app degrades gracefully.
