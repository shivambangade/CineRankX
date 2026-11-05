## 11.7 Package `src/evaluation` (the 23-KPI suite)

| File | Purpose and key logic |
|---|---|
| `metric_names.py` | One constant per KPI (`PRECISION_AT_10`, `MRR`, `COLD_START_ACCURACY`, ...) and group tuples `SEARCH_METRICS`, `RECOMMENDATION_METRICS`, `MACHINE_LEARNING_METRICS`, `SYSTEM_METRICS`. `load_configured_metrics` reads `eval_config.yaml`; `missing_from(report, path)` lists configured KPIs that the report lacks, so a metric cannot silently disappear |
| `search_metrics.py` | `evaluate_search(engine, queries, movies, k=10, relevance_index)`: relevance proxy (shares at least one genre or keyword), `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, latency in ms. `build_relevance_index` parses genre/keyword cells in three formats |
| `recommendation_metrics.py` | Per-user `precision_at_k`, `recall_at_k`, `f1_score_at_k`, `average_precision` (denominator `min(k, #relevant)`), `coverage`, `intra_list_diversity`, `novelty`; `split_user_ratings` (per-user random 80/20); `popularity_baseline_top_k` (observed counts); `evaluate_cold_start` (natural or simulated cohort; returns `mode`); `evaluate_recommendations` ties them together |
| `ml_metrics.py` | `classification_metrics(y_test, y_pred, proba, class_names, y_train)` returns accuracy, weighted precision/recall/F1, per-class table, confusion matrix with labels, one-vs-rest ROC-AUC (returns 0.0 if a class is absent from the test set), majority baseline; `to_report_names` renames to the exact KPI strings |
| `system_metrics.py` | `measure(operation, n_calls, warmup, label)` and `aggregate(measurements)` (see 10.4) |
| `report.py` | `build_report(search, recommendation, ml, system, strict=True)` (raises if any configured KPI is missing) and `render(report)` for the printed table |

`scripts/generate_kpi_report.py` is the driver: it loads the catalog and 400,000 ratings, the IR engine, the classifier and the ranking engine, runs the four groups, writes `reports/kpi_report.json` (plus a text copy when redirected), and reads the classifier results from `data/processed/strategy_eval.json` rather than retraining.

## 11.8 Package `src/benchmark` (frozen comparison)

| File | Purpose and key logic |
|---|---|
| `config.py` | `load_benchmark_config`: seed 20260101, 70/10/20 split, positive threshold 4.0, cohort sizes 2,000 / 1,000 / 1,000, `cold_history` 5, `ks`, `headline_k = 10` |
| `split.py` | `uniform_hash(users, items, seed)` (splitmix64 mapped to [0, 1)), `assign_splits`, `load_interactions` (read `rating.csv` with narrow dtypes, keep catalog movies, add `split` and an ordering key), `build_benchmark` (eligible users, seeded permutation into four disjoint cohorts, cold users' training rows truncated to their first 5 by the second hash, ground-truth dictionaries), `fingerprint` (SHA-256 of cohorts and test truth), `get_benchmark` (pickle cache, verifies the fingerprint) |
| `baseline.py` | `FrozenBaseline(train)`: `ranking` (movies sorted by training rating count, tie by id), `recommend(seen, k)`, `predict(movie)` (damped item mean). Version tag `popularity-itemmean-v1`. Imports only numpy and pandas |
| `metrics.py` | `hit_rate`, `precision`, `recall`, `ndcg`, `average_precision`, `rating_errors`, `summarize` (mean over users with at least one positive, plus coverage/diversity/novelty hooks) |
| `evaluate.py` | `exclusions(bench, phase, users)` (items rated before the scored split are not recommendable), `evaluate_strategy(bench, phase, cohort, recommend_many, predict, ...)`: the *same function* scores any strategy on identical users and truth |
| `cinerankx.py` | `CineRankXStrategy`: EASE pools (`pools`), optional six-objective re-ranking (`recommend_many`), the bias predictor, and the ranker with `build_generator=False` |
| `tuning.py` | `tune_relevance` (grid over lambda and beta by validation NDCG@10) and `tune_weights` (random weight vectors under the diversity floor) |

`scripts/run_benchmark.py` orchestrates: build or load the benchmark, score and verify the frozen baseline (`freeze_baseline` compares code hash, split fingerprint and results with `data/benchmark/baseline_frozen.json`), tune (or reuse `cinerankx_params.json`), evaluate old pipeline (`before_pipeline`) and new pipeline on the test split, write `reports/strategy_comparison.json`, and refit EASE on all ratings for the app.

## 11.9 Package `src/storage` (SQLite)

* `db.py`: `connect`, `db_ready` (true when `user_stats` has a row), `build_database(path, ratings_chunks, movies, merged)`. Tables: `movies(movieId, title, year, genres, original_language, release_date, poster_path)`, `ratings(userId, movieId, rating, ts)`, `user_stats(userId, n_ratings, avg_rating, first_ts, last_ts)`, `recommendation_runs(id, userId, created_at, source, strategy, confidence, k, weights, status, error, elapsed_ms, results)`. The index `idx_ratings_user` on `(userId, ts)` makes one user's history a millisecond lookup.
* `users.py`: `user_summary`, `history_page` (server-side search, genre filter, rating range, sort and `LIMIT/OFFSET`; the sort column is chosen from a whitelist so user input can never be injected into SQL), `rating_distribution`, `genre_preferences`, `user_timeline` (monthly activity, decade mix, language mix), `active_users` (prefix search over user ids).
* `runs.py`: `record_run` (writes strategy, confidence, normalised weights as JSON, status, error text, elapsed time and the ranked results as JSON), `list_runs`.

## 11.10 Package `src/ui` and `app.py` (the web application)

`app.py` (157 lines): `st.set_page_config`, theme and header; **navigation** is a `st.segmented_control` whose key `nav` is synchronised with the URL query parameter `view` (`overview | engines | run | profile | kpi`) through `nav.sync_view`, so refresh and deep links work; `load_engine()` is a `@st.cache_resource` that reads the catalog, the first 400,000 ratings, the IR engine and the classifier and builds the ranker; `recommend(user_id, k, weights)` validates, builds features, predicts the strategy, ranks, logs the run (`_log_run`, best-effort) and returns `(ranked, strategy, confidence)`; `load_report` is a `@st.cache_data` JSON reader keyed by file modification time so the dashboard always shows the latest saved results.

| File | Purpose |
|---|---|
| `theme.py` | Design tokens as CSS variables (porcelain canvas, orange `#F97316`, cobalt, teal), `card()` (bordered container with title), `kpi_row()`, `meter()` (progress bar row) |
| `icons.py` | `icon(name, size)`: Feather icons as inline SVG strings (no icon font, no emoji) |
| `nav.py` | `VIEWS`, `selected_user`, `set_user`, `open_profile`, `sync_view` |
| `pages.py` | `render_overview` (KPI tiles, candidate funnel, objective harmony, strategy routing, telemetry, surfaced candidates) and `render_engines` |
| `runner.py` | The runner: `render_runner`, `_weights_panel` (six sliders, normalisation indicator, reset callback), `_live_controls` (user ID validation), `show_results` (movie cards with a reason and six mini-bars), `_csv_source` (upload validation, 25 MB limit) |
| `results.py` | `prepare_results` (validates and sorts a ranked frame), `output_kpis`, `split_title`, `reason_for` (explanation from the two highest objective scores), `genres_lookup` |
| `profile.py`, `profile_history.py` | User 360 tabs: overview, history (paginated), preferences, past runs, current recommendations |
| `kpi_view.py`, `comparison_view.py` | KPI Benchmarks page: summary cards, top-K chart, selector quality vs chance, strategy comparison panels (frozen baseline vs CineRankX, overall and cold-start), classifier confusion matrix and per-class table, system performance. **Everything shown is read from the saved JSON files** (`reports/kpi_report.json`, `reports/strategy_comparison.json`) |

## 11.11 `scripts/`

| Script | What it does |
|---|---|
| `verify_ir_engine.py`, `verify_strategy_selector.py`, `verify_ranking_engine.py` | Original per-module demonstrations and artifact builders (the IR script saves the TF-IDF files; the selector script trains and saves the early classifier) |
| `generate_kpi_report.py` | The 23-KPI report driver (above) |
| `build_user_db.py` | Streams `rating.csv` in 1M-row chunks into SQLite |
| `build_movielens_only.py` | Builds a demo catalog from MovieLens alone (no TMDB) |
| `train_strategy_selector.py` | Earlier full retraining script (cleaned-label protocol, tuned CV candidates); still runnable |
| `selector_experiments.py` | Fourteen-candidate comparison with Optuna; `--deploy-saved NAME` refits the chosen model on all users and writes `data/processed/strategy_eval.json` and the model files that the app and KPI report use (**this produced the currently deployed selector**) |
| `selector_phase2.py` | Two-objective evaluation: classification and recommendation value, bootstrap CIs, cold-start-by-history thirds, TreeSHAP importances |
| `analyze_selector.py`, `label_experiments.py`, `content_experiments.py` | Diagnostics: label reliability and ceiling; label-definition comparison; content-model variants |
| `run_benchmark.py` | Frozen-benchmark comparison (11.8) |
| `compare_smote.py`, `smote_seed_robustness.py`, `smote_vs_class_weight.py` | The SMOTE investigation (Module 3) |

## 11.12 Where the saved results come from

| Artifact | Created by | Read by |
|---|---|---|
| `reports/kpi_report.json` | `generate_kpi_report.py` | `app.py` -> `kpi_view.render_report` |
| `reports/strategy_comparison.json` | `run_benchmark.py` | `comparison_view.render_comparison` |
| `data/processed/strategy_eval.json` | `selector_experiments.py --deploy-saved` | `generate_kpi_report.py` (classifier KPIs) |
| `data/processed/strategy_classifier.pkl` (+ `strategy_label_encoder.pkl`, `strategy_context.pkl`) | same | `app.recommend` |
| `data/benchmark/baseline_frozen.json` | `run_benchmark.py` (first run) | `run_benchmark.py` (drift check) |
