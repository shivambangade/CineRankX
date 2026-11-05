# Appendix D. Complete Code Inventory (auto-extracted)

Generated from the repository by `docs/_build/extract_code.py` (AST parse). Each row: path, line count, the first line of the module docstring, then classes (methods) and functions. Use it to confirm that no module was left out of Chapter 11.

| File | Lines | Summary | Symbols |
|---|---|---|---|
| `app.py` | 157 | Streamlit UI for live or exported Module 4 recommendation output. | `load_engine`, `strategy_context`, `engine_parts`, `_log_run`, `recommend`, `db_connection`, `load_report` |
| `src/benchmark/baseline.py` | 44 | FROZEN baseline strategy. Do not edit: any change creates a different benchmark. | `FrozenBaseline` (recommend, predict) |
| `src/benchmark/cinerankx.py` | 69 | CineRankX as a benchmark strategy: EASE relevance -> six-objective re-ranking. | `CineRankXStrategy` (fit, pools, recommend_many, predict) ; `weights_from` |
| `src/benchmark/config.py` | 29 | Benchmark settings from eval_config.yaml (the `benchmark:` block). | `load_benchmark_config` |
| `src/benchmark/evaluate.py` | 44 | Strategy-agnostic evaluation: every strategy is scored on identical users and ground truth | `exclusions`, `evaluate_strategy` |
| `src/benchmark/metrics.py` | 67 | Ranking and rating-prediction metrics for the frozen benchmark. | `hit_rate`, `precision`, `recall`, `ndcg`, `average_precision`, `rating_errors`, `summarize` |
| `src/benchmark/split.py` | 141 | Frozen benchmark: deterministic train/validation/test split and fixed evaluation cohorts. | `Benchmark` ; `uniform_hash`, `assign_splits`, `load_interactions`, `_eligible`, `build_benchmark`, `fingerprint`, `get_benchmark` |
| `src/benchmark/tuning.py` | 83 | Hyperparameter tuning on the VALIDATION split only (test interactions are never read). | `_score`, `tune_relevance`, `_weight_candidates`, `tune_weights` |
| `src/evaluation/metric_names.py` | 89 | The exact KPI name strings from eval_config.yaml. | `load_configured_metrics`, `load_k_values`, `missing_from` |
| `src/evaluation/ml_metrics.py` | 132 | Classifier metrics — the "machine_learning" KPI group. | `majority_baseline_accuracy`, `roc_auc`, `classification_metrics`, `to_report_names` |
| `src/evaluation/recommendation_metrics.py` | 331 | Recommendation KPIs, driven off Module 4's final ranked output. | `precision_at_k`, `recall_at_k`, `f1_score_at_k`, `average_precision`, `mean_average_precision`, `coverage`, `intra_list_diversity`, `novelty`, `split_user_ratings`, `popularity_baseline_top_k`, `evaluate_cold_start`, `evaluate_recommendations` |
| `src/evaluation/report.py` | 230 | Assembles and renders the full KPI report. | `build_report`, `_fmt`, `_rule`, `_header`, `render` |
| `src/evaluation/search_metrics.py` | 180 | Search KPIs — Precision@10, Recall@10, MRR, NDCG, Search Latency. | `_tokenize`, `build_relevance_index`, `is_relevant`, `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `evaluate_search` |
| `src/evaluation/system_metrics.py` | 132 | System KPIs — Response Time, Memory Usage, CPU Usage, Throughput. | `measure`, `aggregate` |
| `src/ingestion/build_dataset.py` | 38 | CLI entrypoint: loads raw MovieLens + TMDB data, joins them, cleans text, | `build` |
| `src/ingestion/joiner.py` | 70 | Joins MovieLens movies to TMDB metadata via links.csv, and builds the | `_split_names`, `join_movielens_tmdb` |
| `src/ingestion/loaders.py` | 31 | Loaders for the raw MovieLens 20M and TMDB 930K CSVs. | `load_movielens`, `load_tmdb` |
| `src/ingestion/text_cleaner.py` | 31 | NLTK-based text cleaning: lowercase, stop-word removal, stemming. | `_ensure_nltk_data`, `clean_text` |
| `src/ir_engine/ir_engine.py` | 42 | Unified IR engine combining TF-IDF search and Trie autocomplete. | `IREngine` (fit, save, load, search, autocomplete, similar_to) |
| `src/ir_engine/tfidf_vectorizer.py` | 123 | TF-IDF vectorization and cosine similarity search over movie texts. | `TFIDFEngine` (fit, save, load, search, similar_to) |
| `src/ir_engine/trie.py` | 54 | Trie for prefix-based autocomplete over movie titles. | `TrieNode` (); `Trie` (insert, search) |
| `src/ranking_engine/candidates.py` | 307 | Candidate generation: blends Module 2 (IR) and Module 3 (strategy) output. | `CandidateGenerator` (_build_popularity, _build_collaborative, _content_candidates, _collaborative_can) ; `strategy_source_weights`, `genre_overlap_gate`, `_normalize` |
| `src/ranking_engine/config.py` | 105 | Tunable weights for the six ranking objectives. | `RankingWeights` (items, with_overrides, from_config) ; `load_genre_gate_floor` |
| `src/ranking_engine/ease.py` | 79 | EASE item-item linear model: CineRankX's personalised relevance source. | `EaseModel` (fit, user_matrix, blended_scores, save, load) |
| `src/ranking_engine/objectives.py` | 185 | The six ranking objectives, plus the catalog statistics they read from. | `CatalogStats` (_build_genres, _build_popularity_quality, _build_novelty, genres, popularity_qua) ; `_clean_genre`, `_minmax`, `diversity_score`, `coverage_score` |
| `src/ranking_engine/predictor.py` | 68 | Baseline rating predictor — the source for the predicted_rating objective. | `BaselinePredictor` (fit, predict, predict_normalized) |
| `src/ranking_engine/ranker.py` | 171 | Multi-Objective Hybrid Ranking Engine (Module 4). | `MultiObjectiveRanker` (_similarity_lookup, rank, recommend) ; `_jaccard` |
| `src/storage/db.py` | 84 | SQLite schema and bulk ingestion for ratings, movies, per-user stats and run history. | `connect`, `db_ready`, `_movie_frame`, `build_database` |
| `src/storage/runs.py` | 37 | Persisted recommendation runs: what was generated, when, with which engine and weights. | `record_run`, `list_runs` |
| `src/storage/users.py` | 104 | Server-side user queries: profile summary, paginated history, preferences, timelines. | `user_summary`, `_filters`, `history_page`, `_joined`, `rating_distribution`, `genre_preferences`, `user_timeline`, `active_users` |
| `src/strategy_selector/classifier.py` | 248 | Strategy classifier: predict best-fit strategy per user. | `StrategyClassifier` (fit, fit_full, _resample_training_fold, predict, save, load) |
| `src/strategy_selector/config.py` | 33 | Strategy-selector settings, read from eval_config.yaml (never hardcoded in the logic). | `load_selector_config` |
| `src/strategy_selector/content_model.py` | 90 | Content-based recommender with configurable representation, user profile and quality prior | `ContentModel` (profile_vector, top_items, hit_rate) ; `_quality_percentile` |
| `src/strategy_selector/dataset.py` | 77 | Build (and cache) the labelled training frame: sampled users -> features + backtest scores | `sample_user_ids`, `load_ratings`, `cache_path`, `build_labelled_frame` |
| `src/strategy_selector/estimators.py` | 27 | Importable estimator factory: saved models reference it by module path, so it must not liv | `make` |
| `src/strategy_selector/formulations.py` | 186 | Five ways to turn user features into a strategy choice, behind one sklearn-style interface | `_Constant` (predict_proba); `_Base` (predict); `MulticlassForm` (fit, predict_proba); `OneVsRestForm` (fit, predict_proba); `PairwiseForm` (fit, predict_proba); `RegressionForm` (fit, predict_scores, predict_proba); `RankingForm` (_expand, fit, predict_scores, predict_proba) ; `winner`, `margin`, `_fit`, `_fit_binary`, `_softmax`, `decision_quality` |
| `src/strategy_selector/holdout_labeler.py` | 71 | Leak-free strategy labels: backtest on a per-user holdout the profile never sees. | `split_profile_holdout`, `label_holdout`, `clean_labels` |
| `src/strategy_selector/model_selection.py` | 165 | Cross-validated model comparison and one-shot held-out evaluation for the selector. | `_lgbm`, `_scaled`, `build_estimator`, `fit_candidate`, `tune_candidate`, `_proba`, `_auc`, `cross_validate`, `compare_candidates`, `evaluate_holdout` |
| `src/strategy_selector/profile_features.py` | 152 | Extract user profile features from rating history (vectorized over all users). | `_parse_time`, `_catalog_tables`, `_popularity_percentile`, `_genre_features`, `_time_features`, `extract_profile_features` |
| `src/strategy_selector/stats.py` | 46 | Small statistics helpers: bootstrap confidence intervals and paired significance tests. | `bootstrap_ci`, `paired_bootstrap`, `mcnemar_p`, `proportion_ci` |
| `src/strategy_selector/strategy_context.py` | 207 | Strategy-aware features: a cheap backtest of each strategy on the user's OWN profile. | `LiteContext` (_build_cf_lite, cb_hit_rate, engine_scores, cf_hit_rate) ; `inner_scores`, `build_features`, `save_context`, `load_context` |
| `src/strategy_selector/strategy_labeler.py` | 295 | Backtest strategies to determine the best-fit strategy per user. | `_BacktestContext` (_build_popularity, _build_content_based, _build_collaborative) ; `select_best_strategy`, `_build_cb_profile_vector`, `_content_based_hit_rate`, `_collaborative_hit_rate`, `_popularity_hit_rate`, `label_users_with_best_strategy` |
| `src/strategy_selector/utility.py` | 61 | Utility-based strategy selection with an uncertainty-aware fallback. | `UtilitySelector` (fit, utilities, adjusted, predict_proba, predict) ; `choose_tau` |
| `src/ui/comparison_view.py` | 66 | Strategy Comparison: frozen baseline vs CineRankX on the fixed benchmark (reports/strategy | `_delta_badge`, `_panel`, `render_comparison` |
| `src/ui/icons.py` | 28 | Feather icons (https://feathericons.com, MIT) as inline SVG strings. | `icon` |
| `src/ui/kpi_view.py` | 133 | Evaluation workspace rendered from reports/kpi_report.json (generate_kpi_report.py). | `_num`, `_comparison`, `render_report` |
| `src/ui/nav.py` | 39 | Navigation state shared by screens: section labels, deep links and the selected user. | `selected_user`, `set_user`, `open_profile`, `sync_view` |
| `src/ui/pages.py` | 146 | Overview and engines screens. Everything shown comes from loaded data, config or reports. | `_pool_size`, `_funnel`, `_routing`, `render_overview`, `render_engines` |
| `src/ui/profile.py` | 158 | User 360 screen: summary, history, preferences, past runs and current recommendations. | `_picker`, `_weights_from_session`, `_overview`, `_preferences`, `_current`, `render_profile` |
| `src/ui/profile_history.py` | 91 | Profile tabs backed by the store: paginated rating history and past recommendation runs. | `render_history`, `render_runs` |
| `src/ui/results.py` | 87 | Validate ranked output and summarize metrics available from that output. | `prepare_results`, `output_kpis`, `objective_label`, `split_title`, `reason_for`, `genres_lookup` |
| `src/ui/runner.py` | 192 | Recommendation runner screen: source, user, K, objective weights, ranked results. | `_default_weights`, `_reset_weights`, `_weights_panel`, `_live_controls`, `show_results`, `_csv_source`, `render_runner` |
| `src/ui/theme.py` | 160 | CineRankX visual system: porcelain canvas, orange/cobalt accents, hairline cards. | `apply_theme`, `header`, `card_title`, `card`, `card_html`, `kpi`, `kpi_row`, `meter` |
| `scripts/analyze_selector.py` | 128 | Diagnose why the strategy classifier plateaus: labels, features, errors and attainable acc | `margins`, `noise_ceiling`, `main` |
| `scripts/build_movielens_only.py` | 68 | Build data/processed/ from MovieLens alone (no TMDB), for local demos. | `main` |
| `scripts/build_user_db.py` | 29 | Build data/processed/cinerankx.db (ratings, movies, user stats, run history). | `main` |
| `scripts/compare_smote.py` | 139 | Before/after comparison: Module 3's classifier with and without SMOTE. |  |
| `scripts/content_experiments.py` | 102 | Does a better content representation actually raise held-out hit rate? | `main` |
| `scripts/generate_kpi_report.py` | 186 | End-to-end KPI report — the project's final results table. | `log`, `_plain` |
| `scripts/label_experiments.py` | 96 | Compare label-generation methods: reliability, class mix and what a perfect model could sc | `labels_for`, `classify`, `oracle_ceiling`, `main` |
| `scripts/run_benchmark.py` | 161 | Reproducible strategy comparison: frozen baseline vs. CineRankX on one fixed benchmark. | `code_hash`, `run_baseline`, `freeze_baseline`, `before_pipeline`, `main` |
| `scripts/selector_experiments.py` | 266 | Systematic strategy-selector experiments on the FULL labelled population (no user dropped) | `space`, `evaluate`, `cv_oof`, `tune`, `deploy_winner`, `main` |
| `scripts/selector_phase2.py` | 221 | Strategy selection, evaluated on TWO separate objectives, with uncertainty and significanc | `Reweighted` (predict_proba, predict) ; `per_user_value`, `oof_utilities`, `tune_thresholds`, `main` |
| `scripts/smote_seed_robustness.py` | 119 | Multi-seed robustness check for the SMOTE result. | `run_arm` |
| `scripts/smote_vs_class_weight.py` | 112 | Does SMOTE do anything that cheaper class_weight tuning does not? | `run` |
| `scripts/train_strategy_selector.py` | 147 | Train and honestly evaluate the Adaptive Strategy Selector. | `label_reliability`, `inference_latency`, `main` |
| `scripts/verify_ir_engine.py` | 52 | Verification script for IR engine on real dataset. |  |
| `scripts/verify_ranking_engine.py` | 191 | Verification script for the Multi-Objective Hybrid Ranking Engine (Module 4). | `show`, `spread`, `horror_like` |
| `scripts/verify_strategy_selector.py` | 116 | Verification script for Strategy Selector on real MovieLens data. |  |
| `tests/test_benchmark.py` | 138 | Tests for the frozen benchmark: deterministic split, metrics, frozen baseline, EASE, evalu | 8 test functions |
| `tests/test_evaluation.py` | 604 | Tests for the evaluation suite (all four KPI groups from eval_config.yaml). | 71 test functions |
| `tests/test_formulations.py` | 87 | Tests for the strategy-choice formulations and the decision-quality metric. | 6 test functions |
| `tests/test_ingestion.py` | 103 |  | 6 test functions |
| `tests/test_ir_engine.py` | 192 | Tests for the IR engine: TF-IDF, Trie, and similarity search. | 18 test functions |
| `tests/test_ranking_engine.py` | 539 | Tests for the Multi-Objective Hybrid Ranking Engine (Module 4). | 72 test functions |
| `tests/test_selector_pipeline.py` | 156 | Tests for the richer features, leak-free labelling, margin cleaning and CV evaluation. | 11 test functions |
| `tests/test_strategy_context.py` | 84 | Tests for the strategy-aware features (inner backtest) and their train/serve consistency. | 6 test functions |
| `tests/test_strategy_selector.py` | 741 | Tests for the Adaptive ML Strategy Selection Engine. | 43 test functions |
| `tests/test_ui_results.py` | 81 | Unit tests for output displayed by the recommendation UI. | 9 test functions |
| `tests/test_user_store.py` | 80 | Tests for the SQLite user store: lookup, history paging/filtering, run history. | 6 test functions |
| `tests/test_utility_stats.py` | 77 | Tests for bootstrap statistics, the utility selector with fallback, and the content model. | 6 test functions |
