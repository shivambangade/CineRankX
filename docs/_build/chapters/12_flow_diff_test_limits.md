# Chapter 12. End-to-End Execution Flow

## 12.1 The sequence in the actual implementation

![Sequence of one live recommendation](figs/sequence.png)

```mermaid
sequenceDiagram
  actor U as User
  participant A as app.py / runner.py
  participant E as load_engine (cached)
  participant S as Strategy selector
  participant G as CandidateGenerator
  participant R as MultiObjectiveRanker
  participant D as SQLite run log
  U->>A: enter user ID, K, weights; click Generate
  A->>E: catalog, ratings sample, ranker (cached)
  A->>S: build_features(user ratings) then predict
  S-->>A: strategy and confidence
  A->>G: source weights from strategy_source_weights
  G-->>R: pool of 200 candidates (IR, CF, popularity)
  R->>R: greedy pick, six objectives
  R-->>A: ranked top-K with score breakdown
  A->>D: record_run(weights, strategy, results)
  A-->>U: movie cards and CSV export
```

## 12.2 Stage by stage

| # | Stage | What happens internally (function) |
|---|---|---|
| 1 | **Application initialisation** | `streamlit run app.py` executes the script on every interaction. It applies the theme, draws the header and the section tabs, reads `?view=` from the URL into `st.session_state["nav"]`, and decides `data_ready` (catalog and ratings files exist) and `has_db` (SQLite built) |
| 2 | **Data loading** | `load_engine()` (cached with `@st.cache_resource`, so it runs once per server process): reads `movies_merged.csv`, the first 400,000 rows of the ratings file, loads the IR artifacts, builds `MultiObjectiveRanker` (which builds `CatalogStats`, the bias predictor and the candidate generator's popularity and CF structures), loads the classifier |
| 3 | **Dataset preprocessing** | Already done offline by `src.ingestion.build_dataset` (join, clean). The online path only reads the results |
| 4 | **Feature generation** | `build_features(user_ratings, movies, context)` computes the profile features, popularity-head shares, engine-strength scores and three inner backtests for that one user |
| 5 | **Model loading** | `StrategyClassifier.load()` restores the model and encoder; `strategy_context()` (cached) restores the reference population |
| 6 | **User profile construction** | The user's ratings are selected from the in-memory sample (`ratings[ratings.userId == user_id]`); a user outside the 400k-rating sample cannot be served (the UI explains this) |
| 7 | **Strategy selection** | `classifier.predict(features)` returns `predicted_strategy` and `confidence` (maximum class probability) |
| 8 | **Candidate generation** | `CandidateGenerator.generate(user, strategy, confidence)`: `strategy_source_weights` gives the three source weights; IR neighbours of the user's top-10 rated movies (genre-gated), user-kNN neighbours' favourites, and the 200 most-rated movies are normalised, weighted and summed; rated movies removed; top 200 kept |
| 9 | **Recommendation scoring** | For each candidate the four user/item objectives are computed once (`fixed` dict) |
| 10 | **Multi-objective ranking** | The greedy loop in `MultiObjectiveRanker.rank` selects K movies using diversity and coverage relative to the selected list |
| 11 | **Result presentation** | `runner.show_results` renders movie cards (rank, title, year, genre chips, reason, six objective bars, rank score), summary tiles and a CSV export; `_log_run` writes the run to `recommendation_runs` |
| 12 | **Evaluation and monitoring** | Offline: `generate_kpi_report.py`, `run_benchmark.py` write JSON read by the KPI page. Online: the run's elapsed time is shown in the status line and stored; the Past runs tab lists every run |

## 12.3 The offline training sequence
1. `python -m src.ingestion.build_dataset` -> `movies_merged.csv`.
2. `PYTHONPATH=. python scripts/verify_ir_engine.py` -> TF-IDF artifacts.
3. `PYTHONPATH=. python scripts/build_user_db.py` -> SQLite.
4. `PYTHONPATH=. python scripts/selector_experiments.py 5000 --deploy-saved multiclass/lightgbm` -> classifier, context, `strategy_eval.json` (an earlier full run of the same script with `--trials 20` created the saved comparison `selector_experiments_inner3.json` that `--deploy-saved` reads).
5. `PYTHONPATH=. python scripts/run_benchmark.py` -> `strategy_comparison.json`, EASE model.
6. `PYTHONPATH=. python scripts/generate_kpi_report.py 400000 1000` -> `kpi_report.json`.
7. `streamlit run app.py`.

# Chapter 13. Project Differentiation and Competitive Analysis

> **Rule for this chapter:** we do not claim CineRankX invented any single technique. TF-IDF, cosine similarity, user-kNN, popularity baselines, MMR-style diversity, Bayesian ratings, gradient boosting and algorithm selection are all established. The contribution is in the *combination*, the *honest evaluation*, and the *engineering*.

## 13.1 Comparison table

| Property | Popularity | Collaborative | Content-based | Hybrid (fixed blend) | Single-objective ranker | **CineRankX (as implemented)** |
|---|---|---|---|---|---|---|
| Personalisation | none | high (needs history) | medium (needs item text) | medium-high | depends on score | high: neighbours, content seeds, bias predictor |
| Strategy selection per user | no | no | no | no (one blend for all) | no | **Yes**: classifier predicts strategy and confidence; used as a weighted prior |
| Multi-objective ranking | no | no | no | rarely | no (one score) | **Yes**: six weighted objectives |
| Diversity | low | low | low (over-specialised) | low | low | explicit objective (MMR-style) |
| Novelty | none | low | medium | low | low | explicit objective (self-information) |
| Coverage | very low (0.38% in our benchmark) | medium | medium | medium | low | genre-coverage objective; catalog coverage 5.7% in benchmark (partial: per-item proxy) |
| Cold-start handling | **good** | poor | good for new *items* | partial | partial | **partial**: popularity prior, EASE fold-in from 5 ratings (Hit Rate@10 0.698 vs 0.609 baseline, simulated cohort) |
| Explainability | high | medium | high | low | low | high: score breakdown per movie plus a plain-language reason |
| Adaptability | none | retrain | retrain | retrain | retrain | weights adjustable live; classifier retrained offline; **no online learning** |

## 13.2 Modern platforms
Public descriptions of large streaming recommenders (for example Gomez-Uribe and Hunt, "The Netflix Recommender System", ACM TMIS 2015) describe *many* algorithms combined and a personalised ranking stage, plus attention to diversity and context. We do not know their internal details and make no claim of equivalence. Relevant comparison points: such systems learn from implicit feedback at scale, run online experiments and optimise measured engagement, which CineRankX (offline, 20M ratings, no live users) does not.

## 13.3 What is our actual innovation?
1. **A per-user strategy selector trained on backtested strategy performance**, with the prediction used as a *weighted prior* over candidate sources (not a hard switch). Related literature on predicting when a recommender fails and choosing algorithms per user exists (for example Ekstrand and Riedl, "When Recommenders Fail", RecSys 2012); our implementation and findings are specific to this stack.
2. **Six explainable objectives in one greedy re-ranker** with tunable weights, exposed as sliders, and a score breakdown shown for every movie.
3. **A rigorous, reproducible evaluation culture**: leak-free labels, a hash-based frozen split, a self-contained frozen baseline with a drift check, bootstrap confidence intervals and paired tests, and a systematic diagnosis of why the classifier is limited.
4. **An honest negative result**: with this label definition, per-user "best strategy" is only weakly identifiable (agreement 65-73%, oracle ceiling about 0.76), yet following the selector still gives a statistically significant hit-rate gain over always-popularity.

## 13.4 Why it is not "just another movie recommendation website"
A typical demo returns a list ranked by one score. CineRankX (a) chooses and weights methods per user, (b) shows *why* each movie was ranked (six scores plus a reason), (c) lets the operator change trade-offs live, (d) logs every run with its weights and strategy, (e) provides a User 360 view for each user's history, and (f) reports its own measurement limits.

## 13.5 Why six objectives, and why per-user selection
Accuracy-only lists are repetitive and popular. Each added objective addresses a documented failure: diversity (repetition), novelty (filter bubble), coverage (concentration on few titles), popularity/quality (reliability), predicted rating (the user's likely score). Per-user selection is useful because evidence differs by user: light users gain most (selector hit-rate 0.185 vs 0.152 for popularity among users with 21-40 ratings), heavy users hardly at all (0.070 vs 0.069).

## 13.6 Limitations of the approach (say them first)
Content-based hit-rate is low (0.016); labels are noisy; classifier accuracy is below the majority rate; the strategies in the backtest are simplified versions of the production sources; evaluation is offline (no user feedback); one dataset.

## 13.7 What could make this publishable
(Suggestions, not claims.) Evaluate on several datasets; compare with learned hybrids and published algorithm-selection methods; add online or counterfactual evaluation; formalise the uncertainty-aware label definition and release the labelling protocol; study the trade-off curves among objectives; and test the improved content model and larger-budget labels end to end. A paper-quality claim would rest on the **evaluation methodology and the label-reliability finding**, not on high classifier accuracy.

# Chapter 14. Testing, Debugging and Software Engineering

## 14.1 How to run the tests
```
cd CineRankX
python -m pytest -q            # 272 test cases in 12 files (last full run: all passed)
python -m pytest tests/test_ranking_engine.py -q
```

## 14.2 What each test file validates

| File (cases) | What it protects |
|---|---|
| `test_ingestion.py` (6) | loaders read the right files/schemas; the join builds `text_chunk`; unmatched movies are dropped; `clean_text` lowercases, removes stop-words, stems, handles empty input |
| `test_ir_engine.py` (18) | Trie insert/search is case-insensitive, respects limit, ignores empty titles; TF-IDF `fit/search/similar_to/save/load`; unfitted engine raises; the façade `IREngine` combines both |
| `test_strategy_selector.py` (43) | feature extraction (genre entropy and top-genre share distinguish focused from eclectic users); `select_best_strategy` tie rules; popularity from observed counts, not TMDB; CF candidate budget capped; `_build_cb_profile_vector` caps and weights; classifier fit/predict/save/load, per-class metrics, absent classes flagged; SMOTE only on the training fold and never produces held-out rows |
| `test_ranking_engine.py` (72) | weights normalise and reject invalid input; each objective in [0, 1] and behaves as defined (first pick maximally diverse, near-duplicate scores low, max-not-mean similarity, coverage fractions, Bayesian shrinkage, novelty rewards rare items); predictor fallbacks; `strategy_source_weights` interpolation; candidates exclude rated movies and respect pool size; ranker returns K results with all objective columns |
| `test_evaluation.py` (71) | KPI names equal config; every metric function on hand-computed cases (P@K divides by K, reciprocal rank, NDCG ideal, AP denominator capped at K, coverage, diversity, novelty); the split gives every user train and test; cold-start modes; `build_report(strict)` refuses missing KPIs |
| `test_selector_pipeline.py` (11) | richer features; profile/holdout split is disjoint; holdout labels never use holdout ratings; label cleaning margin from config; cross-validation beats chance on separable data and not on noise; fold shrinkage for rare classes |
| `test_strategy_context.py` (6) | fast content score equals the reference implementation; inner scores deterministic and bounded; a user's own profile is never their neighbour; training and serving produce identical columns |
| `test_formulations.py` (6, 11 with parametrisation) | `winner` tie rule; every formulation learns a planted signal and returns valid probabilities; pairwise ignores ties; ranking expansion keeps three rows per user; decision quality by hand; pickling |
| `test_utility_stats.py` (6) | bootstrap CIs, paired bootstrap, McNemar, Wilson; utility selector fallback; `choose_tau`; content model |
| `test_benchmark.py` (8) | split deterministic and order-independent; cohorts disjoint; cold users truncated; exclusions never use test items; metrics by hand; **baseline is frozen, independent (AST check that it imports only numpy/pandas) and unchanged when CineRankX is retrained**; EASE beats popularity on structured data |
| `test_ui_results.py` (9) | ranked-CSV validation (missing columns, duplicates, invalid ranks), KPI summary, title/year split, explanation text, genre lookup |
| `test_user_store.py` (6) | user summary and unknown users; history pagination/filter/sort with SQL-injection-safe sorting; preferences/timelines; user search; run history round trip |

## 14.3 Types of testing in the project
* **Unit tests** (all of the above), mostly on small synthetic frames so they run in seconds.
* **Integration checks**: Streamlit's `AppTest` was used during development to run each page and a full recommendation without a browser; the verification scripts (`verify_*.py`) run each engine on real data.
* **Model testing**: planted-signal tests (the model must find a known pattern) and noise tests (it must not beat chance on random labels).
* **Data validation**: ranked-CSV validation (`prepare_results`), KPI completeness (`missing_from`), benchmark fingerprint check.
* **API testing / frontend testing**: there is no HTTP API; the UI is tested through `AppTest` and by manual use. **No automated browser test suite exists** (UNVERIFIED claims should not be made about it).

## 14.4 Error handling and logging
* Missing artifacts degrade gracefully: no classifier or context -> strategy `None` and the default prior blend; no database -> the profile page shows the build command; no KPI report -> an info message.
* Invalid input is rejected before running: user ID not in the sample disables the Generate button; all-zero weights raise a clear error; uploaded CSVs are size-limited and column-checked.
* Runs are logged in SQLite (success or error with message). The Python `logging` module is not used; scripts print progress.

## 14.5 Dependencies, serialisation, reproducibility
* `requirements.txt` lists unpinned packages (a reproducibility weakness; exact versions used while writing this guide are in Chapter 4).
* Models are saved with `pickle`; sparse matrices with `save_npz`; EASE with `np.savez`. *Security note:* never unpickle files from untrusted sources. A saved model refers to classes by module path, so factories must live in importable modules (`estimators.py`), not in scripts (a bug found while deploying).
* Seeds: 42 for model training and sampling; 20260101 for the benchmark hash split. Caches (labelled frames, benchmark pickle) are keyed by their settings.

## 14.6 Realistic debugging scenarios

| Symptom | Likely cause | Fix |
|---|---|---|
| Page shows an old behaviour after a code edit | Streamlit caches imported modules; the running process has the old code | Restart `streamlit run app.py` |
| `StreamlitDuplicateElementKey: card_topk` | the same UI block was accidentally rendered twice (a duplicated code section) | Remove the duplicate; keys must be unique per element |
| `ModuleNotFoundError: yaml` | dependencies not installed | `pip install -r requirements.txt` |
| `ValueError: pandas dtypes must be int, float or bool ... niche_fraction: object` | a boolean column produced by `.where` had object dtype | cast with `.astype(float)` (fixed in `profile_features.py`) |
| Classifier cannot be loaded after training (`load()` returns False) | the model was pickled with a factory defined in a script (`__main__`) | move the factory into an importable module (`estimators.py`) |
| "Strategy selector not trained" when generating the KPI report | `strategy_eval.json` or the classifier/context is missing | run `selector_experiments.py ... --deploy-saved` |
| Profile page says the user store is not built | `cinerankx.db` missing | `PYTHONPATH=. python scripts/build_user_db.py` |
| "User N is not in the loaded ratings sample" | live recommendations use only the first 400,000 ratings (about 2,900 users) | choose a smaller user ID, or raise `RATINGS_LIMIT` (more memory and load time) |
| Cold-start KPI is missing | MovieLens has no natural cold users | the evaluation simulates them; check `cold_start_mode` |
| Site stops responding after idling | the preview server was stopped | restart `streamlit run app.py` |

# Chapter 15. Limitations and Future Scope

## 15.1 Real limitations (current implementation)

| Area | Limitation | Evidence |
|---|---|---|
| Labels | per-user best strategy is only about 65-73% reproducible; 60% of winners lie within one standard error | Chapter 9 |
| Classifier | below the majority rate on plain accuracy (0.617 vs 0.644); content-based class only 2.3% of users | `strategy_eval.json` |
| Content-based | hit-rate 0.016; improved to 0.045 in an experiment not yet deployed | `content_experiments.json` |
| Cold start | MovieLens has no natural cold users; evaluation is simulated; gain over popularity is moderate (0.698 vs 0.609) | `strategy_comparison.json` |
| Data sparsity | 0.53% density; CF uses a 5,000-user x 2,000-movie window | `candidates.py` |
| Popularity bias | frozen baseline covers 0.38% of the catalog; CineRankX 5.7% (old pipeline reached 7.9%, so the newer pipeline gave up some coverage) | `strategy_comparison.json` |
| Scalability | the live app loads 400,000 ratings; user-user matrices are quadratic; the in-memory ranker is single-process | `app.py` |
| Evaluation | offline, one dataset; relevance proxy for search; hit-rate depends on holdout size and K | `search_metrics.py` docstring |
| User feedback | no clicks, watches or explicit feedback are collected | repository |
| Security | no authentication layer; pickle persistence | repository |
| Weak search relevance proxy | Recall@10 near zero by construction | KPI report note |
| Reproducibility | dependency versions unpinned; run-to-run variation of about 0.02 accuracy on 999 users | Chapter 9 |

## 15.2 Future work (not implemented)

| Idea | Why it might help | Status |
|---|---|---|
| Utility-based selection with calibrated uncertainty | optimise the quantity users care about (hit-rate) | prototype exists and ties the classifier |
| Contextual bandits / online learning | learn strategy value from real clicks | needs live feedback; **not built** |
| Reinforcement learning | sequential session optimisation | only reasonable with interaction logs; **not built** |
| Fusion of strategy lists (hybrid beyond one prior) | combine strengths per item | **not built** |
| Better content embeddings (sentence embeddings, cast/crew, plot summaries) | stronger content strategy | would relax the "no deep learning" scope |
| Deep recommenders (neural CF, sequence models) | larger accuracy | out of the explainable scope |
| Real-time feedback loop | adapt to the current session | not built |
| Explainable recommendations from neighbours ("because you liked X") | richer explanations | partly present (per-objective reasons) |
| Larger labelling budget and uncertainty-aware labels | more reliable targets | measured (agreement 0.733), not yet deployed |
| Larger evaluation (more users, several datasets, temporal splits) | tighter intervals, generality | not done |
