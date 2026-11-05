# Chapter 4. Technology Stack

Only technologies that the repository actually imports or configures are listed. Versions are those installed in the development environment when this guide was written.

## 4.1 Stack at a glance

| Layer | Technology (version) | Where it is used |
|---|---|---|
| Language | Python 3.14 | Everything |
| Data handling | pandas 3.0, NumPy 2.5, SciPy 1.18 | `src/ingestion`, features, sparse matrices (`csr_matrix`), statistics (`scipy.stats.binomtest`) |
| Classical ML | scikit-learn 1.9 | TF-IDF (`TfidfVectorizer`), cosine similarity, Random Forest/Extra Trees, cross-validation, metrics |
| Gradient boosting | LightGBM 4.7 (deployed); XGBoost, CatBoost (experiments only) | `src/strategy_selector/estimators.py`, `scripts/selector_experiments.py` |
| Hyperparameter search | Optuna | `scripts/selector_experiments.py` (tuning only) |
| Imbalance handling | imbalanced-learn (SMOTE) | `src/strategy_selector/classifier.py` (disabled by default) |
| NLP | NLTK 3.10 (stop-words, Porter stemmer) | `src/ingestion/text_cleaner.py` |
| Configuration | PyYAML | `eval_config.yaml`, loaded by `ranking_engine/config.py`, `strategy_selector/config.py`, `benchmark/config.py` |
| Database | SQLite (Python `sqlite3`) | `src/storage/` (ratings, user stats, recommendation runs) |
| Web UI | Streamlit 1.64 | `app.py`, `src/ui/` |
| System measurement | psutil | `src/evaluation/system_metrics.py` |
| Testing | pytest | `tests/` |
| Icons / fonts | Feather icons (inline SVG), Google Fonts via CSS `@import` | `src/ui/icons.py`, `src/ui/theme.py` |

`requirements.txt` also lists `matplotlib` and `joblib`; no application module imports them (Matplotlib was part of the original plan in `CLAUDE.md`; this guide's figures use it). There is no separate JavaScript frontend, no REST API server and no cloud deployment in the repository: the "backend" is the Python code Streamlit runs.

## 4.2 Technology by technology

### Python
*What:* a general-purpose language with a rich data-science ecosystem. *Why chosen:* the entire team workflow (pandas, scikit-learn, NLTK) is Python. *Alternatives:* R (strong statistics, weaker engineering/UI), Julia (fast, smaller ecosystem). *Trade-off:* Python is slow for per-row loops, which is why many functions here are vectorised.

### pandas and NumPy
*What:* tabular data frames and n-dimensional arrays. *Where:* every pipeline stage. *How it works:* columnar storage with vectorised operations (the 20M-row ratings file is read in 1M-row chunks into SQLite). *Alternatives:* Polars or Spark (faster/bigger scale), Dask. *Why not here:* the working sets (400k ratings in memory for the app, 5,000 users for labelling) fit pandas comfortably.

### SciPy sparse matrices
Ratings and TF-IDF are overwhelmingly zeros. A `csr_matrix` (compressed sparse row) stores only non-zero entries and supports fast row slicing and matrix-vector products. The TF-IDF matrix is saved with `scipy.sparse.save_npz`.

### scikit-learn
Provides `TfidfVectorizer`, `cosine_similarity`, `StratifiedKFold`, `RandomForestClassifier`, `ExtraTreesClassifier`, `HistGradientBoostingClassifier`, metrics and calibration. *Why:* standard, well-tested, consistent API. *Alternatives:* PyTorch/TensorFlow (deep models; the project explicitly excluded them to remain explainable).

### LightGBM
A gradient-boosted decision-tree library with leaf-wise tree growth and histogram binning; chapter 8 explains it from scratch. *Why selected:* among 14 candidate models/formulations evaluated with the same protocol, multiclass LightGBM had the best cross-validated macro-F1 (0.467) in the full comparison, though several models were within noise of it (Chapter 9). *Alternatives tested:* XGBoost, CatBoost, Random Forest, Extra Trees, HistGradientBoosting, stacking, voting.

### NLTK
Used only for English stop-word removal and Porter stemming in `clean_text`. The stop-word list is downloaded on first import if absent. *Alternative:* spaCy (lemmatisation), which is heavier.

### SQLite
A serverless SQL database stored in one file. *Why:* the app needs to read **one user's** ratings out of 20 million rows in milliseconds (an index on `(userId, ts)`), paginate and filter histories server-side, and store recommendation runs. Reading `rating.csv` per request would take many seconds. *Alternatives:* PostgreSQL/MySQL (better for multi-user deployments), DuckDB (fast analytics). *Limitation:* single-writer, local file; no authentication layer (the project has none, so none was added).

### Streamlit
A Python framework that turns scripts into web apps; each user interaction re-runs the script top to bottom. State lives in `st.session_state`; expensive objects use `@st.cache_resource` (engine, SQLite connection, strategy context) and `@st.cache_data` (saved reports). *Alternatives:* Flask/FastAPI + React (more flexible, more code), Dash. *Trade-off:* simplest path from Python models to a UI; limited control over layout (the project injects custom CSS in `src/ui/theme.py`).

### Optuna, XGBoost, CatBoost
Installed for the model-comparison experiments (`scripts/selector_experiments.py`). Only LightGBM's model is deployed.

### psutil
Used by `measure()` in `src/evaluation/system_metrics.py` to read process memory and CPU time while timing operations.

## 4.3 What is *not* used
No deep-learning framework, no embedding model, no large language model, no message queue, no containerisation, no REST API. If asked "why not a neural network / LLM?" see Chapter 16, questions K12-K13.

# Chapter 5. Datasets and Data Engineering

## 5.1 MovieLens

**MovieLens** is a research dataset collected by the GroupLens lab at the University of Minnesota: real people rating real movies on a movie-recommendation website. It is the standard benchmark for recommender systems, so results are comparable to the literature.

**Why MovieLens 20M** (documented in `DATASET_MIGRATION.md`): 20M ratings give enough users and overlap for collaborative filtering and classifier training; the whole file still fits on a laptop; ~27k movies is a manageable catalog; it has `link.csv` mapping to TMDB.

| File | Rows | Columns |
|---|---|---|
| `rating.csv` | 20,000,263 | `userId, movieId, rating, timestamp` (rating 0.5-5.0 in half steps; timestamp is a datetime string such as `2005-04-02 23:53:47`) |
| `movie.csv` | 27,278 | `movieId, title, genres` (genres pipe-separated: `Adventure\|Animation\|Children\|Comedy\|Fantasy`; title contains the year: `Toy Story (1995)`) |
| `link.csv` | 27,278 | `movieId, imdbId, tmdbId` (252 movies have no `tmdbId`) |
| `tag.csv` | 465,564 | `userId, movieId, tag, timestamp` (free-text user tags such as "computer animation") |

Observed in the SQLite copy: 138,493 distinct users, ratings from 0.5 to 5.0, every user has at least 20 ratings (a MovieLens rule; this is why there are **no natural cold-start users**, see 5.7). The files `genome_scores.csv` and `genome_tags.csv` ship with MovieLens but are not used.

A taste of the raw data (first rows of `rating.csv`):

```
userId,movieId,rating,timestamp
1,2,3.5,2005-04-02 23:53:47
1,29,3.5,2005-04-02 23:31:16
1,32,3.5,2005-04-02 23:33:39
```

## 5.2 TMDB

**TMDB (The Movie Database)** is a community-maintained movie database. The project uses the Kaggle export "TMDB Movies Dataset 2023", file `TMDB_movie_dataset_v11.csv` (in the copy used here: 1,515,783 rows; the migration notes quote 1,480,412 for the earlier download, an unexplained size difference between two downloads of the same dataset family: UNVERIFIED cause). Relevant columns: `id, title, overview, genres, keywords, vote_average, vote_count, popularity, release_date, original_language, poster_path`. There is no cast/crew data; the team decided not to backfill it from another file because only a minority of movies would have it, which would make the text feature inconsistent across the catalog.

Raw TMDB row for Toy Story (id 862), abbreviated:

```
title: Toy Story        vote_average: 7.971     vote_count: 17152
release_date: 1995-10-30   original_language: en   popularity: 78.404
genres: Animation, Adventure, Family, Comedy
keywords: rescue, friendship, mission, martial arts, jealousy, bullying, ...
overview: Led by Woody, Andy's toys live happily in his room until ...
```

## 5.3 Joining MovieLens and TMDB

`src/ingestion/joiner.py::join_movielens_tmdb(ml, tmdb)`:

1. Left-join MovieLens movies with `link.csv` on `movieId` to get `tmdbId` (converted to numeric).
2. **Inner-join** with the TMDB table on `tmdbId == id`. Movies without a match are dropped.
3. Parse TMDB genres and keywords (comma-separated text) into lists with `_split_names`.
4. Fill missing overviews with an empty string.
5. Build the raw text: `text_chunk = overview + " " + genres + " " + keywords`.
6. Keep columns: `movieId, tmdbId, title, genres, genre_names, keyword_names, overview, vote_average, vote_count, popularity, text_chunk` plus (added later for the UI) `release_date, original_language, poster_path`.

**Join quality (verified from the pipeline output):**

| Count | Value |
|---|---|
| MovieLens movies | 27,278 |
| with a non-null `tmdbId` | 27,026 |
| successfully joined to TMDB | **26,743 (98.04%)** |
| movies with empty overview in the merged file | 89 |
| movies whose cleaned text is empty | 11 |
| duplicate `movieId` in the merged file | 0 |

The 535 movies lost are those without a TMDB id (252) or whose id is not in the TMDB export. They never appear as candidates because everything downstream uses the merged catalog.

## 5.4 Text cleaning

`src/ingestion/text_cleaner.py::clean_text(text)`:

1. lowercase,
2. keep only alphabetic tokens with the regex `[a-zA-Z]+` (numbers and punctuation vanish),
3. remove English stop-words (NLTK list: *the, and, of...*),
4. **Porter-stem** each remaining token (*living -> live*, *happily -> happili*).

Result for Toy Story (`text_chunk_clean`, start): `led woodi andi toy live happili room andi birthday bring buzz lightyear onto scene afraid lose place andi heart woodi plot buzz ...`. Note the stems are not real words (*woodi*, *happili*): stemming chops suffixes by rule so that *toys / toy* and *plotting / plots* collapse together.

## 5.5 From movies to model-ready features

Two different "feature" worlds exist:

* **Movie features** (for retrieval): the TF-IDF vector of `text_chunk_clean` (Chapter 6, section B), plus genre sets, `vote_average`, `vote_count` and catalog popularity used by the objectives.
* **User features** (for the strategy classifier): numbers summarising one user's ratings (Chapter 8), such as average rating, number of ratings, genre concentration, popularity of what they watch, and "inner backtest" scores.

## 5.6 Splitting data (three different splits in the project)

| Purpose | Where | Rule |
|---|---|---|
| Classifier labels | `holdout_labeler.split_profile_holdout` | Per user: 70% of ratings are the *profile* (features and strategy training), 30% are the hidden *holdout* (the labelling test) |
| Classifier evaluation | `scripts/selector_experiments.py` | Users (not ratings) are split 80/20, stratified by label; 5-fold CV inside the 80%; the 20% is scored once |
| Recommendation benchmark | `src/benchmark/split.py` | Each *rating* is assigned to train / validation / test by a hash of (userId, movieId, seed): 70% / 10% / 20% |
| Legacy KPI report | `recommendation_metrics.split_user_ratings` | Per-user random 80/20 split |

The hash split is worth remembering for viva: because it uses a deterministic hash instead of random shuffling, the split is identical on every run, on every machine, regardless of row order or sample size, and needs no stored file.

## 5.7 Cold-start data and why it is simulated

MovieLens 20M admits only users with at least 20 ratings. The KPI "Cold-Start Accuracy" is defined for users with at most 5 ratings, which therefore **never occur naturally**. The evaluation code *simulates* cold users: real users whose profile is truncated to five ratings and who are scored on their untouched held-out items. The code records `mode: "simulated"` so a simulated number is never presented as natural.

## 5.7b Missing values and duplicates

* Missing overview: 89 movies (filled with empty string). Missing `tmdbId`: 252 movies (dropped by the inner join).
* Duplicate movie ids in the merged catalog: none. Exact duplicate user-feature rows: none (checked by `scripts/analyze_selector.py`: `exact_duplicate_rows = 0`, `missing_values = 0`).
* Highly correlated feature pairs (|r| > 0.95): 2; two redundant/constant features are dropped before training (`cf_cand_score_mean`, `rating_variance`).

## 5.8 Processed artifacts

| File | Created by | Content |
|---|---|---|
| `data/processed/movies_merged.csv` | `python -m src.ingestion.build_dataset` | 26,743 rows, 15 columns |
| `data/processed/ratings_clean.csv` | same | 20,000,263 ratings (the app reads only the first 400,000 rows) |
| `data/processed/tfidf_*.pkl/.npz` | `scripts/verify_ir_engine.py` | vectorizer, sparse matrix, id/title metadata |
| `data/processed/cinerankx.db` | `scripts/build_user_db.py` | SQLite: `ratings`, `movies`, `user_stats`, `recommendation_runs` |
| `data/processed/strategy_classifier.pkl`, `strategy_context.pkl`, `strategy_eval.json` | selector scripts | model, reference population, evaluation results |
| `data/processed/ease_model.npz`, `cinerankx_model.json` | `scripts/run_benchmark.py` | EASE item-item weights and tuned blend |
| `reports/*.json` | KPI/benchmark scripts | saved results the dashboard reads |
