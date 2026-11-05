# Executive Summary

**CineRankX** is a movie recommendation system built for the Department of Artificial Intelligence and Machine Learning at Walchand College of Engineering, Sangli (mini project, AY 2026-27). Its central idea, taken directly from the project README, is that *existing recommender systems fix one strategy and rank by a single similarity score, ignoring diversity and per-user variation*. CineRankX therefore does two things that a conventional recommender does not:

1. It **predicts, for each user, which of three recommendation strategies is likely to work best** (content-based, collaborative filtering, or popularity) using a machine-learning classifier.
2. It **re-ranks candidate movies on six objectives at once** (relevance, diversity, novelty, coverage, popularity/quality and predicted rating) instead of sorting by one similarity number.

The system runs on **MovieLens 20M** (20,000,263 ratings from 138,493 users on 27,278 movies) enriched with **TMDB 2023** metadata (overviews, keywords, genres, posters). It is built in Python with pandas, NumPy, SciPy, scikit-learn, NLTK, LightGBM, SQLite and a Streamlit web application.

## What exists today (verified against the repository)

| Area | State | Evidence |
|---|---|---|
| Data pipeline | Complete: MovieLens + TMDB join (26,743 of 27,278 movies matched, 98.04%), NLTK text cleaning | `src/ingestion/`, `data/processed/movies_merged.csv` |
| Module 2: IR engine | Complete: TF-IDF (5,000 terms, bigrams) + cosine search + Trie autocomplete | `src/ir_engine/` |
| Module 3: strategy selector | Complete but **modest accuracy**: LightGBM multiclass, held-out accuracy 0.617 vs majority class 0.644 | `src/strategy_selector/`, `data/processed/strategy_eval.json` |
| Module 4: ranking engine | Complete: three candidate sources, six-objective greedy re-ranking, tunable weights | `src/ranking_engine/` |
| Evaluation suite | Complete: 23 KPIs named in `eval_config.yaml` | `src/evaluation/`, `reports/kpi_report.json` |
| Web application | Complete: Overview, Engines, Run Engine, User Profile (User 360), KPI Benchmarks | `app.py`, `src/ui/` |
| Frozen benchmark | Complete: deterministic split, frozen popularity baseline, EASE-based CineRankX relevance | `src/benchmark/`, `reports/strategy_comparison.json` |
| Tests | 272 pytest tests passing at last full run | `tests/` |

## The headline numbers, honestly stated

* **Recommendation quality** (frozen benchmark, 2,000 test users): CineRankX reaches **Hit Rate@10 = 0.877**, against **0.601** for the frozen most-popular baseline and **0.751** for the earlier CineRankX pipeline.
* **Strategy classifier** (999 held-out users, none removed): **accuracy 0.617**, **macro-F1 0.472**, **balanced accuracy 0.469**, **ROC-AUC 0.735**. Always predicting "popularity" would score 0.644 accuracy, so *the classifier does not beat the majority class on plain accuracy*.
* **What strategy choice is worth**: the selector's chosen strategy achieves a mean hit-rate of **0.128** against **0.112** for always using popularity and **0.148** for an oracle (an optimistic upper bound). The gain over popularity is **+0.0155, 95% CI [+0.011, +0.020]**, so it is statistically real even though classification accuracy looks weak.
* **Best experiment so far (not deployed)**: with a more reliable label definition (50 recommendations, 40% holdout) and an improved content-based strategy, held-out accuracy rises to **0.677 against a 0.564 majority rate**, macro-F1 to **0.586**, ROC-AUC to **0.807**, and the selector's gain over always-popularity doubles (**+0.037, 95% CI [+0.030, +0.044]**). The two settings use different label definitions, so compare each with its own baseline (section 9.10b).
* **Why accuracy is capped (deployed setting)**: two independent re-labellings of the same users agree on only about **65-73%** of users, and a perfect model that knew each user's true hit-rates would reach only about **0.76**. The labels, not the algorithm, are the bottleneck. This is a central viva point (Chapter 9).

> **How to use this guide.** Read Chapters 1-3 for the story and the system, 5-8 for data and algorithms, 9-10 for evaluation maths, 11-12 for the code, 16 for viva questions and 17 for the demo script. Chapter 18 is a 10-day study plan.

# Chapter 1. Project History and Origin

This chapter reconstructs the story from evidence: the git history (21 commits at the time of writing), `README.md`, `DATASET_MIGRATION.md`, `CLAUDE.md`, `eval_config.yaml` and the code itself. Nothing here is invented; where the repository is silent the text says so.

## 1.1 Why the project exists

The README states the motivation in one sentence: *"Existing recommender systems fix one strategy and rank by a single similarity score, ignoring diversity and per-user variation. CineRankX predicts the best strategy per user via ML classifiers and ranks candidates on six weighted objectives, for relevant, diverse, explainable recommendations."*

`CLAUDE.md` (the project context file) adds the structure of the original plan: **three engines built in a fixed order**:

1. A classical information-retrieval engine (TF-IDF + Trie).
2. An adaptive machine-learning strategy-selection engine.
3. A multi-objective hybrid ranking engine.

The same file records the working conventions the team adopted: one module at a time, pytest tests before moving on, commit after each module passes, ranking weights exposed as configuration (never hard-coded), and no deep learning ("this stays explainable and rule-based (TF-IDF + classical ML classifiers only)").

## 1.2 Questions this chapter answers

| Question | Short answer (details below) |
|---|---|
| What problem were we solving? | Single-strategy, single-score recommenders ignore per-user differences and list diversity |
| What was the original idea? | Predict the best strategy per user, then rank on six objectives |
| What did we implement? | All three engines, an evaluation suite, a web UI, a SQLite user store, a frozen benchmark |
| What changed along the way? | Datasets were migrated; popularity definition was corrected; the classifier and its labels were re-engineered; the UI was redesigned |
| What is incomplete? | Content-based strategy is weak; classifier accuracy is label-limited; no live user feedback; no contextual bandit |

## 1.3 Chronological timeline (from `git log`)

| Date | Commit (author) | What happened |
|---|---|---|
| 2026-08-23 | `eac5871` Initial commit; `f1f836e` Module 1 finished (Shivam Bangade) | Repository created; first data-ingestion module with `ml-latest-small` and TMDB 5000 |
| 2026-08-23 | `ee397bd` Module 1: migrate to TMDB and MovieLens datasets | **Dataset migration** to MovieLens 20M + TMDB 2023 (details in 1.4) |
| 2026-08-23 | `9e393c1` Module 2: Classical IR Engine (18 tests) | TF-IDF + Trie engine |
| 2026-08-23 | `85a586e` Module 3: Adaptive ML Strategy Selection Engine (31 tests) | First classifier (Random Forest) with backtest labels |
| 2026-08-25 | `ce108d5` Module 3: SMOTE investigation (41 tests) | SMOTE tested, **shipped disabled** |
| 2026-08-25 | `11effea` Module 4: Multi-Objective Hybrid Ranking Engine (72 tests) | Six-objective ranker |
| 2026-08-25 | `3583413` Evaluation suite: all 23 KPIs (71 tests) | Evaluation module |
| 2026-08-25 | `c273ba3` Rank popularity by observed rating counts, not TMDB popularity | Important bug fix (1.5) |
| 2026-08-25 | `e536505`, `9acaca7` README with evaluation suite and final KPI report | Documentation milestone |
| 2026-09-30 | `116f21a` Recommendation results dashboard (Shardul Funde, PR #1) | First Streamlit results page |
| 2026-10-01 | `74e14c5` (prathmesh0077) | Comment in `src/ui/results.py` |
| 2026-10-15 | `8ebab03`, `47d423b` (Aditya Patil) | README updates |
| 2026-10-15 | `ef395a9` Redesign UI and add User 360 profile with SQLite user store | Analytics UI, user profiles, run history |
| 2026-11-04 | `d2c3d22` Improve strategy selector: leak-free labels, richer features, CV model selection | Classifier re-engineering |
| 2026-11-05 | `3938f34` Frozen benchmark, EASE relevance, utility selection and selector diagnostics | Benchmark and diagnostics |

> **Note on dates.** Dates are copied from `git log --date=short`. Contributor names are the git author names. Which team member did which part of the later work is not recorded beyond these author fields.

## 1.4 The dataset migration (a major early decision)

`DATASET_MIGRATION.md` documents three stages:

| Stage | Datasets | Join result | Verdict |
|---|---|---|---|
| Before | MovieLens `ml-latest-small` (100,836 ratings, 9,742 movies) + TMDB 5000 | 3,537 of 9,742 joined (**36.3%**) | TMDB 5000 is a fixed 5,000-movie slice, so most MovieLens films are absent |
| Considered, not adopted | MovieLens `ml-32m` + TMDB 5000 | projected **5.3%** (4,634 / 87,585) | More MovieLens movies make the *rate* worse |
| After (current) | MovieLens 20M + TMDB 2023 (near-full TMDB export) | 26,743 of 27,278 joined (**98.04%**) | Solved by upgrading the *bottleneck* dataset |

Two consequences recorded in the same document: the TMDB export has **no cast/crew columns**, so `text_chunk` is overview + genres + keywords only (a deliberate scope decision, not a bug), and TMDB genres/keywords are plain comma-separated strings (not JSON), so the joiner uses a simple split.

## 1.5 Problems found and fixed along the way

These are real engineering events from the history and migration notes:

1. **Popularity defined the wrong way (commit `c273ba3`).** Both the strategy backtest and the candidate generator ranked "popular" by TMDB's `popularity` column. That metric is live and recency-weighted, while MovieLens ratings stop in 2015, so the TMDB top-10 was touched by only 39% of users (87% for the most-rated top-10). Switching to observed rating counts changed the label mix from collaborative 84% / popularity 9% / content 7% to roughly 52% / 45% / 3%, and raised classifier lift over the majority baseline from -0.045 to +0.144.
2. **A flattering cold-start baseline.** The evaluation's popularity baseline used the same TMDB metric, making CineRankX look +0.35 better than it was. With the correct baseline the cold-start lift was -0.03 and, after the popularity fix in both engines, +0.02.
3. **SMOTE investigated and rejected.** Over 10 random seeds SMOTE raised content-based F1 by about 0.038 but left ROC-AUC flat (0.605-0.627) and cost accuracy in 10 of 10 seeds. Conclusion in the migration notes: rebalancing moves the decision threshold; it does not create information.
4. **Label noise.** Multi-split averaging was added to the labeller because single-split winners flipped by chance (popularity "won" for 37.7% of users on one split but 8.9% when averaged over five).

## 1.6 How the project evolved after the first four modules

* **UI (Sept-Oct 2026):** a simple Streamlit results page (`app.py`) was added by a team member; later restyled into a multi-section analytics dashboard with Overview, Engines, Run Engine, User Profile and KPI pages, and a local SQLite store (`data/processed/cinerankx.db`) holding all 20M ratings and every recommendation run.
* **Evaluation integrity (Nov 2026):** the strategy classifier's labels were rebuilt so features can never see the ratings that decide a label (a "profile / holdout" split), and a **frozen benchmark** with an independent baseline was added so improvements cannot move the reference.
* **Better relevance model:** an EASE item-item linear model (`src/ranking_engine/ease.py`) was added as a candidate source in the benchmark pipeline, raising Hit Rate@10 from 0.60 (baseline) to 0.877.
* **Diagnostics:** scripts such as `scripts/analyze_selector.py`, `label_experiments.py`, `content_experiments.py` and `selector_phase2.py` measure *why* the classifier is limited, rather than only trying new algorithms.

## 1.7 Status ledger

| Category | Items |
|---|---|
| **Initially proposed and implemented** | IR engine (TF-IDF + Trie); strategy classifier; six-objective ranker; 23-KPI evaluation suite |
| **Implemented later / improved** | Dataset migration; popularity definition; leak-free labels; richer features; frozen benchmark; EASE relevance; web dashboard; SQLite store; user profiles |
| **Implemented but disabled / not adopted** | SMOTE (`use_smote=False`); utility-based selector with fallback (evaluated, ties with the simple classifier); improved content model and larger-budget labels (evaluated end to end, clearly better in the experiment of section 9.10b, but **not deployed**) |
| **Incomplete or weak** | Content-based strategy hit-rate (0.016); classifier accuracy ceiling; cold-start gain is small |
| **Future possibilities (not built)** | Contextual bandits, real user feedback loop, deep/embedding models, hybrid fusion of strategy lists, cast/crew metadata |

## 1.8 Weekly progress reports (supporting evidence)

The untracked file `demo/weekly_reports_week7-12.md` (and its `.docx` twin) contains the project's weekly sheets for weeks 7 to 12, addressed to the mentor, **Prof. Sahil Nadaf**. It states its own assumptions (the split into Team A data/IR, Team B ML selector, Team C ranking/UI, and that weeks 3 to 6 covered TF-IDF, classifier and ranker are *assumed*, not recorded), so treat team labels as UNVERIFIED. Its content is consistent with the git history:

| Week | Focus recorded in the report |
|---|---|
| 7 | Evaluation suite and the 23 KPIs; cold-start simulated; SMOTE shipped off; popularity switched to observed counts |
| 8 | Streamlit results dashboard merged by pull request; ranked-CSV validation |
| 9 | UI redesign (design tokens, Feather icons), runner with six weight sliders, reset and validation |
| 10 | Full MovieLens 20M + TMDB 2023 load (26,743 movies matched), classifier retrained (accuracy 0.52 at that time, equal to the majority baseline), SQLite store with 20,000,263 ratings |
| 11 | User 360 profile, run history logging, no-authentication note |
| 12 | End-to-end test, KPI report saved as JSON, demo script; **cloud deployment blocked** (no admin access to the GitHub repository; the 1.4 GB data is not in git); plan: demo locally or fork and bundle a small sample |

> **Caution for the viva:** the sheets contain "Mentor response" lines (for example "keep the numbers honest"). These are as written in the report; they are not independently verifiable from the repository, so quote them only if you know they match what your mentor actually said.
