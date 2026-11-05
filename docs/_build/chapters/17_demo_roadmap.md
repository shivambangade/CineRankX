# Chapter 17. Practical Demonstration Preparation

## 17.1 Before the examiner arrives (10-minute checklist)

| Check | Command or action | Expected |
|---|---|---|
| Environment | `cd CineRankX` then `pip install -r requirements.txt` (once) | no errors |
| Data present | `data/processed/movies_merged.csv`, `cinerankx.db`, `strategy_classifier.pkl`, `strategy_context.pkl`, `reports/kpi_report.json`, `reports/strategy_comparison.json` exist | files listed |
| Start the app | `streamlit run app.py` | browser opens at `http://localhost:8501` |
| Warm the cache | open **Run Engine** once and press Generate for user 5 | first load takes several seconds; later runs about 0.2-0.7 s |
| Sample file | `demo/demo_ranked_user5.csv` (a real exported ranked list for user 5) | present |
| Tests (optional) | `python -m pytest -q` | all pass |

## 17.2 Demonstration script (about 10 minutes)

**0:00 Introduce (30 s).** *"CineRankX is a movie recommender that predicts, per user, which strategy fits best and then ranks movies on six objectives. It uses MovieLens 20M, 20 million ratings, with TMDB metadata."*

**0:30 Overview page.** Show the four tiles (catalog titles, ratings in sample, strategy model Ready, search MRR), the candidate funnel and objective weights. *Say:* the weights come from `eval_config.yaml`; the funnel is real counts: catalog, rated titles, 200-candidate pool, latest slate. Mention strategy routing and telemetry come from the saved KPI report.

**2:00 Recommendation Engines.** Three cards: IR engine (Precision@10 0.41, MRR 0.44, search about 29 ms), strategy selector (accuracy, F1, AUC), ranker (MAP, diversity, novelty, coverage). *Say:* all numbers are read from `reports/kpi_report.json`.

**3:00 Run Engine (the main demo).**
1. Source: Live engine. User ID **5**, K **10**. The badge "User 5 found" appears; mention validation (an unknown ID disables the button).
2. Point at the six weight sliders and the "Raw total normalised to 1.00" pill. Explain each objective in one sentence (Chapter 7).
3. Press **Generate recommendations**. Note the elapsed time in the status line.
4. Read the first card: rank, title, year, genre chips, rank score, the *reason* ("ranked on strong relevance and diversity, sourced from collaborative+popularity") and the six bars.
5. **Change one thing**: raise the Diversity slider to 0.6, regenerate, show how the list changes. *Say:* the ranker is greedy and rescored against what is already chosen.
6. Press **Export ranked CSV**. Then switch Source to **Upload ranked CSV**, upload `demo/demo_ranked_user5.csv`: the same cards reappear (validated).
7. Show the **Open full profile** button.

**6:00 User Profile.** Overview tab (ratings count, average, genres, first/latest activity, distribution), History tab (search a title, filter by genre, sort, paginate: done in SQL), Preferences (favourite genres, decades, languages), Past runs (the run you just made, with strategy, weights, time), Current recommendations. Mention MovieLens has no demographics, so none are shown.

**8:00 KPI Benchmarks.** Top cards: *Overall accuracy 0.877* and *Cold-start accuracy 0.698* with the frozen baselines 0.601 and 0.609; stress that this "accuracy" is **Hit Rate@10**, not classifier accuracy. Show the Strategy comparison panels (baseline vs old pipeline vs now), then the classifier card: accuracy 0.617, majority 0.644, macro-F1, balanced accuracy, AUC, confusion matrix, per-class table, and the caption about the label-noise ceiling.

**9:30 Conclude (30 s).** *"CineRankX beats a frozen popularity baseline on held-out interactions, explains each recommendation, and we have measured exactly why the strategy classifier is limited: label noise. Next steps are better labels, a stronger content model and real feedback."*

## 17.3 Questions an examiner may ask while the app is running
* *"What does this slider do?"* It sets the weight of one objective; weights are normalised to sum to 1.
* *"Why did the list change when you raised diversity?"* The greedy ranker penalises candidates similar to those already chosen more heavily.
* *"Which strategy was chosen for this user and how sure was it?"* Shown in the run context line and in Past runs (strategy and confidence).
* *"Is this number accuracy?"* On the KPI page "Overall accuracy" is Hit Rate@10; the classifier accuracy is in the classifier card.
* *"What if I type a user who is not there?"* The Generate button is disabled and a badge says the user is not in the sample.
* *"Show me a cold user."* The cold-start numbers are simulated (five ratings); explain why (MovieLens has none).
* *"Can you show the code?"* `src/ranking_engine/ranker.py::rank` (greedy loop), `src/ranking_engine/candidates.py::strategy_source_weights`, `src/strategy_selector/strategy_context.py::build_features`.

## 17.4 Troubleshooting during the demo

| Problem | Quick fix |
|---|---|
| Page stuck on "CONNECTING..." or blank | The server stopped or restarted: rerun `streamlit run app.py` and refresh |
| "Local datasets are not ready" | Run the ingestion (`python -m src.ingestion.build_dataset`) and `scripts/verify_ir_engine.py`; or demonstrate with Upload ranked CSV |
| "User N is not in the loaded ratings sample" | Use a small user ID (1-2,900) |
| Profile page asks to build the user store | `PYTHONPATH=. python scripts/build_user_db.py` (long) or skip profiles in the demo |
| KPI page shows the "No offline report" note | Run `PYTHONPATH=. python scripts/generate_kpi_report.py 400000 1000` |
| Slow first run | The first Generate loads and caches the engine; warm it before the demo |
| Posters not showing | The History poster column loads images from TMDB's CDN and needs internet |
| A number looks different from this guide | Reports were regenerated; check the date of `reports/*.json`; the deployed classifier numbers are in `data/processed/strategy_eval.json` |

# Chapter 18. Ten-Day Study Roadmap

| Day | Study | Key concepts | Files to open | Practise | You should be able to |
|---|---|---|---|---|---|
| 1 | Overview and architecture | three engines, user journey, data flow, hybrid vs selection | `README.md`, `CLAUDE.md`, `app.py`, Chapters 1-3 | Draw the architecture from memory; viva A, B, C | Explain the project in two minutes and trace one request |
| 2 | Datasets and preprocessing | MovieLens files, TMDB join, text cleaning, splits, cold-start simulation | `src/ingestion/*`, `DATASET_MIGRATION.md`, Chapter 5 | Viva D; compute the join rate; explain `clean_text` on a sentence | Justify every data decision and the three splits |
| 3 | Recommendation algorithms | CF, TF-IDF, cosine, popularity, Bayesian rating, EASE | `candidates.py`, `tfidf_vectorizer.py`, `ease.py`, Chapter 6 | Redo the numerical examples by hand; viva E-I | Derive cosine, TF-IDF, weighted prediction |
| 4 | The six objectives | greedy MMR, each formula, trade-offs, weight tuning | `objectives.py`, `ranker.py`, `config.py`, Chapter 7 | Redo the 3-candidate greedy example; viva J | Explain why each objective exists and how they conflict |
| 5 | Strategy classifier and LightGBM | label creation, features, boosting, gradients, overfitting | `holdout_labeler.py`, `strategy_context.py`, `formulations.py`, Chapter 8 | Explain labels without notes; viva K, L | Describe training, validation, prediction and confidence use |
| 6 | Evaluation and mathematics | all metrics, CV, leakage, bootstrap, significance | `src/evaluation/*`, `src/benchmark/*`, Chapter 10 | Recompute precision, recall, F1, AUC from the confusion matrix; viva M | Compute every metric by hand and say when each misleads |
| 7 | Codebase walkthrough | package responsibilities, key functions | Chapter 11, Appendix D, run a debugger on `rank()` | Open each file and state its input and output | Navigate the repository confidently |
| 8 | Differentiation and contribution | comparison table, innovation, limitations, publishability | Chapters 13, 15 | Viva Q, R, S; write your own contribution statement | Say what is new, what is established, what is weak |
| 9 | Mock viva | accuracy investigation, label noise, honest reporting | Chapter 9, Chapter 16 | Answer 40 random questions aloud; record yourself; peer mock | Answer "why is your classifier worse than baseline?" calmly |
| 10 | Demonstration and revision | demo flow, troubleshooting, key numbers | Chapter 17, Appendix B | Run the full demo twice with a timer; revise Appendix B | Deliver a 10-minute demo and handle failures |
