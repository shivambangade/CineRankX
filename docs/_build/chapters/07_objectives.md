# Chapter 7. The Six Recommendation Objectives

All six objectives are fully implemented in `src/ranking_engine/` (`objectives.py`, `predictor.py`, `ranker.py`, `config.py`). Each objective produces a score in **[0, 1]** for a candidate movie so that the six numbers can be combined with weights that mean what they say.

## 7.1 How the ranker combines them

The weights live in `eval_config.yaml` (`ranking_objectives.weights`) and are normalised to sum to 1 by `RankingWeights`:

| Objective | Default weight |
|---|---|
| relevance | 0.30 |
| diversity | 0.15 |
| novelty | 0.15 |
| coverage | 0.10 |
| popularity_quality | 0.15 |
| predicted_rating | 0.15 |

For a candidate *m*, given the list *S* selected so far:

$$ Score(m | S) = sum over o in six objectives of w_o x obj_o(m, S) $$

Four objectives (relevance, novelty, popularity/quality, predicted rating) depend only on the movie and user, so they are computed once. **Two objectives depend on what is already in the list**: diversity and coverage. That is why `MultiObjectiveRanker.rank` is a **greedy sequential selection**: at every step it rescores all remaining candidates against the current list, picks the best, appends it, and repeats until K movies are chosen. Sorting once by a fixed score would be wrong for these two objectives. The per-objective values stored with each result are the ones that existed *at the moment of selection*, which makes the score breakdown shown in the UI auditable.

## 7.2 Objective 1: Relevance

* **Meaning:** how well the movie matches this user. **Why it matters:** the primary accuracy signal.
* **Calculation:** the candidate's blended score from `CandidateGenerator.generate`: each source's raw scores min-max normalised to [0, 1], multiplied by a strategy-dependent source weight and summed, then normalised again. In the benchmark pipeline the relevance is the min-max normalised EASE/popularity percentile within the 200-candidate pool.
* **Data required:** user ratings, TF-IDF matrix, neighbour ratings, rating counts.
* **Trade-off:** maximising relevance alone yields repetitive lists (nine sequels of the same franchise) and overrepresents popular titles.

## 7.3 Objective 2: Diversity

* **Meaning:** how unlike the movies already selected this one is. Defined by the **MMR** (maximal marginal relevance) idea:

$$ diversity(m | S) = 1 - max over s in S of sim(m, s),   diversity = 1 when S is empty $$

* **Why max, not mean:** one near-duplicate is redundant even if the movie differs from the other nine; a mean would dilute that. 
* **Similarity source:** `MultiObjectiveRanker._similarity_lookup`: cosine similarity of the TF-IDF vectors of the candidates; for candidates missing from the matrix it falls back to **genre Jaccard** so unknown items are not treated as "maximally diverse".
* **Interaction:** pulls against relevance (the most relevant items resemble each other).

## 7.4 Objective 3: Novelty

* **Meaning:** how unexpected an item is, measured by **self-information**: rare items carry more information.

$$ novelty(m) = minmax( -log2 p(m) ),   p(m) = observed(m) / sum of observed(all movies) $$

where `observed = MovieLens rating count + TMDB vote_count` (combining both keeps the measure defined for the whole catalog; counts below 1 are floored at 1). Min-max scaling maps the range to [0, 1].

*Example.* Three movies with 1,000, 100 and 10 observations: p = 0.901, 0.090, 0.009, so -log2 p = 0.151, 3.47, 6.79 and the scaled novelty is 0.00, 0.50, 1.00.

* **Why it matters:** counteracts popularity bias, supports discovery. **Tension:** deliberately opposed to popularity/quality.

## 7.5 Objective 4: Coverage

* **Meaning (per item):** how much of the catalog's *genre space* this movie opens up that the current list does not already cover:

$$ coverage(m | S) = |genres(m) minus covered genres| / |genres(m)| $$

An item whose genres are all already covered scores 0; one with no genre metadata scores 0 (it cannot be shown to broaden coverage).

* **Honest note:** the catalog-level *Coverage* **KPI** (share of the whole catalog that ever appears in any user's list) cannot be a per-item score; the objective is a **per-item proxy** for it, as stated in the docstring of `coverage_score`. This is a *partial* implementation of the idea of catalog coverage and should be described that way in a viva.

## 7.6 Objective 5: Popularity / Quality

* **Meaning:** how well-regarded the movie is by many people, shrunk for reliability, using the Bayesian weighted rating of section 6.C.3, then min-max scaled.
* **Difference between popularity and quality:** *popularity* is how many people engaged (a blockbuster with mediocre reviews is popular); *quality* is how highly they rated (a small gem can be high quality, low popularity). The Bayesian weighted rating blends the two: a high average with many votes scores best; a high average from three votes is pulled toward the mean.
* **Interaction:** pulls against novelty; resolving that tension is exactly what the weights are for.

## 7.7 Objective 6: Predicted Rating

* **Meaning:** the rating we expect this user to give, from the bias model (section 6.D), rescaled to [0, 1] as

$$ predicted(u, m) = (r^(u,m) - 0.5) / (5.0 - 0.5) $$

* **Why rescaled:** the other five objectives are already in [0, 1]; leaving the predictor on a 0.5-5 scale would silently let it dominate.
* **Limitation:** it only knows the global mean and user/item biases, so it cannot say "this user prefers thrillers"; that personalisation comes from relevance.

## 7.8 A worked greedy example (illustrative numbers)

Weights: relevance .30, diversity .15, novelty .15, coverage .10, popularity/quality .15, predicted rating .15. Three candidates (numbers are made up to show the mechanics):

| Candidate | Relevance | Novelty | Pop/Quality | Pred. rating | Genres |
|---|---|---|---|---|---|
| A: *Toy Story 2* | 0.95 | 0.20 | 0.70 | 0.85 | animation, comedy |
| B: *Toy Story 3* | 0.93 | 0.25 | 0.72 | 0.84 | animation, comedy |
| C: *Heat* | 0.60 | 0.45 | 0.65 | 0.75 | action, crime |

**Step 1** (list empty: diversity = 1, coverage = 1 for everyone): A = 0.3(0.95) + 0.15(1) + 0.15(0.20) + 0.10(1) + 0.15(0.70) + 0.15(0.85) = **0.7975**; B = **0.8005**; C = 0.18 + 0.15 + 0.0675 + 0.10 + 0.0975 + 0.1125 = 0.7075. **B is picked.**

**Step 2** (list = {B}; assume sim(A,B) = 0.8 and sim(C,B) = 0.05):
A: diversity 0.2, coverage 0 (genres already covered) -> 0.285 + 0.03 + 0.03 + 0 + 0.105 + 0.1275 = **0.6175**.
C: diversity 0.95, coverage 1 -> 0.18 + 0.1425 + 0.0675 + 0.10 + 0.0975 + 0.1125 = **0.7000**. **C is picked**, although A is far more relevant. That is the multi-objective ranker doing its job: the second slot goes to something different.

## 7.9 Why not maximise relevance alone?
Because the highest-relevance items are mutually similar and popular. The project measured the effect directly (README, "Weight sensitivity"): raising diversity 0.15 -> 0.60 and coverage 0.10 -> 0.40 lifted mean diversity 0.952 -> 0.972 and genre spread 13 -> 16 distinct genres, at the cost of mean relevance 0.814 -> 0.657. Because that trade-off is real, the weights are configurable and exposed as sliders in the Run Engine page.

## 7.10 Tuning the weights (benchmark)
For the benchmark pipeline the weights were *searched* on the validation split (`src/benchmark/tuning.py::tune_weights`): relevance weight sampled in [0.5, 0.95], the rest by a Dirichlet draw, maximising validation NDCG@10 subject to **diversity not falling below the frozen baseline's diversity**. Tuned overall weights: relevance 0.576, diversity 0.064, novelty 0.102, coverage 0.117, popularity/quality 0.110, predicted rating 0.032 (`data/benchmark/cinerankx_params.json`).

## 7.11 Genre-overlap relevance gate (not a seventh objective)
The gate of section 6.B.5 is a *pre-ranking filter on IR relevance*, tunable by `ranking_objectives.genre_relevance_gate.floor` (default 0.25). It exists because TF-IDF measures shared vocabulary, not shared tone.

# Chapter 8. The Strategy Classifier: Deep Technical Explanation

## 8.1 Why a classifier at all?
Because no strategy dominates. If the best strategy differed *randomly* per user a classifier could not help; if it differs *systematically* (with history size, taste concentration, neighbourhood strength) then features describing the user can predict it, and the right strategy can be favoured for that user. The classifier's job is to estimate that relationship.

## 8.2 What it predicts
Three classes, encoded alphabetically (`LabelEncoder`): **0 collaborative, 1 content_based, 2 popularity**. Output: a probability for each class. `confidence = max probability`, and the predicted strategy is the argmax. These two numbers feed `strategy_source_weights` (section 3.4).

## 8.3 How the training labels are made (the most important idea)
A label must say "which strategy actually worked best for this user". CineRankX answers by **backtesting** each strategy on the user's own hidden data:

![Training pipeline](figs/training_pipeline.png)

1. **Sample users.** `dataset.sample_user_ids`: 5,000 random users (seed 42) from `user_stats` with 30 to 1,000 ratings (`min_history`, `max_history`).
2. **Split each user's ratings** (`holdout_labeler.split_profile_holdout`): 70% **profile** (visible), 30% **holdout** (hidden); at least one rating on each side.
3. **Build the three strategies from the profile data only** (`strategy_labeler._BacktestContext`): the popularity list = the most-rated movies in the profile data; the CF similarity structures from the profile ratings; the TF-IDF matrix for content.
4. **Recommend N movies per strategy** (N = `label_top_n` = 30) for each user, excluding movies already in the profile.
5. **Score**: hit-rate = (number of recommended movies that appear in the user's holdout) / (number of holdout movies).
6. **Label** = the strategy with the highest hit-rate; ties go to popularity (`formulations.winner`), because a tie most often means all scores are 0 and popularity is the safe default.

> **The two disjoint worlds.** Features are computed from the *profile* ratings only; labels from the *holdout* only. If features could see the holdout the model could "cheat". Tests `test_features_use_only_the_ratings_given` and `test_holdout_labels_never_use_holdout_ratings` guard this.

## 8.4 The features (47 after removing 2 redundant ones)
Computed by `profile_features.extract_profile_features` (vectorised over users) and `strategy_context.build_features`. One example row from `strategy_labels_*_cbase.csv` is user 104 (698 ratings, average 2.84).

| Group | Features | Intuition |
|---|---|---|
| Activity | `num_ratings`, `log_num_ratings` | more history => CF/content have evidence |
| Rating behaviour | `avg_rating`, `rating_std`, `rating_min`, `rating_max`, `rating_variance`*, `liked_share`, `user_bias` | harsh vs generous raters; `user_bias` = mean(user rating - item mean rating) |
| Genre structure | `num_genres`, `genre_diversity`, `top_genre_share`, `genre_entropy`, 19 x `affinity_<Genre>` | focused fans suit content; eclectic users do not. `genre_entropy` is Shannon entropy of the genre distribution scaled to [0,1] |
| Popularity of taste | `niche_fraction`, `mean_popularity_pct`, `top100_share`, `top1000_share`, `profile_surprisal` | users who watch only blockbusters are well served by popularity |
| Time | `log_active_span_days`, `recent_activity_share` | sustained vs one-sitting raters |
| Engine outputs | `cb_top_cos_mean`, `cb_coherence`, `cf_top_sim_mean`, `cf_top_sim_max`, `cf_cand_score_mean`* | how strongly the content and neighbourhood engines "pull" for this user |
| Strategy-aware ("inner backtest") | `inner_cb`, `inner_cf`, `inner_pop`, `inner_gap_cb_pop`, `inner_gap_cf_pop` | split the *visible* profile again (3 splits, 25% pseudo-holdout) and measure each strategy's hit-rate, an estimate of the same quantity the label measures |

(* removed as redundant, |correlation| > 0.98.) The most influential features by mean absolute TreeSHAP value on the held-out users are `cf_top_sim_max`, `cf_top_sim_mean`, `top100_share`, `profile_surprisal`, `cb_coherence`, `inner_cf`, `top_genre_share`, `affinity_Sci-Fi`.

## 8.5 Training, validation and testing protocol
(`scripts/selector_experiments.py`, `scripts/selector_phase2.py`)

* **All users are kept** (4,993). The earlier practice of dropping "ambiguous" users was removed because it made accuracy look better than it was.
* **Stratified 80/20 split by label** (seed 42): 3,994 development users and 999 held-out users. Held-out users are *never* used to choose anything.
* **5-fold stratified cross-validation** on development users for every candidate; model choice by CV macro-F1.
* **Hyperparameters** tuned by Optuna on development users (TPE sampler, 3-fold macro-F1).
* **Final**: the winner is refit on development users and scored once on the held-out users; the *deployed* model is then refit on all 4,993 users (so the held-out numbers describe a model that never saw those users).

**Deployed LightGBM hyperparameters** (`selector_experiments_inner3.json`): 357 trees, learning rate 0.0219, 27 leaves, minimum 50 samples per leaf, row subsample 0.60 (every iteration), column subsample 0.463, L2 regularisation 0.0035, plus **sample weights** = balanced class weight x (1 + 3.50 x min(label margin / 0.1, 1)), which gives clearer labels more influence while still using every user.

## 8.6 LightGBM explained from scratch

**Decision tree.** A tree asks a sequence of yes/no questions about features ("is `top100_share` < 0.1?") and ends in leaves holding a value. It is easy to read but unstable and overfits.

**Gradient boosting.** Build an *ensemble* of small trees one at a time, each correcting the mistakes of those before it:

$$ F_m(x) = F_(m-1)(x) + eta x f_m(x) $$

*eta* is the **learning rate**; smaller values need more trees but generalise better.

**Loss and gradient/Hessian (multiclass).** The model outputs a raw score per class; probabilities are the **softmax**: p_k = exp(F_k) / sum_j exp(F_j). The loss is the log-loss (cross-entropy), -sum over classes of y_k log p_k. For each sample and class, boosting needs the first derivative (**gradient**) and second derivative (**Hessian**) of the loss with respect to the raw score:

$$ g_k = p_k - y_k,   h_k = p_k (1 - p_k) $$

Intuition: the gradient is "how wrong, and in which direction"; the Hessian says "how confident/curved", and is used to scale the step.

**Splitting and leaf value.** For a set of samples in a leaf, with *G* = sum of gradients and *H* = sum of Hessians, the optimal leaf value and the quality of a split are

$$ w* = - G / (H + lambda) $$
$$ Gain = 1/2 [ G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - (G_L + G_R)^2/(H_L + H_R + lambda) ] $$

*lambda* is the L2 regularisation (`reg_lambda`), which shrinks leaf values and stops tiny leaves from dominating.

**What makes LightGBM different.** (1) **Histogram binning**: continuous features are bucketed into at most 255 bins, so searching for splits is fast. (2) **Leaf-wise growth**: instead of growing every level evenly, it always splits the single leaf with the largest gain, producing deeper, more accurate but more overfit-prone trees; `num_leaves` and `min_child_samples` are the main controls. (3) Optional GOSS/EFB speed-ups (not configured here).

**Overfitting and regularisation.** A model that memorises training noise has high training accuracy and lower test accuracy (here: train 0.77-0.88 vs test 0.61-0.63). Controls: small learning rate, limited leaves, minimum leaf size, row and column subsampling, L2 regularisation, cross-validation for model selection.

## 8.7 Prediction path
`app.py::recommend` -> `strategy_context()` (cached reference population) -> `build_features(user_ratings, movies, context, ...)` -> `classifier.predict(features)` returns `predicted_strategy` and `confidence`. `StrategyClassifier.predict` runs one forward pass (`predict_proba`), takes the argmax class and its probability. Latency: the saved evaluation (`strategy_eval.json`) records a mean of 79 ms per user for features plus prediction (p95 87 ms); an isolated measurement split it into roughly 72 ms of feature building and 17 ms of prediction.

## 8.8 Other approaches genuinely tested
(Chapter 9 has the tables.) Multiclass with six model families (LightGBM, XGBoost, CatBoost, Extra Trees, Random Forest, HistGradientBoosting); one-vs-rest; pairwise comparison (three binary classifiers); regression of expected hit-rate per strategy ("utility"); LambdaRank over the three strategies; soft-voting and stacking ensembles; sigmoid probability calibration; per-class threshold optimisation; SMOTE (early work, disabled); an uncertainty-aware utility selector with a popularity fallback margin (`src/strategy_selector/utility.py`).

## 8.9 Confusion matrix and feature importance
Confusion matrix of the deployed model on 999 held-out users (rows true, columns predicted):

![Confusion matrix](figs/confusion.png)

| true \ predicted | collaborative | content_based | popularity |
|---|---|---|---|
| collaborative | 191 | 6 | 137 |
| content_based | 11 | 4 | 7 |
| popularity | 216 | 6 | 421 |

Per-class precision / recall / F1: collaborative 0.46 / 0.57 / 0.51 (334 users); content_based 0.25 / 0.18 / 0.21 (22 users); popularity 0.75 / 0.65 / 0.70 (643 users). Read it as: popularity and collaborative are partly separable; content_based is almost never the winner, so there is little to learn from.
