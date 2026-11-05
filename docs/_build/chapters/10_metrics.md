# Chapter 10. Model Evaluation and the Mathematics Behind It

CineRankX evaluates three different things with different metric families: **search** (does the IR engine find on-topic movies?), **recommendation lists** (are the lists accurate, diverse, novel?) and **the strategy classifier** (does it predict the right label?). A fourth group measures **system** cost. The names are fixed in `eval_config.yaml` (23 KPIs) and checked by `build_report(strict=True)`, which refuses to finish if any is missing.

| Group | KPIs (exact names) |
|---|---|
| Search (5) | Precision@10, Recall@10, MRR, NDCG, Search Latency |
| Recommendation (8) | Precision@K, Recall@K, F1, MAP, Coverage, Diversity, Novelty, Cold-Start Accuracy |
| Machine learning (6) | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| System (4) | Response Time, Memory Usage, CPU Usage, Throughput |

## 10.1 Classification metrics

Use the real confusion matrix of the deployed strategy classifier on 999 held-out users (rows true, columns predicted):

| | pred collab | pred content | pred popularity | row total |
|---|---|---|---|---|
| **true collaborative** | 191 | 6 | 137 | 334 |
| **true content_based** | 11 | 4 | 7 | 22 |
| **true popularity** | 216 | 6 | 421 | 643 |
| column total | 418 | 16 | 565 | 999 |

### Confusion matrix
A table counting, for every true class, how many users were predicted as each class. The diagonal is correct predictions. It shows *which* mistakes happen (here most true-popularity users are mistaken for collaborative: 216).

### Accuracy
$$ accuracy = (number of correct predictions) / (total) = (191 + 4 + 421) / 999 = 0.6166 $$
*Interpretation:* 61.7% of users were assigned their labelled strategy. *Misleading when:* classes are imbalanced (always predicting popularity scores 643/999 = 0.644).

### Precision and recall (per class)
$$ precision_c = TP_c / (TP_c + FP_c),   recall_c = TP_c / (TP_c + FN_c) $$
For collaborative: TP = 191, FP = 11 + 216 = 227 (predicted collaborative but not), FN = 6 + 137 = 143. Precision = 191/418 = **0.457**; recall = 191/334 = **0.572**. *Precision* answers "when we say collaborative, how often are we right?"; *recall* answers "of the real collaborative users, how many did we find?".

### F1-score
The harmonic mean of precision and recall, which punishes imbalance between them:
$$ F1 = 2 P R / (P + R) = 2 x 0.457 x 0.572 / (0.457 + 0.572) = 0.508 $$
Content-based: P = 4/16 = 0.25, R = 4/22 = 0.182, F1 = **0.211**. Popularity: P = 421/565 = 0.745, R = 421/643 = 0.655, F1 = **0.697**.

### Macro versus weighted averages
* **Macro**: plain mean over classes, treating each class equally: (0.508 + 0.211 + 0.697)/3 = **0.472**. Highlights poor minority-class performance.
* **Weighted**: mean weighted by class size: (334 x 0.508 + 22 x 0.211 + 643 x 0.697)/999 = **0.623** (this is the "F1" shown on the dashboard).
* **Balanced accuracy** = mean of per-class recall = (0.572 + 0.182 + 0.655)/3 = **0.469**.

### ROC curve and AUC
For one class versus the rest, sweep a probability threshold from 1 to 0 and plot the **true-positive rate** (recall) against the **false-positive rate** at each threshold. The area under that curve (AUC) equals the probability that a randomly chosen positive user receives a higher predicted probability than a randomly chosen negative user.

*Numerical example.* Positive scores 0.9, 0.7, 0.4; negative scores 0.6, 0.3, 0.2. Of the 9 positive-negative pairs, the positive is higher in 3 + 3 + 2 = 8, so AUC = 8/9 = **0.889**. Chance = 0.5; perfect = 1.0. For three classes, **one-vs-rest** AUC is computed per class and averaged (macro): 0.735 for the deployed classifier. *Why it matters:* it measures *ranking* quality independent of the threshold, and the downstream ranker consumes probabilities, not hard labels. *Misleading when:* the classes are extremely imbalanced or probabilities are miscalibrated.

## 10.2 Ranking and recommendation metrics

Take one user, the recommended list **[A, B, C, D, E]** and the set of held-out movies the user liked **{B, D, F}** (3 relevant). Hits are at ranks 2 and 4.

| Metric | Formula | Value |
|---|---|---|
| **Precision@K** | relevant in top-K / K | 2/5 = **0.40** |
| **Recall@K** | relevant in top-K / total relevant | 2/3 = **0.667** |
| **Hit Rate@K** | 1 if at least one relevant in top-K, else 0 (averaged over users) | **1** |
| **Average Precision@K** | sum over hit ranks of precision at that rank, divided by min(K, #relevant) | (1/2 + 2/4) / 3 = **0.333** |
| **MAP@K** | mean of AP over users | mean of per-user values |
| **NDCG@K** | DCG / ideal DCG, with DCG = sum of 1/log2(rank + 1) over hits | DCG = 1/log2 3 + 1/log2 5 = 1.062; ideal = 1 + 1/log2 3 + 1/log2 4 = 2.131; NDCG = **0.498** |

**Interpretation and use in CineRankX.** *Precision* is how clean the list is; *recall* is how much of what the user likes was found; *Hit Rate* is the most forgiving ("did we get anything right?"), and is used as the dashboard "accuracy"; *MAP* and *NDCG* reward putting hits early. A "relevant" movie is a held-out movie the user rated **>= 4.0** (`positive_threshold`).

*When misleading.* Recall@K is capped when a user has far more than K relevant items. Precision/Recall depend on K and on how many positives a user has. NDCG with binary relevance ignores how much more the user liked item A than item B. Hit Rate saturates near 1 for users with many liked items.

**The project's definition of "accuracy" in the strategy-comparison dashboard** is Hit Rate@10 on the frozen test split (`src/benchmark/metrics.py` docstring). It is *not* the classifier accuracy. This distinction is the most common viva confusion; keep both separate.

### Coverage, diversity, novelty (list-level metrics)
* **Coverage** = number of distinct movies appearing in any user's top-K / catalog size. Baseline 0.38% vs CineRankX 5.7% on the benchmark.
* **Diversity** (`intra_list_diversity`) = 1 - mean pairwise genre Jaccard similarity inside a list. *Example:* two movies with genres {A,B} and {B,C}: Jaccard = 1/3, diversity = 0.667.
* **Novelty** = mean self-information (-log2 of item popularity), scaled; shared with the novelty objective.

### Search metrics
* **MRR** (mean reciprocal rank) = mean over queries of 1 / (rank of first relevant result): first result relevant -> 1, third -> 0.333.
* **Precision@10**, **Recall@10**, **NDCG** as above, using the documented *proxy* relevance (a result is relevant if it shares at least one genre or keyword with the query). Because on average 1,812 of 26,743 movies count as "relevant" per query, **Recall@10 is bounded above by 10 divided by the number of relevant movies and measured only 0.0101**: it reflects the size of the relevant pool, not retrieval quality (this caveat is printed in the KPI report).
* **Search latency** in milliseconds.

## 10.3 Rating prediction metrics
Predicted ratings [4.0, 3.5, 2.0] versus actual [5.0, 3.0, 2.5]; errors [-1.0, +0.5, -0.5].
$$ MAE = (1/n) sum |error| = (1 + 0.5 + 0.5) / 3 = 0.667 $$
$$ RMSE = sqrt( (1/n) sum error^2 ) = sqrt( (1 + 0.25 + 0.25) / 3 ) = 0.707 $$
RMSE punishes large errors more than MAE. In the benchmark the bias predictor reaches RMSE **0.846** and MAE **0.649** on the test split against **0.936 / 0.725** for the frozen item-mean predictor (`reports/strategy_comparison.json`). *Misleading when:* a model that predicts everyone's rating accurately can still rank poorly.

## 10.4 System metrics
`src/evaluation/system_metrics.py::measure(operation, n_calls, warmup)` times repeated calls with `perf_counter` and uses `psutil` to read process memory and CPU time (derived from `cpu_times()` deltas, not `cpu_percent`). It reports response time in ms per call, memory (MB), CPU (% of one core) and throughput (operations per second) aggregated over the instrumented operations (IR search, IR similar_to, strategy predict, rank top-10). Latest saved: response time 23.4 ms, throughput 46.0 ops/s, memory 398 MB, CPU about 98% of one core.

## 10.5 Evaluation methodology concepts

### Train/validation/test split
*Train* fits parameters; *validation* chooses hyperparameters and models; *test* is touched once for the final report. Reusing the test set to choose a model makes the reported number optimistic (it has been "tuned to"). In CineRankX the benchmark's tuning cohorts are scored on the **validation** split only, and the test split's ground truth is never read during training or tuning.

### Cross-validation and stratified sampling
*k-fold CV* splits the training data into *k* parts, trains on *k-1*, scores on the remaining part, and averages; it estimates generalisation and its variance without spending the test set. *Stratified* folds keep the class proportions the same in every fold, which matters when one class is 2%. The project uses 5-fold stratified CV (`StratifiedKFold`), reducing the fold count automatically when a class is rarer than the fold count (test `test_cross_validation_shrinks_folds_for_rare_classes...`).

### Data leakage (and the concrete cases in this project)
Leakage is information from the evaluation target reaching the model. Cases found and fixed:

1. **Feature/label overlap.** Early features were computed on a user's *whole* history while labels came from backtests on random subsets of the same history; now features use only the profile split.
2. **Collaborative similarity including the test ratings.** The legacy backtest built neighbour structures from all ratings; the leak-free labeller builds everything from the profile ratings only.
3. **Rating predictor fitted on all ratings, including held-out ones.** The benchmark fits the predictor and catalog statistics on the training split only.
4. **SMOTE before splitting.** SMOTE must only see the training fold; the code resamples after the split and tests that no synthetic point is derived from test users.
5. **A user's own row as their neighbour** in CF: explicitly excluded (`sims[own] = -1`).

### Generalisation, bias and variance
*Bias* is error from a model that is too simple; *variance* is error from a model too sensitive to the training sample. Boosted trees with many leaves have low bias and high variance: they fit training labels (accuracy up to 0.88) but generalise less (about 0.62). The gap between training and held-out accuracy is the symptom; regularisation, subsampling and cross-validation are the remedies.

### Hyperparameter tuning
Hyperparameters (learning rate, leaves, ...) are not learned by training. They are chosen by search: Optuna's TPE sampler proposes values, 3-fold CV macro-F1 on development users scores them, the best are kept. Tuning uses only development users; the held-out set stays untouched.

### Confidence intervals and statistical significance
A single metric is a point estimate from a sample of users. The project reports **user-level bootstrap intervals** (`src/strategy_selector/stats.py::bootstrap_ci`): resample users with replacement 2,000 times, recompute the mean, and take the 2.5th/97.5th percentiles. For *improvement* claims it uses a **paired bootstrap** on per-user differences (selector minus popularity); an interval that excludes 0 means the gain is unlikely to be chance. `mcnemar_p` is an exact test for "two classifiers have the same error rate on the same users", and `proportion_ci` is the Wilson interval for accuracy.

> **Worked reading:** selector hit-rate gain over popularity = +0.0155 with 95% CI [+0.011, +0.020]. The interval excludes zero, so the improvement is statistically significant, although small in absolute terms (about 1.5 hit-rate points).

## 10.6 The frozen benchmark protocol (how recommendation accuracy is measured fairly)
`src/benchmark/` implements the protocol that produced the Hit Rate@10 figures.

1. **Deterministic hash split** (`split.assign_splits`): each rating (user, movie) goes to train (70%), validation (10%) or test (20%) by a splitmix64 hash of (user, movie, seed 20260101). No stored file; identical everywhere.
2. **Fixed cohorts**: 2,000 overall test users, 1,000 cold-start test users (profile cut to the first 5 training ratings, ordered by a second hash), plus two disjoint 1,000-user tuning cohorts scored only on validation.
3. **Same ground truth for every strategy**: relevant = held-out ratings >= 4.0; items the user rated before the scored split are excluded from recommendations.
4. **Frozen baseline** (`baseline.py`): rank movies by number of training ratings; predict ratings by item mean with damping 10. The file imports nothing from the rest of the project (a unit test checks this) and `scripts/run_benchmark.py` stores the SHA-256 of the baseline, metrics and split code with the baseline's results and **aborts if the code, split or numbers drift**.
5. **Results** (test split): overall Hit Rate@10 0.601 (baseline) / 0.751 (old CineRankX pipeline) / 0.877 (current); cold-start 0.609 / 0.658 / 0.698; NDCG@10 0.163 -> 0.362; Precision@10 0.131 -> 0.291; RMSE 0.936 -> 0.846.

![Strategy comparison](figs/strategy_comparison.png)

**Why 90-95% accuracy is not realistic here.** Hit Rate@10 asks for at least one liked movie among ten picks out of about 27,000. Hit Rate@20 is already 0.928 overall, but for the *classification* problem, where the label is itself noisy, the ceiling is about 0.76 (Chapter 9). A fairer way to judge the system is the set of ranking metrics plus significance tests, not a single accuracy percentage.
