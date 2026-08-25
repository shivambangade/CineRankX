# Dataset Migration Log

## Before — MovieLens ml-latest-small (100K)

- **Dataset**: MovieLens ml-latest-small — ~100,836 ratings, 9,742 movies.
- **Join result** (`links.csv`: `movieId → tmdbId` → TMDB 5000's `id`):
  **9,742 MovieLens movies → 3,537 successfully joined to TMDB metadata (~36.3%)**.
- **Why the join rate was low**: TMDB 5000 is a fixed 5,000-movie slice of TMDB, not
  full TMDB coverage — most `ml-latest-small` movies simply don't appear in that slice.
- **Verified in Module 1**: `text_chunk_clean` correctly built for matched rows
  (e.g. Toy Story spot-check passed); 6/6 unit tests passing.

## Considered — MovieLens ml-32m (not adopted)

Analyzed switching to **MovieLens ml-32m** (32M ratings, ~87,500 movies) to raise the
join rate via broader MovieLens coverage. Projected join rate against TMDB 5000 was
only **~5.3%** (4,634 / 87,585) — TMDB 5000 is a fixed 5,000-movie slice, so adding
more MovieLens movies mostly adds movies TMDB 5000 doesn't have. Absolute matched
count would have improved (3,536 → 4,634) but the join *rate* got worse, and TMDB 5000
remained the bottleneck either way. Not pursued further.

## After — MovieLens 20M + TMDB Movies Dataset 2023 (930K)

Switched the bottleneck instead: kept a mid-size MovieLens dataset but replaced the
fixed 5,000-movie TMDB slice with a near-full TMDB export.

- **New datasets**:
  - MovieLens 20M (`newdata/movielens/`) — 20,000,263 ratings, 27,278 movies,
    465,564 tags. Also ships `genome_scores.csv`/`genome_tags.csv` (per-movie
    tag-relevance vectors); not loaded by this pipeline.
  - TMDB Movies Dataset 2023 (`newdata/tmdb/TMDB_movie_dataset_v11.csv`) —
    1,480,412 movies, single file, no separate credits file.
- **New join result**: 27,278 MovieLens movies → 26,743 successfully joined to TMDB
  (**98.04%**), vs. the original 36.3%. (27,026 of the 27,278 movies had a non-null
  `tmdbId` in `link.csv` at all; 26,743 of those actually matched a TMDB row.)
- **Schema/column differences handled**:
  - MovieLens 20M filenames are singular (`rating.csv`, `movie.csv`, `link.csv`,
    `tag.csv`) vs. ml-latest-small's plural names — `loaders.py` paths updated.
  - MovieLens 20M `rating.csv`/`tag.csv` timestamps are datetime strings
    (`"2005-04-02 23:53:47"`), not epoch ints — passed through as-is, unused
    downstream by this module.
  - TMDB 930K's `genres`/`keywords` are plain comma-separated strings
    (`"Action, Adventure"`), not TMDB 5000's JSON list-of-dicts — `joiner.py`'s
    extraction switched from `json.loads()`-based parsing to a plain comma-split
    (`_split_names`).
  - **No cast/crew, by decision, not by gap**: TMDB 930K has no credits file and no
    cast/crew columns at all. Backfilling from `tmdb_5000_credits.csv` was considered
    and rejected — it would only cover a minority of the 26,743 joined movies and
    would make `text_chunk` composition inconsistent across the catalog (some movies
    with cast/crew, most without). `text_chunk` is now **overview + genres + keywords
    only**, for every movie, uniformly. `joiner.py` no longer merges a credits frame
    or emits `cast_names`/`crew_names`; `CLAUDE.md`'s IR engine description updated
    to match.
- **Verified in Module 1**:
  - `ratings_clean.csv` row count (20,000,263) matches MovieLens 20M's `rating.csv`
    exactly.
  - Spot-checked Toy Story (`movieId=1`): `text_chunk_clean` is stemmed
    overview + genre + keyword words only (`"woodi andi toy ... anim adventur famili
    comedi rescu friendship mission ..."`) — confirmed no cast/crew names
    (e.g. "hank", "allen", "lasseter") leak into the cleaned chunk.
  - Re-ran Module 1 tests: 6/6 passing, fixtures updated to the new schema (singular
    filenames, comma-separated genres/keywords, no credits/cast/crew fixtures).

## Known limitations — Module 3 (Adaptive ML Strategy Selection Engine)

Two limitations in the strategy selector are documented here rather than fixed. Both
are consequences of label prevalence and data scale, not defects in the classifier,
the profile features, or the backtest. Measurements below are from a 5,000-user sample
(750,512 ratings), `n_splits=5`, 1,000-user held-out test set.

### 1. `content_based` is hard to predict (F1 ≈ 0.12, low recall)

`content_based` wins the backtest for only **347 of 5,000 users (6.9%)**. At that
prevalence the classifier recovers very few of them:

```
content_based    Precision 0.2069   Recall 0.0857   F1 0.1212   support 70
```

Four remedies were tried. Three moved F1 not at all; the fourth (SMOTE) moved it
modestly but bought a decision-threshold shift rather than better discrimination:

| Remedy | Result |
|---|---|
| `class_weight='balanced'` on the RandomForest | No improvement |
| Genre-identity features (`top_genre_share`, `genre_entropy`) | No improvement |
| Multi-split label averaging (`n_splits` 1 → 5) | F1 0.1221 → 0.1212 (flat) |
| SMOTE oversampling of the training fold | F1 +0.038 (10-seed mean), but ROC-AUC flat — see section 3 |

Multi-split averaging was added to test the hypothesis that borderline users' labels
were flipping between `collaborative` and `content_based` by chance on a single
random split. It did **not** improve `content_based`: precision rose (0.148 → 0.207)
while recall fell (0.104 → 0.086), netting out to no change.

Averaging was kept anyway, because it fixed a different and real problem: label
stability. Under a single split, `popularity` won for 1,883 users (37.7%); averaged
over five splits it wins for only 446 (8.9%). Those ~1,400 users were one-shot wins
that disappear once each user's hit-rate is averaged — genuine noise in the ground
truth, now removed.

**Interpretation**: at this scale the limit is the number of `content_based` examples
available to learn from, not the model's capacity to learn them. Note that
`content_based`, despite the worst F1 of the three classes, shows the *strongest*
per-class discrimination relative to its own base rate (see below) — the classifier
is identifying a real minority signal, it is simply being conservative about it.

### 2. Overall accuracy is below the majority-class baseline

```
Accuracy                 0.7960
Majority-class baseline  0.8410     (always predict `collaborative`)
Lift                    -0.0450
ROC-AUC                  0.6421
```

This deficit is partly by design: `class_weight='balanced'` deliberately trades
majority-class accuracy for minority-class recall, so an accuracy figure below the
"always guess the majority" baseline is expected. It is not evidence that the model
has learned nothing — accuracy alone cannot distinguish those two cases on an 84%
-imbalanced problem, which is why the metric is reported alongside ROC-AUC and
per-class figures rather than on its own.

Evidence the model is learning real signal beyond guessing the majority class:

- **ROC-AUC 0.6421**, meaningfully above the 0.50 chance level.
- **Per-class precision vs. that class's own base rate** in the test set:

  | Strategy | Support | Base rate | Precision | Ratio |
  |---|---|---|---|---|
  | `collaborative` | 841 | 84.1% | 0.8540 | 1.02× |
  | `content_based` | 70 | 7.0% | 0.2069 | **2.96×** |
  | `popularity` | 89 | 8.9% | 0.1132 | 1.27× |

  `content_based` predictions are right about three times as often as blind guessing
  at its base rate would be. `popularity` is weakly above chance. `collaborative`'s
  high raw F1 (0.891) is mostly majority-class mass, not discrimination — its 1.02×
  ratio is the honest read of it.

**Framing**: Module 3 delivers **partially effective personalization**, not a solved
classification problem. It reliably identifies the large `collaborative` majority and
extracts a real if low-recall `content_based` signal, while `popularity` sits close to
chance. Downstream (Module 4) should treat the predicted strategy as a weighted prior
rather than a hard routing decision, and should degrade gracefully to `collaborative`
when classifier confidence is low.

Both limitations are accepted as scoped; no further investigation is planned at this
data scale.

### 3. SMOTE oversampling — investigated, implemented, left disabled

SMOTE (imbalanced-learn) was added as a fourth remedy for `content_based`'s low
prevalence, on the hypothesis that synthetic minority examples would give the
classifier a boundary to learn. It is implemented in `classifier.py` behind
`fit(..., use_smote=False)` and **ships disabled**. The investigation is recorded here
because its negative result is the strongest evidence yet for the diagnosis in
section 1.

#### Leakage control

SMOTE is applied strictly **after** `train_test_split`, to the training fold only —
oversampling before the split would interpolate synthetic points from held-out users
and produce inflated, meaningless metrics. In `fit()` the ordering is marked with
explicit comment banners: the split happens first, `X_test`/`y_test` are frozen there
and are never passed to `_resample_training_fold()`. Verified three ways:

- Both arms evaluate an **identical held-out set** (same user IDs, same true labels,
  1,000 users) — a pre-split SMOTE would have put synthetic rows in the test fold and
  changed it.
- `rows_before` reported by the resampler is 4,000 (the train fold), not 5,000.
- On a fixture with deliberately disjoint per-class feature ranges, **every** synthetic
  point falls inside the convex hull of the *training* minority rows, and no synthetic
  row duplicates a test row. SMOTE only interpolates, so a point outside that hull
  would be proof of a leak.

Regression tests covering all three checks live in `TestSMOTEOversampling`.

One related bug was fixed while wiring this up: the majority-class baseline read its
majority off `y_train`, which SMOTE balances. Post-SMOTE that makes "most common label"
an arbitrary tie-break between equal-sized classes, silently corrupting the baseline
and the reported lift. The baseline now reads from `y_train_original`, captured before
resampling.

#### Result: a single split is not enough to judge this

On the documented `random_state=42` split, SMOTE looked like a clear win — `content_based`
F1 0.109 → 0.220, with precision *and* recall both up and ROC-AUC +0.013. Repeating the
identical two-arm comparison across **10 seeds** showed seed 42 was the most favourable
of the ten draws, and that two of those three signals do not survive:

| Metric | Baseline | SMOTE | Δ | SMOTE wins |
|---|---|---|---|---|
| `content_based` F1 | 0.1019 ± 0.0188 | 0.1402 ± 0.0455 | **+0.0383** | 9/10 |
| `content_based` recall | 0.0771 ± 0.0181 | 0.1243 ± 0.0442 | **+0.0471** | 9/10 |
| `content_based` precision | 0.1552 ± 0.0329 | 0.1628 ± 0.0500 | +0.0075 | 4/10 |
| `popularity` F1 | 0.1514 ± 0.0266 | 0.1403 ± 0.0294 | −0.0112 | 3/10 |
| `collaborative` F1 | 0.8879 ± 0.0044 | 0.8778 ± 0.0080 | −0.0100 | 0/10 |
| Accuracy | 0.7920 ± 0.0061 | 0.7743 ± 0.0122 | −0.0177 | 0/10 |
| ROC-AUC | 0.6273 ± 0.0209 | 0.6223 ± 0.0221 | −0.0049 | 4/10 |

Per-seed `content_based` F1 delta ranged from −0.0110 to +0.1116, mean +0.0383 — seed 42's
+0.1116 was roughly three times the typical effect. The precision gain and the ROC-AUC
gain were both seed-42 artifacts; across seeds each is a coin flip.

#### Conclusion: rebalancing moves the threshold, not the probabilities

Higher recall, flat precision and **flat ROC-AUC** together mean the classifier is not
discriminating better — it is pushing the decision boundary toward the minority class.
Confirmed by comparing SMOTE against simply scaling up `class_weight` (10 seeds each):

| Arm | `content_based` F1 | recall | precision | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| A `class_weight='balanced'` (current) | 0.1019 | 0.0771 | 0.1552 | 0.7920 | 0.6273 |
| B SMOTE | 0.1402 | 0.1243 | 0.1628 | 0.7743 | 0.6223 |
| C `balanced` × 2 on minorities | 0.1469 | 0.1543 | 0.1422 | 0.7234 | 0.6258 |
| D `balanced` × 4 on minorities | 0.1756 | 0.2943 | 0.1253 | 0.5701 | 0.6188 |
| E `balanced` × 8 on minorities | 0.1477 | 0.3800 | 0.0917 | 0.3256 | 0.6046 |

**ROC-AUC is flat (0.605–0.627) across every arm.** No rebalancing method — synthetic
oversampling or class weighting — improves the model's ability to rank users by
probability. They only trade accuracy for minority recall, at different exchange rates:

| Arm | Accuracy lost per point of `content_based` recall gained |
|---|---|
| B SMOTE | −0.375 |
| C `balanced` × 2 | −0.889 |
| D `balanced` × 4 | −1.022 |
| E `balanced` × 8 | −1.540 |

SMOTE is the best-priced of the four (~2.4× more efficient than the cheapest
`class_weight` alternative, and the only one that does not degrade `content_based`
precision) — but it is buying a threshold shift, not new information. This is exactly
what section 1 predicted: SMOTE interpolates *within* the 278 real `content_based`
training points, so it cannot manufacture signal that is not already there.

#### Why it ships disabled

Module 4 consumes the classifier's **`confidence`**, not its hard label —
`strategy_source_weights(strategy, confidence)` blends the three candidate sources in
proportion to predicted probability, treating the prediction as a weighted prior. What
matters to that consumer is probability quality, which is precisely what ROC-AUC measures
and precisely what no rebalancing arm improved. A decision-threshold shift is worth
nothing to a downstream stage that never applies a threshold, while costing accuracy in
10/10 seeds.

The code is kept (tested, documented, `use_smote=False`) so the option remains available
if a future consumer needs hard labels rather than probabilities.

Reproduce with `scripts/compare_smote.py` (single-split before/after),
`scripts/smote_seed_robustness.py` (10-seed cross-validation) and
`scripts/smote_vs_class_weight.py` (rebalancing-method comparison).

## Known limitations — evaluation suite (src/evaluation/)

Measurements below are from `scripts/generate_kpi_report.py` on a 900,000-rating sample
(6,034 users), 1,500 users evaluated, K = [5, 10, 20].

### 1. Cold-Start Accuracy: the popularity baseline beats CineRankX

```
Cold-Start Accuracy (CineRankX)   0.5556
Popularity baseline               0.5859
Lift over baseline               -0.0303
```

**On cold-start users, a plain most-popular list outperformed the full pipeline.** This is
a real, reproducible result and is recorded as such rather than omitted or softened.

**Superseded — see "Popularity source correction" below.** After both engines were changed
to rank popularity by observed rating counts, this became `0.6061` vs `0.5859`, a lift of
`+0.0202`. The figures above are the measurement taken *before* that fix and are kept as
the record of what was found.

Interpretation: a user with five ratings gives the content and collaborative sources
almost nothing to work with. Five TF-IDF seeds are a thin profile, and a five-item rating
vector locates a poor collaborative neighbourhood. The blended candidate pool is
therefore mostly noise around a popularity core, and the six-objective ranker then
actively spends relevance on diversity, novelty and coverage — objectives that pull
*away* from the safe popular picks that happen to be the right answer for a user we know
nothing about. The system is paying for personalization it does not yet have the evidence
to perform.

This is consistent with, not contradictory to, Module 3's design: `popularity` exists as a
strategy precisely because it is the right choice for some users. The finding is that the
pipeline does not yet fall back to it hard enough when a profile is this thin.

### 2. Cold-Start Accuracy is measured on a SIMULATED cohort

MovieLens 20M only admits users who rated at least 20 movies. The minimum profile size in
any sample is exactly 20, so the dataset contains **no natural cold-start users at all** and
the KPI as literally defined ("users with <= 5 ratings") can never produce a number on it.

The cohort is therefore simulated: real users have their training profile truncated to 5
ratings and are scored on their untouched held-out items. Every *other* user keeps a full
history, so the collaborative neighbourhood the cold user is matched against stays
realistic — truncating everybody would measure a cold *system*, not a cold *user*.
`evaluate_cold_start()` returns `mode` (`"natural"` / `"simulated"`) and the report prints
an explicit note, so a simulated figure can never be mistaken for a natural one.

### 3. Search relevance is a proxy, and Recall@10 is structurally near zero

There is no hand-labeled relevance set. A result counts as relevant if it shares at least
one genre or keyword with the query. That proxy marks on average **1,812 of the 26,743
catalog movies relevant per query**, so `Recall@10` is bounded above by 10/1,812 and lands
at 0.0101. It measures catalog breadth, not retrieval quality. `Precision@10` (0.4100) is
correspondingly generous — sharing one genre is a low bar for "relevant". Both figures are
reported with this caveat printed directly above them in the KPI report.

### 4. A straw-man baseline initially inverted the cold-start conclusion

Worth recording because the bug was self-flattering and the corrected result is worse.

`popularity_baseline_top_k()` originally ranked by TMDB's `popularity` column. TMDB
popularity is a live, recency-weighted metric; MovieLens 20M's ratings stop in 2015, so the
two disagree almost completely. On the 900k sample the TMDB top-20 shared **1 title** with
the 20 most-rated films and included movies rated by 0 and 5 users respectively, capturing
9,426 ratings against the observed baseline's 46,864.

Measured against that straw man, CineRankX appeared to beat the cold-start baseline by
**+0.3535**. Against a baseline built from observed rating counts, the true figure is
**-0.0303**. The baseline now uses observed counts and falls back to TMDB `popularity`
only when no ratings are available.


## Popularity source correction — Modules 3 and 4

Both engines ranked popularity by TMDB's `popularity` column. TMDB popularity is a live,
recency-weighted metric; MovieLens 20M's ratings stop in 2015, so the two disagree almost
completely. Both now rank by **observed rating counts**, falling back to TMDB `popularity`
only when no ratings are available — the same ordering already used in
`src/evaluation/recommendation_metrics.py`.

Measured on a 900,000-rating sample:

| | TMDB popularity | Observed counts | |
|---|---|---|---|
| Module 3 top-10 covers | 3,520 ratings (0.39%) | 25,819 (2.87%) | 7.3x |
| Module 3 top-10 reaches | 2,361 / 6,034 users (39.1%) | 5,242 / 6,034 (86.9%) | 2.2x |
| Module 4 top-200 covers | 84,755 (9.42%) | 254,031 (28.2%) | 3.0x |
| Module 4 pool overlap | 37 of 200 items shared | | |

### Effect on Module 3 (5,000-user backtest, same scale as section 1)

| Metric | Before | After | |
|---|---|---|---|
| Accuracy | 0.7960 | 0.6600 | |
| Majority-class baseline | 0.8410 | 0.5160 | |
| **Lift over baseline** | **-0.0450** | **+0.1440** | now positive |
| ROC-AUC | 0.6421 | 0.6841 | +0.042 |
| `popularity` F1 | ~0.1103 | **0.6461** | 5.9x |
| `collaborative` F1 | 0.8928 | 0.6951 | |
| `content_based` F1 | 0.1212 | **0.0000** | collapsed |

Label distribution: `collaborative` 84.1% -> 51.6%, `popularity` 8.9% -> **45.3%**,
`content_based` 7.0% -> 3.1%.

**This resolves limitation 2 above.** The classifier now beats the majority-class baseline
by +0.144 — the first time in this project it has done so. Raw accuracy *fell* (0.796 ->
0.660) but the old figure was mostly majority-class mass on an 84%-imbalanced problem; the
class balance is now 52/45/3, so the baseline itself drops to 0.516. ROC-AUC confirms the
improvement independently.

**Why `popularity` improved so much**: the backtest previously scored that strategy against
a top-10 only 39% of users had ever touched, so `pop_score` was 0.0 for the majority and
the label was assigned largely by the all-zero tie-break in `select_best_strategy()` —
noise rather than signal. Against a list 87% of users touch, `popularity` becomes a
genuinely winnable and genuinely learnable class. The hypothesis recorded when this flaw
was first flagged is confirmed.

**Disclosed regression**: `content_based` fell to F1 0.0000 at 3.1% prevalence (156 of
5,000 users) — the classifier now predicts it essentially never. Part of this is honest
reallocation: users previously labeled `content_based` only because `pop_score` was
artificially 0 now correctly resolve to `popularity`. But the class is now below the
threshold of learnability at this data scale, worse than the F1 0.12 recorded in
section 1. Net effect: two of three classes are genuinely predicted where one was before.
Accepted as scoped; no further investigation planned.

### Effect on the KPI report (1,500 users, 900k ratings)

| Metric | Before | After | |
|---|---|---|---|
| Precision@5 | 0.1816 | 0.1944 | +0.013 |
| Precision@10 | 0.1389 | 0.1529 | +0.014 |
| Precision@20 | 0.1008 | 0.1104 | +0.010 |
| Recall@10 | 0.1735 | 0.1732 | flat |
| F1@10 | 0.1259 | 0.1308 | +0.005 |
| MAP | 0.1124 | 0.1174 | +0.005 |
| **Coverage** | 0.0964 | **0.0713** | **-0.025** |
| Diversity | 0.8388 | 0.8322 | -0.007 |
| **Novelty** | 0.2440 | **0.1715** | **-0.073** |
| **Cold-Start Accuracy** | 0.5556 | **0.6061** | +0.051 |
| Cold-start lift vs. baseline | -0.0303 | **+0.0202** | now positive |

**The trade-off is real and expected**: accuracy metrics improved while Coverage and
Novelty fell. The popularity source now proposes genuinely widely-rated films instead of
obscure TMDB-popular ones — those are by definition less novel, and the same titles recur
across users, which narrows catalog coverage. Better hit-rates, less discovery.

### Caveat on the cold-start comparison after this change

The evaluation's cold-start baseline (most-rated top-K) is now a **strict subset of Module
4's own popularity candidate pool** — all 20 baseline items sit inside the ranker's 200-item
pool. The comparison is therefore no longer "CineRankX vs. an independent baseline"; it is
an **ablation**: given a candidate pool that already contains the baseline's exact answer,
does the six-objective ranker order it better than plain popularity rank? The answer is
yes, by +0.0202 — a modest but positive contribution from the ranking layer.

This is a cleaner measurement than before (previously the two drew from disjoint universes,
confounding "different candidate pool" with "different ranking method"), but it licenses a
narrower claim. It is not evidence that the system beats popularity in general — only that
its ranking improves on popularity ordering within a shared pool.
