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

Three remedies were tried and none moved F1 materially:

| Remedy | Result |
|---|---|
| `class_weight='balanced'` on the RandomForest | No improvement |
| Genre-identity features (`top_genre_share`, `genre_entropy`) | No improvement |
| Multi-split label averaging (`n_splits` 1 → 5) | F1 0.1221 → 0.1212 (flat) |

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
