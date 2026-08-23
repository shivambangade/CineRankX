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
