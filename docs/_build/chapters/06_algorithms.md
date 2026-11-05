# Chapter 6. The Recommendation Algorithms, Explained From Scratch

This chapter teaches each algorithm in four steps: intuition, mathematics, a small numerical example, and *what the CineRankX code actually does* (which sometimes differs from the textbook version). Code references are to the repository.

## A. Collaborative Filtering (CF)

### A.1 Intuition
People who agreed in the past tend to agree in the future. If you and another user rated ten movies similarly, then movies *they* loved and you have not seen are good candidates for you. CF needs **no information about the movies themselves**, only who rated what.

### A.2 The user-item matrix and sparsity
Arrange ratings in a matrix *R*: one row per user, one column per movie, entry *R[u,i]* the rating (blank/0 if unrated). For MovieLens 20M the matrix is 138,493 x 27,278 = 3.78 billion cells but only 20,000,263 are filled: **0.53% density, 99.47% sparse**. Sparsity is the central difficulty of CF: two users rarely rate the same movies, so similarities are noisy.

### A.3 Similarity: cosine
Treat each user as a vector of ratings and measure the angle between two vectors:

$$ sim(u, v) = cos(r_u, r_v) = (r_u . r_v) / (||r_u|| ||r_v||) $$

It is 1 when two users rate in proportion, 0 when they share no rated items. (Blanks counted as zero: a simplification the CineRankX code also makes.)

### A.4 Worked example (also the textbook rating prediction)
Items A-E, four users; target user U1 has not rated D or E.

| | A | B | C | D | E |
|---|---|---|---|---|---|
| U1 | 5 | 3 | 4 | ? | ? |
| U2 | 4 | 0 | 5 | 3 | 0 |
| U3 | 1 | 5 | 0 | 4 | 5 |
| U4 | 0 | 4 | 1 | 5 | 4 |

Cosine similarities of U1 with the others: sim(U1,U2) = **0.800**, sim(U1,U3) = **0.346**, sim(U1,U4) = **0.297**.

*Prediction by similarity-weighted average* (user-based CF): for item D the neighbours who rated it are U2 (3), U3 (4), U4 (5):

$$ r^(U1, D) = (0.800 x 3 + 0.346 x 4 + 0.297 x 5) / (0.800 + 0.346 + 0.297) = 3.65 $$

For item E: U3 (5) and U4 (4) give (0.346 x 5 + 0.297 x 4)/(0.346 + 0.297) = **4.54**. So E would be recommended before D.

*Item-based CF* does the same with item columns: cos(D, C) = 0.436 and cos(D, A) = 0.349, so items that co-occur with what U1 liked score higher.

### A.5 What CineRankX actually implements
The CF source in `src/ranking_engine/candidates.py` (`CandidateGenerator._build_collaborative`, `_collaborative_candidates`) is a **user-user nearest-neighbour candidate generator**, not a rating predictor:

1. Take the first `_CF_MAX_USERS = 5,000` users in the ratings frame and their ratings of the `_CF_MAX_MOVIES = 2,000` most-rated movies (bounded so the similarity computation stays cheap).
2. Build the target user's vector from *their own ratings* (so users outside the pool still work), compute cosine against all pooled users, set a perfect self-match to -1.
3. Keep the `_CF_NEIGHBORS = 25` most similar users with positive similarity.
4. Each neighbour contributes their top `_CF_PER_NEIGHBOR = 20` highest-rated movies (from their *full* history, not just the 2,000-movie window). A candidate's score is the **sum of the similarities of the neighbours who propose it**.

$$ score_CF(m) = sum over neighbours n with m in top20(n) of sim(u, n) $$

The strategy labeller has its own, slightly different, CF used only for labelling (`strategy_labeler._collaborative_hit_rate`: 5 neighbours x 5 movies, capped to the top-10 or top-30 by similarity so CF cannot win just by proposing more items than the others).

**Advantages:** simple, explainable, needs no metadata. **Disadvantages:** cold start; popularity-biased neighbours; bounded to a 5,000-user pool (a scalability compromise); no rating normalisation. **Potential viva questions:** why cosine and not Pearson? (Pearson removes each user's mean and is better for rating-scale bias; we kept cosine for sparse vectors and speed; mean-centring is a possible improvement.) Why a bounded pool? (the dense user-user matrix is O(n^2).)

### A.6 EASE: the stronger collaborative model used in the benchmark
The benchmark pipeline (`src/ranking_engine/ease.py`) uses **EASE** (Embarrassingly Shallow Autoencoder, Steck 2019), a closed-form linear item-item model. Let *X* be the binary user-by-item matrix of training interactions (6,000 most-rated movies):

$$ P = (X'X + lambda I)^-1,   B = -P / diag(P),   diag(B) = 0,   score(u) = x_u B $$

Intuition: for each item, ridge-regress its column on all other columns (excluding itself); *B[j,i]* says how strongly owning item *j* predicts owning item *i*. The diagonal is forced to zero so an item cannot trivially predict itself. A user is scored from their own ratings alone (fold-in), including a cold-start user with five ratings. The tuned regulariser is lambda = 1,500. Scores are converted to per-user **percentile ranks** and blended with a popularity percentile with weight beta (1.0 overall, 0.8 cold-start) before the six-objective ranker runs.

An exploratory PureSVD (truncated SVD) run was tried while choosing a relevance model and discarded in favour of EASE; the repository contains no SVD implementation.

## B. Content-Based Filtering

### B.1 Intuition
"You liked *Toy Story*, so here are movies that *read* like it." It uses only item descriptions, so it works for new items and needs no other users, but cannot discover tastes outside what you already like.

### B.2 TF-IDF
Turn each movie's text into a numeric vector. For term *t* in document *d* of a collection of *N* documents:

$$ tf(t,d) = count of t in d $$
$$ idf(t) = ln((1 + N) / (1 + df(t))) + 1 $$
$$ tfidf(t,d) = tf(t,d) x idf(t),   then L2-normalise the vector $$

*Term frequency* rewards words used often in a document; *inverse document frequency* down-weights words that appear in many documents (low discriminating power). The +1 and smoothing are scikit-learn's defaults (`smooth_idf=True`).

**Worked example.** Three tiny documents: d1 = "toy story toy adventure", d2 = "space adventure laser", d3 = "toy robot space". Vocabulary: adventure, laser, robot, space, story, toy. Document frequencies: adventure 2, laser 1, robot 1, space 2, story 1, toy 2, so idf(adventure) = ln(4/3) + 1 = 1.288, idf(laser) = ln(4/2) + 1 = 1.693.

d1: toy 2 x 1.288 = 2.576, story 1.693, adventure 1.288 -> after L2 normalisation (length 3.34): toy **0.771**, story **0.507**, adventure **0.386**.

### B.3 Cosine similarity between movies
After normalisation, the cosine similarity is just the dot product. Here cos(d1,d2) = **0.200** (shared word: adventure), cos(d1,d3) = **0.399** (shared word: toy, which d1 uses twice), cos(d2,d3) = **0.268** (shared word: space).

### B.4 What CineRankX implements (Module 2)
`src/ir_engine/tfidf_vectorizer.py::TFIDFEngine` fits a `TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2), strip_accents='unicode', lowercase=False)` on `text_chunk_clean`:

* `min_df=2` drops terms occurring in fewer than 2 movies (typos, one-off names); `max_df=0.8` drops terms in more than 80% of movies; `ngram_range=(1,2)` adds two-word phrases; `max_features=5000` keeps the most frequent 5,000 terms.
* `search(query)`: vectorise the query with the same vocabulary, cosine against every movie, return top-k with positive similarity.
* `similar_to(movie_id)`: use the movie's own row as the query, exclude itself.
* `src/ir_engine/trie.py::Trie`: a prefix tree over lower-cased titles for autocomplete; `search(prefix)` walks to the prefix node and does a depth-first collection of up to `limit` titles.

### B.5 How content-based candidates are built for a user
`CandidateGenerator._content_candidates`: take the user's 10 highest-rated movies as **seeds** (`_IR_SEEDS = 10`); for each seed fetch 30 similar movies; weight each neighbour as `similarity x rating x gate`, where the **genre-overlap gate** is

$$ gate = floor + (1 - floor) x Jaccard(genres(candidate), genres(seed)),  floor = 0.25 $$

(`genre_overlap_gate`). The gate fixes a real failure: *Toy Story*'s TF-IDF neighbours included *Silent Night, Deadly Night 5: The Toy Maker* purely because of the word "toy". The score of a candidate is the maximum over seeds.

### B.6 The labeller's content strategy
In the backtest used to create classifier labels (`strategy_labeler._content_based_hit_rate`) a user's profile is the **rating-weighted sum of the TF-IDF rows of their 20 best-rated movies**, compared by cosine with every catalog movie; the top-N (10, later 30) unseen movies are the recommendation. This strategy's mean hit-rate is only **0.016**, against about 0.115 for collaborative and popularity (Chapter 9).

### B.7 Experiments to improve it (Chapter 9, section 9.9)
A richer representation (bigram TF-IDF with 20,000 terms, `src/strategy_selector/content_model.py`) raised the held-out hit-rate from 0.0156 to 0.0246 and adding a mild quality prior raised it to 0.0452 (about 2.9 times better, 95% CI of the gain [+0.027, +0.033]); it remains far below popularity (0.119).

## C. Popularity-Based Recommendation

### C.1 Intuition
Recommend what most people rated. It needs no personal data, is hard to beat offline, and is therefore both our fallback strategy and the **frozen baseline** for fair comparison.

### C.2 Raw popularity: rating counts
In the repository popularity means **number of observed ratings** (not TMDB's `popularity` column). `CandidateGenerator._build_popularity` keeps the 200 most-rated movies (`_POPULARITY_POOL = 200`); the labeller keeps the top 10 or 30; the frozen baseline ranks *all* movies by count (ties broken by `movieId`) and removes what the user has seen. This was a deliberate correction (commit `c273ba3`): TMDB's popularity is live and recency-weighted while MovieLens ends in 2015.

### C.3 Quality-adjusted popularity (the "popularity/quality" objective)
Average rating alone is unreliable: a film with three 10/10 votes would beat one rated 8.4 by 20,000 people. `CatalogStats._build_popularity_quality` uses the IMDb-style **Bayesian weighted rating**:

$$ WR = v/(v+m) x R + m/(v+m) x C $$

where *R* is the movie's average (`vote_average`), *v* its vote count, *C* the **unweighted mean** rating of movies that have votes, and *m* the **median** vote count among voted movies. The values are then min-max scaled to [0, 1] because they bunch in a narrow band.

*Example* (C = 6.5, m = 1,000): a classic with R = 8.4 and v = 20,000 gets WR = 8.31; an obscure film with R = 9.5 and v = 3 gets WR = 6.51 (pulled almost to the mean); a mid film R = 7.0, v = 1,000 gets 6.75.

### C.4 Popularity bias
Popular items get more exposure, therefore more ratings, therefore appear more popular. A recommender that leans on popularity concentrates on a few titles (low catalog **coverage**). This is why novelty and coverage are explicit objectives. In our evaluation the frozen baseline covers only 0.38% of the catalog in top-10 lists, while CineRankX covers 5.7% (Chapter 10).

## D. Predicted Rating (the bias baseline)

`src/ranking_engine/predictor.py::BaselinePredictor` predicts a user's rating of a movie as

$$ r^(u,i) = mu + b_u + b_i,   clipped to [0.5, 5.0] $$

with damped biases, fitted in two steps: first the item bias *b_i = sum(r - mu) / (n_i + lambda)* then the user bias on the item-adjusted residual, with damping lambda = 10 to shrink sparse users/items toward 0. An unknown user falls back to mu + b_i; an unknown item to mu + b_u. In the benchmark this predictor has RMSE 0.846 / MAE 0.649 against 0.936 / 0.725 for a frozen item-mean baseline.

## E. Hybrid recommendation: what is and is not implemented

| Concept | Implemented? | Where |
|---|---|---|
| Weighted blend of three candidate *sources* | **Yes** | `CandidateGenerator.generate`: each source's scores are min-max normalised and summed with strategy-dependent weights |
| Strategy prediction steering the blend | **Yes** | `strategy_source_weights(strategy, confidence)` |
| Multi-objective re-ranking of the blended pool | **Yes** | `MultiObjectiveRanker.rank` |
| Hard switching to exactly one strategy | **No** (the prediction is a weighted prior) | |
| Fusion of complete ranked lists from strategies | No (identified as future work) | |

**Source blending detail.** For each source, scores are min-max normalised to [0, 1] (otherwise popularity's raw counts would swamp cosine similarities), multiplied by that source's weight, and **summed**, so an item proposed by two sources outranks one proposed by a single source at the same strength. The pool is then normalised again and cut to 200 candidates.

## F. Summary table

| Algorithm | Needs history? | Needs metadata? | Strength | Weakness | Code |
|---|---|---|---|---|---|
| Popularity | no | no | strong offline, cold-start safe | no personalisation, popularity bias | `candidates._build_popularity`, `benchmark/baseline.py` |
| User-kNN CF | yes | no | personalised, serendipitous | sparse, cold start, bounded pool | `candidates._collaborative_candidates` |
| EASE | yes (any amount) | no | strongest accuracy in our benchmark | needs top-N item cut; no content | `ranking_engine/ease.py` |
| TF-IDF content | seeds only | yes | works for new items | over-specialised; weakest strategy here | `ir_engine`, `candidates._content_candidates` |
| Bias predictor | yes | no | gives rating estimate | not personalised beyond biases | `ranking_engine/predictor.py` |

## G. Likely viva questions for this chapter
Chapter 16 sections E-I contain model answers. Practise: *What is cosine similarity and why is it preferred over Euclidean distance for TF-IDF? What does IDF do? Why do you min-max normalise each source before blending? Why is a Bayesian average better than a plain average? Why is popularity a strong baseline?*
