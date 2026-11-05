# Chapter 16. Viva Preparation: 164 Questions with Model Answers

**How to use this chapter.** Each question has a difficulty tag: **[Basic]** what is it; **[Intermediate]** how does it work; **[Advanced]** why did we choose it; **[Expert]** what if we change it; **[Critical]** what is the weakness. Answers are written to be *spoken*, in first person plural, and tied to CineRankX facts. Numbers come from saved artifacts (Chapter 9-10). Do not memorise answers word for word: understand the reasoning, then say it in your own words.

## A. Project introduction and motivation

**A1 [Basic] What is CineRankX in one minute?**
CineRankX is a movie recommender built on MovieLens 20M and TMDB metadata. It does two things a normal recommender does not: it predicts for each user which recommendation strategy (content-based, collaborative or popularity) is likely to work best, and it ranks candidate movies on six objectives (relevance, diversity, novelty, coverage, popularity/quality, predicted rating) instead of a single similarity score. A Streamlit web app lets you run it, inspect any user's history and view the evaluation results.

**A2 [Basic] What problem does it solve?**
Most recommenders use one fixed method and rank by one score. That ignores that different users give different methods different amounts of evidence (a user with five ratings versus one with five hundred), and it produces repetitive lists. CineRankX addresses per-user variation and list quality.

**A3 [Basic] What are the three engines?**
Module 2, the classical IR engine (TF-IDF with cosine similarity plus a Trie for title autocomplete); Module 3, the adaptive strategy selector (a classifier on user features); Module 4, the multi-objective ranking engine (candidate generation and greedy six-objective re-ranking). Module 1 is data ingestion. An evaluation suite and a UI sit around them.

**A4 [Intermediate] How did the project start and evolve?**
It began in August 2026 with data ingestion and a decision to move from a small MovieLens plus TMDB 5000 pairing (36% join rate) to MovieLens 20M plus TMDB 2023 (98% join). Modules 2, 3 and 4 and the 23-KPI evaluation suite followed within days, then a Streamlit dashboard, a redesigned multi-page UI with a SQLite user store, and finally a leak-free labelling pipeline, a frozen benchmark and diagnostic experiments on the classifier. (Chapter 1 has the dated timeline.)

**A5 [Intermediate] What did you actually build versus plan?**
Everything in the original plan exists: the three engines, the evaluation suite and an interface. What we did not achieve is a high-accuracy strategy classifier: we planned a classifier, built it, and then showed that its accuracy is limited by label reliability.

**A6 [Advanced] Why did you choose movie recommendation?**
It has an open, standard dataset (MovieLens) with enough users and ratings for collaborative filtering and for training a per-user model, a rich metadata source (TMDB) for content-based methods, and clearly measurable offline evaluation. It also lets us demonstrate each recommender family and compare them fairly.

**A7 [Expert] If you were told to remove one of the three engines, which would you remove?**
The classical IR engine's Trie autocomplete is not used by the recommendation path, but TF-IDF similarity is used for content candidates and for the diversity objective, so I would remove the Trie first. Removing the strategy selector would still leave a working recommender with the fixed prior blend, but we would lose the per-user idea that is the project's core.

**A8 [Critical] What is the biggest weakness of the project?**
The strategy classifier does not beat the majority-class rate on plain accuracy (0.617 vs 0.644) because its labels are noisy. We measured this rather than hiding it, and showed that the strategy choice still gives a statistically significant hit-rate gain over always-popularity.

## B. Problem statement and research gap

**B1 [Basic] State the problem statement.**
Given a user's possibly short rating history and a movie catalog with text and metadata, choose the most suitable strategy for that user, generate candidates accordingly, and rank them on six objectives to produce a top-K list that is accurate on held-out interactions while remaining diverse, novel and explainable.

**B2 [Basic] What is information overload?**
Too many items for a person to browse. MovieLens has 27,278 movies and TMDB about 1.5 million, so users need a short, relevant list.

**B3 [Intermediate] Why is one strategy not enough for everyone?**
Evidence differs per user: collaborative filtering needs history and neighbours, content-based needs a coherent taste profile, popularity needs nothing. In our data the best strategy on a hidden holdout was popularity for 64% of users, collaborative for 33% and content-based for 2%, and the mix varies with history size (the selector gains most for users with 21-40 ratings).

**B4 [Intermediate] What is the research gap?**
Many hybrid systems blend strategies with one global weighting, and many re-rankers optimise one or two objectives. CineRankX studies user-specific strategy selection together with a six-objective re-ranker in one explainable pipeline and, importantly, measures how reliable strategy labels actually are.

**B5 [Advanced] Why not just use the best strategy overall (popularity)?**
Popularity is strong offline, which is why it is our frozen baseline, but it is not personalised, has very low catalog coverage (0.38% in our benchmark) and cannot serve users with distinct tastes. The point is to beat it where evidence allows. On the benchmark CineRankX reaches Hit Rate@10 of 0.877 versus 0.601.

**B6 [Expert] What if every user's best strategy were popularity?**
Then a selector would add nothing and a classifier would equal the majority rule. We see something close to this: 64% of labels are popularity. That is why we compare against the majority baseline and report hit-rate gains: the classifier is useful only to the extent that it identifies users who gain from a different strategy.

**B7 [Critical] Is your research contribution really new?**
Not as individual techniques; each is established and we say so. The contribution is the combination, the leak-free labelling and frozen-benchmark methodology, and the documented negative result on label reliability. Related work on predicting recommender failure and per-user algorithm selection exists (for example Ekstrand and Riedl, RecSys 2012).

## C. Architecture and system design

**C1 [Basic] Describe the architecture.**
Five layers: raw data and SQLite; ingestion producing `movies_merged.csv`; three recommendation engines; an evaluation layer (23-KPI suite plus frozen benchmark); and a Streamlit presentation layer. (Figure in section 3.5.)

**C2 [Basic] Where does a recommendation request flow?**
`runner.py` collects inputs; `app.recommend` builds features and predicts a strategy; `MultiObjectiveRanker.rank` generates candidates and ranks greedily; results are shown and logged to SQLite.

**C3 [Intermediate] Why is there both `src/evaluation` and `src/benchmark`?**
`src/evaluation` is the original suite that computes all 23 named KPIs on a sample (per-user random splits). `src/benchmark` was added later because we wanted a stricter, frozen protocol: a hash-based split that cannot change, identical cohorts for every strategy, and a baseline that cannot drift.

**C4 [Intermediate] Why SQLite for the user store?**
We need one user's ratings out of 20 million rows in milliseconds, with server-side filtering and pagination. SQLite with an index on `(userId, ts)` does that in a single file without a server. A multi-user production system would use PostgreSQL.

**C5 [Advanced] Why Streamlit rather than Flask plus React?**
Time and scope: Streamlit lets Python models become a working interface with little code, which fitted a mini-project. The costs are less layout control and a script re-run model; we managed these with caching (`st.cache_resource`) and custom CSS.

**C6 [Advanced] How does caching work and why is it needed?**
Streamlit re-executes the script on every interaction. `load_engine()` is decorated with `@st.cache_resource`, so the catalog, ratings sample and ranker load once per server process. The KPI JSON uses `@st.cache_data` keyed by file modification time so the page shows new results as soon as a report is regenerated.

**C7 [Expert] What happens if the classifier file is missing?**
`StrategyClassifier.load()` returns False, `recommend` leaves `strategy = None`, and `strategy_source_weights` returns the fixed prior (collaborative 0.60, content 0.20, popularity 0.20). The app keeps working; there is a unit test for the missing-strategy path.

**C8 [Critical] What are the scalability limits?**
The live app loads only 400,000 ratings (about 2,900 users) and the collaborative source uses a 5,000-user by 2,000-movie window, because the user-user similarity is quadratic. Profiles use the database and cover all 138,493 users, but live recommendations are limited to the sample. For millions of users we would need approximate nearest neighbours or a learned model, a feature store and a serving layer.

## D. Dataset and preprocessing

**D1 [Basic] What datasets do you use?**
MovieLens 20M (20,000,263 ratings, 138,493 users, 27,278 movies, 465,564 tags) and the TMDB Movies Dataset 2023 (about 1.5 million movies with overviews, genres, keywords, vote statistics and poster paths).

**D2 [Basic] Why MovieLens 20M?**
It is a standard benchmark, large enough for collaborative filtering and classifier training, small enough for a laptop, and it ships `link.csv` mapping movies to TMDB ids. We also considered `ml-latest-small` (too small) and `ml-32m` (the join rate to TMDB 5000 would fall to about 5%).

**D3 [Basic] Why integrate TMDB?**
MovieLens has only titles, genres and tags. TMDB adds plot overviews, keywords and vote statistics, which give the content-based engine real text and the quality objective a vote count and average.

**D4 [Intermediate] How do you join them?**
`link.csv` gives `tmdbId` for each MovieLens movie; we inner-join on `tmdbId == TMDB id`. 26,743 of 27,278 movies (98.04%) match; the rest have no TMDB id or are missing from the export.

**D5 [Intermediate] How is text prepared?**
We concatenate overview, genre names and keywords into `text_chunk`, then `clean_text`: lowercase, keep alphabetic tokens, remove English stop-words and Porter-stem. Cast and crew are not available in this TMDB export, and we deliberately did not backfill them.

**D6 [Intermediate] What data problems did you find?**
Missing overviews (89 movies), 252 movies without a TMDB id, singular versus plural filenames across MovieLens releases, datetime-string timestamps instead of integers, and TMDB genres/keywords as comma-separated strings instead of JSON. No duplicate movie ids and no missing values in the classifier feature table.

**D7 [Advanced] How do you split data and why?**
Three splits serve different purposes: a per-user profile/holdout split to create strategy labels; a user-level 80/20 stratified split plus 5-fold CV to evaluate the classifier; and a hash-based train/validation/test split of ratings (70/10/20) for the recommendation benchmark. Splitting by user for the classifier prevents the same user appearing in train and test.

**D8 [Expert] Why a hash split instead of random shuffling?**
A hash of (user, movie, seed) gives every rating a fixed uniform number, so the split is identical on every run and machine, independent of row order and sample size, and needs no stored file. That is what lets us call the test set "frozen".

**D9 [Critical] What are the weaknesses of your data?**
MovieLens ratings stop in 2015 (so newer movies and tastes are absent); users have at least 20 ratings so there are no natural cold-start users; ratings come from one site's volunteers; the join drops 2% of movies; content is limited to overview, genres and keywords.

**D10 [Critical] How do you handle cold start in evaluation?**
Since no natural cold users exist, we simulate them: take real users, keep only their first five training ratings (by a deterministic order) and score them on their untouched held-out items. The code records `mode = "simulated"` so it is never presented as natural.

## E. Collaborative filtering

**E1 [Basic] What is collaborative filtering?**
Recommending items that similar users liked, using only the pattern of who rated what.

**E2 [Basic] User-based versus item-based?**
User-based finds similar users and recommends what they liked; item-based finds items similar (by co-ratings) to what the user liked. Our `candidates.py` is user-based nearest-neighbour; our EASE model in the benchmark is item-item.

**E3 [Intermediate] Explain the user-item matrix and sparsity.**
Rows are users, columns are movies, entries ratings. MovieLens 20M has 20,000,263 ratings among 138,493 x 27,278 possible cells, about 0.53% filled, so most pairs of users share few rated movies.

**E4 [Intermediate] How is similarity computed?**
Cosine similarity between rating vectors: dot product divided by the product of norms, with unrated entries as zero. Worked example in 6.A.4: for user U1 the similarity with U2 is 0.80.

**E5 [Intermediate] How does your CF candidate generator work?**
We find the 25 most similar users in a pool of 5,000 users and 2,000 most-rated movies, take each neighbour's 20 highest-rated movies, and score a candidate by the sum of the similarities of the neighbours who propose it.

**E6 [Advanced] Why did you not use matrix factorisation?**
We kept the explainable classical scope. Matrix factorisation (SVD) was explored informally and discarded in favour of EASE, which is also classical, closed-form and strong on top-N tasks. The repository contains no SVD implementation.

**E7 [Advanced] What is EASE and why did you choose it?**
EASE is a linear item-item model: ridge regression of every item on all other items with a zero diagonal, in closed form `B = -P / diag(P)` with `P = (X'X + lambda I)^-1`. We chose it because it trains in seconds on the 6,000 most-rated movies, scores a cold user from only five ratings by fold-in, and beat popularity clearly on our benchmark (Hit Rate@10 0.877 vs 0.601).

**E8 [Expert] What happens if you use Pearson correlation instead of cosine?**
Pearson mean-centres each user, removing rating-scale bias (harsh vs generous raters) and usually helps on dense data, but on very sparse vectors the centred overlap is small and noisy. It is a reasonable improvement we have not tested.

**E9 [Critical] Weaknesses of CF here?**
Cold start; popularity-biased neighbours; the 5,000-user window means most users are not neighbours of each other; no normalisation of ratings; quadratic similarity cost.

## F. Content-based filtering

**F1 [Basic] What is content-based filtering?**
Recommend items similar in content (text, genres) to items the user liked, without using other users.

**F2 [Basic] What content do you use?**
`text_chunk_clean`: stemmed overview, genre names and keywords from TMDB; plus genre sets for gating and diversity.

**F3 [Intermediate] How is a user profile built?**
In the label backtest, as the rating-weighted sum of the TF-IDF rows of the user's 20 best-rated movies; the movies most cosine-similar to that vector (excluding rated ones) are recommended. In the ranking engine, the user's top-10 rated movies seed `similar_to` queries.

**F4 [Intermediate] What is the genre gate?**
A multiplier `floor + (1 - floor) x Jaccard(genres)` applied to IR candidates so that keyword-only matches (such as toy-themed horror films for *Toy Story*) are suppressed. Default floor 0.25.

**F5 [Advanced] Why is content-based your weakest strategy?**
Mean hit-rate 0.016 against about 0.115 for collaborative and popularity. Reasons: the text is short and generic; the profile averages many different movies; random-split offline evaluation rewards methods that capture popularity and co-rating; and TF-IDF over overview words says little about quality.

**F6 [Expert] How did you try to improve it?**
We tested representations on 2,000 users with a development/confirmation split. Bigram TF-IDF with 20,000 terms raised hit-rate from 0.0156 to 0.0246; adding a quality prior (similarity times a quality percentile to the power 2) raised it to 0.0452, about 2.9 times better (95% CI of the gain +0.027 to +0.033). Mean-centred profiles and appended genre blocks made it worse. It is still far below popularity (0.119).

**F7 [Critical] Is the quality prior still "content-based"?**
Partly. It makes the strategy lean towards well-rated similar movies, so some gain is a generic quality effect. We say so explicitly and keep it as a separate, labelled variant.

**F8 [Critical] Why is cast/director similarity missing?**
The TMDB export in use has no cast or crew; backfilling it for only a minority of movies would make the feature inconsistent across the catalog.
