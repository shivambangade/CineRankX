## G. Popularity-based recommendation

**G1 [Basic] What is popularity-based recommendation?**
Recommend the items that most users rated. No personal data required.

**G2 [Basic] How do you define popularity?**
By observed rating count in the data (the 200 most-rated movies for the candidate pool, all movies sorted by count for the frozen baseline), not by TMDB's `popularity` column.

**G3 [Intermediate] Why not TMDB popularity?**
TMDB popularity is a live, recency-weighted number, while MovieLens ratings stop in 2015. The TMDB top-10 was touched by only 39% of users against 87% for the most-rated top-10, which made the popularity strategy look artificially weak and the cold-start baseline artificially easy to beat. We found and fixed this (commit `c273ba3`).

**G4 [Intermediate] What is the Bayesian weighted rating?**
`WR = v/(v+m) R + m/(v+m) C`, with `R` the movie's average, `v` its votes, `m` the median vote count and `C` the mean average. It pulls ratings of rarely voted movies towards the mean so three 10/10 votes cannot beat a film rated 8.4 by 20,000 people. Used in the popularity/quality objective, then min-max scaled.

**G5 [Advanced] Why is popularity such a strong baseline?**
In offline evaluation with random held-out ratings, many users' future ratings fall on already popular movies, so recommending the most-rated ones scores well. This is a known effect in the literature on top-N evaluation (for example Cremonesi, Koren and Turrin, RecSys 2010). In our benchmark it reaches Hit Rate@10 of 0.601, which is why we freeze it as the reference.

**G6 [Expert] What is popularity bias and how do you counter it?**
Popular items collect more exposure and ratings and so appear more popular; recommendations concentrate on few titles. We counter it with novelty (self-information), coverage and the quality objective, and we measure catalog coverage (baseline 0.38%, CineRankX 5.7%).

**G7 [Critical] Isn't a baseline that wins 64% of labels a sign the project adds little?**
It is a sign that popularity is hard to beat *for many users*, which we report openly. Where personalisation can help, it does: Hit Rate@10 0.877 vs 0.601 overall, and the selector adds +0.0155 over always-popularity (CI +0.011 to +0.020).

## H. Hybrid recommendation

**H1 [Basic] What is a hybrid recommender?**
One that combines several recommendation methods.

**H2 [Basic] Is CineRankX hybrid?**
Yes in two senses: it blends three candidate sources with weights, and it combines six objectives in the ranker.

**H3 [Intermediate] Hybrid versus strategy selection versus switching?**
Hybrid blends scores from several methods; strategy selection predicts which method suits a user; switching uses one method chosen by a rule. In CineRankX the prediction is a weighted prior over sources (so it is selection feeding a blend), not a hard switch.

**H4 [Intermediate] Exactly how does the prediction change the blend?**
`strategy_source_weights`: strength `s = clip((confidence - 1/3)/(2/3), 0, 1)`; weights `(1 - s) x prior + s x one_hot(strategy)` with prior (collaborative 0.60, content 0.20, popularity 0.20). At confidence 0.7 and strategy collaborative: collaborative 0.82, content 0.09, popularity 0.09.

**H5 [Advanced] Why a weighted prior instead of a hard switch?**
The classifier's probabilities are uncertain and its labels noisy, so a hard decision would amplify errors. A prior that smoothly moves towards the predicted strategy as confidence grows degrades gracefully; at chance confidence it reverts to a sensible default.

**H6 [Expert] Why not average the three ranked lists directly (list fusion)?**
It is a good idea and listed as future work. It needs per-strategy ranked lists and a fusion rule (for example reciprocal rank fusion); we did not implement it, and the repository contains no such code.

**H7 [Critical] Does the sources' sum favour popularity?**
Each source's scores are min-max normalised first, so raw magnitudes (popularity counts) do not swamp cosine scores. An item proposed by two sources outranks one proposed by a single source at equal strength. Normalisation is a design choice that could be varied.

## I. TF-IDF and cosine similarity

**I1 [Basic] What is TF-IDF?**
A weighting that scores a word highly if it appears often in a document (term frequency) but rarely across the collection (inverse document frequency).

**I2 [Basic] What is cosine similarity?**
The cosine of the angle between two vectors: 1 for the same direction, 0 for orthogonal. For L2-normalised TF-IDF vectors it equals the dot product.

**I3 [Intermediate] Give the formulas.**
`idf(t) = ln((1+N)/(1+df)) + 1` (scikit-learn smoothing), `tfidf = tf x idf`, then L2-normalise. `cos(a,b) = a.b / (|a||b|)`. Worked example in 6.B.2: d1 = "toy story toy adventure" gives toy 0.771, story 0.507, adventure 0.386.

**I4 [Intermediate] Which TF-IDF settings do you use and why?**
`max_features=5000, min_df=2, max_df=0.8, ngram_range=(1,2)`. `min_df=2` removes one-off terms, `max_df=0.8` removes near-universal terms, bigrams capture phrases, 5,000 terms limit memory.

**I5 [Advanced] Why cosine instead of Euclidean distance?**
Document length changes vector magnitude but not direction; cosine measures topical similarity independent of length. Euclidean distance would penalise longer overviews.

**I6 [Advanced] Why stemming?**
It merges inflections (*toys*, *toy*) so the same concept is one term, shrinking the vocabulary. The cost is non-words (*woodi*) and occasional over-merging.

**I7 [Expert] Would word embeddings be better?**
Likely for semantic similarity (synonyms), but they would break the explainable, no-deep-learning scope and need a pretrained model; TF-IDF is transparent and cheap. It would be a natural extension.

**I8 [Critical] What does TF-IDF miss?**
Synonyms and context, plot structure, tone, cast and director (not in our data), and quality. That is why the genre gate was necessary.

## J. Multi-objective recommendation

**J1 [Basic] List the six objectives.**
Relevance, diversity, novelty, coverage, popularity/quality, predicted rating.

**J2 [Basic] How are they combined?**
A weighted sum with weights normalised to 1 (defaults 0.30, 0.15, 0.15, 0.10, 0.15, 0.15), computed greedily: each step picks the candidate with the highest total given the list so far.

**J3 [Intermediate] Why greedy and not sort once?**
Diversity and coverage depend on what is already chosen; the best second item depends on the first. Greedy selection is the standard MMR-style approach.

**J4 [Intermediate] Define diversity.**
`1 - max similarity to already selected items`; the first item gets 1. Similarity is TF-IDF cosine, with genre Jaccard as a fallback.

**J5 [Intermediate] Define novelty.**
Self-information `-log2 p(item)` of observed engagement (rating count plus TMDB votes), min-max scaled: rarely seen items score near 1.

**J6 [Advanced] Why use max rather than mean similarity for diversity?**
One near-duplicate in the list makes a candidate redundant even if it differs from the other nine; a mean would hide that.

**J7 [Advanced] Is coverage a real implementation of catalog coverage?**
It is a per-item genre-coverage proxy (fraction of the item's genres not yet in the list). True catalog coverage is a property of all users' lists together and is reported as a KPI; we state clearly that the objective is only a proxy.

**J8 [Expert] What happens if you set all weight on relevance?**
The list becomes the most relevant but most similar and popular movies. In the benchmark, relevance-only ranking scored a slightly higher NDCG@10 than the tuned multi-objective version (0.373 vs 0.362 overall) but lower diversity (0.789 vs 0.809): the objectives cost a little accuracy and buy variety.

**J9 [Expert] How did you choose the weights?**
Defaults come from `eval_config.yaml`. For the benchmark we searched random weight vectors on the validation split, maximising NDCG@10 subject to diversity not falling below the baseline's. The tuned relevance weight is 0.576.

**J10 [Critical] Weighted sums assume trade-offs are linear. Is that a problem?**
Yes: a linear scalarisation cannot reach points on non-convex parts of the Pareto front and weights are not interpretable as exchange rates. Pareto-based or constrained methods are alternatives; we chose the weighted sum because it is transparent and adjustable by sliders.

## K. The strategy classifier

**K1 [Basic] What does it predict?**
For each user, which of three strategies (collaborative, content-based, popularity) is most likely to work best, with a probability for each.

**K2 [Basic] How are training labels created?**
By backtesting: split each user's ratings 70/30 into profile and holdout; run each strategy on the profile; count how many of its recommendations land in the holdout; the best hit-rate is the label (ties go to popularity).

**K3 [Intermediate] What features does it use?**
47 numbers per user: activity, rating statistics, genre concentration (entropy, top-genre share, 19 genre affinities), popularity of taste, time span, engine-strength scores and three "inner backtest" hit-rates computed on the visible profile (section 8.4).

**K4 [Intermediate] Why "inner backtest" features?**
The label measures how each strategy performs on a hidden holdout; the best available predictor of that is how the same strategies perform when the *visible* history is split again. It uses only ratings the system can see at prediction time.

**K5 [Advanced] How do you prevent leakage?**
Features come from the profile ratings only; labels from the holdout only; the user-level train/test split is made before any model fitting; SMOTE (disabled) only sees the training fold; CV and tuning use development users only; the held-out users are scored once.

**K6 [Advanced] Why LightGBM?**
It handles mixed numeric features, needs little preprocessing, trains fast and was among the best of fourteen candidates (CV macro-F1 0.467), though several models were statistically tied. We chose it for performance, speed and simplicity, not because it won decisively.

**K7 [Advanced] How is the confidence used?**
The maximum predicted probability becomes the `confidence` in `strategy_source_weights`, deciding how strongly the predicted strategy shifts the source blend.

**K8 [Expert] What if you drop the inner-backtest features?**
In an ablation (5-fold CV, balanced random forest, on the cleaned-label subset used at that stage) macro-F1 fell from about 0.47-0.48 with them to 0.43 without them, so they carry real information. (That ablation predates the full-population evaluation.)

**K9 [Expert] What if you drop all ambiguous users from the data?**
Accuracy rises (it was 0.636 on a cleaned subset) but only because we removed hard cases; it does not describe real users. We report the full-population number (0.617).

**K10 [Critical] Why is your classifier worse than the majority baseline?**
On plain accuracy it scores 0.617 against 0.644 for always predicting popularity. The labels are only about 65-73% reproducible and a perfect model would reach about 0.76; with a dominant class, a model that tries to recover minority classes pays in accuracy. It beats the baseline on balanced accuracy (0.469 vs 0.333), AUC (0.735 vs 0.5) and recommendation value (+0.0155 hit-rate).

**K11 [Critical] If accuracy is only 61.7%, why should we trust the system?**
Because the system's value is not the classifier's accuracy but the quality of the recommendations. The ranking pipeline reaches Hit Rate@10 0.877 against 0.601 for the frozen baseline, and following the selector gives a significant gain over always using popularity. The classifier's weakness is measured, explained and bounded by label noise, not hidden.

**K12 [Critical] Why not use a neural network for the classifier?**
The data is tabular with about 5,000 labelled users, where gradient-boosted trees are usually at least as good as neural networks, and the bottleneck is label noise rather than model capacity (fourteen models landed in a narrow band). A neural network would reduce explainability and add cost without addressing the bottleneck.

## L. LightGBM and machine learning

**L1 [Basic] What is a decision tree?**
A model that splits data by feature thresholds to reach leaf predictions.

**L2 [Basic] What is gradient boosting?**
Building trees sequentially, each one fitted to the errors (gradients) of the ensemble so far, and adding it with a small learning rate.

**L3 [Intermediate] What are gradient and Hessian here?**
For softmax cross-entropy, per class `g = p - y` and `h = p(1-p)`: the first and second derivatives of the loss with respect to the raw score. Leaf value `w = -G/(H + lambda)`.

**L4 [Intermediate] What is leaf-wise growth?**
LightGBM always splits the leaf with the largest loss reduction rather than growing all leaves at one depth, giving more accurate but more overfit-prone trees; `num_leaves` and `min_child_samples` control it.

**L5 [Intermediate] Which hyperparameters did you tune and how?**
`n_estimators` (357), `learning_rate` (0.022), `num_leaves` (27), `min_child_samples` (50), row and column subsampling (0.60, 0.46) and `reg_lambda`, by Optuna's TPE sampler with 3-fold CV macro-F1 on development users only, plus a sample-weight multiplier based on label clarity.

**L6 [Advanced] What is overfitting and how do you see it here?**
Memorising training noise: training accuracy 0.77-0.88 against 0.61-0.63 on held-out users. Remedies: small learning rate, limited leaves, minimum leaf size, subsampling, L2 regularisation, CV.

**L7 [Advanced] Why macro-F1 for model selection?**
Because accuracy would reward predicting the majority class. Macro-F1 averages per-class F1 equally, so poor minority-class performance is visible. It is noisy with only 22 content-based test users, so we also look at balanced accuracy, AUC and hit-rate gain.

**L8 [Expert] What does SMOTE do and why is it disabled?**
It creates synthetic minority samples by interpolating between real neighbours. Across ten seeds it raised content-based F1 by about 0.038 but left AUC flat and lowered accuracy in 10 of 10 seeds, meaning it only moved the decision threshold. We kept the code, tested for leakage, and disabled it.

**L9 [Expert] Why did probability calibration make it worse?**
Sigmoid calibration (3-fold) flattened the class probabilities on a small, noisy dataset; held-out accuracy fell from 0.617 to 0.495 and its hit-rate gain over popularity vanished (+0.0001). Calibration needs more data than we have.

**L10 [Critical] How do you know the model is not overfitting?**
We do not claim it is immune: the train-test gap shows overfitting. What we control is that evaluation is on users never used for fitting or selection (stratified 80/20 split, CV inside the 80%), and that conclusions rest on held-out numbers with confidence intervals.
