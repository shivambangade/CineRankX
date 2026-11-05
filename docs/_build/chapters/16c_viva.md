## M. Evaluation metrics

**M1 [Basic] Define precision and recall.**
Precision: of what we predicted/recommended, how much was right. Recall: of what was truly relevant, how much we found. For the collaborative class: precision 191/418 = 0.457, recall 191/334 = 0.572.

**M2 [Basic] What is F1?**
The harmonic mean `2PR/(P+R)`, high only when both precision and recall are high.

**M3 [Basic] What is a confusion matrix?**
Counts of true class versus predicted class; the diagonal is correct. It shows which mistakes happen, such as 216 true-popularity users predicted collaborative.

**M4 [Intermediate] Macro versus weighted F1?**
Macro averages classes equally (0.472), weighted by class size (0.623). Macro exposes weak minority classes; weighted looks like accuracy.

**M5 [Intermediate] What is ROC-AUC?**
The probability that a random positive receives a higher score than a random negative; 0.5 is chance. For three classes we use one-vs-rest and average (0.735). Example: scores positives {0.9, 0.7, 0.4}, negatives {0.6, 0.3, 0.2}: 8 of 9 pairs ordered correctly, AUC 0.889.

**M6 [Intermediate] Define Precision@K, Recall@K and Hit Rate@K.**
P@K = hits in top K divided by K; R@K = hits divided by the number of relevant items; HR@K = share of users with at least one hit in the top K.

**M7 [Intermediate] Define NDCG and MAP.**
NDCG sums `1/log2(rank+1)` over hits and divides by the ideal sum, rewarding early hits. MAP averages, over users, the precision at each hit rank divided by min(K, #relevant).

**M8 [Advanced] What is a "relevant" movie in your evaluation?**
A held-out rating of at least 4.0 (`positive_threshold`). Movies the user rated poorly do not count as hits.

**M9 [Advanced] What is the difference between accuracy and hit rate?**
Accuracy (classification) is the share of users whose predicted strategy equals the labelled strategy. Hit Rate@10 (recommendation) is the share of users for whom at least one of the ten recommended movies was a movie they later liked. The dashboard's "accuracy" in the strategy comparison is Hit Rate@10, which is different from the classifier accuracy.

**M10 [Expert] Why can Recall@10 be near zero for search?**
There is no hand-labelled relevance, so relevance is a proxy (shares a genre or keyword). On average 1,812 of 26,743 movies are relevant per query, so recall is bounded by 10 divided by the number of relevant movies and measured 0.010. It says more about the relevant pool size than about the engine.

**M11 [Expert] When is RMSE misleading?**
A model can predict ratings accurately (low RMSE) and still rank poorly, since ranking depends on the order among high-rated items. We therefore report both (RMSE 0.846, MAE 0.649 for the bias predictor, and the ranking metrics).

**M12 [Critical] Are the hit-rate numbers comparable across your experiments?**
Only within the same protocol: hit-rate depends on holdout size, K and the number of positives. The benchmark's 0.877 (K=10, test split) and the classifier labelling's 0.128 (30 recommendations vs a 30% holdout) are different quantities.

## N. Accuracy and label noise

**N1 [Basic] What is label noise?**
Labels that are not reliably correct. Here, a user's "best strategy" is the maximum of three noisy sample hit-rates computed from a small holdout (median 26 movies).

**N2 [Intermediate] What is a standard error and how do you use it?**
The typical sampling error of an estimate; for a hit-rate `p` from `m` holdout items it is about `sqrt(p(1-p)/m)`. We compared the gap between the top two strategies with their combined standard error: for 60.3% of users the gap is within one standard error, so the winner is not reliably identified.

**N3 [Intermediate] What is label agreement?**
Label the same users twice with two independent random holdouts and measure how often the labels match: 65.3% in our setting (73.3% with a larger holdout and budget).

**N4 [Intermediate] What is the accuracy ceiling and how was it estimated?**
We simulated an oracle that knows each user's true hit-rates (taking the observed values as truth), drew fresh noisy labels from the holdout sizes and computed how often the oracle's choice matches them: about 0.76.

**N5 [Advanced] Why is training accuracy higher than test accuracy?**
Boosted trees fit patterns, including noise, in the training labels; noise does not repeat in new data. Train 0.77-0.88 versus held-out 0.62.

**N6 [Advanced] Why did changing algorithms not help?**
Fourteen models and formulations all landed between 0.41 and 0.47 CV macro-F1 and 0.61-0.65 accuracy. When the target is only about two-thirds reproducible, model choice matters little; improving the label (budget, holdout size) and the information (features) matters more.

**N7 [Advanced] What did improving the label definition do?**
A 50-recommendation / 40%-holdout definition raised label agreement from 0.65 to 0.73. Combined with the improved content strategy, the experiment (not deployed) reached held-out accuracy 0.677 against a 0.564 majority rate, macro-F1 0.586 and ROC-AUC 0.807.

**N8 [Expert] Is it legitimate to change the label definition to get better accuracy?**
Only if the change makes the labels *more reliable and meaningful* and not merely easier. We justified it by agreement and uncertainty measures, we report both settings with their own baselines, and we never changed test labels to flatter results.

**N9 [Expert] What is an "indistinct" user and how is it handled?**
A user whose best strategy does not beat the runner-up by a one-sided 95% margin. 72-85% of users are indistinct depending on the setting. Instead of inventing a confident label for them, this representation records the uncertainty; among users with a clear winner in both labellings, agreement is 92-99%.

**N10 [Critical] Why not remove ambiguous users to raise accuracy?**
That would measure performance only on easy cases and overstate how well the system works for real users. We did it at one stage (accuracy 0.636) and then reverted to the full population because removing difficult samples is not honest.

**N11 [Critical] Why can the classifier have a low accuracy yet improve recommendations?**
Misclassifying a user whose strategies perform almost equally costs almost nothing; the value comes from the cases where the strategies differ clearly. The selector's hit-rate is 0.128 vs 0.112 (popularity), with a significant paired gain.

**N12 [Critical] Is the oracle hit-rate of 0.148 a fair target?**
No: it is the mean of each user's best of three noisy numbers, so it is biased upward and unreachable. We use it only as a rough upper bound; the selector captures about 43% of the gap between popularity and that bound.

## O. Software architecture and implementation

**O1 [Basic] How is the code organised?**
By package: `ingestion`, `ir_engine`, `strategy_selector`, `ranking_engine`, `evaluation`, `benchmark`, `storage`, `ui`; scripts and tests alongside. Tunable settings live in `eval_config.yaml`.

**O2 [Intermediate] How does `MultiObjectiveRanker.rank` work?**
It generates or accepts a candidate pool, precomputes the four per-item objectives, then loops K times: score every remaining candidate (adding diversity and coverage relative to the selected set), pick the best, update the selected list and covered genres. It returns a frame with a column per objective.

**O3 [Intermediate] How is the UI connected to the model?**
`app.py` wraps loading and prediction; `runner.py` collects input, calls `recommend`, and renders results; `profile.py` reuses the same runner and result display for per-user recommendations.

**O4 [Advanced] Why must training and serving use the same feature function?**
Any difference (a column computed differently) silently shifts inputs and degrades predictions. `build_features` is used both to build the training table and in `app.recommend`, and a test checks they produce identical columns.

**O5 [Advanced] How do you serialise models and what is the risk?**
Pickle for scikit-learn/LightGBM objects, `save_npz` for sparse matrices, `np.savez` for EASE. Pickle can execute code on load, so files must come from a trusted source; pickled classes must live in importable modules.

**O6 [Expert] What would you change for production?**
Pin dependencies; add authentication; replace pickle with a safer format or ONNX; move features to a feature store; serve from a REST API with a model registry; add monitoring and online experiments; replace the in-process cache with a service.

**O7 [Critical] What is the weakest part of the engineering?**
Unpinned requirements, no authentication, no automated browser tests, and a ranking pipeline whose heavy parts run in one Streamlit process.

**O8 [Critical] Where does the frozen baseline live and how is it protected from changes?**
`src/benchmark/baseline.py` is self-contained (imports only numpy and pandas; a test checks this). `scripts/run_benchmark.py` stores a SHA-256 hash of the baseline, metrics and split code plus the baseline's results; if either the code or the numbers change, the run aborts.

## P. Testing and debugging

**P1 [Basic] How many tests do you have?**
272 test cases in 12 files; all passed at the last full run.

**P2 [Intermediate] Give an example of a test that prevents leakage.**
`test_holdout_labels_never_use_holdout_ratings` and `test_features_use_only_the_ratings_given`; SMOTE tests assert synthetic rows are never derived from test users.

**P3 [Intermediate] How do you test a probabilistic model?**
With planted-signal tests (the model must find a known pattern) and noise tests (it must not beat chance on random labels).

**P4 [Advanced] Describe a bug you found and how.**
The popularity definition: both engines used TMDB popularity; we noticed the backtest popularity score was zero for most users and that the TMDB top-10 touched only 39% of users, so labels were assigned by tie-break. Switching to observed rating counts fixed the labels and the baseline.

**P5 [Advanced] Another bug?**
A saved classifier that could not be loaded because its estimator factory had been defined in a script and pickled as `__main__.make`; we moved it to `src/strategy_selector/estimators.py`.

**P6 [Critical] What is not tested?**
End-to-end browser behaviour, load/performance tests, and the correctness of external data. Pages were exercised with Streamlit's `AppTest` and manually.

## Q. Project differentiation

**Q1 [Basic] How is CineRankX different from a standard recommender website?**
It chooses and weights methods per user, ranks on six objectives, explains each recommendation with scores, lets the operator change trade-offs live, logs runs, and reports its own measurement limits.

**Q2 [Intermediate] What is your actual contribution?**
The combination (per-user strategy prior plus six-objective explainable ranking), a leak-free and frozen evaluation methodology, and the documented finding that per-user best-strategy labels are only weakly reproducible. We do not claim a new algorithm.

**Q3 [Intermediate] Why six objectives?**
Each one counters a specific failure of relevance-only lists: repetition (diversity), filter bubbles (novelty), concentration (coverage), unreliable ratings (quality), and ignoring the user's likely score (predicted rating).

**Q4 [Advanced] Why is per-user strategy selection useful?**
Evidence differs by user. The selector gains most for light users (0.185 vs 0.152 for popularity among users with 21-40 ratings) and almost nothing for heavy users (0.070 vs 0.069).

**Q5 [Advanced] How do you compare against modern platforms?**
We do not claim equivalence; public descriptions (for example Gomez-Uribe and Hunt 2015) describe many algorithms, personalised ranking and online experiments. CineRankX is an offline, explainable, research-scale system.

**Q6 [Expert] What would make it publishable?**
Multi-dataset evaluation, comparison with learned hybrids and published algorithm-selection methods, online or counterfactual evaluation, a formal uncertainty-aware label protocol, and the label-reliability analysis as the central finding.

**Q7 [Critical] Isn't this just a hybrid recommender?**
It contains hybrid elements, but a standard hybrid uses one blend for everyone. Here the blend depends on a per-user prediction, and the final list is the product of a multi-objective ranking step, both measured.

## R. Limitations and future scope

**R1 [Basic] Name three limitations.**
Weak content-based strategy, noisy strategy labels limiting classifier accuracy, and offline-only evaluation with no user feedback.

**R2 [Intermediate] How would you improve content-based recommendation?**
Richer text representation (bigrams helped), embeddings, metadata such as cast and director, a user taste model that weights liked and disliked items, and a quality prior; we measured the first and last.

**R3 [Intermediate] How would you handle users with no history?**
Fall back to popularity or quality-weighted popularity, ask for genre preferences, and use metadata-based content recommendations; EASE can score a user from even five ratings.

**R4 [Advanced] How would you scale to millions of users?**
Approximate nearest-neighbour search or learned embeddings instead of dense user-user similarity, a precomputed item-item model (EASE restricted to top items), a feature store, batch scoring and a serving service, plus incremental retraining.

**R5 [Advanced] What is a contextual bandit and would it help?**
A bandit learns online which action (here, strategy) pays off in a given context (user features) using observed rewards (clicks or watches). It would remove the label-noise problem by learning from real feedback, but needs live users; our system collects none.

**R6 [Expert] Could reinforcement learning help?**
Possibly for sequential session optimisation if interaction logs existed; with static ratings it adds complexity without clear benefit.

**R7 [Critical] If given six more months?**
Deploy the improved label definition and content strategy; add real feedback collection and a bandit; evaluate on several datasets with temporal splits; add embeddings and cast/crew metadata; productionise (API, authentication, monitoring).

## S. Advanced research questions

**S1 [Expert] How would you formally estimate the benefit of strategy selection?**
Per-user off-policy comparison: for each user, estimate each strategy's expected utility with uncertainty, then compare the selector's choices against always-popularity with a paired bootstrap on per-user utilities (what we did) or with doubly robust estimators on logged data.

**S2 [Expert] What is regression to the mean in the labels?**
Picking the maximum of three noisy estimates overstates the winner's true advantage and makes labels unstable. It is why the oracle number is optimistic and why labels agree only 65-73%.

**S3 [Expert] Why not use the continuous hit-rates as soft labels?**
We tried predicting each strategy's expected hit-rate directly (utility regression with a fallback margin); it matched but did not beat the classifier. Soft targets reduce noise sensitivity but cannot add information the features lack.

**S4 [Expert] How would you account for multiple comparisons across fourteen models?**
Differences among top models were inside run-to-run noise; selection by CV macro-F1 on development users and a single held-out evaluation limit optimism. A strict approach would use nested CV or a pre-registered single comparison.

**S5 [Expert] Could the benchmark's 0.877 be inflated by popularity?**
Popularity is in the pool through the blend, but the frozen baseline gets 0.601 on the same ground truth; the gain is substantial in NDCG@10 (0.163 to 0.362) and precision (0.131 to 0.291). Coverage 5.7% vs 0.38% shows items beyond the head are being recommended.

**S6 [Expert] What are the threats to validity?**
One dataset; random (not temporal) splits that favour popularity; simulated cold start; proxy relevance for search; the classifier's simplified strategies in the backtest differ from the production sources; small held-out sets for rare classes.

**S7 [Critical] Where could the evaluation still leak?**
Candidate generators for the old pipeline use ratings of other users in the training pool; hyper-parameters of EASE were tuned on a validation cohort drawn from the same user population; the 5,000-user classifier labelling sample was drawn independently of the benchmark cohorts, so some users may appear in both (not checked). None uses the test interactions of the evaluated users, but these are points to mention honestly.

**S8 [Critical] Defend the project in one sentence.**
CineRankX shows that per-user strategy selection plus explainable multi-objective ranking can beat a frozen popularity baseline on held-out interactions (Hit Rate@10 0.877 vs 0.601), while honestly demonstrating that the strategy labels, not the algorithms, limit classifier accuracy.
