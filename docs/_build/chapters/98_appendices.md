# Appendix A. Glossary

| Term | Meaning |
|---|---|
| **Accuracy** | share of correct predictions (classification) |
| **AUC / ROC-AUC** | probability a random positive outranks a random negative; 0.5 = chance |
| **Backtest** | evaluating a strategy on data held back from it (here: the user's hidden holdout) |
| **Balanced accuracy** | mean per-class recall; chance = 1/(number of classes) |
| **Bayesian weighted rating** | rating shrunk towards a global mean in proportion to how few votes it has |
| **Bias (predictor)** | a user's or item's systematic offset from the global mean rating |
| **Bootstrap** | resampling users with replacement to estimate uncertainty |
| **Candidate generation** | producing a pool of possible recommendations before ranking |
| **Class imbalance** | classes with very different frequencies |
| **Cold start** | a user or item with little or no history |
| **Collaborative filtering** | recommending from the behaviour of similar users or co-rated items |
| **Confidence (selector)** | maximum predicted class probability; steers the source blend |
| **Confusion matrix** | table of true versus predicted classes |
| **Content-based filtering** | recommending items similar in content to what the user liked |
| **Cosine similarity** | cosine of the angle between two vectors |
| **Coverage** | share of the catalog that appears in recommendations |
| **Cross-validation** | repeated train/test splits inside the training data to estimate generalisation |
| **Data leakage** | information from the evaluation target reaching training |
| **DCG / NDCG** | position-discounted gain of hits; NDCG normalises by the ideal ordering |
| **Diversity** | how dissimilar the items in one list are |
| **EASE** | closed-form linear item-item recommender (Steck 2019) |
| **F1 score** | harmonic mean of precision and recall |
| **Feature** | a numeric input describing a user or item |
| **Frozen baseline** | a reference method that is never retrained or changed |
| **Gradient boosting** | building trees sequentially, each fitting the previous errors |
| **Greedy selection** | choosing items one at a time, always the best given the current list |
| **Hessian** | second derivative of the loss; scales boosting steps |
| **Hit rate@K** | share of users with at least one relevant item in the top K |
| **Holdout** | data hidden from the model, used as the test |
| **Hyperparameter** | a setting chosen before training (learning rate, leaves) |
| **IDF** | inverse document frequency; down-weights common terms |
| **Jaccard similarity** | size of intersection divided by size of union of two sets |
| **Label noise** | labels that are not reliably correct |
| **Label agreement** | how often two independent labelings give the same label |
| **LightGBM** | gradient-boosting library with leaf-wise trees and histogram binning |
| **MAP** | mean (over users) of average precision |
| **Macro / weighted average** | unweighted / size-weighted mean over classes |
| **MMR** | maximal marginal relevance: balance relevance and novelty to already chosen items |
| **MRR** | mean reciprocal rank of the first relevant result |
| **Majority-class baseline** | accuracy of always predicting the most common class |
| **Novelty** | how unexpected (rare) an item is; self-information |
| **Oracle** | an idealised selector that always picks the best option (an upper bound) |
| **Overfitting** | fitting training noise; high training, lower test performance |
| **Popularity bias** | tendency to over-recommend already popular items |
| **Precision@K / Recall@K** | share of the top K that is relevant / share of relevant items found |
| **Profile / holdout** | the visible and hidden parts of a user's ratings |
| **Regularisation** | penalties that limit model complexity |
| **RMSE / MAE** | root mean squared / mean absolute error of rating predictions |
| **SMOTE** | oversampling by interpolating between minority samples |
| **Sparsity** | the fraction of empty cells in the user-item matrix |
| **Standard error** | typical sampling error of an estimate |
| **Stratified sampling** | keeping class proportions in each split |
| **TF-IDF** | term frequency times inverse document frequency weighting |
| **Trie** | prefix tree for fast prefix lookup |
| **TreeSHAP** | exact Shapley-value feature attributions for tree models |
| **Utility (strategy)** | expected recommendation quality of a strategy for a user |

# Appendix B. Quick Revision Notes (one page)

**The idea.** Predict the best strategy per user (content / collaborative / popularity), blend candidate sources by that prediction, re-rank on six objectives.

**Key numbers (verified).**
* Data: 20,000,263 ratings, 138,493 users, 27,278 movies; joined catalog 26,743 (98.04%).
* IR engine: Precision@10 0.41, MRR 0.44, NDCG 0.41, latency about 29 ms.
* Frozen benchmark (test): Hit Rate@10 baseline 0.601, old CineRankX 0.751, now **0.877** (cold 0.609, 0.658, 0.698); NDCG@10 0.163 -> 0.362; RMSE 0.936 -> 0.846.
* Classifier (999 held-out users, none dropped): accuracy **0.617**, majority 0.644, macro-F1 **0.472**, balanced accuracy **0.469**, ROC-AUC **0.735**.
* Selector value: hit-rate **0.128** vs popularity **0.112** vs oracle 0.148; gain +0.0155 [0.011, 0.020].
* Noise: winner within 1 SE for 60%; label agreement 65-73%; oracle ceiling about 0.76.
* Content-based hit-rate 0.016 (improved variant 0.045, not deployed).
* Best experiment: improved content + 50 recs / 40% holdout: accuracy 0.677 vs 0.564 majority; macro-F1 0.586; AUC 0.807; gain +0.037.
* System: about 23 ms per instrumented operation, about 46 ops/s, 398 MB.

**Formulas.**
* cosine: a.b / (|a||b|); idf = ln((1+N)/(1+df)) + 1.
* strategy weights: s = clip((conf - 1/3)/(2/3), 0, 1); w = (1-s) prior + s one-hot; prior (0.6, 0.2, 0.2).
* objectives: diversity = 1 - max sim; novelty = -log2 p; coverage = new genres / genres; quality = Bayesian WR; predicted = (r^ - 0.5)/4.5; score = sum of weight x objective (greedy).
* EASE: B = -P/diag(P), P = (X'X + lambda I)^-1.
* F1 = 2PR/(P+R); NDCG = DCG/IDCG; AUC = P(score+ > score-).
* LightGBM: g = p - y, h = p(1-p), leaf = -G/(H + lambda).

**Say these sentences.** "Accuracy is label-limited, not model-limited." "Hit rate and classification accuracy are different." "The baseline is frozen and self-contained." "Prediction is a weighted prior, not a hard switch." "Cold start is simulated because MovieLens has no cold users."

**Never say.** That the classifier is highly accurate; that CineRankX invented TF-IDF, CF or MMR; that the improved content model/labels are deployed; that cold-start results are natural.

# Appendix C. References and Source Paths

## C.1 Repository sources used for this guide
`README.md`, `CLAUDE.md`, `DATASET_MIGRATION.md`, `eval_config.yaml`, `requirements.txt`, `app.py`, all of `src/`, `scripts/`, `tests/`, `git log`, `demo/weekly_reports_week7-12.md`, and the saved artifacts: `data/processed/strategy_eval.json`, `selector_phase2_base.json`, `selector_phase2_rq_t30_h0.3.json`, `selector_phase2_rq_t50_h0.4.json`, `selector_analysis.json`, `selector_experiments_inner3.json`, `label_experiments.json`, `content_experiments.json`, `data/benchmark/baseline_frozen.json`, `data/benchmark/cinerankx_params.json`, `reports/kpi_report.json`, `reports/strategy_comparison.json`.

## C.2 Literature (recalled from memory; verify details before citing in a report)
* F. M. Harper and J. A. Konstan, "The MovieLens Datasets: History and Context", ACM TiiS, 2015.
* K. Spärck Jones, "A statistical interpretation of term specificity and its application in retrieval", Journal of Documentation, 1972 (IDF); G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval", 1988 (TF-IDF).
* J. Carbonell and J. Goldstein, "The use of MMR, diversity-based reranking...", SIGIR 1998.
* H. Steck, "Embarrassingly Shallow Autoencoders for Sparse Data", WWW 2019 (EASE).
* P. Cremonesi, Y. Koren, R. Turrin, "Performance of recommender algorithms on top-N recommendation tasks", RecSys 2010.
* M. D. Ekstrand and J. Riedl, "When Recommenders Fail: Predicting Recommender Failure for Algorithm Selection and Combination", RecSys 2012.
* C. A. Gomez-Uribe and N. Hunt, "The Netflix Recommender System: Algorithms, Business Value, and Innovation", ACM TMIS, 2015.
* G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017.
* N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique", JAIR 2002.
* S. Lundberg and S. Lee, "A unified approach to interpreting model predictions", NeurIPS 2017 (SHAP); LightGBM's `pred_contrib` gives TreeSHAP values.
