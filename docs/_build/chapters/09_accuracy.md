# Chapter 9. The Classifier Accuracy Investigation

This chapter explains why the strategy classifier scores modestly, what was tried, and how to talk about it honestly. Every number below was read from a saved artifact (file names in brackets) and can be regenerated with the scripts named.

## 9.1 Reproducing the headline numbers

| Quantity | Value | Source artifact |
|---|---|---|
| Held-out accuracy (999 users) | **0.617** (0.6166) | `data/processed/strategy_eval.json`, `selector_phase2_base.json` |
| Majority-class baseline (always "popularity") | **0.644** (0.6436) | same |
| Macro-F1 | **0.472** | same |
| Balanced accuracy (macro recall) | **0.469** (chance = 0.333) | same |
| ROC-AUC (one-vs-rest, macro) | **0.735** | same |
| Selector realised hit-rate | **0.128** (0.1276; 95% CI [0.121, 0.135]) | `selector_phase2_base.json` |
| Always-popularity hit-rate | **0.112** (0.1125) | same |
| Oracle hit-rate (upper bound) | **0.148** (0.1482) | same |
| Training accuracy of the same model | 0.765 (analysis run) to 0.876 (earlier run) | `selector_analysis.json`, `selector_experiments.json` |

**Discrepancies worth knowing** (an examiner may compare reports):

1. *The accuracy of the same configuration has been reported as 0.617, 0.634 and 0.636 in different runs.* The held-out set has only 999 users and just 22 content-based users; run-to-run variation of about +/-0.02 in accuracy and +/-0.015 in macro-F1 is normal at this size. The exact internal source of the variation between runs is **UNVERIFIED**; the saved, deployed result is 0.617.
2. *The dashboard's "F1 = 0.623" is the **weighted** F1, while the "macro-F1 = 0.472" is the unweighted mean.* They answer different questions (section 9.8).
3. *Label counts differ slightly between two scripts* (collaborative 1,668 / content 113 / popularity 3,212 in `formulations.winner`; 1,617 / 164 / 3,212 in the legacy `select_best_strategy`). The only difference is the tie rule when content-based and collaborative tie above popularity (`winner` prefers collaborative; the legacy function prefers the first tied key, content-based). Popularity is identical in both.
4. *The earlier accuracy of 0.636 versus a 0.550 baseline* was measured after dropping about 30% of users whose winner was ambiguous. That subset has a different class mix. It was replaced by the "no user dropped" evaluation above because dropping hard cases flatters accuracy.

## 9.2 Why the accuracy is low: the ranked root causes

### Cause 1 (largest): the labels are noisy
A user's label is the argmax of three **sample** hit-rates computed on their hidden holdout, which holds on average 44 movies (min 9, median 26, max 299). A hit-rate from so few items is a noisy estimate. Evidence:

* **Winner inside the noise.** Treat each hit-rate as a proportion with standard error *SE = sqrt(p(1-p)/m)* where *m* is the holdout size. For **60.3%** of users the gap between the best and second-best strategy is smaller than one combined standard error (`winner_within_1_standard_error`).
* **Label agreement.** Labelling the *same* 3,000 users twice with two independent random holdouts gives the same label only **65.3%** of the time (old 10-recommendation / 20% setting and current 30 / 30% setting) and **73.3%** with a larger 50-recommendation / 40% holdout (`label_experiments.json`).
* **Accuracy ceiling.** Simulate an oracle that knows each user's *true* hit-rates (taking the observed ones as truth) and is then scored against a freshly noisy label: its expected accuracy is only **0.761** (`noise_ceiling_full_population`; 0.825 / 0.768 / 0.793 for the three labelling settings).
* **Accuracy tracks label clarity** (figure below): the quarter of held-out users with the most ambiguous labels is classified correctly 52% of the time; the clearest quarter 70%.

![Accuracy by label clarity](figs/label_margin.png)

> **Plain-language version for the viva:** *"We ask the model to guess a label that is itself only about two-thirds reproducible. Even a perfect model would be marked wrong on the noisy third."*

### Cause 2: the information gap
The distance between the ceiling (0.76) and what we reach (0.62-0.65) is the part the available features do not explain. Features describing the user's taste and history are only mildly informative about which backtest happens to win. Nearest-neighbour check: in the closest 10% of user pairs in feature space, the labels disagree 31% of the time against 48% expected by chance, so the features do carry real signal but far from enough.

### Cause 3: content-based is a rare and weak strategy
Mean strategy hit-rates over 4,993 users: collaborative 0.115, popularity 0.116, content-based **0.016**; the oracle (best of three per user) 0.152. Content-based wins for only 2.3% of users (113), so the held-out set has just 22 of them. A class with so few examples is almost unlearnable; this was already noted in `DATASET_MIGRATION.md` (SMOTE could not fix it because it interpolates between the same few real points).

### Cause 4: overfitting, which is visible
Training accuracy (0.77-0.88) is far above cross-validated (0.61-0.63) and held-out accuracy. Boosted trees can memorise label noise; regularisation limits but cannot remove it.

### Cause 5: a strong majority class
64.4% of users are labelled popularity. A classifier that learns anything useful about the minority classes must sacrifice some majority accuracy, so plain accuracy can fall below 0.644 even while macro-F1, balanced accuracy and AUC rise.

## 9.3 Why the majority baseline can beat the classifier on accuracy
Accuracy counts every user equally; if 64% of users are one class, "always predict that class" scores 64% while being useless for the other 36%. On the held-out users the deployed-style model predicts popularity for 67% of users, collaborative for 32% and content-based for 0.2%, while the true label mix is 64% / 33% / 2%. It recovers 57% of the collaborative users (recall 0.57) at the price of some wrong majority-class calls: a deliberate trade of plain accuracy for minority recall. Balanced accuracy (0.469 vs 0.333 for a majority guesser or random guesser) and ROC-AUC (0.735 vs 0.5) show that it does carry signal.

## 9.4 Why changing algorithms did not help
Fourteen candidates were compared under one protocol (5-fold CV on 3,994 development users, then once on the held-out users; `selector_experiments_inner3.json`):

![Model comparison](figs/model_zoo.png)

| Candidate | CV macro-F1 | CV accuracy | CV realised hit-rate |
|---|---|---|---|
| **multiclass LightGBM (deployed)** | 0.467 | 0.633 | 0.131 |
| multiclass XGBoost | 0.457 | 0.629 | 0.130 |
| multiclass CatBoost | 0.454 | 0.640 | 0.131 |
| multiclass Extra Trees | 0.466 | 0.620 | 0.130 |
| multiclass Random Forest | 0.456 | 0.613 | 0.130 |
| multiclass HistGradientBoosting | 0.447 | 0.630 | 0.130 |
| one-vs-rest LightGBM | 0.436 | 0.645 | 0.130 |
| pairwise LightGBM | 0.458 | 0.628 | 0.131 |
| regression (expected utility) LightGBM | 0.453 | 0.638 | 0.131 |
| LambdaRank over strategies | 0.421 | 0.636 | 0.129 |
| soft voting (LGB + ET + Cat) | 0.453 | 0.646 | 0.131 |
| **stacking (LGB + ET + Cat)** | 0.410 | 0.524 | 0.117 |

All sensible models land in a band of 0.43-0.47 macro-F1 and 0.61-0.65 accuracy; the differences between the top candidates are inside run-to-run noise (rerunning the same two candidates moved their CV macro-F1 by about 0.01). **Therefore the bottleneck is the data (labels and information), not the algorithm.** Deploying LightGBM is justified by simplicity, speed and being among the best, not by a decisive win.

## 9.5 Further experiments on the labelled data (`selector_phase2_base.json`)

| Candidate | Held-out accuracy | Macro-F1 | Balanced acc. | Hit-rate [95% CI] | Gain vs popularity [95% CI] |
|---|---|---|---|---|---|
| A. multiclass LightGBM | 0.617 | 0.472 | 0.469 | 0.128 [0.121, 0.135] | +0.0152 [+0.010, +0.020] |
| B. + sigmoid calibration | 0.495 | 0.401 | 0.571 | 0.113 [0.106, 0.120] | +0.0001 [-0.005, +0.005] (not significant) |
| B2. + tuned class thresholds | 0.653 | 0.482 | 0.491 | 0.125 [0.118, 0.132] | +0.0125 [+0.009, +0.017] |
| D. pairwise | 0.636 | 0.483 | 0.497 | 0.129 [0.122, 0.136] | +0.0166 [+0.012, +0.022] |
| E. LambdaRank | 0.642 | 0.403 | 0.406 | 0.127 [0.120, 0.133] | +0.0143 [+0.010, +0.019] |
| C. utility regression | 0.643 | 0.440 | 0.434 | 0.128 [0.121, 0.135] | +0.0156 [+0.011, +0.020] |
| F. utility + popularity fallback (margin 0.005) | 0.658 | 0.443 | 0.433 | 0.128 [0.121, 0.135] | +0.0155 [+0.011, +0.020] |

Observations: (1) probability calibration *hurt* here (it flattened class separation); (2) tuning per-class thresholds is the only change that lifted plain accuracy above the majority rate (0.653 vs 0.644) while keeping macro-F1 up, but it did not improve recommendation value; (3) all selectors achieve essentially the same hit-rate, differences far inside the confidence intervals.

## 9.6 What the selector is actually worth

![What strategy choice is worth](figs/decision_value.png)

The decision-quality view asks a different question: *regardless of whether the label is "right", does the chosen strategy produce more hits?* For each held-out user take the hit-rate of the strategy the selector picked. Mean 0.128 against 0.112 (always popularity) and 0.148 (oracle). The gain over always-popularity, +0.0155, has a 95% bootstrap CI [+0.011, +0.020] (user-level paired bootstrap), so it is statistically significant. The selector therefore captures roughly (0.128-0.112)/(0.148-0.112) = **43%** of the achievable gap. **Caution:** the oracle is optimistic by construction, because it takes the best of three *noisy* scores per user; no real selector can reach it.

By history size (held-out users in thirds by number of profile ratings): users with 21-40 ratings, selector 0.185 vs popularity 0.152; 41-89 ratings, 0.130 vs 0.117; 90+ ratings, 0.070 vs 0.069 (no meaningful gain). Strategy choice matters most for light users, which is where it should.

## 9.7 Did a better label definition help?
Three labelling settings (`label_experiments.json`, 3,000 users, two independent holdouts each):

| Setting | Label agreement | Oracle ceiling | Class mix (collab / content / pop) |
|---|---|---|---|
| old: 10 recommendations, 20% holdout | 0.653 | 0.825 | 21% / 3.1% / 75.6% |
| current: 30 recommendations, 30% holdout | 0.653 | 0.768 | 31.6% / 2.3% / 66.1% |
| larger: 50 recommendations, 40% holdout | **0.733** | 0.793 | 37.5% / 1.6% / 60.9% |

An **uncertainty-aware** variant marks a user *indistinct* unless the winner beats the runner-up by a one-sided 95% margin. Result: 72% (old), 85% (current), 80% (larger) of users are indistinct; among the few users with a clear winner in both labellings, agreement is 92-99% (3-11% of users). This is the strongest evidence that most per-user "best strategy" labels are statistically indistinguishable. A larger evaluation budget gives more repeatable labels (0.653 -> 0.733) but squeezes the content-based class further (1.6%).

## 9.8 Metric glossary for this chapter

| Term | What it measures | Where it can mislead |
|---|---|---|
| **Classification accuracy** | share of users whose predicted strategy equals the label | inflated by a dominant class; ignores label noise |
| **Majority-class baseline** | accuracy of always predicting the most common label (0.644) | the number a useful classifier should be compared with, not zero |
| **Macro-F1** | unweighted mean of per-class F1 | very sensitive to the tiny content-based class (22 test users) |
| **Weighted F1** | per-class F1 weighted by class size | close to accuracy; dominated by the big classes |
| **Balanced accuracy** | mean per-class recall | chance level = 1/3 for three classes |
| **ROC-AUC (OvR)** | ability to rank users of a class above others | insensitive to calibration and to the decision threshold |
| **Recommendation hit-rate** | of the movies a strategy recommends, how many appear in the holdout (hits / holdout size), averaged over users | depends on holdout size and recommendation budget |
| **Oracle hit-rate** | mean of each user's best of three hit-rates | optimistic: selection from noisy numbers |

**Why improving classification accuracy does not necessarily improve recommendation quality.** When two strategies perform almost the same for a user, mislabelling that user costs almost nothing in hit-rate; the costly mistakes are the rare cases where one strategy is clearly better. A classifier can therefore gain accuracy by getting many "ties" right without changing what users see, or lose accuracy yet improve hit-rate. That is why both objectives are reported separately.

## 9.9 Improving the content-based strategy (`content_experiments.json`)
Hypothesis: content-based is weak because its representation is poor. Protocol: label backtest (70/30 split, 30 recommendations), 2,000 users; variants chosen on a 1,000-user development half and confirmed on 1,000 *different* users.

![Content variants](figs/content_variants.png)

| Variant (confirmation half) | Hit-rate [95% CI] | Gain vs original |
|---|---|---|
| original (2,000 terms, top-20-rated profile) | 0.0156 [0.0138, 0.0174] | n/a |
| bigram TF-IDF, 20,000 terms ("rich") | 0.0246 [0.0221, 0.0273] | +0.0090 |
| quality prior only (alpha = 2) | 0.0383 | +0.0227 |
| **rich + quality prior (alpha = 2) (selected on dev)** | **0.0452 [0.0420, 0.0484]** | **+0.0296 [+0.027, +0.033]** |
| mean-centred ("liked-centred") profile | 0.0093 | -0.0063 (worse) |
| genre block appended (weight 1.0) | 0.0122 | -0.0034 (worse) |
| reference: popularity | 0.1188 | |
| reference: collaborative | 0.1130 | |

Improving the representation nearly **triples** content-based hit-rate, but it stays far below popularity. Two caveats: the quality prior makes content-based partly "popular among similar-looking movies", so part of the gain comes from a generic quality signal; and in MovieLens-style random-split evaluation popularity dominates any method not explicitly built on collaborative signals.

## 9.10 End-to-end effect of the improved content strategy (`selector_phase2_rq_t30_h0.3.json`)
Rebuilding labels and features with the improved content model changes the problem: the mean content-based hit-rate rises from 0.016 to 0.049 and the content-based class grows from 2.3% (113) to 10.6% (527 users). Results on the new labels (999 held-out users): CV macro-F1 about 0.49-0.50 (up from 0.47), held-out accuracy 0.546 against a *lower* majority rate of 0.585, macro-F1 0.47-0.48, selector hit-rate 0.130 vs always-popularity 0.116 (gain +0.0129 to +0.0149, CIs excluding zero) and an oracle of 0.158. Interpretation: the minority class becomes more learnable, but the decision value of choosing a strategy does not change. **This variant is an experiment; it is not what is deployed.**

## 9.10b The best experiment so far: improved content + larger label budget (`selector_phase2_rq_t50_h0.4.json`)
Combining the improved content strategy with the more reliable label definition (50 recommendations per strategy, 40% holdout) changes the picture. All users are kept; evaluation is the same 80/20 user split with 999 held-out users:

| Quantity | Deployed setting (30 recs, 30%, original content) | Experiment (50 recs, 40%, improved content) |
|---|---|---|
| Class mix (collab / content / popularity) | 33% / 2% / 64% | 36% / 7% / 56% |
| Majority-class accuracy | 0.644 | 0.564 |
| Held-out accuracy (multiclass LightGBM) | 0.617 | **0.677** (+11 points above the majority rate) |
| Macro-F1 | 0.472 | **0.586** |
| Balanced accuracy | 0.469 | **0.606** |
| ROC-AUC | 0.735 | **0.807** |
| Selector hit-rate / always-popularity / oracle | 0.128 / 0.112 / 0.148 | 0.199 / 0.162 / 0.218 |
| Gain over always-popularity [95% CI] | +0.0152 [+0.010, +0.020] | **+0.0368 [+0.030, +0.044]** |

By history size the experiment shows the same pattern, more strongly: users with 18-36 ratings: selector 0.297 vs popularity 0.210; 37-79 ratings: 0.196 vs 0.173; 80+ ratings: 0.107 vs 0.105.

**How to read this honestly.** (1) The two columns use *different label definitions* (budget, holdout size, content model) and different class mixes, so their hit-rates and accuracies are not directly comparable; compare each column with *its own* majority rate and popularity reference. (2) What does carry over is the conclusion: **better labels and a better content strategy make the classifier clearly beat the majority class** (accuracy +11 points, balanced accuracy 0.61, AUC 0.81) and double the gain over popularity (+0.037 vs +0.015). That supports the diagnosis that the earlier ceiling came from label quality. (3) This setting has **not been deployed**; the app still uses the deployed model of 9.1. Adopting it means rebuilding labels and features (`selector_phase2.py --top-n 50 --holdout 0.4` builds the frame; deployment would use the same deployment function as before) and then regenerating the KPI report.

## 9.11 Status of the more advanced strategy-selection ideas

| Idea | Status in the repository |
|---|---|
| Utility-based selection (predict each strategy's expected hit-rate, pick the best) | Implemented and evaluated (`src/strategy_selector/utility.py`, candidates C and F above); statistically tied with the classifier; not deployed |
| Pairwise strategy comparison | Implemented and evaluated (`PairwiseForm`); tied |
| Learning-to-rank over strategies (LambdaRank) | Implemented and evaluated (`RankingForm`); lower macro-F1, equal hit-rate |
| Uncertainty-aware fallback | Implemented as the utility margin *tau* (0.005); did not improve over *tau* = 0 |
| Contextual bandit | **Not implemented**; needs online user feedback that the project does not collect |
| Hybrid fusion of strategy lists | **Not implemented** |
| Improved content-based model | Implemented as `content_model.py`; evaluated; not yet the deployed content strategy |

## 9.12 How to defend the result in one paragraph
*"Our classifier's accuracy is 61.7%, below the 64.4% majority rate, and we report that openly. We showed that the cause is label reliability: two independent re-labellings of the same users agree only about 65-73% of the time, and a perfect model could reach only about 76%. Changing algorithms (fourteen of them) did not move the result, which confirms the bottleneck is the data. What matters for users is recommendation quality: following the selector's choice gives a hit-rate of 0.128 versus 0.112 for always-popularity, a gain that is statistically significant (95% CI +0.011 to +0.020)."*
