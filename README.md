# Cracking the Rift: Predicting Pro League of Legends Match Outcomes from the 15-Minute Mark

**Name:** Yifei Chen

**Course:** DSC 80, Practice and Application of Data Science, UCSD (Spring 2026)

---

## Introduction

League of Legends is the most-watched competitive video game in the world, with a professional scene spanning dozens of regional leagues and tens of millions of viewers tuning in to global tournaments. In each match, two five-player teams race to destroy each other's base, gaining advantages through kills, gold, experience, and map objectives (dragons, heralds, barons, towers). One of the oldest questions in the LoL community is this: how much does the early game decide the outcome? Veteran analysts constantly debate whether a sizeable lead at 15 minutes is essentially game-over, or whether enough comebacks happen that the late game still meaningfully matters.

This project investigates that question using data from every professional LoL match played in 2022, collected by [Oracle's Elixir](https://oracleselixir.com/). The central question I am trying to answer is:

> **Can we predict the winner of a professional League of Legends match using only information available at the 15-minute mark, and which early-game indicators matter most?**

This question matters to anyone who has watched a pro game and felt a result was "decided" before it should have been. Sports broadcasters, esports analysts, betting markets, and team coaches all have practical reasons to want a calibrated answer.

### The dataset

The raw dataset contains **150,348 rows × 165 columns**, structured as 12 rows per match (5 players per team plus 2 team-summary rows), covering **12,529 matches** across 80+ regional leagues. After filtering to team-summary rows (the unit of analysis for predicting team outcomes), we have **25,058 rows**, equal to 12,529 matches times 2 teams.

The columns most relevant to the central question are:

- **`gameid`**: unique match identifier
- **`league`**: regional league (LCK, LPL, LEC, LCS, etc.); 80+ values ranging from tier-1 majors to academy leagues
- **`side`**: `Blue` or `Red`; blue side has a small but documented historical advantage
- **`result`**: 1 if the team won, 0 otherwise (the target variable for the prediction model)
- **`gamelength`**: total match duration in seconds
- **`golddiffat15`, `xpdiffat15`, `csdiffat15`**: team's lead over its opponent in gold, experience, and creep score at the 15-minute mark
- **`killsat15`, `opp_killsat15`**: team and opponent kill counts at 15 minutes
- **`firstblood`, `firstdragon`, `firstherald`, `firsttower`**: indicators for being first to secure each early-game objective
- **`ckpm`**: combined kills per minute across both teams, a standard proxy for "action" in a match
- **`datacompleteness`**: `complete` or `partial`; as shown later, this column structurally explains all of the missingness in the timeline-at-15 features

---

## Data Cleaning and Exploratory Data Analysis

### Cleaning steps

The cleaning recipe addresses the special considerations called out by Oracle's Elixir's data layout, plus a few that come up naturally during analysis:

1. **Filter to team-summary rows.** Each match has 12 rows (5 per team plus 2 team summaries). Since we're predicting *team* outcomes, we keep only `position == 'team'`. This collapses the dataset from 150,348 rows to 25,058 rows (2 per match).
2. **Cast boolean-like columns.** Columns like `firstblood`, `firstdragon`, `firsttower`, `playoffs`, and `result` are stored as integer 0/1 but represent yes/no facts. We cast them to `boolean` so they read correctly and play nicely with grouping.
3. **Parse `date` to datetime** for time-trend studies.
4. **Engineer a `tier` column** marking matches as `Tier 1` (LCK, LPL, LEC, LCS, PCS, VCS, CBLOL, LJL, LLA, LCO plus global events MSI, Worlds, and EWC) or `Other`. The Oracle's Elixir dataset has no built-in region/tier column, so we add it. Tier 1 corresponds to the franchised major leagues that send teams to the World Championship.
5. **Engineer `killdiffat15`** as `killsat15 − opp_killsat15`, the team's net kill lead at 15 minutes.
6. **Convert `gamelength` from seconds to minutes** for readability.

We do **not** drop rows with missing timeline data at this stage; the missingness itself is the subject of the next section.

### Head of the cleaned DataFrame

| gameid               | league   | tier   | side   | result   |   gamelength_min |   golddiffat15 |   killdiffat15 | firstblood   | firstdragon   |
|:---------------------|:---------|:-------|:-------|:---------|-----------------:|---------------:|---------------:|:-------------|:--------------|
| ESPORTSTMNT01_2690210 | LCK CL   | Other  | Blue   | True     |             32.0 |           1373 |              0 | True         | False         |
| ESPORTSTMNT01_2690210 | LCK CL   | Other  | Red    | False    |             32.0 |          -1373 |              0 | False        | True          |
| ESPORTSTMNT01_2690219 | LCK CL   | Other  | Blue   | False    |             34.5 |          -1234 |             -2 | False        | True          |
| ESPORTSTMNT01_2690219 | LCK CL   | Other  | Red    | True     |             34.5 |           1234 |              2 | True         | False         |
| ESPORTSTMNT01_2690227 | LCK CL   | Other  | Blue   | True     |             27.5 |           2143 |              1 | True         | True          |

### Univariate analysis

The first feature we examine is **game length**, that is, how long a typical pro match runs.

<iframe src="assets/01_gamelength.html" width="840" height="480" frameborder="0"></iframe>

Game lengths cluster around **30 to 32 minutes**, with a long right tail of matches running 40+ minutes. The shape matches the intuition that most pro games end shortly after the first baron buff (available at 20 minutes) snowballs into a base push.

Next we look at the distribution of **gold difference at 15 minutes**, the single most-discussed early-game variable in pro LoL.

<iframe src="assets/02_golddiff15.html" width="840" height="480" frameborder="0"></iframe>

The distribution is roughly symmetric around zero by construction (every gold lead one team has is its opponent's deficit). The bulk of team-rows fall within ±3,000 gold (about 68%), corresponding to small advantages worth roughly one extra kill plus a few extra creep waves. The tails reaching ±8,000+ gold (~2% of team-rows) represent the "stomps" that pros and viewers often complain about.

### Bivariate analysis

Now we look at how the strongest early-game economic signal, **gold difference at 15 minutes**, relates to the outcome.

<iframe src="assets/03_golddiff_by_outcome.html" width="840" height="480" frameborder="0"></iframe>

Winning teams hold a median gold lead of roughly **+1,500 gold** at the 15-minute mark, while losing teams sit at roughly **−1,500 gold**. The distributions overlap substantially (there is no clean gold-lead threshold that guarantees a win), but the central tendency is clearly separated. This suggests gold-diff-at-15 alone should be a moderately strong predictor in the model later.

We can also look at how taking specific early-game **objectives** swings win rate:

<iframe src="assets/04_winrate_objectives.html" width="840" height="500" frameborder="0"></iframe>

Taking the **first tower** swings win rate the most (~68% vs ~32%), followed by **first blood** (~61% vs ~39%). **First herald** (~58% vs ~42%) and **first dragon** (~58% vs ~42%) are nearly tied as the weakest signals among these four. Each objective still meaningfully tilts the game, but securing a structure clearly dominates the others.

### Interesting aggregates

To see how the meta varies across the most-active leagues in 2022:

| league   |   Matches |   Blue win rate |   Mean ckpm |   Avg game length (min) |
|:---------|----------:|----------------:|------------:|------------------------:|
| LCK      |       467 |           0.507 |       0.700 |                   33.67 |
| UPL      |       405 |           0.551 |       1.362 |                   28.61 |
| VCS      |       325 |           0.495 |       1.052 |                   30.03 |
| LCS      |       306 |           0.493 |       0.733 |                   33.03 |
| PCS      |       273 |           0.604 |       0.832 |                   30.95 |

Three observations stand out. **Blue-side advantage is real but uneven.** Most leagues sit above 50%, with PCS at 60.4% and the LCK at 50.7%, but a few (LCS at 49.3%, VCS at 49.5%) actually showed a small red-side edge in 2022. **Action varies wildly.** The UPL averaged 1.36 combined kills per minute, nearly **double** the LCK's 0.70. And **major leagues play longer.** The LCK and LCS sit at the top of the game-length distribution, consistent with their methodical, scaling-focused playstyle. We test the action observation formally in the hypothesis-testing section below.

---

## Assessment of Missingness

### NMAR analysis

Among columns with appreciable missingness in the team-rows dataset (`golddiffat15`, `xpdiffat15`, `csdiffat15`, etc.), I believe **none are NMAR**. NMAR would require that the probability of being missing depends on the *missing value itself*, even after conditioning on everything else observed.

The timeline-at-15 columns are missing in bulk for entire leagues. The LDL (Chinese amateur league) and the LPL (Chinese pro league) both have 100% missing `golddiffat15` in 2022. The cause is structural: Riot Games doesn't always release the live-event timeline data to data scrapers for these regions. That's a property of the **data-collection process**, not of the underlying gold-diff value. Concretely, if Oracle's Elixir were granted timeline access to all LPL matches, those gold-diff values would fill in regardless of what they turned out to be. This is a textbook **MAR** situation: the missingness depends entirely on observable columns (`datacompleteness` and `league`).

If I instead wanted to argue for NMAR, the additional data I'd need is information about *why* a specific match's timeline data is unavailable on a per-game basis (for instance, whether Riot's API failed on close games more often than blowouts). I have no reason to believe this is the case, and Oracle's Elixir's `datacompleteness` flag points directly at the structural explanation. So I conclude these columns are MAR, not NMAR.

### Missingness dependency tests

I analyzed the missingness of `golddiffat15` (about 15% of team-rows are missing it) using permutation tests with two candidate dependency columns.

**Dependency we found.** Missingness of `golddiffat15` **does depend on `gamelength_min`** (observed mean difference ≈ 0.77 minutes, p < 0.001 from 1,000 permutations). Matches missing the timeline tend to be slightly longer than matches with full timelines. This makes sense, since the leagues with `datacompleteness == partial` (notably the LPL and LDL) average longer games on slower-paced patches, and *all* of their timeline data is missing. The observed difference falls far outside the null distribution from the permutation test:

<iframe src="assets/05_missingness_null.html" width="840" height="480" frameborder="0"></iframe>

**Independence we found.** Missingness of `golddiffat15` **does not depend on `result`** (observed difference essentially 0, p ≈ 1.0). Win and loss rows are equally likely to be missing, confirming the missingness is structural-by-league, not outcome-dependent.

This is exactly the MAR pattern we predicted: the missingness can be fully explained by observable characteristics (`league`, `datacompleteness`), and is conditionally independent of the missing values themselves once you know the league.

---

## Hypothesis Testing

### Question

In the EDA, the pivot table revealed that combined kills per minute (`ckpm`) varies dramatically across leagues. The Ukrainian UPL averaged 1.36 ckpm while the LCK barely cleared 0.70. We test that observation formally: **is the amount of in-game action in tier-one professional leagues meaningfully different from the action level in everything else?**

I used `ckpm` as the operationalization of "action" because it directly captures how many champion takedowns happen per minute of game time, which is the dominant action signal viewers and analysts use.

### Hypotheses

- **H₀ (null):** The mean combined kills-per-minute (`ckpm`) is the **same** in Tier 1 leagues (LCK, LPL, LEC, LCS, PCS, VCS, CBLOL, LJL, LLA, LCO plus international events) as in all other leagues.
- **H₁ (alternative):** The mean `ckpm` **differs** between Tier 1 and non-Tier 1 leagues (two-sided).

### Test choice

- **Test:** Permutation test (two-sided), 10,000 permutations.
- **Test statistic:** Difference in mean `ckpm` between Tier 1 and Other (`mean_T1 − mean_Other`).
- **Significance level:** α = 0.05.
- **Why permutation rather than a t-test?** `ckpm` is right-skewed (the long tail of high-action matches violates the t-test's normality assumption), and the two groups have very different sample sizes. The permutation test makes no distributional assumption.

### Result

<iframe src="assets/06_hypothesis_null.html" width="840" height="480" frameborder="0"></iframe>

- **Tier 1 mean ckpm:** 0.834 (n = 3,492 matches)
- **Other mean ckpm:** 0.971 (n = 9,037 matches)
- **Observed difference:** −0.136
- **Two-sided p-value:** < 0.0001

The observed difference falls far outside the null distribution.

### Conclusion

Since p < α = 0.05, **we reject the null hypothesis**. The evidence is consistent with the alternative: combined kills per minute differs significantly between Tier 1 and other leagues, with Tier 1 games actually being **less** action-packed by this measure. This direction may be counterintuitive (one might expect the best players in the world to play the most exciting games), but it matches what experienced viewers describe: top professional teams play more methodically, value vision and macro setups, and are less willing to risk skirmishes than amateur or semi-pro leagues. We cannot prove that Tier 1 games are *categorically* less action-packed; we can only say that the data we observed would be very surprising under the null hypothesis that the two means are equal.

---

## Framing a Prediction Problem

We frame the prediction task that motivated the entire project:

> **Given only information that would be known at the 15-minute mark of a professional League of Legends match, predict whether the team will win.**

This is a **binary classification** problem. The **response variable** is `result` ∈ {0, 1} (1 = win), aggregated to one row per match by taking the blue-side perspective. So `result` represents "did the blue side win?".

I chose `result` because it's the single most consequential variable in competitive LoL. Predicting it from early-game indicators has real-world stakes: betting markets, broadcast win-probability graphics, and team analytics all rely on exactly this kind of model.

### Time of prediction

I must be careful about what information would be known **at the time of prediction** (minute 15 of the match). Any feature recorded after 15 minutes is off limits since using it would be data leakage. The features I allow:

- `golddiffat15`, `xpdiffat15`, `csdiffat15` (economic and lane state)
- `killsat15`, `opp_killsat15`, and their difference (combat state)
- `firstblood`, `firstdragon`, `firstherald` (early-objective indicators; all decided in the first ~15 minutes)
- `league`, `patch` (match context)

The features I forbid: `gamelength`, total `kills`/`deaths`/`assists`, total `dragons`/`barons`/`towers`/`inhibitors`, `gspd`, `gpr`, `firstbaron`, and `firsttower`. Baron isn't available until 20 minutes; first tower can theoretically fall before 15 but its semantics in this dataset aren't strictly tied to a ≤15-min boundary, so I conservatively exclude it.

### Evaluation metric

I use **accuracy** as the primary metric. This is justified for two reasons.

1. **The classes are exactly balanced** (50% win rate by construction, since every match has one winner). With balanced classes, accuracy is meaningful and interpretable.
2. **The cost of false positives and false negatives is symmetric** in this application. There's no asymmetric harm from mis-predicting a win as a loss vs. the reverse. F1, precision, and recall would all be valid here too, but accuracy most directly expresses the underlying question.

I use a **75/25 train/test split** with stratification on `result` (preserves balance) and `random_state=42` so the baseline and final models are compared on identical test sets.

---

## Baseline Model

### Setup

I restrict to matches with complete timeline-at-15 data (i.e., `golddiffat15` is not missing), since the prediction question by definition requires those features. This drops the 15% of team-rows where `datacompleteness == 'partial'`. I then take one row per match, from the **blue side's perspective**, so the target variable is "did blue win?".

### Features and model

The baseline uses **two features**:

1. **`golddiffat15`**: quantitative, passed through as-is. The single strongest early-game signal in LoL.
2. **`league`**: nominal, one-hot encoded. Different leagues have different baseline blue-side advantages and play styles.

By feature type: **1 quantitative**, **1 nominal**, **0 ordinal**.

The model is **logistic regression** (the simplest probabilistic linear classifier for binary outcomes). All preprocessing and fitting happen in a single `sklearn` Pipeline, with a `ColumnTransformer` that one-hot-encodes `league` and passes `golddiffat15` through unchanged.

### Performance

- **Train accuracy:** 0.7420
- **Test accuracy:** 0.7467
- **Naive baseline (always predict blue wins):** 0.5230

The baseline achieves **~74.7% test accuracy**, a substantial lift over the naive baseline of 52.3%. The train and test accuracies are within a few hundredths of each other, suggesting no meaningful overfitting at this complexity level.

Is this model "good"? It's **a reasonable starting point**, but not satisfying. Two features is the minimum the spec asks for, and I'm leaving signal on the table. Kill diff, XP diff, CS diff, and the first-objective flags all encode independent information about who's ahead at 15 minutes. The final model adds those.

---

## Final Model

### Features added

I added **five new engineered features** on top of the baseline:

1. **`xpdiffat15`** (quantitative, standardized). Experience lead is partially correlated with gold lead but captures *level* advantages, which translate into ability ranks and stat thresholds that gold can't directly measure.
2. **`csdiffat15`** (quantitative, standardized). Creep score difference captures lane farming dominance, which correlates with but isn't redundant with gold (gold counts plate gold, kill gold, dragon gold; CS only counts minions).
3. **`killdiff15`**, defined as `killsat15 − opp_killsat15` (quantitative, standardized). Net kill lead at 15. I used the *difference* rather than both raw kills columns because a team with 5-3 in kills is in a similar position to one at 7-5, and the difference compresses two correlated columns into one informative one.
4. **`firstdragon`** (binary). Although first dragon is among the weaker individual signals from EDA, it captures jungle-side priority and a small permanent stat boost that compounds across the rest of the game. In combination with the gold/XP features, it can help disambiguate teams with similar leads.
5. **`firstherald`** (binary). Herald spawns at 8 minutes and contests are usually resolved by ~14 minutes, so it's unambiguously known at the 15-minute mark.
6. **`patch`** (nominal, one-hot encoded). LoL patches change item costs and champion power weekly. Patches within the same year can shift average game pace substantially, and a tree-based model can learn patch-specific calibration.

These additions are justified from the **data-generating perspective**. Each captures a dimension of early-game state that the gold-diff feature alone misses: XP captures levels, CS captures lane macro, kill diff captures fight-by-fight aggression, the objective binaries encode discrete events that gold smooths over, and patch captures the contemporary meta context.

### Modeling algorithm

I moved from logistic regression to a **`RandomForestClassifier`**. Random forests handle non-linear interactions between features (e.g., a 1,500 gold lead might mean very different things when paired with a kill lead vs. a kill deficit), mixed numerical/categorical inputs gracefully, and are robust to a few noisy features.

### Hyperparameter search

I tuned three random-forest hyperparameters using `GridSearchCV` with 3-fold cross-validation, scored on accuracy:

- **`n_estimators`** ∈ {100, 300}: number of trees. More trees reduce variance at diminishing returns.
- **`max_depth`** ∈ {5, 10, None}: depth limit. Controls overfitting.
- **`min_samples_leaf`** ∈ {1, 5}: minimum samples per leaf. Larger values smooth predictions and reduce variance.

Twelve configurations × 3 folds = 36 fits. The same train/test split as the baseline (`random_state=42`, 75/25, stratified) is preserved so the comparison isolates the model change from any data-split lottery.

### Result

**Best hyperparameters found:** `max_depth=10`, `min_samples_leaf=5`, `n_estimators=300`. Best 3-fold CV accuracy: 0.7457.

| Model | Test accuracy |
|---|---|
| Baseline (logistic, 2 features) | 0.7467 |
| Final (Random Forest, 7 features) | **0.7546** |
| Absolute improvement | +0.0079 |
| Relative error reduction | +3.1% |

The final model improved test accuracy from **74.7% to 75.5%**, about a **3% reduction in classification error**.

The confusion matrix on the held-out test set shows the structure of the model's errors:

<iframe src="assets/07_confusion_matrix.html" width="640" height="540" frameborder="0"></iframe>

The model is slightly more accurate predicting blue wins than blue losses (precision 76% vs 75%, recall 78% vs 72%), partly because blue side is the majority class (52% of matches in the held-out set). The asymmetry is small and consistent with a well-calibrated classifier on a near-balanced problem.

Why the improvement isn't larger: gold-diff-at-15 is *already* such a strong feature that the remaining 25% of misclassifications are largely matches where the trailing team mounts a comeback. That's something we fundamentally cannot predict from minute-15 numbers, because those matches "look" exactly like other matches with similar minute-15 stats but happen to flip. The extra features each shave a little error off the margins but can't dissolve this irreducible noise.

The best configuration favored a **moderate tree depth** with **300 trees** and **`min_samples_leaf=5`**, which prevents the model from carving the feature space into tiny noise-fitted regions; the ensemble of 300 such trees then averages out individual-tree variance.

---

## Fairness Analysis

### Question

Does the final model predict matches in **Tier 1 leagues** (LCK, LPL, LEC, LCS, PCS, VCS, CBLOL, LJL, LLA, LCO plus international events) as accurately as it predicts matches in other leagues?

This is a fairness question with practical bite. Tier 1 leagues are watched by tens of millions of people. If a win-probability graphic shown during a broadcast is systematically worse at Tier 1 (because pros mount comebacks more often, breaking the minute-15 → outcome relationship), broadcasters need to know.

### Setup

- **Group X:** Tier 1 league matches
- **Group Y:** Other league matches
- **Evaluation metric:** accuracy

### Hypotheses

- **H₀ (null, fair):** The model's accuracy on Tier 1 matches equals its accuracy on Other matches. Any observed difference is due to chance.
- **H₁ (alternative, unfair):** The model's accuracy on Tier 1 matches **differs** from its accuracy on Other matches.

Test details:

- **Test statistic:** signed difference in accuracy, `acc(Tier 1) − acc(Other)`.
- **Significance level:** α = 0.05.
- **Procedure:** permutation test. The model and its predictions are held fixed (no retraining); the Tier-1/Other labels are repeatedly shuffled across the test set and the accuracy difference is recomputed under each random assignment. 10,000 permutations.

### Result

| Group | Accuracy | n |
|---|---|---|
| Tier 1 | 0.7525 | 691 |
| Other | 0.7553 | 1,970 |

- **Observed difference (Tier 1 − Other):** −0.0028
- **Two-sided p-value:** 0.9246

The observed difference sits right in the center of the null distribution, indicating no evidence that the model's accuracy differs by tier:

<iframe src="assets/08_fairness_null.html" width="840" height="480" frameborder="0"></iframe>

### Conclusion

We **fail to reject H₀** at α = 0.05. The evidence is fully consistent with the model being equally accurate on Tier 1 and Other league matches. This doesn't *prove* the model is perfectly fair (failure to reject is not proof of equivalence), but with 2,661 held-out matches we have enough sample size that a meaningful disparity would have been likely to surface, and none did.

A practical caveat: this fairness check uses the model the way it was trained (one global model across all leagues) and asks whether its accuracy is uniform across the tier dichotomy. Other fairness lenses (calibration parity, false-positive/false-negative balance, or per-league rather than per-tier accuracy) would be useful follow-ups but are outside the scope of this analysis.
