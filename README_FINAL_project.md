# Dota 2 Project — Revised Cleaning Rules

This revision is based on the user's original project.

## Data-cleaning rules used

### Step 1 — Duplicate match IDs
The code checks and reports duplicate `match_id` values.

It does **not** automatically delete them.

### Step 2 — Missing values
The code reports missing-value counts and percentages for each column.

It does **not** automatically delete missing rows.

### Step 3 — `radiant_win`
The only valid values are:

- `1` = Radiant win
- `0` = Dire win

The code checks and reports invalid or missing target values.

It does **not** automatically delete them.

### Step 4 — All-zero hero composition
The code removes the complete match row when either:

- Radiant is `[0, 0, 0, 0, 0]`, or
- Dire is `[0, 0, 0, 0, 0]`.

This is the only automatic deletion rule in the cleaning section.

## Why the modeling section may be skipped

Because Steps 1–3 are check-only, the code does not silently change those records. If duplicate IDs, missing required values, or invalid `radiant_win` values remain, the code reports them and skips model training instead of crashing or silently changing the study rules.

## Colab optimization

The revised code avoids creating hundreds of binary columns directly inside `df_matches_clean`. It uses `MultiLabelBinarizer(..., sparse_output=True)` separately for Radiant and Dire and then joins the two sparse matrices.

This keeps:
- Radiant heroes separate from Dire heroes,
- memory usage lower,
- Logistic Regression input numeric,
- feature creation faster than repeated `.apply()` calls.

## Logistic Regression correction

Logistic Regression uses `model.coef_` for interpretation.

`feature_importances_` is only used for Decision Tree and Random Forest.

## Model quality

The script compares:
- Baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Bernoulli Naive Bayes
- SGD Classifier

using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Log Loss

A model should not be called good from Accuracy alone. It should meaningfully improve on the baseline, show useful ROC-AUC, maintain balanced class metrics, and ideally remain stable under additional validation.

## Research improvements

Recommended next steps:
1. Add patch and match-time information.
2. Use cross-validation after the base pipeline is stable.
3. Tune hyperparameters using training data only.
4. Add hero-pair and hero-vs-hero interaction features.
5. Add player rank/MMR, roles, draft order, or team context if available.
6. Validate on later matches or a later patch.
7. Treat model feature importance as association, not causation.


## Additional models added

### Bernoulli Naive Bayes

Bernoulli Naive Bayes is included because the hero-composition predictors are binary presence/absence features. It is computationally light and works well with sparse matrices, making it suitable for Google Colab.

### SGD Classifier

The SGD Classifier uses `loss="log_loss"` so it behaves as a scalable linear probabilistic classifier. It is especially useful for large sparse datasets because it trains incrementally using stochastic gradient updates.

### Why these models were added

The final comparison now represents several different modeling approaches:

1. Logistic Regression — interpretable linear probabilistic model.
2. Decision Tree — nonlinear tree model.
3. Random Forest — ensemble of decision trees.
4. Bernoulli Naive Bayes — probabilistic model designed for binary features.
5. SGD Classifier — large-scale sparse linear classifier.

The purpose is not to select a model only because it has the highest Accuracy. All models should be compared using Accuracy, Precision, Recall, F1 Score, ROC-AUC, Log Loss, and the confusion matrices.
