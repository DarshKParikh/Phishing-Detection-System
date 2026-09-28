# Methodology

## Current model

The project currently uses a Random Forest classifier with the feature columns
listed in `docs/features.md`.

The rule-based scorer is retained as a baseline.

## Evaluation metrics

Report at least:

- precision
- recall
- F1 score
- confusion matrix
- false-positive rate
- false-negative rate

ROC-AUC can also be reported when probability outputs and an appropriate
evaluation set are available.

## Avoiding misleading results

The current dataset is relatively small and is built from a live phishing feed
plus a curated legitimate list. Treat current performance measurements as
prototype results.

Future evaluation should include:

1. a larger legitimate sample
2. multiple phishing sources
3. source-aware validation
4. time-aware validation
5. difficult benign examples
6. known phishing examples
7. a held-out test set never used for model selection

## Useful experiments

### Model comparison

Compare:

```text
Rule-based baseline
Logistic Regression
Random Forest
Gradient Boosting
```

using the same held-out test set.

### Feature ablation

Compare:

```text
URL-only
URL + domain
URL + domain + HTML
URL + domain + HTML + brand signals
```

This shows which parts of the pipeline actually contribute useful information.
