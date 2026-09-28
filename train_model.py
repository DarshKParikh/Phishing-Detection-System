"""
Trains a classifier on labeled_dataset.csv (produced by phishing_scraper.py)
and compares it against the hand-built rule_based_score() baseline.

Run phishing_scraper.py first to generate the dataset, then run this.
"""

import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


FEATURE_COLUMNS = [
    'url_length',
    'num_dots',
    'num_hyphens',
    'has_at_symbol',
    'uses_https',
    'has_ip_as_domain',
    'on_free_hosting',
    'has_suspicious_keyword',
    'subdomain_length',
    'subdomain_entropy',
    'num_forms',
    'has_password_field',
    'form_posts_externally',
    'fetch_succeeded',
    'domain_age_known',
    'brand_edit_distance',
    'impersonates_brand',
]
# domain_age_days itself is left out — it's often missing (None) for real
# phishing URLs, and scikit-learn can't handle NaN directly. domain_age_known
# (whether we got an answer at all) is included instead as a cheap stand-in.
# matched_brand is text (the brand name itself), not a number, so it's left
# out too — brand_edit_distance and impersonates_brand already capture what
# matters about it numerically.


def load_dataset(filename='labeled_dataset.csv'):
    df = pd.read_csv(filename)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}\n"
            f"Did you run the latest phishing_scraper.py before this?"
        )
    return df


def get_domain_group(url):
    """Return a simple hostname group so one domain stays in one data split."""
    hostname = (urlparse(str(url)).hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def train_and_evaluate(df, test_size=0.3, random_state=42):
    """
    Train and evaluate while keeping the same hostname out of both train and test.

    This is a stricter evaluation than a random row split because multiple URLs
    from one website can otherwise make the test set too similar to training.
    """
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    groups = df["url"].map(get_domain_group)

    print(
        f"Dataset size: {len(df)} rows "
        f"({int(y.sum())} phishing, {int(len(y) - y.sum())} legit)"
    )
    print(f"Unique hostname groups: {groups.nunique()}")

    if len(df) < 1000:
        print(
            "\nDataset note: this is still a relatively small dataset for "
            "general website classification. Treat the metrics as experimental "
            "and keep expanding both legitimate and phishing examples."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(
        f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | "
        f"Train domains: {groups.iloc[train_idx].nunique()} | "
        f"Test domains: {groups.iloc[test_idx].nunique()}"
    )

    # Grouped cross-validation keeps each hostname entirely within a fold.
    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits >= 2:
        cv = GroupKFold(n_splits=n_splits)
        cv_scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                class_weight="balanced",
            ),
            X,
            y,
            cv=cv,
            groups=groups,
            scoring="f1",
        )
        print(f"\n--- {n_splits}-fold domain-grouped cross-validation (F1) ---")
        print(f"Folds: {[round(score, 3) for score in cv_scores]}")
        print(
            f"Mean: {cv_scores.mean():.3f} | "
            f"Std dev: {cv_scores.std():.3f}"
        )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    phishing_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Performance on unseen-domain test set ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["legit", "phishing"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, cols = predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=["actual: legit", "actual: phishing"],
            columns=["pred: legit", "pred: phishing"],
        )
    )

    print("\n--- Feature importance ---")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).round(3))

    # False positives are especially important for this project: these are
    # legitimate sites that the model incorrectly calls phishing.
    false_positive_mask = (y_test.to_numpy() == 0) & (y_pred == 1)
    false_positive_positions = [
        position
        for position, is_fp in enumerate(false_positive_mask)
        if is_fp
    ]

    print("\n--- False positives: legitimate sites predicted as phishing ---")
    if not false_positive_positions:
        print("None in this test split.")
    else:
        for position in false_positive_positions:
            idx = X_test.index[position]
            print(
                f"\n{df.loc[idx, 'url']} | "
                f"phishing probability={phishing_prob[position]:.1%}"
            )
            fired = {
                feature: X_test.loc[idx, feature]
                for feature in FEATURE_COLUMNS
                if bool(X_test.loc[idx, feature])
                and feature not in {"uses_https", "fetch_succeeded", "domain_age_known"}
            }
            print(f"  notable feature values: {fired}")
            if "risk_score" in df.columns:
                print(f"  rule risk score: {df.loc[idx, 'risk_score']}")
            if "risk_reasons" in df.columns:
                print(f"  rule reasons: {df.loc[idx, 'risk_reasons']}")

    print("\n--- False negatives: phishing sites predicted as legitimate ---")
    false_negative_mask = (y_test.to_numpy() == 1) & (y_pred == 0)
    false_negative_positions = [
        position
        for position, is_fn in enumerate(false_negative_mask)
        if is_fn
    ]
    if not false_negative_positions:
        print("None in this test split.")
    else:
        for position in false_negative_positions:
            idx = X_test.index[position]
            print(
                f"{df.loc[idx, 'url']} | "
                f"phishing probability={phishing_prob[position]:.1%}"
            )

    return model, X_test, y_test

def compare_to_rule_based_baseline(df, threshold=2):
    """How well would the original point-based scorer alone have done?"""
    predicted = (df['risk_score'] >= threshold).astype(int)
    actual = df['label']

    print(f"\n--- Rule-based baseline (risk_score >= {threshold} = phishing) ---")
    print(classification_report(actual, predicted, target_names=['legit', 'phishing']))


def sweep_thresholds(df, thresholds=range(0, 12)):
    """
    Tries every threshold instead of guessing one, so you can see the
    actual precision/recall trade-off and pick a defensible cutoff
    instead of an arbitrary number.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    print("\n--- Threshold sweep (risk_score >= T counts as phishing) ---")
    print(f"{'T':>3} {'precision':>10} {'recall':>8} {'f1':>6}")
    best_f1, best_t = -1, None
    for t in thresholds:
        predicted = (df['risk_score'] >= t).astype(int)
        actual = df['label']
        p = precision_score(actual, predicted, zero_division=0)
        r = recall_score(actual, predicted, zero_division=0)
        f1 = f1_score(actual, predicted, zero_division=0)
        marker = ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
            marker = "  <- best F1 so far"
        print(f"{t:>3} {p:>10.2f} {r:>8.2f} {f1:>6.2f}{marker}")
    print(f"\nBest single threshold by F1: {best_t} (F1={best_f1:.2f})")
    return best_t


if __name__ == '__main__':
    df = load_dataset('labeled_dataset.csv')

    print("=" * 70)
    print("RULE-BASED BASELINE")
    print("=" * 70)
    compare_to_rule_based_baseline(df)
    sweep_thresholds(df)

    print("\n" + "=" * 70)
    print("MACHINE LEARNING MODEL")
    print("=" * 70)
    model, X_test, y_test = train_and_evaluate(df)

    joblib.dump(model, 'phishing_model.joblib')
    print("\nModel saved to phishing_model.joblib")
    print("Load it later with: model = joblib.load('phishing_model.joblib')")