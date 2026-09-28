# Phishing Website Scraper

A Python phishing-analysis project that extracts website signals, builds a labelled dataset, trains a Random Forest classifier, and provides a single Gradio interface for testing URLs with the trained model.

## What I built

The project combines rule-based phishing signals with a machine-learning classifier.

`phishing_scraper.py` extracts signals such as:

- URL length, dots, hyphens, `@` symbols, and HTTPS usage
- raw IP addresses used as domains
- suspicious keywords
- free-hosting detection
- subdomain length and entropy
- HTML forms and password fields
- forms that submit externally
- domain-age information
- brand matching and possible brand impersonation
- a rule-based risk score and readable risk reasons

It can build `labeled_dataset.csv`, which is then used to train the machine-learning model.

## Training the phishing model

`train_model.py` trains a `RandomForestClassifier` using the feature columns produced by `phishing_scraper.py`.

The training script performs a train/test split, cross-validation, classification reporting, a confusion matrix, feature-importance reporting, and comparison with the rule-based baseline.

After training, it saves the trained classifier as:

```text
phishing_model.joblib
```

To retrain the model:

```bash
python3 phishing_scraper.py
python3 train_model.py
```

## User interface

The project now has **one user interface: `ui.py`**.

The previous Flask `app.py` interface has been removed so there is only one way to run the interactive application.

`ui.py`:

1. loads `phishing_model.joblib`;
2. calls `analyze_url()` from `phishing_scraper.py`;
3. converts the extracted signals into the same feature columns used during training;
4. calls the trained model's `predict()` and `predict_proba()` methods;
5. displays the phishing/legitimate prediction and confidence.

This means the UI prediction comes from the trained Random Forest model rather than creating a separate detection implementation.

Run the interface with:

```bash
python3 ui.py
```

The Gradio application starts on port `7860`.

## Model flow

```text
phishing_scraper.py
        |
        v
labeled_dataset.csv
        |
        v
train_model.py
        |
        v
phishing_model.joblib
        |
        v
ui.py
        |
        v
Phishing / Legitimate prediction
```

## Optional GPT analysis

`gpt_phishing_analyzer.py` remains a separate optional analysis tool. It reuses the structured evidence produced by `phishing_scraper.py`; it is not the main UI and does not replace the trained Random Forest used by `ui.py`.

## Installation

```bash
pip install -r requirements.txt
```

Development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Main files

| File | Purpose |
| --- | --- |
| `phishing_scraper.py` | Extracts phishing signals and builds labelled data |
| `train_model.py` | Trains and evaluates the Random Forest classifier |
| `phishing_model.joblib` | Saved trained phishing-detection model |
| `ui.py` | The single Gradio user interface; uses the trained model for predictions |
| `gpt_phishing_analyzer.py` | Optional OpenAI-based analysis |
| `labeled_dataset.csv` | Labelled feature dataset used for model training |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Development dependencies |

## Notes

The UI and training code must use the same feature columns and ordering. The current `ui.py` keeps the same feature list as `train_model.py` and builds the model input in that order.

The scraper performs network requests to URLs being analysed.
