# Repository cleanup migration

This package is a safe first-pass cleanup for the existing repository.

## Before running

Make a backup or commit your current working tree.

## What the migration does

- creates `data/raw`
- creates `data/processed`
- creates `models`
- creates `scripts`
- creates `tests`
- creates `docs`
- moves files from the old `csv files of data/` directory into `data/raw/`
- moves `phishing_model.joblib` into `models/`
- removes development artefacts such as `__pycache__`, `.DS_Store` and `DEBUG files`
- writes `.gitignore`, requirements files and documentation

## What it deliberately does NOT do

It does not split or rewrite the existing Python modules.

This is intentional because `app.py`, `ui.py`, `gpt_phishing_analyzer.py` and
`train_model.py` currently import `phishing_scraper.py` directly and expect the
model to be in the repository root. Those paths should be changed together after
tests are added.

## After migration

The next refactor should:

1. move feature extraction into `src/phishing_detector/features.py`
2. move domain/RDAP/WHOIS logic into `domain.py`
3. move brand checks into `brand_detection.py`
4. move rule scoring into `rules.py`
5. create one shared `FEATURE_COLUMNS` definition
6. update training and serving code to import that shared definition
7. move the model to `models/`
8. update the Flask/Gradio paths
9. add unit tests for every feature group
10. only then remove the compatibility layout
