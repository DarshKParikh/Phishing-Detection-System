# Architecture

## Current architecture

The existing implementation intentionally keeps the scraper and feature
extraction pipeline together. This is being preserved during the first cleanup
pass to avoid breaking imports.

```text
                    +----------------+
                    |      URL       |
                    +-------+--------+
                            |
                            v
                 +---------------------+
                 | phishing_scraper.py |
                 +----------+----------+
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
     URL features      HTML features     Domain/brand
          |                 |             information
          +-----------------+-----------------+
                            |
                            v
                  Rule-based risk score
                            |
                            v
                  Random Forest model
                            |
                 +----------+----------+
                 |                     |
                 v                     v
             Prediction            Evidence
                 |                     |
                 +----------+----------+
                            |
                            v
                    Optional GPT
                    explanation
```

## Planned architecture

Once tests cover the current behaviour, the large scraper can safely be split
into:

```text
src/phishing_detector/
├── scraper.py
├── features.py
├── domain.py
├── brand_detection.py
├── rules.py
├── model.py
├── predictions.py
└── utils.py
```

The important rule is that refactoring should not change feature definitions
or model input order unless the model is retrained and evaluated again.
