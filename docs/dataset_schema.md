# Dataset Schema

## Recommended columns

```text
url
label
label_name
label_source
collected_at
first_seen
last_seen
verification_status
```

followed by the feature columns listed in `features.md`.

## Labels

Use:

```text
0 = legitimate
1 = phishing
```

for the current binary model.

For analysis and auditing, also retain:

```text
label_name
```

with values such as:

```text
legitimate
phishing
```

The source should be recorded separately:

```text
openphish
phishtank
tranco
curated
manual
```

Do not treat the source name as the label.

## Why provenance matters

A model trained on one phishing feed and a small hand-curated legitimate list can
learn properties of those sources rather than general phishing characteristics.

Record collection time and source so later evaluations can be reproduced.

## Recommended future dataset split

Prefer a time-aware or source-aware evaluation in addition to a random split.

For example:

```text
training data
      |
      v
older collection period

test data
      |
      v
newer collection period
```

This gives a more realistic indication of performance against changing phishing
campaigns.
