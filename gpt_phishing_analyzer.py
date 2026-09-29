"""
GPT phishing analyzer.

Uses the features from phishing_scraper.py and sends them to OpenAI
for a second phishing assessment.

You can use it with one URL, a dataset, or a text file of URLs.

Examples:

    python gpt_phishing_analyzer.py https://example.com

    python gpt_phishing_analyzer.py --dataset labeled_dataset.csv

    python gpt_phishing_analyzer.py --urls urls.txt

You can also change the model with:

    OPENAI_MODEL=gpt-5.6-luna

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pathlib import Path
from typing import Any

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI


# Reuse the feature extraction from the main scraper
try:
    from phishing_scraper import analyze_url

except ImportError as exc:
    print(
        "ERROR: Could not import phishing_scraper.py.\n"
        "Make sure gpt_phishing_analyzer.py is in the same directory as "
        "phishing_scraper.py."
    )

    raise SystemExit(1) from exc


DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_OUTPUT = "gpt_phishing_results.csv"


# These are the same main features used when training the ML model
FEATURE_COLUMNS = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "has_at_symbol",
    "uses_https",
    "has_ip_as_domain",
    "on_free_hosting",
    "has_suspicious_keyword",
    "subdomain_length",
    "subdomain_entropy",
    "num_forms",
    "has_password_field",
    "form_posts_externally",
    "fetch_succeeded",
    "domain_age_known",
    "brand_edit_distance",
    "impersonates_brand",
]


# Tell GPT exactly how to judge the features and return the result
SYSTEM_PROMPT = """
You are a defensive cybersecurity classifier. Your task is to assess whether
a website is likely legitimate or phishing using structured evidence produced
by a URL/page scraper.

This is a classification aid, not proof of maliciousness. Do not claim that
a feature alone proves phishing. HTTPS, passwords, forms, free hosting, and
keywords can all occur on legitimate sites. Consider the combination of
signals and the limitations of the evidence.

Return ONLY valid JSON matching the requested schema.

Required fields:
- classification: exactly "phishing", "legitimate", or "uncertain"
- confidence: integer from 0 to 100
- risk_level: exactly "low", "medium", "high", or "critical"
- reasons: array of 1 to 5 concise strings
- recommendation: exactly "allow", "review", or "block"
- summary: one concise sentence

Use "uncertain" when the supplied evidence is insufficient for a confident
decision. Do not invent WHOIS, DNS, page content, reputation, registration,
or visual information that is not supplied.
"""


# Makes sure GPT always gives us the same response structure
JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,

    "properties": {
        "classification": {
            "type": "string",
            "enum": [
                "phishing",
                "legitimate",
                "uncertain",
            ],
        },

        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },

        "risk_level": {
            "type": "string",
            "enum": [
                "low",
                "medium",
                "high",
                "critical",
            ],
        },

        "reasons": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 1,
            "maxItems": 5,
        },

        "recommendation": {
            "type": "string",
            "enum": [
                "allow",
                "review",
                "block",
            ],
        },

        "summary": {
            "type": "string"
        },
    },

    "required": [
        "classification",
        "confidence",
        "risk_level",
        "reasons",
        "recommendation",
        "summary",
    ],
}


def make_client() -> OpenAI:
    """Create the OpenAI client using the API key."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n\n"
            "Set it as an environment variable or put it in a .env file:\n"
            "OPENAI_API_KEY=your_api_key_here"
        )

    return OpenAI(
        api_key=api_key
    )


def get_model() -> str:
    """Get the model from .env or use the default one."""
    load_dotenv()

    return os.getenv(
        "OPENAI_MODEL",
        DEFAULT_MODEL
    )


def clean_value(value: Any) -> Any:
    """Turn pandas values into something JSON can handle."""

    # Missing pandas values become normal Python None
    if pd.isna(value):
        return None

    # Convert numpy values into normal Python values
    if hasattr(value, "item"):
        try:
            return value.item()

        except (ValueError, TypeError):
            pass

    return value


def build_evidence(
    features: dict[str, Any]
) -> dict[str, Any]:
    """Build the information that will be sent to GPT."""

    # Start with the useful rule-based information
    evidence: dict[str, Any] = {
        "url": features.get(
            "url",
            ""
        ),

        "risk_score": clean_value(
            features.get("risk_score")
        ),

        "risk_reasons": features.get(
            "risk_reasons",
            ""
        ),

        "matched_brand": features.get(
            "matched_brand",
            ""
        ),
    }

    # Only send the features we actually care about
    evidence["features"] = {
        column: clean_value(
            features.get(column)
        )

        for column in FEATURE_COLUMNS

        if column in features
    }

    # Domain age is useful extra information but isn't part of the ML features
    evidence["domain_age_days"] = clean_value(
        features.get("domain_age_days")
    )

    return evidence


def classify_with_gpt(
    client: OpenAI,
    features: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    """Send the website features to GPT for classification."""

    evidence = build_evidence(
        features
    )

    model = model or get_model()

    # Give GPT only the evidence collected by the scraper
    user_prompt = (
        "Assess this website using ONLY the supplied evidence.\n\n"
        "Website evidence:\n"
        + json.dumps(
            evidence,
            indent=2,
            ensure_ascii=False
        )
    )

    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,

        text={
            "format": {
                "type": "json_schema",
                "name": "phishing_assessment",
                "strict": True,
                "schema": JSON_SCHEMA,
            }
        },
    )

    raw = response.output_text.strip()

    # Turn GPT's JSON response back into a Python dictionary
    try:
        result = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OpenAI returned invalid JSON:\n{raw}"
        ) from exc

    # Double-check that nothing important is missing
    required = {
        "classification",
        "confidence",
        "risk_level",
        "reasons",
        "recommendation",
        "summary",
    }

    missing = required - set(result)

    if missing:
        raise RuntimeError(
            f"OpenAI response is missing fields: "
            f"{sorted(missing)}"
        )

    return result


def analyze_one_url(
    client: OpenAI,
    url: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the scraper first, then send its results to GPT."""

    print(
        f"\nAnalyzing URL: {url}"
    )

    print(
        "Running existing scraper/feature extraction..."
    )

    # Get all the features using the main scraper
    features = analyze_url(
        url
    )

    print(
        "Sending structured evidence to OpenAI..."
    )

    # Ask GPT to make its own assessment
    gpt_result = classify_with_gpt(
        client,
        features,
        model=model
    )

    # Keep the original scraper results and add GPT's result
    result = dict(features)

    result["gpt_classification"] = (
        gpt_result["classification"]
    )

    result["gpt_confidence"] = (
        gpt_result["confidence"]
    )

    result["gpt_risk_level"] = (
        gpt_result["risk_level"]
    )

    result["gpt_recommendation"] = (
        gpt_result["recommendation"]
    )

    result["gpt_reasons"] = " | ".join(
        gpt_result["reasons"]
    )

    result["gpt_summary"] = (
        gpt_result["summary"]
    )

    return result


def print_result(
    result: dict[str, Any]
) -> None:
    """Print the result in an easier-to-read format."""

    print(
        "\n" + "=" * 70
    )

    print(
        "GPT PHISHING ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"URL:            "
        f"{result.get('url', '')}"
    )

    print(
        f"Rule score:     "
        f"{result.get('risk_score', 'N/A')}"
    )

    print(
        f"Rule reasons:   "
        f"{result.get('risk_reasons', '') or 'none'}"
    )

    print()

    print(
        f"Classification: "
        f"{str(result.get('gpt_classification', '')).upper()}"
    )

    print(
        f"Confidence:     "
        f"{result.get('gpt_confidence', '')}%"
    )

    print(
        f"Risk level:     "
        f"{str(result.get('gpt_risk_level', '')).upper()}"
    )

    print(
        f"Recommendation: "
        f"{str(result.get('gpt_recommendation', '')).upper()}"
    )

    print()

    print(
        "GPT reasons:"
    )

    for reason in str(
        result.get(
            "gpt_reasons",
            ""
        )
    ).split(" | "):

        if reason:
            print(
                f"  - {reason}"
            )

    print()

    print(
        f"Summary: "
        f"{result.get('gpt_summary', '')}"
    )

    print(
        "=" * 70
    )


def analyze_dataset(
    client: OpenAI,
    filename: str,
    output: str,
    model: str | None = None,
    delay: float = 0.0,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Run GPT on a dataset that has already been made by the scraper.

    This uses the saved features instead of downloading every site again.
    """

    df = pd.read_csv(
        filename
    )

    # We need the URL column to know which website each row belongs to
    if "url" not in df.columns:
        raise ValueError(
            f"{filename} does not contain a 'url' column."
        )

    # Useful when I only want to test a small part of the dataset
    if limit is not None:
        df = df.head(
            limit
        ).copy()

    rows: list[dict[str, Any]] = []

    print(
        f"Loaded {len(df)} rows from {filename}"
    )

    print(
        f"OpenAI model: "
        f"{model or get_model()}"
    )

    # Go through each website one at a time
    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        print(
            f"\n[{position}/{len(df)}] "
            f"{row['url']}"
        )

        # Clean the CSV values before sending them to GPT
        features = {
            key: clean_value(value)

            for key, value
            in row.to_dict().items()
        }

        try:
            gpt_result = classify_with_gpt(
                client,
                features,
                model=model,
            )

            # Keep the old row and add GPT's results
            output_row = features.copy()

            output_row["gpt_classification"] = (
                gpt_result["classification"]
            )

            output_row["gpt_confidence"] = (
                gpt_result["confidence"]
            )

            output_row["gpt_risk_level"] = (
                gpt_result["risk_level"]
            )

            output_row["gpt_recommendation"] = (
                gpt_result["recommendation"]
            )

            output_row["gpt_reasons"] = " | ".join(
                gpt_result["reasons"]
            )

            output_row["gpt_summary"] = (
                gpt_result["summary"]
            )

            output_row["gpt_error"] = ""

            print(
                f"    GPT: "
                f"{gpt_result['classification']} "
                f"({gpt_result['confidence']}%)"
            )

        except Exception as exc:
            print(
                f"    ERROR: {exc}"
            )

            # Keep the row even if the API request fails
            output_row = features.copy()

            output_row["gpt_classification"] = ""
            output_row["gpt_confidence"] = ""
            output_row["gpt_risk_level"] = ""
            output_row["gpt_recommendation"] = ""
            output_row["gpt_reasons"] = ""
            output_row["gpt_summary"] = ""
            output_row["gpt_error"] = str(exc)

        rows.append(
            output_row
        )

        # Optional pause between API requests
        if delay > 0 and position < len(df):
            time.sleep(
                delay
            )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved GPT results to: {output}"
    )

    # Compare GPT against the existing labels if they're available
    if "label" in result_df.columns:
        print_dataset_comparison(
            result_df
        )

    return result_df


def print_dataset_comparison(
    df: pd.DataFrame
) -> None:
    """Compare GPT's answers against the labels in the dataset."""

    # Only compare rows where GPT made a clear decision
    valid = df[
        df["gpt_classification"].isin(
            [
                "phishing",
                "legitimate"
            ]
        )
        & df["label"].isin(
            [0, 1]
        )
    ].copy()

    if valid.empty:
        print(
            "\nNo comparable labelled GPT results available."
        )

        return

    # Turn GPT's text answer into the same 0/1 format as the dataset
    predicted = (
        valid["gpt_classification"]
        == "phishing"
    ).astype(int)

    actual = valid[
        "label"
    ].astype(int)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "GPT VS DATASET LABELS"
    )

    print(
        "=" * 70
    )

    print(
        f"Comparable rows: {len(valid)}"
    )

    print(
        f"Accuracy:        "
        f"{accuracy_score(actual, predicted):.3f}"
    )

    print(
        f"Precision:       "
        f"{precision_score(actual, predicted, zero_division=0):.3f}"
    )

    print(
        f"Recall:          "
        f"{recall_score(actual, predicted, zero_division=0):.3f}"
    )

    print(
        f"F1:              "
        f"{f1_score(actual, predicted, zero_division=0):.3f}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            actual,
            predicted,
            target_names=[
                "legit",
                "phishing"
            ],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix "
        "(rows = actual, cols = GPT prediction):"
    )

    print(
        pd.DataFrame(
            confusion_matrix(
                actual,
                predicted
            ),
            index=[
                "actual: legit",
                "actual: phishing"
            ],
            columns=[
                "pred: legit",
                "pred: phishing"
            ],
        )
    )


def load_urls_file(
    filename: str
) -> list[str]:
    """Load URLs from a text file."""

    urls = []

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            url = line.strip()

            # Ignore empty lines and commented-out URLs
            if not url or url.startswith("#"):
                continue

            urls.append(
                url
            )

    return urls


def analyze_url_file(
    client: OpenAI,
    filename: str,
    output: str,
    model: str | None = None,
    delay: float = 0.0,
    limit: int | None = None,
) -> pd.DataFrame:
    """Analyse all the URLs stored in a text file."""

    urls = load_urls_file(
        filename
    )

    # Handy for testing without running the whole file
    if limit is not None:
        urls = urls[:limit]

    if not urls:
        raise ValueError(
            f"No URLs found in {filename}"
        )

    rows = []

    print(
        f"Loaded {len(urls)} URLs from {filename}"
    )

    # Run the normal analysis on each URL
    for position, url in enumerate(
        urls,
        start=1
    ):

        print(
            f"\n[{position}/{len(urls)}]"
        )

        try:
            result = analyze_one_url(
                client,
                url,
                model=model
            )

            rows.append(
                result
            )

            print_result(
                result
            )

        except Exception as exc:
            print(
                f"ERROR analyzing {url}: {exc}"
            )

        # Optional pause so API requests aren't sent back-to-back
        if delay > 0 and position < len(urls):
            time.sleep(
                delay
            )

    result_df = pd.DataFrame(
        rows
    )

    # Only create the file if we actually got some results
    if not result_df.empty:
        result_df.to_csv(
            output,
            index=False
        )

        print(
            f"\nSaved results to: {output}"
        )

    return result_df


def parse_args() -> argparse.Namespace:
    """Read the options passed in from the terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a separate OpenAI phishing assessment using the "
            "features produced by phishing_scraper.py."
        )
    )

    # The user should choose one input method
    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "url",
        nargs="?",
        help="Analyze a single URL.",
    )

    source.add_argument(
        "--dataset",
        help=(
            "Analyze an existing labeled_dataset.csv "
            "without refetching URLs."
        ),
    )

    source.add_argument(
        "--urls",
        help=(
            "Analyze a text file containing one URL per line."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"OpenAI model. Defaults to OPENAI_MODEL "
            f"or {DEFAULT_MODEL}."
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            f"CSV output filename "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows/URLs.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between API requests.",
    )

    return parser.parse_args()


def main() -> int:
    """Run whichever analysis mode was chosen."""

    args = parse_args()

    try:
        # Set up the API connection
        client = make_client()

        model = (
            args.model
            or get_model()
        )

        print(
            f"Using OpenAI model: {model}"
        )

        # Analyse one URL
        if args.url:
            result = analyze_one_url(
                client,
                args.url,
                model=model,
            )

            print_result(
                result
            )

            # Save a copy so the result can be checked later
            pd.DataFrame(
                [result]
            ).to_csv(
                args.output,
                index=False
            )

            print(
                f"\nSaved result to: {args.output}"
            )

        # Analyse an existing dataset
        elif args.dataset:
            analyze_dataset(
                client,
                filename=args.dataset,
                output=args.output,
                model=model,
                delay=args.delay,
                limit=args.limit,
            )

        # Analyse URLs from a text file
        elif args.urls:
            analyze_url_file(
                client,
                filename=args.urls,
                output=args.output,
                model=model,
                delay=args.delay,
                limit=args.limit,
            )

        return 0

    # Allow Ctrl+C to stop the program cleanly
    except KeyboardInterrupt:
        print(
            "\nStopped by user."
        )

        return 130

    # Catch anything else and show a readable error
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )