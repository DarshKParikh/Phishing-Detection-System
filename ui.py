"""
PhishScan user interface.

This is the project's single user-facing interface. It loads the trained
phishing_model.joblib model, extracts the same features used during training
through phishing_scraper.analyze_url(), and uses the trained classifier to
predict whether a submitted website is phishing or legitimate.

Run:
    python3 ui.py
"""

import os
import joblib
import pandas as pd
import gradio as gr

# ── Import existing pipeline ──────────────────────────────────────────
try:
    from phishing_scraper import analyze_url
except ImportError:
    raise SystemExit(
        "ERROR: phishing_scraper.py not found.\n"
        "Make sure ui.py is in the same directory as phishing_scraper.py."
    )

# ── Load trained model ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "phishing_model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise SystemExit(
        "ERROR: phishing_model.joblib not found.\n"
        "Run train_model.py first, then start the UI."
    )

# ── Feature columns — must exactly match train_model.py ──────────────
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


def features_to_df(features: dict) -> pd.DataFrame:
    """Builds the exact DataFrame shape the model was trained on."""
    row = {col: (features.get(col) if features.get(col) is not None else 0)
           for col in FEATURE_COLUMNS}
    return pd.DataFrame([row])


# ── Core prediction function ──────────────────────────────────────────

def scan_url(url: str):
    """
    Called by Gradio on every button click.

    Runs the full pipeline from phishing_scraper.py, then feeds the
    resulting features into the trained model. Returns four values
    that map to the four Gradio output components below.
    """
    url = url.strip()
    if not url:
        return "⚠ Please enter a URL.", {}, "", ""

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        # ── Step 1: full feature extraction (same as training) ──
        features = analyze_url(url)

        # ── Step 2: ML prediction ──
        X = features_to_df(features)
        prediction   = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        phishing_prob = round(float(probabilities[1]) * 100, 1)
        legit_prob    = round(float(probabilities[0]) * 100, 1)

        # ── Step 3: format verdict ──
        # Avoid turning a borderline model probability into an absolute verdict.
        # The trained model remains the source of the probability; these bands
        # only make the UI more cautious about uncertain predictions.
        if phishing_prob >= 75:
            verdict = f"🚨  LIKELY PHISHING  ({phishing_prob}% phishing probability)"
        elif phishing_prob >= 40:
            verdict = (
                f"⚠️  NEEDS REVIEW  "
                f"({phishing_prob}% phishing / {legit_prob}% legitimate)"
            )
        else:
            verdict = f"✅  LIKELY LEGITIMATE  ({legit_prob}% legitimate probability)"

        # ── Step 4: probability breakdown for the label component ──
        label_data = {
            "Phishing":    phishing_prob / 100,
            "Legitimate":  legit_prob    / 100,
        }

        # ── Step 5: rule-based reasons (from phishing_scraper.py) ──
        reasons_raw = features.get("risk_reasons", "")
        if reasons_raw:
            reasons_text = "\n".join(
                f"  {r.strip()}" for r in reasons_raw.split(";") if r.strip()
            )
        else:
            reasons_text = "  No rules fired — URL passed all heuristic checks."

        # ── Step 6: feature detail table ──
        detail_rows = [
            ("HTTPS",              "✓ Yes" if features.get("uses_https") else "✗ No"),
            ("Free hosting",       "⚠ Yes" if features.get("on_free_hosting") else "✓ No"),
            ("Brand impersonation","⚠ Yes" if features.get("impersonates_brand") else "✓ No"),
            ("Matched brand",      features.get("matched_brand") or "—"),
            ("Suspicious keyword", "⚠ Yes" if features.get("has_suspicious_keyword") else "✓ No"),
            ("Page fetched",       "✓ Yes" if features.get("fetch_succeeded") else "✗ No"),
            ("Domain age",         f"{features['domain_age_days']} days"
                                   if features.get("domain_age_days") is not None
                                   else "Unknown"),
            ("Subdomain entropy",  str(features.get("subdomain_entropy", "—"))),
            ("Rule-based score",   str(features.get("risk_score", "—"))),
            ("Forms on page",      str(features.get("num_forms", "—"))),
            ("Password field",     "Yes" if features.get("has_password_field") else "No"),
            ("External form",      "Yes" if features.get("form_posts_externally") else "No"),
        ]

        table = "\n".join(f"  {k:<22} {v}" for k, v in detail_rows)

        return verdict, label_data, reasons_text, table

    except Exception as exc:
        return f"❌  Error: {exc}", {}, "", ""


# ── Gradio interface ──────────────────────────────────────────────────

with gr.Blocks(
    title="PhishScan — URL Threat Analyzer",
    theme=gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:

    gr.Markdown("""
# 🔍 PhishScan — URL Threat Analyzer
Paste any URL and your trained **RandomForestClassifier** will assess it
using the same feature extraction pipeline from `phishing_scraper.py`.
No new logic — the same model, the same signals.
    """)

    with gr.Row():
        url_input = gr.Textbox(
            placeholder="https://example.com",
            label="URL to analyze",
            scale=4,
        )
        scan_button = gr.Button("Analyze", variant="primary", scale=1)

    verdict_output = gr.Textbox(
        label="Model assessment",
        interactive=False,
        lines=1,
    )

    confidence_output = gr.Label(
        label="Model confidence",
        num_top_classes=2,
    )

    with gr.Row():
        with gr.Column():
            reasons_output = gr.Textbox(
                label="Rules fired (rule-based scorer)",
                interactive=False,
                lines=6,
            )
        with gr.Column():
            detail_output = gr.Textbox(
                label="Signal breakdown",
                interactive=False,
                lines=12,
            )

    gr.Markdown("""
---
**How this works:**
The URL is passed to `analyze_url()` from `phishing_scraper.py` — which fetches the page,
runs WHOIS/RDAP, and extracts all features. Those features are then fed directly into
`phishing_model.joblib` (your trained RandomForest). Nothing here is reimplemented;
this UI is just a window into the existing pipeline.
    """)

    # Wire up inputs → function → outputs
    outputs = [verdict_output, confidence_output, reasons_output, detail_output]

    scan_button.click(fn=scan_url, inputs=url_input, outputs=outputs)
    url_input.submit(fn=scan_url, inputs=url_input, outputs=outputs)

    # Example URLs to try straight from the interface
    gr.Examples(
        examples=[
            ["https://www.google.com"],
            ["https://open-instagram.vercel.app/"],
            ["https://www.roblox.com.ml/users/123/profile"],
            ["http://securebankofamerica.vercel.app/"],
        ],
        inputs=url_input,
        label="Example URLs to try",
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )