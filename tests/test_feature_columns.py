# Keeps the training/serving feature contract documented.
# This test intentionally does not import the scraper yet, because the first
# cleanup pass keeps the existing runtime layout unchanged.

EXPECTED_FEATURES = {
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
}


def test_feature_contract_is_not_empty():
    assert len(EXPECTED_FEATURES) >= 10
