# Feature Dictionary

The current model uses these features:

| Feature | Type | Meaning |
|---|---|---|
| url_length | numeric | Total URL length |
| num_dots | numeric | Number of `.` characters |
| num_hyphens | numeric | Number of `-` characters |
| has_at_symbol | boolean | Whether `@` appears in the URL |
| uses_https | boolean | Whether the URL uses HTTPS |
| has_ip_as_domain | boolean | Whether the hostname is a raw IP |
| on_free_hosting | boolean | Whether the hostname matches a configured free-hosting domain |
| has_suspicious_keyword | boolean | Whether configured suspicious keywords occur |
| subdomain_length | numeric | Length of the selected hostname target |
| subdomain_entropy | numeric | Character entropy of the selected hostname target |
| num_forms | numeric | Number of HTML forms found |
| has_password_field | boolean | Whether a password input was found |
| form_posts_externally | boolean | Whether a form posts to another host |
| fetch_succeeded | boolean | Whether the page could be fetched |
| domain_age_known | boolean | Whether a domain-age result was obtained |
| brand_edit_distance | numeric | Levenshtein distance to a known brand |
| impersonates_brand | boolean | Whether brand impersonation heuristics fired |

## Important interpretation notes

These are signals, not proof.

For example:

- HTTPS is common on legitimate and phishing sites.
- Login/password forms are common on legitimate websites.
- Third-party forms can be legitimate.
- Free hosting can be legitimate.
- Domain-age lookup can fail for legitimate technical reasons.
- An unreachable page is unverifiable, not automatically malicious.

The UI and documentation should preserve these distinctions.
