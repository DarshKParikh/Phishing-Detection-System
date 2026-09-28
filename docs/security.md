# Scraper Security Considerations

The application fetches URLs supplied by users or collected from external feeds.
That creates security considerations beyond ordinary web scraping.

## Recommended controls

- Validate URL schemes.
- Reject localhost and private IP addresses.
- Re-check destinations after redirects.
- Limit redirect count.
- Enforce connection and read timeouts.
- Limit response size.
- Validate content types where practical.
- Rate-limit requests.
- Avoid executing untrusted JavaScript in the scraper.
- Keep API keys out of source control.
- Run automated analysis in an isolated environment when possible.

## SSRF consideration

Checking only the original hostname is not enough if redirects are followed.
A public URL may redirect to an internal address.

The destination after each redirect should therefore be validated before the
request is allowed to continue.
