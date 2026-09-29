"""
Phishing website scraper.

Collects phishing and legitimate URLs, checks them for suspicious
features and saves the results so they can be used to train the model.

Run this before train_model.py.
"""

import requests
import csv
import math
import ipaddress
import threading

from collections import Counter
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from datetime import datetime


# Optional package for WHOIS lookups
try:
    import whois
except ImportError:
    whois = None
    print("python-whois not installed — skipping domain-age checks. (pip install python-whois)")


# Used to properly split domains and suffixes
try:
    import tldextract

    # Keep this offline so it doesn't make another network request
    _tld_extractor = tldextract.TLDExtract(suffix_list_urls=())
except ImportError:
    tldextract = None
    _tld_extractor = None
    print("tldextract not installed — WHOIS lookups will be less reliable on multi-part domains. (pip install tldextract)")


# Only needed when saving an Excel version of the dataset
try:
    import pandas as pd
except ImportError:
    pd = None
    print("pandas not installed — skipping Excel export. (pip install pandas openpyxl)")


# Give each thread its own session to avoid connection problems
_thread_local = threading.local()


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def get_session():
    """Give each thread its own requests session."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = s

    return _thread_local.session


def load_openphish_urls(limit=10, timeout=10):
    """Get some current phishing URLs from OpenPhish."""
    try:
        response = requests.get(
            "https://openphish.com/feed.txt",
            timeout=timeout
        )
        response.raise_for_status()

    except requests.RequestException as e:
        print(f"Could not fetch OpenPhish feed: {e}")
        return []

    urls = [
        line.strip()
        for line in response.text.strip().split('\n')
        if line.strip()
    ]

    return urls[:limit]


# Legitimate sites to compare against the phishing URLs
LEGIT_TEST_URLS = [
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.python.org",
    "https://www.wikipedia.org",
    "https://www.github.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.paypal.com",
    "https://www.netflix.com",
    "https://www.linkedin.com",
    "https://www.reddit.com",
    "https://www.spotify.com",
    "https://www.dropbox.com",
    "https://www.adobe.com",
    "https://www.stackoverflow.com",
    "https://www.nytimes.com",
    "https://www.bbc.com",
    "https://www.cloudflare.com",
    "https://www.mozilla.org",
    "https://www.ibm.com",
    "https://www.oracle.com",
    "https://www.salesforce.com",
    "https://www.zoom.us",
    "https://www.slack.com",
    "https://www.shopify.com",
    "https://www.ebay.com",
    "https://www.twitch.tv",
    "https://www.airbnb.com",
    "https://www.uber.com",
    "https://www.chase.com",
    "https://www.walmart.com",
    "https://www.target.com",
    "https://www.bankofamerica.com",
    "https://www.wellsfargo.com",
    "https://www.instagram.com",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.pinterest.com",
    "https://www.wordpress.com",
    "https://www.yahoo.com",
    "https://www.bing.com",
    "https://www.aws.amazon.com",
    "https://www.digitalocean.com",
    "https://www.atlassian.com",
    "https://www.notion.so",
    "https://www.figma.com",
    "https://www.canva.com",
    "https://www.trello.com",
    "https://www.asana.com",
]


def is_private_or_local(url):
    """Check if a URL points to a private or local address."""
    hostname = urlparse(url).hostname

    if not hostname:
        return True

    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_private or ip.is_loopback or ip.is_link_local

    except ValueError:
        # If this happens it's a normal domain rather than an IP
        return False


def hostname_is_ip(hostname):
    """Check if the hostname is an IP address."""
    if not hostname:
        return False

    try:
        ipaddress.ip_address(hostname)
        return True

    except ValueError:
        return False


def sanitize_for_spreadsheet(value):
    """Stop spreadsheet values being treated as formulas."""
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value

    return value


def sanitize_row(row):
    """Sanitize every value before saving it."""
    return {
        key: sanitize_for_spreadsheet(value)
        for key, value in row.items()
    }


# Free hosting is common with phishing pages because it's quick and easy to set up
FREE_HOSTING_DOMAINS = [
    'weebly.com',
    'pages.dev',
    'gitbook.io',
    'azurefd.net',
    'cloudclusters.net',
    'amplifyapp.com',
    'godaddysites.com',
    'netlify.app',
    'vercel.app',
    'herokuapp.com',
    '000webhostapp.com',
    'github.io',
    'firebaseapp.com',
    'web.app',
    'wixsite.com',
    'blogspot.com',
    'sites.google.com',
    'repl.co',
    'glitch.me',
    'replit.app',
    'edgeone.dev',
    'laravel.cloud',
    'wasmer.app',
    'typedream.app',
    'flutterflow.app',
    'workers.dev',
    'staticdomains.app',
]


# Words that commonly appear in phishing URLs
SUSPICIOUS_KEYWORDS = [
    'login',
    'verify',
    'secure',
    'account',
    'update',
    'confirm',
    'banking',
    'signin',
    'support',
    'wallet',
]


# Brands that phishing sites commonly try to copy
# Store the real suffix as well so fake versions can be spotted
PHISHING_TARGET_BRAND_SUFFIXES = {
    'google': 'com',
    'microsoft': 'com',
    'apple': 'com',
    'amazon': 'com',
    'paypal': 'com',
    'netflix': 'com',
    'facebook': 'com',
    'instagram': 'com',
    'twitter': 'com',
    'linkedin': 'com',
    'roblox': 'com',
    'chase': 'com',
    'wellsfargo': 'com',
    'bankofamerica': 'com',
    'dropbox': 'com',
    'adobe': 'com',
    'ebay': 'com',
    'airbnb': 'com',
    'uber': 'com',
    'spotify': 'com',
    'wordpress': 'com',
    'github': 'com',
    'coinbase': 'com',
    'trustwallet': 'com',
    'dhl': 'com',
    'fedex': 'com',
    'ups': 'com',
    'usps': 'com',
    'meta': 'com',
    'exodus': 'com',
    'phantom': 'app',
    'binance': 'com',
}


def _build_known_brand_suffixes():
    """Build a list of known brands and their real domain suffixes."""
    suffixes = dict(PHISHING_TARGET_BRAND_SUFFIXES)

    # Add the real suffixes from our legitimate test sites
    if _tld_extractor is not None:
        for url in LEGIT_TEST_URLS:
            hostname = urlparse(url).hostname
            ext = _tld_extractor(hostname)

            if ext.domain:
                suffixes[ext.domain] = ext.suffix

    return suffixes


KNOWN_BRAND_SUFFIXES = _build_known_brand_suffixes()
KNOWN_BRANDS = list(KNOWN_BRAND_SUFFIXES.keys())


def levenshtein_distance(a, b):
    """Count how many character changes are needed to turn a into b."""
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous_row = range(len(b) + 1)

    for i, char_a in enumerate(a):
        current_row = [i + 1]

        for j, char_b in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (char_a != char_b)

            current_row.append(
                min(insertions, deletions, substitutions)
            )

        previous_row = current_row

    return previous_row[-1]


def check_brand_impersonation(url):
    """
    Check if the URL looks like it's pretending to be a known brand.

    Free hosting sites are checked using their chosen subdomain.
    Normal sites are checked using their main domain.
    """

    # On free hosting, this is the part of the URL chosen by the page owner
    if is_on_free_hosting(url):
        target = split_subdomain_target(url).lower()

        if not target:
            return None, None, False

        # Catch brand names hidden inside subdomains like "open-instagram"
        for brand in KNOWN_BRANDS:
            if len(brand) >= 4 and brand in target:
                return brand, 0, True

        # Also look for small misspellings of brand names
        best_brand = None
        best_distance = None

        for brand in KNOWN_BRANDS:
            distance = levenshtein_distance(target, brand)

            if best_distance is None or distance < best_distance:
                best_brand = brand
                best_distance = distance

        close_typo = (
            best_distance is not None
            and 0 < best_distance <= 2
        )

        return best_brand, best_distance, close_typo

    if _tld_extractor is None:
        return None, None, False

    # For normal sites, check the actual registered domain
    ext = _tld_extractor(urlparse(url).hostname or '')
    domain = (ext.domain or '').lower()
    suffix = ext.suffix or ''

    if not domain:
        return None, None, False

    # Exact brand name but using the wrong domain ending
    if domain in KNOWN_BRAND_SUFFIXES:
        real_suffix = KNOWN_BRAND_SUFFIXES[domain]
        return domain, 0, (suffix != real_suffix)

    # Look for small typos of known brand names
    best_brand = None
    best_distance = None

    for brand in KNOWN_BRANDS:
        distance = levenshtein_distance(domain, brand)

        if best_distance is None or distance < best_distance:
            best_brand = brand
            best_distance = distance

    if best_distance is None:
        return None, None, False

    close_typo = 0 < best_distance <= 2

    return best_brand, best_distance, close_typo


def is_on_free_hosting(url):
    """Check if the site is using one of the free hosting platforms."""
    hostname = urlparse(url).hostname or ''

    return any(
        hostname == domain or hostname.endswith('.' + domain)
        for domain in FREE_HOSTING_DOMAINS
    )


def split_subdomain_target(url):
    """Get the useful part of the subdomain that we want to analyse."""
    hostname = urlparse(url).hostname or ''

    matched_base = next(
        (
            base
            for base in FREE_HOSTING_DOMAINS
            if hostname == base or hostname.endswith('.' + base)
        ),
        None
    )

    if matched_base:
        target = (
            hostname[:-(len(matched_base) + 1)]
            if hostname != matched_base
            else ''
        )

        if target.startswith('www.'):
            target = target[4:]

    else:
        target = hostname.split('.')[0]

    return target


def subdomain_entropy(url):
    """
    Measure how random the subdomain looks.

    Random-looking names can be another sign of a phishing page.
    """
    target = split_subdomain_target(url)

    if not target:
        return 0.0

    counts = Counter(target)
    length = len(target)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def fetch_page(url, timeout=5):
    """Try to download and parse the page."""

    # Don't allow requests to local or private addresses
    if is_private_or_local(url):
        print(f"Skipping {url} — points to a private/internal address")
        return None

    try:
        response = get_session().get(url, timeout=timeout)
        response.raise_for_status()

    except requests.RequestException as e:
        print(f"Could not fetch {url}: {e}")
        return None

    return BeautifulSoup(response.text, 'html.parser')


def extract_features(url, soup):
    """Pull out the features that will be used by the detector."""
    parsed = urlparse(url)
    features = {"url": url}

    # Basic URL features
    features['fetch_succeeded'] = soup is not None
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_hyphens'] = url.count('-')
    features['has_at_symbol'] = '@' in url
    features['uses_https'] = parsed.scheme == 'https'
    features['has_ip_as_domain'] = hostname_is_ip(parsed.hostname)
    features['on_free_hosting'] = is_on_free_hosting(url)

    # Check for suspicious words in the URL
    features['has_suspicious_keyword'] = any(
        keyword in url.lower()
        for keyword in SUSPICIOUS_KEYWORDS
    )

    # Look at the subdomain
    subdomain_target = split_subdomain_target(url)
    features['subdomain_length'] = len(subdomain_target)
    features['subdomain_entropy'] = round(
        subdomain_entropy(url),
        2
    )

    # Check if the site looks like it's copying a known brand
    brand, brand_distance, impersonating = check_brand_impersonation(url)

    features['matched_brand'] = brand or ''
    features['brand_edit_distance'] = (
        brand_distance if brand_distance is not None else -1
    )
    features['impersonates_brand'] = impersonating

    if soup:
        # Look for forms and password fields on the page
        forms = soup.find_all('form')

        features['num_forms'] = len(forms)
        features['has_password_field'] = bool(
            soup.find('input', {'type': 'password'})
        )

        # A form sending data somewhere else can be suspicious
        page_host = parsed.hostname
        external_form = False

        for form in forms:
            action_host = urlparse(
                form.get('action', '')
            ).hostname

            if action_host and action_host != page_host:
                external_form = True

        features['form_posts_externally'] = external_form

    else:
        features['num_forms'] = 0
        features['has_password_field'] = False
        features['form_posts_externally'] = False

    return features


def get_registrable_domain(hostname):
    """Get the main domain used for WHOIS/RDAP lookups."""
    if _tld_extractor is not None:
        ext = _tld_extractor(hostname)

        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"

    return (
        hostname[4:]
        if hostname.startswith('www.')
        else hostname
    )


def get_domain_age_days(url):
    """Find roughly how old the domain is."""
    if is_private_or_local(url):
        return None

    domain = get_registrable_domain(
        urlparse(url).netloc
    )

    # Try RDAP first
    age = _get_domain_age_rdap(domain)

    if age is not None:
        return age

    # Use WHOIS as a backup
    return _get_domain_age_whois(domain)


def _get_domain_age_rdap(domain, timeout=6):
    """Try to get the domain age using RDAP."""
    try:
        response = requests.get(
            f"https://rdap.org/domain/{domain}",
            timeout=timeout
        )
        response.raise_for_status()

        data = response.json()

        for event in data.get('events', []):
            if event.get('eventAction') == 'registration':
                date_str = event.get('eventDate')

                if date_str:
                    creation_date = datetime.fromisoformat(
                        date_str.replace('Z', '+00:00')
                    )

                    if creation_date.tzinfo is not None:
                        creation_date = creation_date.replace(
                            tzinfo=None
                        )

                    return (
                        datetime.now() - creation_date
                    ).days

    except Exception as e:
        print(f"RDAP lookup failed for {domain}: {e}")

    return None


def _get_domain_age_whois(domain):
    """Try WHOIS if RDAP couldn't find the domain age."""
    if whois is None:
        return None

    try:
        w = whois.whois(domain)
        creation_date = w.creation_date

        if isinstance(creation_date, list):
            creation_date = min(
                date
                for date in creation_date
                if date is not None
            )

        if creation_date:
            if creation_date.tzinfo is not None:
                creation_date = creation_date.replace(
                    tzinfo=None
                )

            return (
                datetime.now() - creation_date
            ).days

    except Exception as e:
        print(f"WHOIS fallback also failed for {domain}: {e}")

    return None


def rule_based_score(features, domain_age_days):
    """Give the URL a simple risk score based on suspicious features."""
    score = 0
    reasons = []

    # Helper for adding points and keeping track of why
    def add(points, why):
        nonlocal score
        score += points

        if points:
            reasons.append(f"+{points} {why}")

    # Basic suspicious URL checks
    if not features['uses_https']:
        add(1, "no HTTPS")

    if features['has_ip_as_domain']:
        add(2, "IP used as domain")

    if features['has_at_symbol']:
        add(2, "'@' in URL")

    # If we couldn't reach the page, we can't properly verify it
    if not features['fetch_succeeded']:
        add(1, "page unreachable (unverifiable)")

    if features['has_suspicious_keyword']:
        add(2, "suspicious keyword in URL")

    # Brand impersonation is a stronger phishing signal
    if features['impersonates_brand']:

        if (
            features['on_free_hosting']
            and features['brand_edit_distance'] == 0
        ):
            add(
                4,
                f"brand name '{features['matched_brand']}' "
                "found in the subdomain (free hosting)"
            )

        elif features['brand_edit_distance'] == 0:
            add(
                4,
                f"exact brand name '{features['matched_brand']}' "
                "on wrong TLD"
            )

        else:
            add(
                3,
                f"looks like a typo of "
                f"'{features['matched_brand']}'"
            )

    # Password details being sent somewhere else is especially suspicious
    if (
        features['form_posts_externally']
        and features['has_password_field']
    ):
        add(
            4,
            "password field + external form "
            "(credential harvesting pattern)"
        )

    elif features['form_posts_externally']:
        add(1, "form posts to another host")

    if features['on_free_hosting']:
        add(2, "free hosting")

    # Only flag random-looking subdomains when they're long enough
    if (
        features['subdomain_entropy'] >= 3.0
        and features['subdomain_length'] >= 10
    ):
        add(2, "gibberish subdomain")

    # Don't use domain age for free hosting because we'd only
    # be measuring the age of Vercel, GitHub, Netlify etc.
    if not features['on_free_hosting']:
        if (
            domain_age_days is not None
            and domain_age_days < 30
        ):
            add(3, "domain under 30 days old")

    return score, reasons


def analyze_url(url):
    """Run all of the checks on one URL."""

    # Download the page and collect its features
    soup = fetch_page(url)
    features = extract_features(url, soup)

    # Add domain age information
    domain_age_days = get_domain_age_days(url)

    features['domain_age_days'] = domain_age_days
    features['domain_age_known'] = int(
        domain_age_days is not None
    )

    # Add the rule-based score and reasons
    score, reasons = rule_based_score(
        features,
        domain_age_days
    )

    features['risk_score'] = score
    features['risk_reasons'] = "; ".join(reasons)

    return features


def analyze_urls(urls, max_workers=5):
    """Analyse several URLs at the same time."""
    results = []

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_to_url = {
            executor.submit(analyze_url, url): url
            for url in urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]

            try:
                results.append(future.result())

            except Exception as e:
                print(f"Failed to analyze {url}: {e}")

    return results


def build_labeled_dataset(
    phishing_urls,
    legit_urls,
    max_workers=5
):
    """Label phishing as 1 and legitimate as 0."""
    phishing_set = set(phishing_urls)

    all_urls = (
        list(phishing_urls)
        + list(legit_urls)
    )

    results = analyze_urls(
        all_urls,
        max_workers=max_workers
    )

    # Add the correct label to each analysed URL
    for result in results:
        result['label'] = (
            1 if result['url'] in phishing_set else 0
        )

    return results


def save_to_csv(
    rows,
    filename='phishing_features.csv',
    fieldnames=None
):
    """Save the finished dataset as a CSV file."""
    if not rows:
        print("No rows to save.")
        return

    # Clean values before putting them into a spreadsheet
    sanitized_rows = [
        sanitize_row(row)
        for row in rows
    ]

    with open(
        filename,
        mode='w',
        newline='',
        encoding='utf-8'
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=(
                fieldnames
                or list(sanitized_rows[0].keys())
            )
        )

        writer.writeheader()
        writer.writerows(sanitized_rows)

    print(f"Saved {len(rows)} rows to {filename}")


def save_to_excel(
    rows,
    filename='phishing_features.xlsx'
):
    """Save another copy of the dataset as an Excel file."""
    if not rows:
        print("No rows to save.")
        return

    if pd is None:
        print(
            "pandas isn't installed, so I can't write Excel files. "
            "(pip install pandas openpyxl)"
        )
        return

    # Clean the values before saving them
    sanitized_rows = [
        sanitize_row(row)
        for row in rows
    ]

    df = pd.DataFrame(sanitized_rows)
    df.to_excel(filename, index=False)

    print(f"Saved {len(rows)} rows to {filename}")


if __name__ == '__main__':

    # Get fresh phishing examples
    phishing_urls = load_openphish_urls(limit=150)

    print(
        f"Loaded {len(phishing_urls)} URLs from OpenPhish"
    )

    # Add the known legitimate examples
    legit_urls = LEGIT_TEST_URLS

    print(
        f"Using {len(legit_urls)} known-legit URLs"
    )

    # Analyse everything and build the dataset
    dataset = build_labeled_dataset(
        phishing_urls,
        legit_urls
    )

    # Print the results so they're easy to check
    print(
        f"\n{'LABEL':10} | {'SCORE':5} | URL"
    )
    print("-" * 80)

    for result in sorted(
        dataset,
        key=lambda x: x['risk_score'],
        reverse=True
    ):
        label = (
            "PHISHING"
            if result['label'] == 1
            else "LEGIT"
        )

        print(
            f"{label:10} | "
            f"{result['risk_score']:5} | "
            f"{result['url']}"
        )

    # Compare the average scores for both groups
    phishing_scores = [
        result['risk_score']
        for result in dataset
        if result['label'] == 1
    ]

    legit_scores = [
        result['risk_score']
        for result in dataset
        if result['label'] == 0
    ]

    if phishing_scores and legit_scores:
        avg_phishing = (
            sum(phishing_scores)
            / len(phishing_scores)
        )

        avg_legit = (
            sum(legit_scores)
            / len(legit_scores)
        )

        print(
            f"\nAverage risk score — "
            f"phishing: {avg_phishing:.2f} | "
            f"legit: {avg_legit:.2f}"
        )

    # Show why each URL got its score so it's easier to tune the rules
    print("\n--- Score breakdown ---")

    for result in sorted(
        dataset,
        key=lambda x: x['risk_score']
    ):
        if (
            result['risk_score'] == 0
            or result['risk_reasons']
        ):
            label = (
                "PHISHING"
                if result['label'] == 1
                else "LEGIT"
            )

            why = (
                result['risk_reasons']
                or "no rules fired"
            )

            print(
                f"[{label:8}] "
                f"{result['risk_score']:3}  "
                f"{result['url']}\n"
                f"           -> {why}"
            )

    # Save the finished dataset
    save_to_csv(
        dataset,
        "labeled_dataset.csv"
    )

    save_to_excel(
        dataset,
        "labeled_dataset.xlsx"
    )
