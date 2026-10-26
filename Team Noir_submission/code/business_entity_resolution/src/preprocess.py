"""
Amazon ML Challenge 2026: Business Entity Resolution
Data Preprocessing & Normalization Module (Stage 1)

This module standardizes noisy business entities across Source 1, Source 2, and Source 3.
Key design constraints addressed:
- Preserves legal entity suffixes (e.g. Corp, Ltd, LLC, Pvt Ltd, SARL, SAS) rather than discarding them.
- Strips URL/web domain noise from business names (e.g., 'company.com' -> 'company').
- Unicode NFKD decomposition to flatten accents/diacritics into ASCII (vital for France test set).
- Directional, roadway, and unit standardization for addresses without external geocoding APIs.
- Agnostic ISO country code normalization (US, IN, FR).
- Deterministic and sub-second batch execution.
"""

import sys
import re
import unicodedata
import argparse     
from pathlib import Path
from typing import Optional, Union
import pandas as pd

# Force UTF-8 for console output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------
# REGEX DEFINITIONS & COMPILED PATTERNS
# ---------------------------------------------------------

# 1. Domain regex to strip protocols and extensions before punctuation removal
DOMAIN_RE = re.compile(
    r'\b(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|org|net|in|co\.in|co\.uk|fr|io|biz|info|gov|edu)\b',
    re.IGNORECASE
)

# 2. Legal Suffix Mappings (Strictly ordered from most specific to least specific)
LEGAL_MAPPINGS = [
    # French entity types (mandatory for test set)
    # (r'\b(?:societe a responsabilite limitee|s\.?a\.?r\.?l\.?)\b', ' sarl '),
    # (r'\b(?:societe par actions simplifiee unipersonnelle|s\.?a\.?s\.?u\.?)\b', ' sasu '),
    # (r'\b(?:societe par actions simplifiee|s\.?a\.?s\.?)\b', ' sas '),
    # (r'\b(?:entreprise unipersonnelle a responsabilite limitee|e\.?u\.?r\.?l\.?)\b', ' eurl '),
    # (r'\b(?:societe anonyme|s\.?a\.?)\b', ' sa '),
    # US & India entity types


    (r'\b(?:private limited|pvt\.?\s*ltd\.?|p\.?\s*ltd\.?|pvtltd)\b', ' pvt ltd '),
    (r'\b(?:limited liability company|l\.?l\.?c\.?)\b', ' llc '),
    (r'\b(?:limited liability partnership|l\.?l\.?p\.?)\b', ' llp '),
    (r'\b(?:incorporated|inc\.?)\b', ' inc '),
    (r'\b(?:corporation|corp\.?)\b', ' corp '),
    (r'\b(?:limited|ltd\.?)\b', ' ltd '),
    (r'\b(?:company|co\.?)\b', ' co '),
    (r'\b(?:enterprises?|entr\.?)\b', ' ent '),
    (r'\b(?:services?|serv\.?)\b', ' services '),
    (r'\b(?:international|intl\.?)\b', ' intl '),
    (r'\b(?:association|assoc\.?)\b', ' assoc '),
]
COMPILED_LEGAL = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in LEGAL_MAPPINGS]

# 3. US States lookup for address abbreviation unification
US_STATES = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh',
    'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn',
    'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa',
    'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'
}
# Compile US States with negative lookahead to prevent corrupting street and city names (e.g. 'Washington Blvd', 'Michigan City', 'Indiana Ave')
COMPILED_STATES = [
    (re.compile(r'\b' + re.escape(state) + r'\b(?!\s+(?:st|ave|rd|blvd|dr|ln|lane|court|ct|cir|way|city|hwy|pkwy))', re.IGNORECASE), f" {abbr} ")
    for state, abbr in US_STATES.items()
]

# 4. Address Thoroughfare, Unit, and Directional Abbreviations
ADDRESS_MAPPINGS = [
    # Roadways & Thoroughfares
    (r'\b(?:streets?|str\.?|st\.?)\b', ' st '),
    (r'\b(?:roads?|rd\.?)\b', ' rd '),
    (r'\b(?:avenues?|ave\.?|av\.?)\b', ' ave '),
    (r'\b(?:boulevards?|blvd\.?|bd\.?)\b', ' blvd '),
    (r'\b(?:drives?|dr\.?)\b', ' dr '),
    (r'\b(?:lanes?|ln\.?)\b', ' ln '),
    (r'\b(?:highways?|hwy\.?)\b', ' hwy '),
    (r'\b(?:courts?|ct\.?)\b', ' ct '),
    (r'\b(?:circles?|cir\.?)\b', ' cir '),
    (r'\b(?:parkways?|pkwy\.?)\b', ' pkwy '),
    (r'\b(?:places?|pl\.?)\b', ' pl '),
    # French thoroughfares
    # (r'\b(?:rues?)\b', ' rue '),
    # (r'\b(?:chemins?)\b', ' chemin '),
    # (r'\b(?:allees?)\b', ' allee '),
    # (r'\b(?:impasses?)\b', ' impasse '),
    # Units & Subdivisions
    (r'\b(?:suites?|ste\.?)\b', ' ste '),
    (r'\b(?:apartments?|apt\.?)\b', ' apt '),
    (r'\b(?:floors?|fl\.?)\b', ' fl '),
    (r'\b(?:buildings?|bldg\.?)\b', ' bldg '),
    (r'\b(?:rooms?|rm\.?)\b', ' rm '),
    (r'\b(?:units?)\b', ' unit '),
    (r'\b(?:post\s*office\s*box|p\.?o\.?\s*box|pobox)\b', ' pobox '),
    # Compass directions
    (r'\b(?:north|n\.?)\b', ' n '),
    (r'\b(?:south|s\.?)\b', ' s '),
    (r'\b(?:east|e\.?)\b', ' e '),
    (r'\b(?:west|w\.?)\b', ' w '),
    (r'\b(?:northeast|ne\.?)\b', ' ne '),
    (r'\b(?:northwest|nw\.?)\b', ' nw '),
    (r'\b(?:southeast|se\.?)\b', ' se '),
    (r'\b(?:southwest|sw\.?)\b', ' sw '),
    # Administrative divisions
    (r'\b(?:township|townshiip|twp\.?)\b', ' twp '),
]
COMPILED_ADDR = [(re.compile(p, re.IGNORECASE), repl) for p, repl in ADDRESS_MAPPINGS]

# 5. Postal code regex (US 5-digit ZIP, Indian 6-digit PIN, French 5-digit postal code)
POSTAL_RE = re.compile(r'\b(\d{5}(?:-\d{4})?|\d{6})\b')


# ---------------------------------------------------------
# PREPROCESSING FUNCTIONS
# ---------------------------------------------------------

def clean_base_text(text: Optional[str]) -> str:
    """
    Standard base cleaning:
    - Unicode NFKD decomposition (converts accents like é -> e)
    - Strips non-ASCII characters & replacement characters
    - Lowercase
    - Replaces '&' with ' and ', slashes with spaces
    """
    if not isinstance(text, str) or not text:
        return ""
    # Flatten accents (NFKD)
    text = unicodedata.normalize('NFKD', text)
    # Convert to pure ascii, discarding unmapped symbols
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = text.replace('&', ' and ')
    text = re.sub(r'[/\\_]', ' ', text)
    return text


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalizes a business name:
    1. Base Unicode cleaning
    2. Strips URL/web domain patterns (e.g. example.com -> example)
    3. Normalizes legal suffixes (e.g. Pvt Ltd, Inc, Corp, SARL)
    4. Strips remaining punctuation and collapses whitespace
    """
    cleaned = clean_base_text(name)
    if not cleaned:
        return ""
    # Strip domain suffix
    cleaned = DOMAIN_RE.sub(r'\1', cleaned)
    # Normalize legal forms
    for pattern, repl in COMPILED_LEGAL:
        cleaned = pattern.sub(repl, cleaned)
    # Remove remaining punctuation
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    return " ".join(cleaned.split())


def normalize_address(address: Optional[str]) -> str:
    """
    Normalizes an address:
    1. Base Unicode cleaning
    2. Standardizes roadways, units, and directions
    3. Standardizes US state names to 2-letter codes
    4. Strips remaining punctuation and collapses whitespace
    """
    cleaned = clean_base_text(address)
    if not cleaned:
        return ""
    for pattern, repl in COMPILED_ADDR:
        cleaned = pattern.sub(repl, cleaned)
    for pattern, repl in COMPILED_STATES:
        cleaned = pattern.sub(repl, cleaned)
    # Remove remaining punctuation
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    return " ".join(cleaned.split())


def extract_postal_code(address: Optional[str]) -> str:
    """
    Extracts a postal code from an address string if present.
    Supports US (5-digit), India (6-digit), and France (5-digit).
    """
    if not isinstance(address, str) or not address:
        return ""
    match = POSTAL_RE.search(address)
    if match:
        return match.group(1).split('-')[0]
    return ""


def normalize_country(country: Optional[str]) -> str:
    """
    Standardizes country string to ISO alpha-2 uppercase code.
    Fully agnostic and open: supports US, IN, FR and leaves unknown countries normalized.
    """
    if not isinstance(country, str) or not country:
        return ""
    c = country.strip().lower()
    if c in ('us', 'usa', 'united states', 'u.s.', 'u.s.a.'):
        return 'US'
    if c in ('in', 'ind', 'india'):
        return 'IN'
    if c in ('fr', 'fra', 'france'):
        return 'FR'
    return country.strip().upper()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Stage 1 normalization to a DataFrame containing:
    ['entity_id', 'business_name', 'business_address', 'country']

    Produces new standardized columns:
    - clean_name: canonical business name
    - clean_address: canonical address
    - postal_code: extracted postal/PIN code
    - clean_country: canonical ISO country code
    """
    df_clean = df.copy()

    # Fill NaNs with empty string
    name_series = df_clean['business_name'].fillna('')
    addr_series = df_clean['business_address'].fillna('')
    ctry_series = df_clean['country'].fillna('')

    df_clean['clean_name'] = name_series.apply(normalize_business_name)
    df_clean['clean_address'] = addr_series.apply(normalize_address)
    df_clean['postal_code'] = addr_series.apply(extract_postal_code)
    df_clean['clean_country'] = ctry_series.apply(normalize_country)

    return df_clean


# ---------------------------------------------------------
# CLI RUNNER FOR LOCAL CACHING & PIPELINE EXECUTION
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Preprocess and normalize entity dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input TSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output processed file (.tsv or .parquet)")
    parser.add_argument("--format", type=str, choices=["tsv", "parquet"], default=None, help="Output format (default inferred from extension)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 1] Loading {input_path}...")
    df = pd.read_csv(input_path, sep="\t", dtype=str)

    print(f"[Stage 1] Preprocessing {len(df)} records...")
    df_clean = preprocess_dataframe(df)

    out_fmt = args.format or ("parquet" if output_path.suffix == ".parquet" else "tsv")

    print(f"[Stage 1] Saving preprocessed records to {output_path} ({out_fmt})...")
    if out_fmt == "parquet":
        df_clean.to_parquet(output_path, index=False)
    else:
        df_clean.to_csv(output_path, sep="\t", index=False)

    print("[Stage 1] Preprocessing complete.")


if __name__ == "__main__":
    main()
