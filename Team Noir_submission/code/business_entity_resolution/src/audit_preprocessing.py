"""
Amazon ML Challenge 2026: Preprocessing Audit & Verification Tool
Compares raw records against preprocessed records and generates a detailed
quantitative breakdown of:
- Total rows altered vs unchanged per column
- Exact manipulation types triggered (legal suffixes, domain stripping, roadway standardizations, postal code extraction, etc.)
- Concrete before-and-after change examples
"""

import sys
import re
import unicodedata
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Force UTF-8 encoding for Windows output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import core preprocessing routines
from preprocess import (
    clean_base_text,
    normalize_business_name,
    normalize_address,
    extract_postal_code,
    normalize_country,
    DOMAIN_RE,
    COMPILED_LEGAL,
    COMPILED_ADDR,
    COMPILED_STATES,
    POSTAL_RE
)


def analyze_name_manipulation(raw: str, clean: str) -> List[str]:
    """Identifies the specific transformations applied to a business name."""
    if not raw or pd.isna(raw):
        return ["Null / Empty"]
    tags = []
    if raw == clean:
        return ["Unchanged"]
    
    # 1. Case change
    if raw.lower() != raw:
        tags.append("Case Lowered")
    # 2. Unicode / Accents
    if unicodedata.normalize('NFKD', raw) != raw or any(ord(c) >= 128 for c in raw):
        tags.append("Unicode / Accent Normalized")
    # 3. Domain stripping
    if DOMAIN_RE.search(raw):
        tags.append("Domain Stripped")
    # 4. Legal suffix mapping
    for pat, _ in COMPILED_LEGAL:
        if pat.search(raw):
            tags.append("Legal Suffix Standardized")
            break
    # 5. Ampersand / Slash
    if '&' in raw:
        tags.append("Ampersand Expanded")
    if '/' in raw or '\\' in raw:
        tags.append("Slash Separated")
    # 6. Punctuation stripped
    if re.search(r'[^\w\s]', raw):
        tags.append("Punctuation Cleaned")
    
    return tags if tags else ["Whitespace Normalized"]


def analyze_address_manipulation(raw: str, clean: str, postal: str) -> List[str]:
    """Identifies the specific transformations applied to an address."""
    if not raw or pd.isna(raw):
        return ["Null / Empty"]
    tags = []
    if raw == clean and not postal:
        return ["Unchanged"]
    
    if raw.lower() != raw:
        tags.append("Case Lowered")
    if unicodedata.normalize('NFKD', raw) != raw or any(ord(c) >= 128 for c in raw):
        tags.append("Unicode / Accent Normalized")
    
    # Roadway abbreviations
    for pat, _ in COMPILED_ADDR[:15]:
        if pat.search(raw):
            tags.append("Roadway Standardized")
            break
    
    # Units / Directions
    for pat, _ in COMPILED_ADDR[15:]:
        if pat.search(raw):
            tags.append("Unit/Direction Standardized")
            break
            
    # State mapping
    for pat, _ in COMPILED_STATES:
        if pat.search(raw):
            tags.append("State Abbreviated")
            break

    if postal:
        tags.append("Postal Code Extracted")

    if '&' in raw:
        tags.append("Ampersand Expanded")
    if '/' in raw or '\\' in raw:
        tags.append("Slash Separated")
    if re.search(r'[^\w\s]', raw):
        tags.append("Punctuation Cleaned")

    return tags if tags else ["Whitespace Normalized"]


def analyze_country_manipulation(raw: str, clean: str) -> List[str]:
    """Identifies transformations applied to country."""
    if not raw or pd.isna(raw):
        return ["Null / Empty"]
    if raw == clean:
        return ["Unchanged"]
    return ["Country Standardized to ISO"]


def run_audit(input_path: Path, sample_size: int = 50000, output_report_path: Optional[Path] = None):
    print(f"Reading dataset: {input_path} (auditing up to {sample_size:,} rows)...")
    df = pd.read_csv(input_path, sep="\t", nrows=sample_size, dtype=str)
    
    total_rows = len(df)
    
    # Track statistics
    stats = {
        "name": {"altered": 0, "tags": {}},
        "address": {"altered": 0, "tags": {}},
        "country": {"altered": 0, "tags": {}},
        "postal_codes_found": 0
    }
    
    sample_changes = []

    print("Analyzing transformations row by row...")
    for idx, row in df.iterrows():
        raw_name = "" if pd.isna(row.get('business_name')) else str(row['business_name'])
        raw_addr = "" if pd.isna(row.get('business_address')) else str(row['business_address'])
        raw_ctry = "" if pd.isna(row.get('country')) else str(row['country'])

        clean_name = normalize_business_name(raw_name)
        clean_addr = normalize_address(raw_addr)
        postal = extract_postal_code(raw_addr)
        clean_ctry = normalize_country(raw_ctry)

        name_tags = analyze_name_manipulation(raw_name, clean_name)
        addr_tags = analyze_address_manipulation(raw_addr, clean_addr, postal)
        ctry_tags = analyze_country_manipulation(raw_ctry, clean_ctry)

        # Update Name stats
        if raw_name != clean_name:
            stats["name"]["altered"] += 1
        for tag in name_tags:
            stats["name"]["tags"][tag] = stats["name"]["tags"].get(tag, 0) + 1

        # Update Address stats
        if raw_addr != clean_addr:
            stats["address"]["altered"] += 1
        for tag in addr_tags:
            stats["address"]["tags"][tag] = stats["address"]["tags"].get(tag, 0) + 1

        if postal:
            stats["postal_codes_found"] += 1

        # Update Country stats
        if raw_ctry != clean_ctry:
            stats["country"]["altered"] += 1
        for tag in ctry_tags:
            stats["country"]["tags"][tag] = stats["country"]["tags"].get(tag, 0) + 1

        # Collect diverse sample rows for the report
        if len(sample_changes) < 15:
            if raw_name != clean_name or raw_addr != clean_addr or raw_ctry != clean_ctry:
                sample_changes.append({
                    "entity_id": row.get('entity_id', f"ROW-{idx}"),
                    "raw_name": raw_name,
                    "clean_name": clean_name,
                    "raw_addr": raw_addr,
                    "clean_addr": clean_addr,
                    "postal": postal,
                    "raw_ctry": raw_ctry,
                    "clean_ctry": clean_ctry,
                    "name_tags": ", ".join(name_tags),
                    "addr_tags": ", ".join(addr_tags)
                })

    # Build Markdown report
    report_lines = []
    report_lines.append("# Preprocessing Audit Report")
    report_lines.append(f"- **Source File**: `{input_path.name}`")
    report_lines.append(f"- **Total Rows Audited**: `{total_rows:,}`\n")

    report_lines.append("## 1. High-Level Alteration Summary")
    report_lines.append("| Field | Total Records | Altered Records | % Altered | Unchanged Records | % Unchanged |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for field, key in [("business_name", "name"), ("business_address", "address"), ("country", "country")]:
        alt = stats[key]["altered"]
        unchanged = total_rows - alt
        alt_pct = (alt / total_rows) * 100 if total_rows > 0 else 0
        unch_pct = 100.0 - alt_pct
        report_lines.append(f"| **{field}** | {total_rows:,} | {alt:,} | {alt_pct:.2f}% | {unchanged:,} | {unch_pct:.2f}% |")

    report_lines.append(f"\n- **Postal / PIN Codes Extracted**: `{stats['postal_codes_found']:,}` ({(stats['postal_codes_found']/total_rows)*100:.2f}% of rows)\n")

    report_lines.append("## 2. Detailed Breakdown of Manipulations Applied")
    report_lines.append("\n### A. `business_name` Transformations")
    report_lines.append("| Manipulation Type | Rows Affected | % of Audited Rows |")
    report_lines.append("| :--- | :--- | :--- |")
    for tag, count in sorted(stats["name"]["tags"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_rows) * 100
        report_lines.append(f"| {tag} | {count:,} | {pct:.2f}% |")

    report_lines.append("\n### B. `business_address` Transformations")
    report_lines.append("| Manipulation Type | Rows Affected | % of Audited Rows |")
    report_lines.append("| :--- | :--- | :--- |")
    for tag, count in sorted(stats["address"]["tags"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_rows) * 100
        report_lines.append(f"| {tag} | {count:,} | {pct:.2f}% |")

    report_lines.append("\n### C. `country` Transformations")
    report_lines.append("| Manipulation Type | Rows Affected | % of Audited Rows |")
    report_lines.append("| :--- | :--- | :--- |")
    for tag, count in sorted(stats["country"]["tags"].items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_rows) * 100
        report_lines.append(f"| {tag} | {count:,} | {pct:.2f}% |")

    report_lines.append("\n## 3. Concrete Before vs After Examples")
    for i, ex in enumerate(sample_changes[:10], 1):
        report_lines.append(f"\n#### Example {i} (`{ex['entity_id']}`)")
        report_lines.append(f"- **Raw Name**: `{ex['raw_name']}`")
        report_lines.append(f"  - **Clean Name**: `{ex['clean_name']}` *(Tags: {ex['name_tags']})*")
        report_lines.append(f"- **Raw Address**: `{ex['raw_addr']}`")
        report_lines.append(f"  - **Clean Address**: `{ex['clean_addr']}` *(Tags: {ex['addr_tags']})*")
        report_lines.append(f"  - **Postal Code**: `{ex['postal'] or '[None]'}`")
        report_lines.append(f"- **Country**: `{ex['raw_ctry']}` -> `{ex['clean_ctry']}`")

    report_text = "\n".join(report_lines)

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nAudit report successfully written to: {output_report_path}")

    # Also output to stdout
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Audit and compare raw vs preprocessed data")
    parser.add_argument("--input", type=str, default="_student_resource/student_resource/dataset/train/train_source1.tsv", help="Path to input TSV file")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of rows to audit (default: 10,000)")
    parser.add_argument("--report", type=str, default="audit_report.md", help="Path to save the Markdown report table")
    args = parser.parse_args()

    run_audit(
        input_path=Path(args.input),
        sample_size=args.sample_size,
        output_report_path=Path(args.report)
    )


if __name__ == "__main__":
    main()
