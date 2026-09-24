#!/usr/bin/env python3
"""
Step 1: Preprocessing & Feature Extraction
Cleans and normalizes business names and addresses for Entity Resolution.
"""
import re
import unicodedata
import pandas as pd
from typing import Optional


# ── Legal suffix normalization ────────────────────────────────────────────────
LEGAL_SUFFIXES = {
    r"\bpvt\.?\s*ltd\.?\b": "pvt ltd",
    r"\bprivate\s+limited\b": "pvt ltd",
    r"\bltd\.?\b": "ltd",
    r"\blimited\b": "ltd",
    r"\binc\.?\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bcorp\.?\b": "corp",
    r"\bcorporation\b": "corp",
    r"\bllc\.?\b": "llc",
    r"\bllp\.?\b": "llp",
    r"\bco\.?\b": "co",
    r"\bcompany\b": "co",
    r"\benterprises?\b": "ent",
    r"\bservices?\b": "svc",
    r"\bsolutions?\b": "sol",
    r"\bindustries?\b": "ind",
    r"\btrading\b": "trd",
    r"\bgroup\b": "grp",
    r"\bholdings?\b": "hld",
    r"\bassociates?\b": "assoc",
    r"\bconsultants?\b": "cons",
    r"\btechnologies?\b": "tech",
    r"\binternational\b": "intl",
    r"\bnational\b": "natl",
    r"\bglobal\b": "glbl",
}

ADDRESS_ABBREVS = {
    r"\bstreet\b": "st",
    r"\broad\b": "rd",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
    r"\bdrive\b": "dr",
    r"\blane\b": "ln",
    r"\bplace\b": "pl",
    r"\bcourt\b": "ct",
    r"\bcircle\b": "cir",
    r"\bsquare\b": "sq",
    r"\bnorth\b": "n",
    r"\bsouth\b": "s",
    r"\beast\b": "e",
    r"\bwest\b": "w",
    r"\bapartment\b": "apt",
    r"\bsuite\b": "ste",
    r"\bfloor\b": "fl",
    r"\bbuilding\b": "bldg",
    r"\bblock\b": "blk",
    r"\bnear\b": "",
    r"\bopposite\b": "opp",
    r"\bbehind\b": "",
    r"\bnext to\b": "",
}


def normalize_unicode(text: str) -> str:
    """Convert unicode to ASCII-equivalent where possible."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_name(name: Optional[str]) -> str:
    """Normalize business name for matching."""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    text = name.lower().strip()
    text = normalize_unicode(text)
    # Replace & with and
    text = re.sub(r"&", "and", text)
    # Remove punctuation except spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Apply legal suffix normalization
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_address(addr: Optional[str]) -> str:
    """Normalize business address for matching."""
    if pd.isna(addr) or not isinstance(addr, str):
        return ""
    text = addr.lower().strip()
    text = normalize_unicode(text)
    text = re.sub(r"[^\w\s,]", " ", text)
    # Apply address abbreviations
    for pattern, replacement in ADDRESS_ABBREVS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Normalize zip/pin codes (keep digits)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_name_tokens(name: str) -> set:
    """Get meaningful word tokens from a normalized name."""
    stopwords = {"the", "a", "an", "of", "for", "in", "at", "by", "and", "or"}
    tokens = set(name.split()) - stopwords
    # Remove very short tokens (single chars) unless they're meaningful
    return {t for t in tokens if len(t) > 1}


def extract_numeric_tokens(text: str) -> set:
    """Extract all number sequences (useful for address numbers, pin codes)."""
    return set(re.findall(r"\d+", text))


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalization steps to a source dataframe."""
    df = df.copy()
    df["name_clean"] = df["business_name"].apply(clean_name)
    df["addr_clean"] = df["business_address"].apply(clean_address)
    df["country_clean"] = df["country"].str.lower().str.strip().fillna("")
    df["name_tokens"] = df["name_clean"].apply(extract_name_tokens)
    df["addr_tokens"] = df["addr_clean"].apply(extract_numeric_tokens)
    # First token of name (often the most distinctive part)
    df["name_prefix"] = df["name_clean"].apply(lambda x: x.split()[0] if x else "")
    # First digit sequence in address (building/street number)
    df["addr_num"] = df["addr_clean"].apply(
        lambda x: re.search(r"\d+", x).group() if re.search(r"\d+", x) else ""
    )
    return df


if __name__ == "__main__":
    # Quick sanity check
    test_cases = [
        ("McDonald's Corporation", "Mcdonalds corp"),
        ("Pvt. Ltd.", "pvt ltd"),
        ("Tech Solutions International", "tech sol intl"),
    ]
    for inp, _ in test_cases:
        print(f"  '{inp}' -> '{clean_name(inp)}'")
