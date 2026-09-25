#!/usr/bin/env python3
"""
Step 1: Preprocessing & Feature Extraction  (vectorized, scale-aware)
Cleans and normalizes business names and addresses using fully vectorized
pandas string operations — avoids slow row-by-row .apply() at 10M+ scale.

Performance target: preprocess 10M rows in < 60 seconds.
"""
import re
import unicodedata

import pandas as pd
import numpy as np


# ── Legal suffix normalization table ──────────────────────────────────────────
# Applied as a single compiled regex alternation for speed
_LEGAL_PATTERNS = [
    (r"private\s+limited",   "pvt ltd"),
    (r"pvt\.?\s*ltd\.?",     "pvt ltd"),
    (r"limited",             "ltd"),
    (r"ltd\.?",              "ltd"),
    (r"incorporated",        "inc"),
    (r"inc\.?",              "inc"),
    (r"corporation",         "corp"),
    (r"corp\.?",             "corp"),
    (r"llc\.?",              "llc"),
    (r"llp\.?",              "llp"),
    (r"company",             "co"),
    (r"co\.?",               "co"),
    (r"enterprises?",        "ent"),
    (r"services?",           "svc"),
    (r"solutions?",          "sol"),
    (r"industries?",         "ind"),
    (r"technologies?",       "tech"),
    (r"international",       "intl"),
    (r"national",            "natl"),
    (r"global",              "glbl"),
    (r"associates?",         "assoc"),
    (r"consultants?",        "cons"),
    (r"trading",             "trd"),
    (r"holdings?",           "hld"),
    (r"group",               "grp"),
]

_ADDR_PATTERNS = [
    (r"\bstreet\b",    "st"),
    (r"\broad\b",      "rd"),
    (r"\bavenue\b",    "ave"),
    (r"\bboulevard\b", "blvd"),
    (r"\bdrive\b",     "dr"),
    (r"\blane\b",      "ln"),
    (r"\bplace\b",     "pl"),
    (r"\bcourt\b",     "ct"),
    (r"\bcircle\b",    "cir"),
    (r"\bsquare\b",    "sq"),
    (r"\bnorth\b",     "n"),
    (r"\bsouth\b",     "s"),
    (r"\beast\b",      "e"),
    (r"\bwest\b",      "w"),
    (r"\bapartment\b", "apt"),
    (r"\bsuite\b",     "ste"),
    (r"\bfloor\b",     "fl"),
    (r"\bbuilding\b",  "bldg"),
    (r"\bblock\b",     "blk"),
    (r"\bnear\b",      ""),
    (r"\bopposite\b",  "opp"),
    (r"\bbehind\b",    ""),
    (r"\bnext\s+to\b", ""),
]

# Pre-compile all patterns with word-boundary flags
_LEGAL_RE  = [(re.compile(p, re.IGNORECASE), r) for p, r in _LEGAL_PATTERNS]
_ADDR_RE   = [(re.compile(p, re.IGNORECASE), r) for p, r in _ADDR_PATTERNS]
_PUNCT_RE  = re.compile(r"[^\w\s]")
_WS_RE     = re.compile(r"\s+")
_DIGIT_RE  = re.compile(r"\d+")


# ── Vectorized Unicode normalization ──────────────────────────────────────────

def _make_ascii_table():
    """Build a 256-wide translation table for stripping accents after NFKD."""
    # We'll use str.encode('ascii', 'ignore') approach via pandas
    pass


def _normalize_series(s: pd.Series) -> pd.Series:
    """
    Vectorized unicode normalization: NFKD → encode ASCII ignore → decode.
    Much faster than unicodedata.normalize row-by-row at 10M scale.
    """
    return (
        s.fillna("")
         .str.normalize("NFKD")
         .str.encode("ascii", errors="ignore")
         .str.decode("ascii")
    )


# ── Vectorized name cleaning ──────────────────────────────────────────────────

def _clean_name_series(s: pd.Series) -> pd.Series:
    """Fully vectorized business name normalization."""
    s = _normalize_series(s).str.lower().str.strip()
    # & → and
    s = s.str.replace("&", " and ", regex=False)
    # Remove punctuation
    s = s.str.replace(_PUNCT_RE, " ", regex=True)
    # Apply legal suffix replacements using numpy object array (avoids Unicode OOM)
    arr = s.to_numpy(dtype=object)  # object dtype = Python str refs, no fixed width
    for pat, repl in _LEGAL_RE:
        arr = np.array([pat.sub(repl, x) for x in arr], dtype=object)
    s = pd.Series(arr, index=s.index, dtype=str)
    # Collapse whitespace
    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


def _clean_addr_series(s: pd.Series) -> pd.Series:
    """Fully vectorized address normalization."""
    s = _normalize_series(s).str.lower().str.strip()
    # Use character-class replace via numpy to avoid pandas 80MB mask on 10M rows
    arr = s.to_numpy(dtype=object)
    _non_word = re.compile(r"[^\w\s,]")
    arr = np.array([_non_word.sub(" ", x) for x in arr], dtype=object)
    for pat, repl in _ADDR_RE:
        arr = np.array([pat.sub(repl, x) for x in arr], dtype=object)
    s = pd.Series(arr, index=s.index, dtype=str)
    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


# ── Token extraction (vectorized where possible) ──────────────────────────────

def _first_token(s: pd.Series) -> pd.Series:
    """First whitespace-separated token of each string."""
    return s.str.split(n=1).str[0].fillna("")


def _first_digit_seq(s: pd.Series) -> pd.Series:
    """First sequence of digits in each string (for address blocking key)."""
    return s.str.extract(r"(\d+)", expand=False).fillna("")


def _consonant_key(s: pd.Series) -> pd.Series:
    """Vowel-stripped prefix key for transliteration tolerance."""
    return (
        s.str[:6]
         .str.replace(r"[aeiou\s]", "", regex=True)
         .str[:4]
         .fillna("")
    )


# ── Main public API ───────────────────────────────────────────────────────────

def preprocess_df(df: pd.DataFrame, chunk_size: int = 500_000) -> pd.DataFrame:
    """
    Apply all normalization steps to a source dataframe.
    Processes in chunks to avoid pandas memory spikes on 10M+ row DataFrames.
    chunk_size=500K keeps peak memory under ~2GB per chunk.
    """
    if len(df) <= chunk_size:
        return _preprocess_chunk(df)

    chunks = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start: start + chunk_size]
        chunks.append(_preprocess_chunk(chunk))
    return pd.concat(chunks, ignore_index=True)


def _preprocess_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization to a single chunk."""
    df = df.copy()
    df["name_clean"]    = _clean_name_series(df["business_name"])
    df["addr_clean"]    = _clean_addr_series(df["business_address"])
    df["country_clean"] = df["country"].str.lower().str.strip().fillna("")
    df["name_prefix"]   = _first_token(df["name_clean"])
    df["addr_num"]      = _first_digit_seq(df["addr_clean"])
    df["consonant_key"] = _consonant_key(df["name_clean"])
    return df


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    test_data = {
        "entity_id":        ["S1-001", "S1-002", "S1-003", "S1-004"],
        "business_name":    [
            "McDonald's Corporation",
            "Reliance Pvt. Ltd.",
            "Tech Solutions International",
            "R&D Enterprises Incorporated",
        ],
        "business_address": [
            "123 Main Street, Suite 4, Phoenix AZ",
            "Near SBI ATM, MG Road, Kolkata 700001",
            "Block 4, Sector 12, Noida",
            "42 Boulevard, North Side, Paris",
        ],
        "country": ["US", "India", "India", "France"],
    }

    df = pd.DataFrame(test_data)
    t0 = time.time()
    result = preprocess_df(df)
    print(f"Processed {len(df)} rows in {time.time()-t0:.3f}s")
    print(result[["entity_id", "name_clean", "addr_clean", "name_prefix", "addr_num", "consonant_key"]].to_string())

    # Benchmark on synthetic large dataset
    print("\nBenchmark: 100,000 rows...")
    big_df = pd.concat([df] * 25000, ignore_index=True)
    big_df["entity_id"] = [f"S1-{i:07d}" for i in range(len(big_df))]
    t0 = time.time()
    _ = preprocess_df(big_df)
    print(f"  100K rows preprocessed in {time.time()-t0:.2f}s")
