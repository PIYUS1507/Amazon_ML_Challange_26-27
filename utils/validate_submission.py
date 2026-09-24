#!/usr/bin/env python3
"""
Submission validator for Business Entity Resolution Challenge.
Checks both matching_results.tsv and candidate_pairs.tsv for format correctness.
Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""
import argparse
import sys
import os


def load_tsv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append(parts)
    return header, rows


def load_entity_ids(test_dir):
    """Load all valid S2/S3 entity IDs from test source files."""
    valid_s2, valid_s3, valid_s1 = set(), set(), set()
    for fname, target in [
        ("test_source1.tsv", valid_s1),
        ("test_source2.tsv", valid_s2),
        ("test_source3.tsv", valid_s3),
    ]:
        path = os.path.join(test_dir, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping ID validation for {fname}")
            continue
        with open(path, encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                eid = line.strip().split("\t")[0]
                target.add(eid)
    return valid_s1, valid_s2, valid_s3


def validate_file(path, col_id, col_matches, valid_s1, valid_s23, label):
    issues = []
    if not os.path.exists(path):
        return [f"{label}: file not found at {path}"]

    header, rows = load_tsv(path)
    if header[0] != col_id or (len(header) < 2 or header[1] != col_matches):
        issues.append(f"{label}: unexpected header {header}, expected [{col_id}, {col_matches}]")

    seen_s1 = set()
    for i, row in enumerate(rows, start=2):
        if len(row) < 2:
            row = row + [""]  # treat missing column as empty

        s1_id = row[0].strip()
        matched_str = row[1].strip() if len(row) > 1 else ""

        # Check S1 ID validity
        if valid_s1 and s1_id not in valid_s1:
            issues.append(f"{label} row {i}: S1 ID '{s1_id}' not in test source1")

        # Duplicate S1 row
        if s1_id in seen_s1:
            issues.append(f"{label} row {i}: duplicate source1_entity_id '{s1_id}'")
        seen_s1.add(s1_id)

        if not matched_str:
            continue

        ids = [x.strip() for x in matched_str.split(",")]
        seen_ids = set()
        for eid in ids:
            if not (eid.startswith("S2-") or eid.startswith("S3-")):
                issues.append(f"{label} row {i}: ID '{eid}' is not S2-/S3-")
            if valid_s23 and eid not in valid_s23:
                issues.append(f"{label} row {i}: ID '{eid}' not found in test S2/S3 files")
            if eid in seen_ids:
                issues.append(f"{label} row {i}: duplicate ID '{eid}' in list")
            seen_ids.add(eid)

    # Check all S1 entities are present
    if valid_s1:
        missing = valid_s1 - seen_s1
        if missing:
            issues.append(f"{label}: {len(missing)} Source 1 entities missing from submission: {list(missing)[:5]}...")

    return issues


def check_subset(matching_path, candidate_path):
    """Check that all matched IDs appear in candidate_pairs."""
    issues = []
    try:
        _, m_rows = load_tsv(matching_path)
        _, c_rows = load_tsv(candidate_path)
    except Exception:
        return issues

    candidate_map = {}
    for row in c_rows:
        s1 = row[0].strip()
        ids = set()
        if len(row) > 1 and row[1].strip():
            ids = set(x.strip() for x in row[1].split(","))
        candidate_map[s1] = ids

    for row in m_rows:
        s1 = row[0].strip()
        if len(row) < 2 or not row[1].strip():
            continue
        matched = set(x.strip() for x in row[1].split(","))
        candidates = candidate_map.get(s1, set())
        not_in_candidates = matched - candidates
        if not_in_candidates:
            issues.append(f"Matched IDs not in candidate set for {s1}: {not_in_candidates}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()

    valid_s1, valid_s2, valid_s3 = load_entity_ids(args.test_dir)
    valid_s23 = valid_s2 | valid_s3

    all_issues = []

    all_issues += validate_file(
        args.matching, "source1_entity_id", "matched_entity_ids",
        valid_s1, valid_s23, "matching_results.tsv"
    )
    all_issues += validate_file(
        args.candidate, "source1_entity_id", "candidate_entity_ids",
        valid_s1, valid_s23, "candidate_pairs.tsv"
    )
    all_issues += check_subset(args.matching, args.candidate)

    if all_issues:
        print(f"FAIL — {len(all_issues)} issue(s) found:\n")
        for idx, issue in enumerate(all_issues, 1):
            print(f"  {idx}. {issue}")
        sys.exit(1)
    else:
        print("PASS — submission files are valid and ready to upload.")
        sys.exit(0)


if __name__ == "__main__":
    main()
