#!/usr/bin/env python3

import re

INPUT_FILE = "pseudosequence.2023.dat"
OUTPUT_FILE = "pseudosequence.pvac.dat"


def split_four_digits(digits):
    """Convert 0101 -> 01:01"""
    return f"{digits[:2]}:{digits[2:]}"


def convert_allele(name):
    """
    Convert pseudosequence allele names to pVACtools format.
    Returns None if the allele should be excluded.
    """

    name = name.strip()

    # Remove trailing allele suffixes like N/L/Q
    name = re.sub(r'[NLQ]$', '', name)

    # Exclude mouse alleles
    if name.startswith("H-2-"):
        return None

    # -----------------------------
    # DRB-style (DRB1_0101)
    # -----------------------------
    m = re.match(r"(DRB\d)_(\d{4})$", name)
    if m:
        gene, digits = m.groups()
        return f"{gene}*{split_four_digits(digits)}"

    # -----------------------------
    # DRB-style with 5 digits (BoLA)
    # -----------------------------
    m = re.match(r"BoLA-(DRB\d)_(\d{5})$", name)
    if m:
        gene, digits = m.groups()
        return f"{gene}*{digits[:2]}:{digits[2:4]}"

    # -----------------------------
    # HLA paired chains (DQA/DQB)
    # -----------------------------
    m = re.match(r"HLA-(DQA\d)(\d{4})-(DQB\d)(\d{4})$", name)
    if m:
        g1, a1, g2, a2 = m.groups()
        return f"{g1}*{split_four_digits(a1)}-{g2}*{split_four_digits(a2)}"

    # -----------------------------
    # HLA paired chains (DPA/DPB)
    # -----------------------------
    m = re.match(r"HLA-(DPA\d)(\d{4})-(DPB\d)(\d{4,5})$", name)
    if m:
        g1, a1, g2, a2 = m.groups()
        return f"{g1}*{split_four_digits(a1)}-{g2}*{a2[:2]}:{a2[2:4]}"

    # If already formatted or unknown, pass through
    return name


def main():
    with open(INPUT_FILE) as infile, open(OUTPUT_FILE, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 2:
                continue

            allele = parts[0]
            sequence = parts[1]

            new_allele = convert_allele(allele)

            if new_allele is None:
                continue

            outfile.write(f"{new_allele}\t{sequence}\n")


if __name__ == "__main__":
    main()