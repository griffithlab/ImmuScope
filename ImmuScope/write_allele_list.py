#!/usr/bin/env python3

INPUT_FILE = "pseudosequence.pvac.dat"
OUTPUT_FILE = "Immuscope.txt"


def main():
    with open(INPUT_FILE) as infile, open(OUTPUT_FILE, "w") as outfile:
        for line in infile:
            line = line.strip()

            if not line:
                continue

            allele = line.split()[0]
            outfile.write(f"{allele}\n")


if __name__ == "__main__":
    main()