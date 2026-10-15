#!/usr/bin/env python3
"""Repair the NCBI_lineage of a single samplesheet row.

The lineage is the last column of the samplesheet, so the edit is done on the
raw line and only that field changes - re-serialising the whole file with the
csv module would rewrite every row's quoting and produce a huge diff.

The operation is idempotent and writes a ``.bak-lineagefix`` backup.

Usage
-----
    # the MAG was classified only to genus: extend it to a submittable taxon
    python3 submission/scripts/05_fix_lineage.py samplesheet.csv \
        --id SRR24759607_binette_bin44 --append 589

    # replace the lineage outright
    python3 submission/scripts/05_fix_lineage.py samplesheet.csv \
        --id SRR24759607_binette_bin44 --set "Unclassified Bacteria"

Pick the taxid with the help of ``04_check_ena_taxa.py``: it suggests the
level(s) that sibling MAGs of the same lineage use.  Verify it is a
submittable taxon first, e.g.

    curl -s "https://www.ebi.ac.uk/ena/taxonomy/rest/tax-id/589"
"""

import argparse
import csv
import shutil
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samplesheet")
    ap.add_argument("--id", required=True, help="MAG id (first column) of the row to repair")
    ap.add_argument("--append", help="taxid to append to the existing lineage")
    ap.add_argument("--set", dest="new_lineage", help="replace the lineage with this value")
    args = ap.parse_args()

    if bool(args.append) == bool(args.new_lineage):
        sys.exit("pass exactly one of --append or --set")

    with open(args.samplesheet) as fh:
        lines = fh.read().splitlines()

    hits = [i for i, line in enumerate(lines) if line.startswith(args.id + ",")]
    if len(hits) != 1:
        sys.exit(f"expected exactly 1 row starting with {args.id!r}, found {len(hits)}")
    i = hits[0]

    current = lines[i].rsplit(",", 1)[-1]
    if args.new_lineage:
        wanted = args.new_lineage
        if current == wanted:
            print(f"ALREADY CORRECT: {args.id} -> {wanted}")
            return
    else:
        if current.split(";")[-1] == args.append:
            print(f"ALREADY CORRECT: {args.id} -> {current}")
            return
        wanted = f"{current};{args.append}"
    print(f"{args.id}\n  {current}\n  -> {wanted}")

    shutil.copyfile(args.samplesheet, args.samplesheet + ".bak-lineagefix")
    lines[i] = lines[i][: len(lines[i]) - len(current)] + wanted
    with open(args.samplesheet, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    with open(args.samplesheet, newline="") as fh:
        rows = list(csv.DictReader(fh))
    row = next(r for r in rows if r["id"] == args.id)
    assert row["NCBI_lineage"] == wanted, f"verification failed: {row['NCBI_lineage']}"
    print(f"OK: {len(rows)} rows, backup at {args.samplesheet}.bak-lineagefix")


if __name__ == "__main__":
    main()
