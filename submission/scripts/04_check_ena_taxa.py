#!/usr/bin/env python3
"""Pre-flight check that every NCBI_lineage in a samplesheet ends at a taxon ENA accepts.

ENA rejects a sample whose taxon is not "submittable" (in practice: below the
rank of genus).  A lineage that stops early therefore fails the whole batch,
because the pipeline sends all genomes of a run in a single XML:

    <ERROR>In sample, alias: "SRR24759607_binette_bin44".
           Taxon Bacteria id 2 is not submittable.</ERROR>

Nothing gets registered in that case, so a failed attempt is cheap - but the
diagnosis is not obvious.  This script makes it explicit.  It checks that

1. all numeric taxid lineages have the same number of levels - a shorter chain
   almost always means the classification stopped at genus level, and
2. the last taxid of every lineage resolves to ``submittable: true`` on the
   ENA taxonomy service (it is a *string*, not a boolean).

Rows that fail are reported together with a repair suggestion: for a chain
that ends too early, other rows sharing the same prefix usually continue one
level further, and that taxid is a valid species/placeholder taxon.

Exit code is 1 when a problem is found, so the script can gate a submission.

Usage
-----
    python3 submission/scripts/04_check_ena_taxa.py samplesheet.csv
    python3 submission/scripts/04_check_ena_taxa.py samplesheet.csv --offline
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

NUMERIC = re.compile(r"^[0-9]+(;[0-9]+)*$")
ENA_TAXONOMY = "https://www.ebi.ac.uk/ena/taxonomy/rest/tax-id/{}"

# taxa that genome_uploader falls back to for a not-classified MAG
FALLBACK_TAXA = {"77133": "uncultured bacterium", "115547": "uncultured archaeon"}


def fetch_taxon(taxid: str) -> dict:
    with urlopen(ENA_TAXONOMY.format(taxid), timeout=30) as response:
        import json

        data = json.loads(response.read().decode())
    if isinstance(data, list):
        data = data[0] if data else {}
    return data


def is_submittable(taxon: dict) -> bool:
    # the ENA API returns the flag as the string "true"/"false"
    return str(taxon.get("submittable", "")).strip().lower() == "true"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samplesheet")
    ap.add_argument("--offline", action="store_true", help="skip the ENA taxonomy lookups")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.samplesheet, newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"{args.samplesheet}: {len(rows)} rows")

    numeric, placeholders = [], []
    for row in rows:
        lineage = (row.get("NCBI_lineage") or "").strip()
        if NUMERIC.match(lineage):
            numeric.append((row["id"], lineage))
        else:
            placeholders.append((row["id"], lineage))

    print(f"  numeric taxid lineages : {len(numeric)}")
    print(f"  placeholder lineages   : {len(placeholders)}")
    for name, count in Counter(l for _, l in placeholders).most_common():
        print(f"      {name!r}: {count}")

    problems = []
    taxa = {}

    # 1. chain length consistency ------------------------------------------
    lengths = Counter(len(l.split(";")) for _, l in numeric)
    print(f"  lineage level counts   : {dict(sorted(lengths.items()))}")
    if len(lengths) > 1:
        expected = lengths.most_common(1)[0][0]
        for mid, lineage in numeric:
            if len(lineage.split(";")) != expected:
                problems.append((mid, lineage, f"has {len(lineage.split(';'))} levels, expected {expected}"))

    # 2. last taxon must be submittable ------------------------------------
    if not args.offline:
        last_by_lineage = {lineage: lineage.split(";")[-1] for _, lineage in numeric}
        taxids = sorted(set(last_by_lineage.values()))
        print(f"  distinct final taxids  : {len(taxids)} (querying ENA taxonomy ...)")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            taxa = dict(zip(taxids, pool.map(fetch_taxon, taxids)))

        bad_taxa = {t: d for t, d in taxa.items() if not is_submittable(d)}
        for taxid, taxon in sorted(bad_taxa.items()):
            print(f"      NOT submittable: taxid {taxid} ({taxon.get('scientificName')}, rank {taxon.get('rank')})")

        for mid, lineage, reason in list(problems):
            problems.remove((mid, lineage, reason))
        for mid, lineage in numeric:
            taxon = taxa.get(last_by_lineage[lineage], {})
            if not is_submittable(taxon):
                problems.append(
                    (mid, lineage, f"final taxon {lineage.split(';')[-1]} "
                                   f"({taxon.get('scientificName')}, rank {taxon.get('rank')}) is not submittable")
                )

        # sanity check the not-classified fallbacks
        prefixes = {l for _, l in placeholders}
        for taxid, name in FALLBACK_TAXA.items():
            if not prefixes:
                break
            taxon = fetch_taxon(taxid)
            if not is_submittable(taxon):
                problems.append((f"<{name}>", "-", f"ENA fallback taxon {taxid} is not submittable"))

    if not problems:
        print("\nOK: every lineage ends at a submittable taxon")
        return

    # 3. suggest a repair ---------------------------------------------------
    by_prefix = defaultdict(set)
    for _, lineage in numeric:
        parts = lineage.split(";")
        for i in range(1, len(parts)):
            by_prefix[";".join(parts[:i])].add(parts[i])

    print(f"\n{len(problems)} row(s) need attention:")
    for mid, lineage, reason in problems:
        print(f"  {mid}\n      lineage: {lineage}\n      problem: {reason}")
        deeper = sorted(by_prefix.get(lineage, set()) - {lineage.split(";")[-1]}, key=int)
        if not deeper:
            continue

        named = []
        for taxid in deeper[:8]:
            if args.offline:
                named.append((taxid, "?", "?"))
                continue
            taxon = taxa.get(taxid) or fetch_taxon(taxid)
            named.append((taxid, taxon.get("scientificName", "?"), str(taxon.get("submittable", "")).lower()))

        print("      candidate continuations (used by sibling MAGs of this lineage):")
        for taxid, name, sub in named:
            print(f"          taxid {taxid:<10} {name:<40} submittable={sub}")
        # prefer the generic "<Genus> sp." placeholder when a sibling used one
        pick = next((t for t, name, sub in named if name.endswith(" sp.") and sub == "true"), named[0][0])
        print(f"      fix: submission/scripts/05_fix_lineage.py {args.samplesheet} "
              f"--id {mid} --append {pick}")
    sys.exit(1)


if __name__ == "__main__":
    main()
