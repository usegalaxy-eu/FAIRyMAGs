#!/usr/bin/env python3
"""Build an nf-core/seqsubmit samplesheet for a FAIRyMAGs use case.

Every column comes from data already versioned in this repository
(``data/use-cases/<use-case>/``) plus the MAG FASTA files fetched with
``01_fetch_mags_from_galaxy.py``:

======================  ======================================================
column                  source
======================  ======================================================
id                      MAG name = <run accession>_<binner>_bin<N>
fasta                   --fasta-dir/<id>.fasta.gz
accession               run accession (the MAG id prefix), from metadata.tsv
assembly_software       --assembly-software
binning_software        --binning-software
binning_parameters      --binning-parameters
stats_generation_...    --stats-generation-software
completeness            checkm2.tsv   (Completeness)
contamination           checkm2.tsv   (Contamination)
genome_coverage         coverm.tsv    (column named after the run accession)
metagenome              --metagenome
co-assembly             false (single-sample assemblies)
broad_environment       --broad-environment
local_environment       --local-environment
environmental_medium    --environmental-medium
RNA_presence            empty -> the pipeline runs tRNA/rRNA detection
NCBI_lineage            kmetashot.tsv rank columns (NCBI taxids) joined with
                        ';', e.g. 2;1224;1236;91347;1903414;586;587
                        fallback: gtdb.tsv classification -> Unclassified X
======================  ======================================================

The taxid lineage deserves a warning: ENA only accepts a taxon that is
"submittable", which in practice means species rank or below.  The last taxid
of every lineage must therefore be a species/strain taxon.  Run
``04_check_ena_taxa.py`` on the result before submitting.

Usage
-----
    python3 submission/scripts/02_build_samplesheet.py \
        --use-case-dir data/use-cases/bee-gut \
        --fasta-dir mags \
        --out samplesheet.csv \
        --metagenome "honeybee metagenome" \
        --broad-environment apiary-biome \
        --local-environment pain-cages \
        --environmental-medium Honeybee
"""

import argparse
import csv
import os
import sys
from collections import Counter

# rank order used to build the semicolon-separated NCBI taxid lineage
RANKS = ("superkingdom", "phylum", "class", "order", "family", "genus", "species")

COLUMNS = [
    "id",
    "fasta",
    "accession",
    "fastq_1",
    "fastq_2",
    "assembly_software",
    "binning_software",
    "binning_parameters",
    "stats_generation_software",
    "completeness",
    "contamination",
    "genome_coverage",
    "metagenome",
    "co-assembly",
    "broad_environment",
    "local_environment",
    "environmental_medium",
    "RNA_presence",
    "NCBI_lineage",
]


def strip_fasta(name: str) -> str:
    """'SRR1_binette_bin4.fasta.fasta' -> 'SRR1_binette_bin4'."""
    n = (name or "").strip()
    while n.endswith((".fasta", ".fna", ".fa")):
        n = n.rsplit(".", 1)[0]
    return n


def read_tsv(path: str, key: str) -> dict:
    with open(path, newline="") as fh:
        return {strip_fasta(r[key]): r for r in csv.DictReader(fh, delimiter="\t")}


def read_csv(path: str, key: str) -> dict:
    with open(path, newline="") as fh:
        return {strip_fasta(r[key]): r for r in csv.DictReader(fh)}


def build_lineage(mid: str, kmetashot: dict, gtdb: dict) -> str:
    """NCBI taxid lineage from kmetashot, or an 'Unclassified X' placeholder.

    kmetashot.tsv stores NCBI *taxids* in its rank columns; ENA resolves them
    to names itself.  When kmetashot has no superkingdom the MAG was not
    classified, and the 'uncultured bacterium'/'uncultured archaeon' style
    fallback that genome_uploader applies is triggered by these placeholders.
    """
    row = kmetashot.get(mid, {})
    if row.get("superkingdom", "0") not in ("", "0", None):
        return ";".join(row[r] for r in RANKS if row.get(r, "0") not in ("", "0"))

    classification = (gtdb.get(mid, {}).get("classification") or "").strip()
    if "Archaea" in classification:
        return "Unclassified Archaea"
    if "Eukaryota" in classification:
        return "Unclassified Eukaryota"
    return "Unclassified Bacteria"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--use-case-dir", required=True, help="dir with checkm2.tsv, kmetashot.tsv, gtdb.tsv, coverm.tsv")
    ap.add_argument("--fasta-dir", default="mags", help="dir holding <id>.fasta.gz (default: mags)")
    ap.add_argument("--out", required=True, help="samplesheet to write")
    ap.add_argument("--expected-rows", type=int, default=None, help="assert this many rows (safety net)")
    ap.add_argument("--assembly-software", default="MEGAHIT_v1.2.9")
    ap.add_argument("--binning-software", required=True, help="comma-separated binner/refiner names with versions")
    ap.add_argument("--binning-parameters", required=True)
    ap.add_argument("--stats-generation-software", required=True)
    ap.add_argument("--metagenome", required=True)
    ap.add_argument("--broad-environment", required=True)
    ap.add_argument("--local-environment", required=True)
    ap.add_argument("--environmental-medium", required=True)
    args = ap.parse_args()

    d = args.use_case_dir
    checkm2 = read_tsv(os.path.join(d, "checkm2.tsv"), "Name")
    kmetashot = read_csv(os.path.join(d, "kmetashot.tsv"), "bin")
    gtdb = read_tsv(os.path.join(d, "gtdb.tsv"), "user_genome")

    coverm = {}
    with open(os.path.join(d, "coverm.tsv"), newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        for row in reader:
            coverm[strip_fasta(row[0])] = dict(zip(header[1:], row[1:]))

    rows = []
    # keep the checkm2.tsv row order so re-runs are reproducible
    for mid, checkm in checkm2.items():
        run = mid.split("_")[0]
        if mid not in coverm:
            sys.exit(f"{mid}: not in coverm.tsv")
        if run not in coverm[mid]:
            sys.exit(f"{mid}: coverm.tsv has no column for run {run}")
        rows.append(
            {
                "id": mid,
                "fasta": os.path.join(args.fasta_dir, f"{mid}.fasta.gz"),
                "accession": run,
                "fastq_1": "",
                "fastq_2": "",
                "assembly_software": args.assembly_software,
                "binning_software": args.binning_software,
                "binning_parameters": args.binning_parameters,
                "stats_generation_software": args.stats_generation_software,
                "completeness": checkm["Completeness"].strip(),
                "contamination": checkm["Contamination"].strip(),
                "genome_coverage": coverm[mid][run].strip(),
                "metagenome": args.metagenome,
                "co-assembly": "false",
                "broad_environment": args.broad_environment,
                "local_environment": args.local_environment,
                "environmental_medium": args.environmental_medium,
                "RNA_presence": "",
                "NCBI_lineage": build_lineage(mid, kmetashot, gtdb),
            }
        )

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # ---- validation -------------------------------------------------------
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate MAG ids"
    if args.expected_rows is not None:
        assert len(rows) == args.expected_rows, f"expected {args.expected_rows} rows, got {len(rows)}"
    for r in rows:
        assert r["id"] and r["fasta"] and r["accession"], f"incomplete row: {r}"
        assert "\t" not in r["binning_parameters"] and "\n" not in r["binning_parameters"]

    classified = Counter(r["NCBI_lineage"].startswith("Unclassified") for r in rows)
    print(f"wrote {args.out}: {len(rows)} rows, {len(COLUMNS)} columns")
    print(f"  classified lineages  : {classified[False]}")
    print(f"  placeholders         : {classified[True]}")
    print(f"  example row          : {rows[0]['id']} -> {rows[0]['NCBI_lineage']}")


if __name__ == "__main__":
    main()
