#!/usr/bin/env python3
"""Rewrite the binning/statistics provenance columns of a samplesheet.

The ENA sample metadata must state which software produced the MAGs.  Those
values are read from the FAIRyMAGs workflow invocation (per-step tool ids and
the workflow-level inputs) and are identical for every MAG of a use case, so
they are applied to all rows in one pass.

Run this after ``02_build_samplesheet.py`` (which writes placeholder values).

Usage
-----
    python3 submission/scripts/03_update_binning_metadata.py samplesheet.csv

The values default to the bee-gut run; override with --binning-software,
--binning-parameters and --stats-generation-software for another use case.
"""

import argparse
import csv
import shutil
import sys

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

DEFAULT_BINNING_SOFTWARE = (
    "MetaBAT2_v2.18.23,MaxBin2_v2.2.7,SemiBin2_v2.1.0,"
    "CONCOCT_v1.1.0,COMEBin_v1.0.4,Binette_v1.2.1"
)

DEFAULT_BINNING_PARAMETERS = (
    "defaults except: SemiBin2 environment=global; CONCOCT read_length=150; "
    "Binette contamination_weight=2; MAG filters min_length=50000 bp, "
    "min_completeness=0%, max_contamination=100%; dRep dereplication ANI=0.95; "
    "MEGAHIT min_contig_length=200 bp. MAGs generated with the FAIRyMAGs "
    "workflows (https://github.com/usegalaxy-eu/FAIRyMAGs) on usegalaxy.eu"
)

DEFAULT_STATS_SOFTWARE = "CheckM2_v1.1.0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samplesheet")
    ap.add_argument("--binning-software", default=DEFAULT_BINNING_SOFTWARE)
    ap.add_argument("--binning-parameters", default=DEFAULT_BINNING_PARAMETERS)
    ap.add_argument("--stats-generation-software", default=DEFAULT_STATS_SOFTWARE)
    args = ap.parse_args()

    # the pipeline writes the metadata as a tab-separated table, so no value
    # may contain a tab or a newline
    for name in ("binning_software", "binning_parameters", "stats_generation_software"):
        value = getattr(args, name)
        if "\t" in value or "\n" in value:
            sys.exit(f"{name} must not contain tabs or newlines")

    with open(args.samplesheet, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            sys.exit(f"unexpected header: {reader.fieldnames}")
        rows = list(reader)

    changed = 0
    for row in rows:
        before = (row["binning_software"], row["binning_parameters"], row["stats_generation_software"])
        row["binning_software"] = args.binning_software
        row["binning_parameters"] = args.binning_parameters
        row["stats_generation_software"] = args.stats_generation_software
        if before != (row["binning_software"], row["binning_parameters"], row["stats_generation_software"]):
            changed += 1

    shutil.copyfile(args.samplesheet, args.samplesheet + ".bak")
    with open(args.samplesheet, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with open(args.samplesheet, newline="") as fh:
        check = list(csv.DictReader(fh))
    assert len(check) == len(rows), f"row count changed: {len(rows)} -> {len(check)}"
    assert {r["binning_software"] for r in check} == {args.binning_software}
    assert {r["binning_parameters"] for r in check} == {args.binning_parameters}
    assert {r["stats_generation_software"] for r in check} == {args.stats_generation_software}
    assert all(r["id"] and r["fasta"] and r["accession"] for r in check)
    assert all(r["fasta"].endswith(r["id"] + ".fasta.gz") for r in check)

    print(f"rows={len(check)} changed={changed}")
    print(f"  binning_software     = {args.binning_software}")
    print(f"  stats_generation_... = {args.stats_generation_software}")
    print(f"  backup written       = {args.samplesheet}.bak")


if __name__ == "__main__":
    main()
