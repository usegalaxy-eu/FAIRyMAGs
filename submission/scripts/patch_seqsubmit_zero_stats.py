#!/usr/bin/env python3
"""Work around the nf-core/seqsubmit bug that re-runs CheckM2 for MAGs with a 0.0 statistic.

Background
----------
``workflows/genomesubmit.nf`` decides whether a MAG still needs genome
evaluation (CheckM2, and therefore a ~1 GB database download) with a
truthiness test:

    genome_evaluation_input: !(meta.completeness) || !(meta.contamination) ||
                             !(meta.stats_generation_software)

In Groovy/Nextflow a value of ``0.0`` is falsy, so a MAG that is legitimately
*perfect* on one metric - ``contamination = 0.0`` is the desired result - is
treated as "no measurement available" and CheckM2 is re-run on it.  In the
bee-gut set 81 of 738 MAGs have ``contamination = 0.0``.

The fix is a null check:

    genome_evaluation_input: meta.completeness == null || meta.contamination == null ||
                             meta.stats_generation_software == null

``assets/schema_input_genome.json`` permits ``completeness``/``contamination``
``>= 0``, so 0 is a valid measurement and must not be confused with a missing
one (``genome_coverage`` on the other hand has ``exclusiveMinimum: 0``).

Run this on a seqsubmit checkout before running the pipeline:

    python3 submission/scripts/patch_seqsubmit_zero_stats.py /path/to/seqsubmit
    python3 submission/scripts/patch_seqsubmit_zero_stats.py /path/to/seqsubmit --check
    python3 submission/scripts/patch_seqsubmit_zero_stats.py /path/to/seqsubmit --revert

The patch is idempotent and is a one-line change, so ``nextflow -resume``
keeps every cached task.
"""

import argparse
import sys
from pathlib import Path

RELATIVE = "workflows/genomesubmit.nf"

OLD = "genome_evaluation_input: !(meta.completeness) || !(meta.contamination) || !(meta.stats_generation_software)"
NEW = "genome_evaluation_input: meta.completeness == null || meta.contamination == null || meta.stats_generation_software == null"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipeline", help="path to a nf-core/seqsubmit checkout")
    ap.add_argument("--check", action="store_true", help="report status, change nothing")
    ap.add_argument("--revert", action="store_true", help="restore the original line")
    args = ap.parse_args()

    path = Path(args.pipeline) / RELATIVE
    if not path.exists():
        sys.exit(f"not found: {path}")

    text = path.read_text()
    patched = NEW in text
    original = OLD in text

    if args.check:
        if patched:
            print(f"PATCHED   {path}")
        elif original:
            print(f"UNPATCHED {path}")
        else:
            print(f"UNKNOWN   {path}: neither the buggy nor the patched line was found")
            sys.exit(1)
        return

    if args.revert:
        if not patched:
            print("already unpatched")
            return
        path.write_text(text.replace(NEW, OLD))
        print(f"REVERTED {path}")
        return

    if patched:
        print("ALREADY PATCHED")
        return
    if not original:
        sys.exit(f"expected line not found in {path} - has upstream changed it already?")

    count = text.count(OLD)
    if count != 1:
        sys.exit(f"expected exactly 1 occurrence, found {count}")
    path.write_text(text.replace(OLD, NEW))
    print(f"PATCHED {path}")


if __name__ == "__main__":
    main()
