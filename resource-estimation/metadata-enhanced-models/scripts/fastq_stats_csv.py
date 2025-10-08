#!/usr/bin/env python3
import argparse
import csv
import gzip
import os
import random
from collections import defaultdict
from math import sqrt
from statistics import median
from typing import Dict, Optional

try:
    from Bio import SeqIO
except ImportError as e:
    raise SystemExit(
        "Biopython is required. Install with: pip install biopython\n"
        f"Original error: {e}"
    )


class FastqStats:
    """Streaming FASTQ statistics with Biopython's SeqIO.parse."""

    def __init__(self, path: str):
        self.path = path
        self._reads = 0
        self._len_hist = defaultdict(int)
        self._total_bases = 0
        self._gc_sample = []

    def file_size_bytes(self) -> int:
        return os.path.getsize(self.path)

    def _seq_iter(self):
        if self.path.endswith(".gz"):
            with gzip.open(self.path, "rt") as handle:
                yield from SeqIO.parse(handle, "fastq")
        else:
            with open(self.path, "r") as handle:
                yield from SeqIO.parse(handle, "fastq")

    def _scan(self, sample_frac: float, seed: Optional[int], exclude_ns: bool):
        rng = random.Random(seed)
        gc_vals = [] if sample_frac > 0 else None
        len_hist = defaultdict(int)
        reads = 0
        total_bases = 0

        for rec in self._seq_iter():
            seq = str(rec.seq)
            L = len(seq)
            reads += 1
            total_bases += L
            len_hist[L] += 1

            if gc_vals is not None and L > 0 and rng.random() < sample_frac:
                s = seq.upper()
                if exclude_ns:
                    atgc_count = sum(b in "ATGC" for b in s)
                    if atgc_count == 0:
                        continue
                    gc = sum(b in "GC" for b in s) / atgc_count
                else:
                    gc = (s.count("G") + s.count("C")) / L
                gc_vals.append(gc)

        self._reads = reads
        self._total_bases = total_bases
        self._len_hist = len_hist
        self._gc_sample = gc_vals or []

    def compute(self, sample_frac: float, seed: Optional[int], exclude_ns: bool) -> Dict[str, Optional[float]]:
        self._scan(sample_frac=sample_frac, seed=seed, exclude_ns=exclude_ns)

        # read-length stats
        if self._reads == 0:
            rl = dict(min=None, max=None, mean=None, median=None, N50=None, L50=None, total_bases=0)
        else:
            hist = self._len_hist
            total_reads = self._reads
            total_bases = self._total_bases
            lengths = sorted(hist.keys())

            # count-median
            cumulative = 0
            mid1 = (total_reads + 1) // 2
            mid2 = (total_reads + 2) // 2
            m1 = m2 = None
            for L in lengths:
                prev = cumulative
                cumulative += hist[L]
                if m1 is None and prev < mid1 <= cumulative:
                    m1 = L
                if m2 is None and prev < mid2 <= cumulative:
                    m2 = L
                if m1 and m2:
                    break
            median_len = (m1 + m2) / 2

            # N50 / L50
            half_bases = total_bases / 2
            cum_bases = 0
            reads_counted = 0
            N50 = L50 = None
            for L in sorted(hist.keys(), reverse=True):
                cnt = hist[L]
                block = L * cnt
                if cum_bases + block >= half_bases:
                    N50 = L
                    remaining = half_bases - cum_bases
                    L50 = reads_counted + int(remaining // L) + 1
                    break
                cum_bases += block
                reads_counted += cnt

            rl = dict(
                min=min(lengths),
                max=max(lengths),
                mean=total_bases / total_reads,
                median=median_len,
                N50=N50,
                L50=L50,
                total_bases=total_bases,
            )

        # GC sample stats
        vals = self._gc_sample
        if vals:
            mean_gc = sum(vals) / len(vals)
            std_gc = sqrt(sum((x - mean_gc) ** 2 for x in vals) / len(vals))
            gc = dict(mean=mean_gc, median=median(vals), std=std_gc, n_sampled=len(vals))
        else:
            gc = dict(mean=None, median=None, std=None, n_sampled=0)

        return {
            "file": self.path,
            "file_size_bytes": self.file_size_bytes(),
            "number_of_reads": self._reads,
            "min_read_len": rl["min"],
            "max_read_len": rl["max"],
            "mean_read_len": rl["mean"],
            "median_read_len": rl["median"],
            "N50": rl["N50"],
            "L50": rl["L50"],
            "total_bases": rl["total_bases"],
            "gc_mean": gc["mean"],
            "gc_median": gc["median"],
            "gc_std": gc["std"],
            "gc_n_sampled": gc["n_sampled"],
        }


def main():
    p = argparse.ArgumentParser(
        description="Compute FASTQ stats (Biopython parser) and write CSV."
    )
    p.add_argument("fastqs", nargs="+", help="FASTQ/FASTQ.GZ files")
    p.add_argument("-o", "--output", default="-", help="Output CSV file (default: stdout)")
    p.add_argument("--sample-frac", type=float, default=0.2, help="Fraction of reads to sample for GC (0..1)")
    p.add_argument("--seed", type=int, default=1, help="Random seed for sampling")
    p.add_argument("--include-ns", action="store_true", help="Include Ns in GC denominator (default: exclude)")
    args = p.parse_args()

    if not (0.0 <= args.sample_frac <= 1.0):
        raise SystemExit("--sample-frac must be between 0 and 1")

    exclude_ns = not args.include_ns

    rows = []
    for fq in args.fastqs:
        if not os.path.isfile(fq):
            print(f"Warning: '{fq}' not found, skipping.", flush=True)
            continue
        stats = FastqStats(fq).compute(
            sample_frac=args.sample_frac, seed=args.seed, exclude_ns=exclude_ns
        )
        rows.append(stats)

    # CSV output
    fieldnames = [
        "file",
        "file_size_bytes",
        "number_of_reads",
        "min_read_len",
        "max_read_len",
        "mean_read_len",
        "median_read_len",
        "N50",
        "L50",
        "total_bases",
        "gc_mean",
        "gc_median",
        "gc_std",
        "gc_n_sampled",
    ]

    out_fh = None
    try:
        if args.output == "-" or args.output.lower() == "stdout":
            out_fh = None  # csv.writer can use sys.stdout via print workaround
            import sys
            writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        else:
            with open(args.output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
    finally:
        if out_fh:
            out_fh.close()


if __name__ == "__main__":
    main()
