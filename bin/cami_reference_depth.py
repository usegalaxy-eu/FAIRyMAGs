#!/usr/bin/env python3
"""Expected sequencing depth of every CAMI II plant reference genome per sample.

CAMISIM simulated each of the 21 short-read samples with a fixed size
(``[ReadSimulator] size`` in config.ini, 5 Gbp) and drew reads from each genome
in proportion to ``abundance x genome size`` (abundance<sample>.tsv). The depth
of genome g in sample s is therefore

    depth[g, s] = size_bp * (a[g, s] * L[g] / sum_h a[h, s] * L[h]) / L[g]

Genome sizes come from the CheckM2 report of the reference genomes in the dRep
benchmark (``<genome>_ref`` rows). On sample 0 these depths match the reads
CAMISIM actually wrote (gsa_mapping.tsv, Spearman 0.998, median ratio 1.0).

Inputs come from rhimgCAMI2_setup.tar.gz
(https://frl.publisso.de/data/frl:6425521/plant_associated/short_read/), only:
simulation_short_read/{config.ini,genome_to_id.tsv,abundance*.tsv}

    python bin/cami_reference_depth.py <simulation_short_read dir> \
        data/benchmark/checkm2.tsv data/benchmark/cami_reference_depth.tsv
"""

import argparse
import configparser
from pathlib import Path

import pandas as pd


def reference_depth(setup_dir, checkm2_path):
    setup_dir = Path(setup_dir)
    config = configparser.ConfigParser(strict=False)  # CAMISIM repeats [Main]
    config.read(setup_dir / "config.ini")
    size_bp = float(config["ReadSimulator"]["size"]) * 1e9

    g2i = pd.read_csv(setup_dir / "genome_to_id.tsv", sep="\t", header=None, names=["genome_id", "path"])
    # The benchmark names each reference genome after its FASTA file, plus "_ref".
    g2i["genome"] = g2i.path.map(lambda p: Path(p).name.rsplit(".fasta", 1)[0] + "_ref")

    checkm2 = pd.read_csv(checkm2_path, sep="\t", comment="#")
    sizes = checkm2.loc[checkm2.Name.str.endswith("_ref")].set_index("Name").Genome_Size
    missing = set(g2i.genome) - set(sizes.index)
    if missing:
        raise ValueError(f"{len(missing)} reference genomes have no CheckM2 genome size, e.g. {sorted(missing)[:3]}")
    length = g2i.set_index("genome_id").genome.map(sizes)

    samples = sorted(int(p.stem.removeprefix("abundance")) for p in setup_dir.glob("abundance*.tsv"))
    abundance = pd.concat({s: pd.read_csv(setup_dir / f"abundance{s}.tsv", sep="\t", header=None,
                                          index_col=0)[1] for s in samples}, axis=1)
    if set(abundance.index) != set(length.index):
        raise ValueError("abundance files and genome_to_id.tsv list different genomes")
    length = length.loc[abundance.index]

    weight = abundance.mul(length, axis=0)
    depth = (weight / weight.sum()).mul(size_bp).div(length, axis=0)
    depth.columns = [f"depth_sample_{s}" for s in samples]

    out = g2i.set_index("genome_id").loc[depth.index, ["genome"]]
    out["genome_size"] = length.astype(int)
    out["max_sample_depth"] = depth.max(axis=1)
    out["pooled_depth"] = depth.sum(axis=1)
    out["samples_ge_5x"] = (depth >= 5).sum(axis=1)
    out = out.join(depth.round(4))
    out.index.name = "genome_id"
    return out.sort_values("max_sample_depth", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setup_dir")
    parser.add_argument("checkm2")
    parser.add_argument("out")
    args = parser.parse_args()
    table = reference_depth(args.setup_dir, args.checkm2)
    table.to_csv(args.out, sep="\t", float_format="%.4f")
    print(f"{len(table)} genomes; >=5x in some sample: {(table.max_sample_depth >= 5).sum()}, "
          f">=10x: {(table.max_sample_depth >= 10).sum()}")
