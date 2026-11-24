#!/usr/bin/env python3
"""Compare single- and multi-coverage binning of one use case.

Panel A: high- and medium-quality MAGs after dereplication per setup.
Panel B: CheckM2 completeness vs contamination of those MAGs.

Reads <data_dir>/single/checkm2.tsv and <data_dir>/multi/checkm2.tsv (the
CheckM2 quality report of the dereplicated MAGs, as exported by the campaign)
and writes <out>.png and <out>.svg.

    python bin/plot_single_vs_multi_coverage.py                   # bee gut defaults
    python bin/plot_single_vs_multi_coverage.py --data-dir data/use-cases/<case> \
        --out results/use-cases/<case>/single_vs_multi_coverage --title "<case>"
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
HQC, MQC = "#1c5cab", "#86b6ef"
COLORS = {"Single-coverage": "#2a78d6", "Multi-coverage": "#eb6834"}


def load(data_dir):
    data = {}
    for label, setup in [("Single-coverage", "single"), ("Multi-coverage", "multi")]:
        ck = pd.read_csv(f"{data_dir}/{setup}/checkm2.tsv", sep="\t")
        hq = (ck.Completeness >= 90) & (ck.Contamination <= 5)
        mq = (ck.Completeness >= 50) & (ck.Contamination <= 10) & ~hq
        data[label] = dict(hq=int(hq.sum()), mq=int(mq.sum()), n=len(ck), good=ck[hq | mq])
    return data


def plot(data, out, title):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.linewidth": 0.6})
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURF,
                               gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.42})
    for ax in (a, b):
        ax.set_facecolor(SURF)
        ax.spines[["top", "right"]].set_visible(False)

    labels = list(data)
    y = np.array([1, 0])
    xmax = max(d["hq"] + d["mq"] for d in data.values())
    for yi, label in zip(y, labels):
        d = data[label]
        a.barh(yi, d["hq"], height=0.42, color=HQC, edgecolor=SURF, linewidth=2)
        a.barh(yi, d["mq"], left=d["hq"], height=0.42, color=MQC, edgecolor=SURF, linewidth=2)
        a.text(d["hq"] / 2, yi, str(d["hq"]), ha="center", va="center", color="white", fontweight="bold")
        a.text(d["hq"] + d["mq"] / 2, yi, str(d["mq"]), ha="center", va="center", color=INK)
        a.text(d["hq"] + d["mq"] + xmax * 0.02, yi, f"{d['hq'] + d['mq']}  (of {d['n']} MAGs)",
               va="center", color=INK2, fontsize=8)
    a.set_yticks(y, labels)
    a.tick_params(axis="y", length=0)
    a.set_ylim(-0.55, 1.55)
    a.set_xlim(0, xmax * 1.6)
    a.set_xlabel("MAGs after dereplication")
    a.xaxis.grid(True, color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.spines["left"].set_visible(False)
    a.legend([Patch(color=HQC), Patch(color=MQC)],
             ["High quality (≥90% compl., ≤5% contam.)", "Medium quality (≥50%, ≤10%)"],
             frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    a.set_title("A  Quality of recovered MAGs", loc="left", fontsize=10, fontweight="bold", color=INK)

    b.axvspan(90, 101.5, ymax=(5 + 0.3) / 10.6, color="#eef4fc", lw=0, zorder=0)
    b.text(90.6, 5.25, "HQ", fontsize=7.5, color=HQC, fontweight="bold", va="bottom")
    for label, marker in zip(labels, ["o", "s"]):
        g = data[label]["good"]
        b.scatter(g.Completeness, g.Contamination, s=34, marker=marker, color=COLORS[label], alpha=0.85,
                  edgecolor=SURF, linewidth=1, clip_on=False, zorder=3,
                  label=f"{label} (median {g.Completeness.median():.1f}%)")
    b.set_xlim(49, 101.5)
    b.set_ylim(-0.3, 10.3)
    b.set_xlabel("Completeness (%)")
    b.set_ylabel("Contamination (%)")
    b.grid(True, color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
             title="Medium + high quality MAGs", title_fontsize=7.5)
    b.set_title("B  CheckM2 estimates", loc="left", fontsize=10, fontweight="bold", color=INK)

    fig.suptitle(title, x=0.02, ha="left", fontsize=9, color=INK2, y=1.02)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", facecolor=SURF)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/use-cases/bee-gut-microbiome")
    parser.add_argument("--out", default="results/use-cases/bee-gut-microbiome/single_vs_multi_coverage")
    parser.add_argument("--title", default="Bee gut microbiome (21 samples) · IWC MAGs generation v1.0 · "
                                           "single- vs multi-coverage binning")
    args = parser.parse_args()
    plot(load(args.data_dir), args.out, args.title)
