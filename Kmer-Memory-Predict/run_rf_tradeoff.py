#!/usr/bin/env python3
"""
Run Random Forest on k-mer stats and plot trade-off between underpredicted samples
and excess memory across several models (including Random Forest).

Usage:
  python run_rf_tradeoff.py \
    --stats-csv updated_mgnify_assemblies_stats_v3.15.3_metaspades_kmer_stats.csv \
    --meta-csv input/updated_mgnify_assemblies_stats_v3.15.3_metaspades_subset.csv \
    --output plot_tradeoff.png

Assumes the stats CSV was exported with sample IDs as the index (first column).
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

sns.set(style="whitegrid")


def load_stats(stats_path: Path) -> pd.DataFrame:
    # try load with first column as index (common when saving DataFrame.to_csv())
    df = pd.read_csv(stats_path, index_col=0)
    # ensure numeric columns only (drop metrics that are non-numeric)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_metadata(meta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_path)
    # try common names
    if "srr_id" in df.columns and "peak_mem_in_gbs" in df.columns:
        df2 = df.set_index("srr_id")[["peak_mem_in_gbs"]]
        return df2

    # fallback: try to find the best column names
    candidates = [c for c in df.columns if "srr" in c.lower() or "run" in c.lower()]
    mem_cols = [c for c in df.columns if "mem" in c.lower() or "peak" in c.lower()]
    if candidates and mem_cols:
        idx_col = candidates[0]
        mem_col = mem_cols[0]
        return df.set_index(idx_col)[[mem_col]].rename(columns={mem_col: "peak_mem_in_gbs"})

    raise ValueError("Could not find 'srr_id' and 'peak_mem_in_gbs' columns in metadata."
                     " Please provide a metadata CSV with these columns.")


def compute_tradeoff(y_true: np.ndarray, y_pred_base: np.ndarray, adjustment_steps: np.ndarray):
    underpred_percent_list = []
    excess_memory_list = []
    n_samples = len(y_true)

    for adj in adjustment_steps:
        y_pred_adj = y_pred_base + adj
        under_pred_mask = y_pred_adj < y_true
        under_pred_count = np.sum(under_pred_mask)
        under_pred_percent = (under_pred_count / n_samples) * 100

        # for samples where prediction >= true, compute excess memory
        excess_memory = y_pred_adj[~under_pred_mask] - y_true[~under_pred_mask]
        total_excess_memory = np.sum(excess_memory)

        underpred_percent_list.append(under_pred_percent)
        excess_memory_list.append(total_excess_memory)

    return np.array(underpred_percent_list), np.array(excess_memory_list)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stats-csv", default="updated_mgnify_assemblies_stats_v3.15.3_metaspades_kmer_stats.csv",
                   help="CSV with k-mer stats (samples as index)")
    p.add_argument("--meta-csv", default="input/updated_mgnify_assemblies_stats_v3.15.3_metaspades_subset.csv",
                   help="Metadata CSV with srr_id and peak_mem_in_gbs columns")
    p.add_argument("--output", default="tradeoff_models.png", help="Output plot file (PNG)")
    p.add_argument("--adjust-min", type=float, default=-100.0, help="Min adjustment (GB)")
    p.add_argument("--adjust-max", type=float, default=100.0, help="Max adjustment (GB)")
    p.add_argument("--adjust-step", type=float, default=5.0, help="Adjustment step (GB)")
    args = p.parse_args(argv)

    stats_path = Path(args.stats_csv)
    meta_path = Path(args.meta_csv)

    if not stats_path.exists():
        print(f"Stats CSV not found: {stats_path}")
        sys.exit(2)
    if not meta_path.exists():
        print(f"Metadata CSV not found: {meta_path}")
        sys.exit(2)

    print("Loading stats...")
    stats = load_stats(stats_path)
    print(f"Stats shape: {stats.shape}")

    print("Loading metadata...")
    meta = load_metadata(meta_path)
    print(f"Metadata shape: {meta.shape}")

    # Align indices: try to match sample names
    common = stats.index.intersection(meta.index)
    if len(common) == 0:
        # try stripping extensions or prefixes
        stats_idx = stats.index.astype(str).str.replace(r"\..*$", "", regex=True)
        meta_idx = meta.index.astype(str).str.replace(r"\..*$", "", regex=True)
        common = stats_idx[stats_idx.isin(meta_idx)].index
        # remap stats to stripped
        stats.index = stats_idx
    
    if len(common) == 0:
        # try case-insensitive match
        stats_idx = stats.index.astype(str).str.lower()
        meta_idx = meta.index.astype(str).str.lower()
        common_vals = set(stats_idx).intersection(meta_idx)
        common = [s for s in stats.index if s.lower() in common_vals]

    if len(common) == 0:
        raise ValueError("No common sample IDs found between stats and metadata after several heuristics.")

    # reindex both to common intersection
    stats = stats.loc[common].sort_index()
    meta = meta.loc[common].sort_index()

    # drop columns from stats that are all-nan or non-numeric
    stats = stats.dropna(axis=1, how="all")

    # target
    y = meta["peak_mem_in_gbs"].astype(float)

    # feature sets
    features_1 = ["std_count"] if "std_count" in stats.columns else [stats.columns[0]]
    features_2 = ["file_size"] if "file_size" in stats.columns else [stats.columns[0]]
    features_all = stats.columns.tolist()

    # build model list
    models = [
        ("Model 1: Std. kmer count (Linear)", features_1, LinearRegression()),
        ("Model 2: File size (Linear)", features_2, LinearRegression()),
        ("Model 3: Predict 250 (Dummy)", features_all, DummyRegressor(strategy="constant", constant=250)),
        ("Model 4: All features (Random Forest)", features_all, RandomForestRegressor(random_state=42)),
        ("Model 5: All features (Gradient Boosting)", features_all, GradientBoostingRegressor(random_state=42)),
    ]

    if HAS_XGB:
        models.append(("Model 6: All features (XGBoost)", features_all, xgb.XGBRegressor(random_state=42, verbosity=0)))
    else:
        warnings.warn("xgboost not available; skipping XGBoost model in plot.")

    adjustment_steps = np.arange(args.adjust_min, args.adjust_max + 0.0001, args.adjust_step)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    plt.figure(figsize=(12, 8))

    for title, feat_list, model in models:
        print(f"Processing: {title} with {len(feat_list)} feature(s)")
        X = stats[feat_list].astype(float)

        # cross-validated predictions
        try:
            y_pred_base = cross_val_predict(model, X, y, cv=cv)
        except Exception as e:
            print(f"Model {title} failed during cross_val_predict: {e}")
            continue

        underpred_percent_list, excess_memory_list = compute_tradeoff(y.values, y_pred_base, adjustment_steps)

        plt.plot(underpred_percent_list, excess_memory_list, label=title, marker='o')

    plt.xlabel("Percentage of Underpredicted Samples (%)")
    plt.ylabel("Total Excess Memory (GB)")
    plt.title("Trade-off Between Underpredicted Samples and Excess Memory Across Models")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 40)
    # auto-ylim but keep a reasonable lower bound
    plt.ylim(bottom=0)

    out_path = Path(args.output)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved trade-off plot to: {out_path}")


if __name__ == '__main__':
    main()
