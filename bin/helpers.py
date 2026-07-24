"""Helper functions for the aeromicrobiome exploration notebook."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence

import pandas as pd


def notebook_display(obj: object) -> None:
    """Display an object in notebook environments with a safe console fallback.

    Parameters
    ----------
    obj:
        Object to render in a notebook output cell or print in a non-notebook context.

    Returns
    -------
    None
        This function is used for side effects only.
    """
    try:
        ipy_display = importlib.import_module("IPython.display")
        ipy_display.display(obj)
    except ImportError:
        print(obj)

TOTAL_LENGTH_COLUMN = "Total length"
UNCLASSIFIED_CLUSTERS_COLUMN = "Unclassified clusters"
TAXONOMY_COLS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def load_dfs(data_dp_2: Path, result_dp: Path, rep_fn: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load metadata, representative MAG table, and coverage table.

    Parameters
    ----------
    data_dp_2:
        Path to the use-case data directory containing `metadata.tsv` and `coverm.tsv`.
    result_dp:
        Path to the use-case results directory containing the representative MAG table.
    rep_fn:
        File name of the representative MAG table (for example `reps_cloud.tsv`).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Metadata, representative MAGs, and coverage DataFrames in that order.
    """
    metadata_fp = data_dp_2 / "metadata.tsv"
    if not metadata_fp.exists():
        print(f"Metadata file not found: {metadata_fp}")
        metadata_df = pd.DataFrame()
    else:
        metadata_df = pd.read_csv(metadata_fp, sep="\t")

    reps_fp = result_dp / rep_fn
    reps_df = pd.read_csv(reps_fp, sep="\t")
    for column in ["Completeness", "Contamination"]:
        if column in reps_df.columns:
            reps_df[column] = pd.to_numeric(
                reps_df[column].astype(str).str.replace("%", "", regex=False).str.strip(),
                errors="coerce",
            )

    coverage_fp = data_dp_2 / "coverm.tsv"
    coverage_df = pd.read_csv(coverage_fp, sep="\t")
    return metadata_df, reps_df, coverage_df


def print_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for all numeric columns in a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame containing numeric and non-numeric columns.

    Returns
    -------
    None
        This function prints formatted statistics and does not return a value.
    """
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            print(
                f"{column}: {df.loc['mean', column]:.2f} ± {df.loc['std', column]:.2f}, "
                f"Median: {df.loc['50%', column]:.2f}, "
                f"IQR: {df.loc['25%', column]:.2f}-{df.loc['75%', column]:.2f}, "
                f"Range: {df.loc['min', column]:.2f}-{df.loc['max', column]:.2f}"
            )


def compute_print_stats(df: pd.DataFrame) -> None:
    """Print core MAG quality summary statistics for a representative table.

    The function summarizes `Cluster members`, `Contamination`, `Completeness`, and
    `Total length` (reported in Mb), then prints missing value counts.

    Parameters
    ----------
    df:
        Representative MAG DataFrame containing quality-related columns.

    Returns
    -------
    None
        This function prints summary statistics and does not return a value.
    """
    columns = ["Cluster members", "Contamination", "Completeness", TOTAL_LENGTH_COLUMN]
    stats = df[columns].describe()
    stats[TOTAL_LENGTH_COLUMN] = stats[TOTAL_LENGTH_COLUMN] / 1000000
    stats = stats.T
    stats["missing_values"] = df.isnull().sum()

    print(f"Total number: {stats.loc['Cluster members', 'count']}")
    print_stats(stats.T)


def explore_species_level_clusters(df: pd.DataFrame, contamination_threshold: float = 100) -> None:
    """Print cluster summary for an optional contamination threshold.

    Parameters
    ----------
    df:
        Representative MAG DataFrame.
    contamination_threshold:
        Maximum contamination percentage to filter clusters. Use `100` to skip filtering.

    Returns
    -------
    None
        This function prints cluster summaries and does not return a value.
    """
    if contamination_threshold != 100:
        selected_reps_df = df.query(f"Contamination < {contamination_threshold}")
        print(f"Species-level clusters with contamination < {contamination_threshold}%")
        compute_print_stats(selected_reps_df)
        print()
    else:
        print("Species-level clusters with no contamination threshold")
        compute_print_stats(df)
        print()


def explore_species_level_clusters_all(df: pd.DataFrame) -> None:
    """Print cluster summaries across predefined contamination thresholds.

    Parameters
    ----------
    df:
        Representative MAG DataFrame.

    Returns
    -------
    None
        This function prints cluster summaries and does not return a value.
    """
    explore_species_level_clusters(df, 100)
    explore_species_level_clusters(df, 5)
    explore_species_level_clusters(df, 10)


def compute_taxo_classification_summary(
    df: pd.DataFrame,
    taxonomy_cols: Sequence[str] = TAXONOMY_COLS,
) -> pd.DataFrame:
    """Compute classified vs unclassified cluster counts per taxonomy rank.

    Parameters
    ----------
    df:
        Input MAG DataFrame with taxonomy columns.
    taxonomy_cols:
        Ordered taxonomy rank names to evaluate.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by taxonomy level with counts and percentages.
    """
    existing_taxonomy_cols = [col for col in taxonomy_cols if col in df.columns]

    if not existing_taxonomy_cols:
        raise KeyError("No taxonomy columns found in the DataFrame.")

    unclassified_mask = df[existing_taxonomy_cols].apply(
        lambda s: s.astype("string").str.strip().str.lower().eq("unclassified")
    )

    summary_df = pd.DataFrame(index=existing_taxonomy_cols)
    summary_df[UNCLASSIFIED_CLUSTERS_COLUMN] = unclassified_mask.sum(axis=0)
    summary_df["Classified clusters"] = len(df) - summary_df[UNCLASSIFIED_CLUSTERS_COLUMN]
    summary_df["Unclassified clusters %"] = (
        summary_df[UNCLASSIFIED_CLUSTERS_COLUMN] / len(df) * 100
    ).round(2)
    summary_df["Classified clusters %"] = (100 - summary_df["Unclassified clusters %"]).round(2)

    return summary_df


def get_level_counts(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Aggregate cluster and MAG counts for one taxonomy level.

    Parameters
    ----------
    df:
        Input MAG DataFrame.
    level:
        Taxonomy column name (for example `Phylum` or `Genus`).

    Returns
    -------
    pd.DataFrame
        Summary with cluster counts, percentages, MAG totals, and a TOTAL row.
    """
    level_group = df.groupby(level)

    level_counts = level_group.size().sort_values(ascending=False).to_frame("Cluster")
    level_counts["Cluster %"] = 100 * level_counts["Cluster"] / level_counts["Cluster"].sum()

    level_mag_counts = level_group["Cluster members"].sum().sort_values(ascending=False).to_frame(
        "Total MAG count"
    )

    level_summary = pd.concat([level_counts, level_mag_counts], axis=1)
    level_summary.sort_values(by="Total MAG count", ascending=False, inplace=True)

    level_summary.loc["TOTAL"] = level_summary.sum(numeric_only=True)
    level_summary.loc["TOTAL", "Cluster %"] = 100.0

    return level_summary


def get_all_taxo_levels(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute and display taxonomic summaries for all predefined ranks.

    Parameters
    ----------
    df:
        Input MAG DataFrame with taxonomy columns.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from taxonomy rank to per-rank summary DataFrame.
    """
    taxo_levels: dict[str, pd.DataFrame] = {}
    for level in TAXONOMY_COLS:
        taxo_levels[level] = get_level_counts(df, level)
        print(f"\nLevel: {level}")
        notebook_display(taxo_levels[level])
    return taxo_levels


def get_relative_abundance(df: pd.DataFrame, coverage_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sample relative abundance grouped by family/genus/species.

    Parameters
    ----------
    df:
        Input MAG DataFrame.
    coverage_df:
        Coverage DataFrame containing `Genome` and sample coverage columns.

    Returns
    -------
    pd.DataFrame
        Relative abundance table (%) with a multi-index of Family/Genus/Species.
    """
    species_idx = df.columns.get_loc("Species")
    cov_taxo_df = df.copy()
    cov_taxo_df = cov_taxo_df.iloc[:, : species_idx + 1].copy()

    cov_taxo_df = cov_taxo_df.merge(
        coverage_df,
        left_on="MAG",
        right_on="Genome",
        how="left",
    )
    cov_taxo_df = cov_taxo_df.drop(columns=["Genome"], errors="ignore")

    abund_df = cov_taxo_df.groupby(["Family", "Genus", "Species"]).sum(numeric_only=True)
    abund_df = abund_df.div(abund_df.sum(axis=0), axis=1) * 100
    return abund_df


def get_relative_abund_taxo_levels(df: pd.DataFrame, coverage_df: pd.DataFrame) -> pd.DataFrame:
    """Display and return abundance summaries for each taxonomic index level.

    Parameters
    ----------
    df:
        Input MAG DataFrame with taxonomy columns.
    coverage_df:
        Coverage DataFrame containing `Genome` and sample coverage columns.

    Returns
    -------
    pd.DataFrame
        Relative abundance table (%) indexed by Family/Genus/Species.
    """
    relative_abund_df = get_relative_abundance(df, coverage_df)
    for level in relative_abund_df.index.names:
        taxo_level_df = relative_abund_df.groupby(level=level).sum()
        print(f"\nLevel: {level}")
        notebook_display(taxo_level_df.T.describe().T.sort_values(by="mean", ascending=False))
    return relative_abund_df


def get_bakta_annot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Bakta annotation columns and normalize column names.

    Parameters
    ----------
    df:
        Input MAG DataFrame potentially containing `bakta_`-prefixed columns.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only Bakta annotation columns with the prefix removed.
    """
    bakta_annot_df = df.filter(regex="^bakta_").copy()
    bakta_annot_df.columns = bakta_annot_df.columns.str.replace("^bakta_", "", regex=True)
    return bakta_annot_df


def get_kegg_path_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return non-empty KEGG completeness columns after filtering zeros.

    The function removes the `kegg_` prefix, fills NaN values with 0, and drops
    rows/columns composed entirely of zeros.

    Parameters
    ----------
    df:
        Input MAG DataFrame potentially containing `kegg_`-prefixed columns.

    Returns
    -------
    pd.DataFrame
        Filtered KEGG completeness DataFrame containing only non-zero rows and columns.
    """
    kegg_path_df = df.filter(regex="^kegg_").copy()
    kegg_path_df.columns = kegg_path_df.columns.str.replace("^kegg_", "", regex=True)
    kegg_path_df = kegg_path_df.fillna(0)
    print("Before removing rows and columns with only zeros:")
    print(f"Clusters: {kegg_path_df.shape[0]}")
    print(f"KEGG modules: {kegg_path_df.shape[1]}")
    kegg_path_df = kegg_path_df.loc[:, (kegg_path_df != 0).any(axis=0)]
    kegg_path_df = kegg_path_df.loc[(kegg_path_df != 0).any(axis=1), :]
    print("\nAfter removing rows and columns with only zeros:")
    print(f"Clusters: {kegg_path_df.shape[0]}")
    print(f"KEGG modules: {kegg_path_df.shape[1]}")
    print()
    non_zero_per_row = (kegg_path_df != 0).sum(axis=1)
    print_stats(non_zero_per_row.describe().to_frame("KEGG modules"))
    return kegg_path_df
