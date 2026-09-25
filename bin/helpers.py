"""Helper functions for the aeromicrobiome exploration notebook."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Sequence

import pandas as pd
from Bio import Entrez

# Column name constants reused across summary functions to avoid typos/duplication
TOTAL_LENGTH_COLUMN = "Total length"
UNCLASSIFIED_CLUSTERS_COLUMN = "Unclassified clusters"
TAXONOMY_COLS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]

# Required by NCBI Entrez API to identify the requester
Entrez.email = "berenice.batut@gmail.com"


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
        # Only import IPython when needed, so this module also works outside notebooks
        ipy_display = importlib.import_module("IPython.display")
        ipy_display.display(obj)
    except ImportError:
        # Fallback for plain Python scripts / terminals without IPython installed
        print(obj)


def resolve_uc_path(uc_name: str) -> tuple[Path, Path]:
    """Resolve the data and result directories for a given use-case name.
    
    Parameters
    ----------
    uc_name:
        Name of the use-case.

    Returns
    -------
    tuple[Path, Path]
        Data and result directories for the use-case.
    """
    # Paths are relative to the repository root, following the project's data/result layout
    data_dp = Path("../data/use-cases/") / uc_name
    result_dp = Path("../results/use-cases/") / uc_name
    return data_dp, result_dp


def tax_label(classification):
    """Return lowest resolved rank with GTDB prefix if not species-level.
    
    Parameters
    ----------
    classification:
        GTDB classification string.

    Returns
    -------
    str
        Lowest resolved taxonomic rank with GTDB prefix if not species-level, or "no GTDB hit" if unavailable.
    """
    if pd.isna(classification):
        return ""
    # Check ranks from most specific (species) to least specific (phylum);
    # species is returned without a prefix, others get their GTDB prefix as a label
    ranks = [("s__", None), ("g__", "g__"), ("f__", "f__"), ("p__", "p__")]
    for prefix, label in ranks:
        for part in str(classification).split(";"):
            part = part.strip()
            if part.startswith(prefix):
                name = part[len(prefix):].strip()
                if name:
                    return name if label is None else f"{label}{name}"
    return ""


def clean_genome_name(genome_series: pd.Series, pattern: str = r"\.fasta$") -> pd.Series:
    """Remove specified suffix from genome names in a pandas Series.

    Parameters
    ----------
    genome_series:
        Pandas Series containing genome names.
    pattern:
        Regex pattern to match the suffix to remove (default is `.fasta`).

    Returns
    -------
    pd.Series
        Pandas Series with `.fasta` suffix removed from genome names.
    """
    # Different tools (checkm2, bakta, kmetashot, etc.) suffix genome/bin names differently,
    # so this is used to normalize names before merging tables on genome identifiers
    return genome_series.str.replace(pattern, "", regex=True)


def load_df(df_dp: Path, sep="\t", index_col: int = -1, genome_name_col: str = "", to_tranpose=False) -> pd.DataFrame:
    """Load a DataFrame and optionally clean genome names.

    Parameters
    ----------
    df_dp:
        Path to the CSV file containing the DataFrame to load.
    sep:
        Delimiter to use for parsing the CSV file (default is tab).
    index_col:
        Column index to use as the row labels of the DataFrame. If -1, no index column is used.
    genome_name_col:
        Column name to clean genome names in by removing `.fasta` suffix. If empty, no cleaning is done.
    to_transpose:
        If True, transpose the DataFrame after loading and clean genome names in the index.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with optional genome name cleaning.
    """
    # index_col=-1 is a sentinel meaning "no explicit index column"
    if index_col == -1:
        df = pd.read_csv(df_dp, sep=sep)
    else:
        df = pd.read_csv(df_dp, sep=sep, index_col=index_col)

    if to_tranpose:
        # Some tool outputs (e.g. QUAST, Bakta) are wide with samples as columns;
        # transposing puts genomes as rows for consistent merging
        df = df.T
        df.index = clean_genome_name(df.index)

    if genome_name_col != "" and genome_name_col in df.columns:
        df[genome_name_col] = clean_genome_name(df[genome_name_col])
    elif genome_name_col != "" and genome_name_col not in df.columns:
        print(f"Warning: Column '{genome_name_col}' not found in DataFrame. No cleaning applied.")

    return df


def load_dfs(uc_name) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataframes from the use-case data directories and merge them for analysis.

    Parameters
    ----------
    uc_name:
        Name of the use-case.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Metadata, representative MAGs, and coverage DataFrames in that order.
    """
    # This is the main entry point orchestrating the whole ETL pipeline for a use case:
    # 1. locate data/result directories
    # 2. load simple standalone tables (metadata, coverage)
    # 3. build and enrich the "representative MAGs" table step by step
    # 4. persist the final table and return everything needed for downstream analysis
    data_dp, result_dp = resolve_uc_path(uc_name)

    metadata_df = _load_metadata(data_dp)
    coverage_df = load_df(data_dp / "coverm.tsv")

    # Step 1: merge CheckM2 + GTDB + kmetashot into a base representative table
    reps_df, drep_df = _build_base_reps_df(data_dp)
    # Step 2: resolve taxonomy at every rank (GTDB primary, kmetashot fallback)
    reps_df = _add_taxonomy_ranks(reps_df)
    # Step 3: add dRep cluster size information
    reps_df = _add_cluster_sizes(reps_df, drep_df)

    # Step 4: merge in additional annotation sources (assembly stats, gene annotation,
    # legacy CheckM stats, coverage, and KEGG pathway completeness)
    merged_cols: list[str] = []
    for merge_fn in (_merge_quast, _merge_bakta, _merge_checkm_v1, _merge_coverm, _merge_kegg):
        reps_df, cols = merge_fn(reps_df, data_dp)
        merged_cols += cols

    # Step 5: rename/reorder columns into the final, user-facing schema
    reps_df = _finalize_reps_df(reps_df, merged_cols)

    # Persist the final representative table so it can be reused without recomputation
    reps_fp = result_dp / "reps.tsv"
    reps_df.to_csv(reps_fp, sep="\t", index=False)

    return metadata_df, reps_df, coverage_df


def _load_metadata(data_dp: Path) -> pd.DataFrame:
    """Load metadata table, returning an empty DataFrame if missing.
    
    Parameters
    ----------
    data_dp:
        Path to the use-case data directory containing `metadata.tsv`.  
        
    Returns
    -------
    pd.DataFrame
        Metadata DataFrame if the file exists, otherwise an empty DataFrame.
    """
    metadata_fp = data_dp / "metadata.tsv"
    if not metadata_fp.exists():
        # Metadata is optional for some use cases; fail gracefully instead of raising
        print(f"Metadata file not found: {metadata_fp}")
        return pd.DataFrame()
    return load_df(metadata_fp)


def _build_base_reps_df(data_dp: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CheckM2, GTDB, and drep tables, and merge them into a base reps DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The merged representative MAGs DataFrame and the drep DataFrame.
    """
    # drep.tsv defines species-level clusters (used later for "Cluster members" counts)
    drep_df = load_df(data_dp / "drep.tsv", genome_name_col="genome", sep=",")
    # checkm2.tsv provides completeness/contamination quality metrics
    checkm_df = load_df(data_dp / "checkm2.tsv", genome_name_col="Name")
    # gtdb.tsv provides the primary GTDB taxonomic classification
    gtdb_df = load_df(data_dp / "gtdb.tsv", genome_name_col="user_genome")

    # kmetashot.tsv is an alternative/backup taxonomic classifier (taxon IDs per rank),
    # used later to fill gaps where GTDB has no classification
    kmetashot_df = load_df(data_dp / "kmetashot.tsv", sep=",", index_col=0, genome_name_col="bin").add_prefix("kmetashot_")
    kmetashot_df["kmetashot_bin"] = clean_genome_name(kmetashot_df["kmetashot_bin"])
    # Normalize to string/NA so mixed int/NaN taxon ID columns can be handled uniformly downstream
    kmetashot_df = kmetashot_df.astype(str).replace("nan", pd.NA)

    # Left joins on genome identifiers: keep every CheckM2 genome, adding GTDB/kmetashot info where available
    reps_df = pd.merge(checkm_df, gtdb_df, left_on="Name", right_on="user_genome", how="left")
    reps_df = pd.merge(reps_df, kmetashot_df, left_on="Name", right_on="kmetashot_bin", how="left")

    # Sort by completeness (best MAGs first) and round quality metrics for readability
    reps_df = reps_df.sort_values("Completeness", ascending=False).copy()
    reps_df["Completeness"] = reps_df["Completeness"].round(1)
    reps_df["Contamination"] = reps_df["Contamination"].round(2)

    return reps_df, drep_df


def get_ncbi_scientific_name(taxon_id: int) -> str:
    """Retrieve the scientific name for a given NCBI taxon ID using Entrez.

    Parameters
    ----------
    taxon_id:
        NCBI taxon ID to look up.
    
    Returns
    -------
    str
        Scientific name corresponding to the taxon ID, or "unclassified" if not found.
    """
    if taxon_id != 0:
        try:
            # Query NCBI's Taxonomy database over the network via Entrez
            stream = Entrez.efetch(db="Taxonomy", id=str(taxon_id), retmode="xml")
            records = Entrez.read(stream)
            return records[0]["ScientificName"]
        except Exception as e:
            # Network/parsing errors shouldn't crash the whole pipeline
            print(f"Error retrieving scientific name for taxon ID {taxon_id}: {e}")
            return "unclassified"
    else:
        # Taxon ID 0 conventionally means "no classification" in kmetashot output
        return "unclassified"


def _resolve_kmetashot_lineage(reps_df: pd.DataFrame, row_idx: int, rank_idx: int, taxo_ranks: list[str]) -> None:
    """Fill kmetashot rank columns for a row using NCBI lineage of its taxon ID.

    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame being updated in place.
    row_idx:
        Row index to update.
    rank_idx:
        Index of the current rank within `taxo_ranks`.
    taxo_ranks:
        Ordered list of taxonomy ranks (from species to domain).
    """
    # NCBI taxonomy uses "superkingdom" instead of GTDB's "domain"
    kmeta_rank = "superkingdom" if taxo_ranks[rank_idx] == "domain" else taxo_ranks[rank_idx]
    col = f"kmetashot_{kmeta_rank}"
    raw_value = reps_df.iat[row_idx, reps_df.columns.get_loc(col)]

    # kmetashot stores taxon IDs as numeric strings; skip if already resolved to a name
    # or otherwise not a valid ID (e.g. already "unclassified" or missing)
    if not str(raw_value).isdigit():
        return

    taxon_id = int(raw_value)
    if taxon_id == 0 or pd.isna(taxon_id):
        reps_df.iat[row_idx, reps_df.columns.get_loc(col)] = "unclassified"
        return

    # Fetch the full lineage once per taxon ID, then propagate names to this rank
    # and any higher (less specific) ranks that are still unresolved
    stream = Entrez.efetch(db="Taxonomy", id=str(taxon_id), retmode="xml")
    records = Entrez.read(stream)
    lineage = records[0].get("Lineage", "")
    lineage_parts = [part.strip() for part in lineage.split(";")]

    # Lineage is ordered from most general to most specific; reverse to align with
    # taxo_ranks, which goes from species (index 0) to domain (last index)
    for offset, part in enumerate(reversed(lineage_parts)):
        target_idx = rank_idx + offset
        if target_idx >= len(taxo_ranks):
            break
        target_rank = taxo_ranks[target_idx]
        if target_rank == "domain":
            target_rank = "superkingdom"
        reps_df.iat[row_idx, reps_df.columns.get_loc(f"kmetashot_{target_rank}")] = part


def _fill_kmetashot_rank(reps_df: pd.DataFrame, rank_idx: int, taxo_ranks: list[str]) -> None:
    """Resolve kmetashot taxon IDs to names for every row at a given rank.

    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame being updated in place.
    rank_idx:
        Index of the current rank within `taxo_ranks`.
    taxo_ranks:
        Ordered list of taxonomy ranks (from species to domain).
    """
    # Row-by-row processing is required because each row may need its own NCBI lookup
    for row_idx in range(len(reps_df)):
        _resolve_kmetashot_lineage(reps_df, row_idx, rank_idx, taxo_ranks)


def _add_taxonomy_ranks(reps_df: pd.DataFrame) -> pd.DataFrame:
    """Extract all taxonomy ranks from GTDB classification and handle missing species.

    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame containing a `classification` column.

    Returns
    -------
    pd.DataFrame
        Updated reps_df with separate columns for each taxonomy rank and missing species filled with "no GTDB hit".
    """
    # GTDB provides two possible classification columns depending on the method used;
    # prefer the closest-genome-based one, falling back to pplacer-based placement
    taxonomy = reps_df["closest_genome_taxonomy"].combine_first(
        reps_df["pplacer_taxonomy"]
    )
    # Process from species up to domain, since kmetashot lineage resolution needs
    # to propagate names from more specific to less specific ranks
    taxo_ranks = list(reversed([x.lower() for x in TAXONOMY_COLS]))

    for rank_idx, rank in enumerate(taxo_ranks):
        # Extract this rank's name directly from the GTDB classification string
        reps_df[f"gtdb_{rank}"] = taxonomy.apply(lambda value: extract_rank(value, rank))

        # For MAGs without a GTDB result at this rank, resolve kmetashot taxon IDs via NCBI
        kmeta_rank = "superkingdom" if rank == "domain" else rank
        _fill_kmetashot_rank(reps_df, rank_idx, taxo_ranks)

        # Final rank column: prefer GTDB, fall back to kmetashot, default to "unclassified"
        reps_df[rank] = reps_df[f"gtdb_{rank}"].combine_first(reps_df[f"kmetashot_{kmeta_rank}"])
        reps_df[rank] = reps_df[rank].fillna("unclassified")

    return reps_df


def _add_cluster_sizes(reps_df: pd.DataFrame, drep_df: pd.DataFrame) -> pd.DataFrame:
    """Compute species-level cluster sizes and add them to reps_df.
    
    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame containing a `Name` column.
    drep_df:
        dRep clustering DataFrame containing `genome` and `secondary_cluster` columns.

    Returns
    -------
    pd.DataFrame
        Updated reps_df with a "Cluster members" column indicating species-level cluster sizes.
    """
    # secondary_cluster groups genomes at ~95% ANI (species level, per dRep convention)
    cluster_size_map = drep_df.groupby("secondary_cluster").size().to_dict()
    genome_to_cluster = drep_df.set_index("genome")["secondary_cluster"].to_dict()
    # For each representative MAG, look up its cluster, then the size of that cluster
    reps_df["Cluster members"] = reps_df["Name"].map(
        lambda n: cluster_size_map.get(genome_to_cluster.get(n), 0)
    )
    return reps_df


def _merge_quast(reps_df: pd.DataFrame, data_dp: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load QUAST table and merge assembly stats into reps_df.
    
    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame containing a `Name` column.
    data_dp:
        Path to the use-case data directory containing `quast.tsv`.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Updated reps_df with QUAST assembly stats merged, and a list of the added column names
    """
    # QUAST output is wide (metrics as rows, genomes as columns); transpose to genomes-as-rows
    quast_df = load_df(data_dp / "quast.tsv", index_col=0, genome_name_col="", to_tranpose=True)
    # "Assembly" is a redundant identifier column, drop it before merging
    quast_cols = [c for c in quast_df.columns if c != "Assembly"]
    quast_df = quast_df[quast_cols]
    reps_df = pd.merge(reps_df, quast_df, left_on="Name", right_index=True, how="left")
    return reps_df, quast_df.columns.tolist()


def _merge_bakta(reps_df: pd.DataFrame, data_dp: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load Bakta annotations and merge prefixed counts into reps_df.
    
    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame containing a `Name` column.
    data_dp:
        Path to the use-case data directory containing `bakta.tsv`.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Updated reps_df with Bakta annotation counts merged, and a list of the added column names
    """
    bakta_df = load_df(data_dp / "bakta.tsv", index_col=0, to_tranpose=True)
    # Bakta output columns are suffixed with "_Count"; strip it to recover genome names
    bakta_df.index = clean_genome_name(bakta_df.index, pattern=r"\.fasta_Count$")
    # Drop non-numeric/metadata columns before merging
    bakta_cols = [c for c in bakta_df.columns if c not in ("Annotation", "Count")]
    # Prefix all Bakta columns so they're clearly distinguishable in the final table
    bakta_df = bakta_df[bakta_cols].add_prefix("bakta_")
    reps_df = pd.merge(reps_df, bakta_df, left_on="Name", right_index=True, how="left")
    return reps_df, bakta_df.columns.tolist()


def _merge_checkm_v1(reps_df: pd.DataFrame, data_dp: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load CheckM (v1) stats and merge selected columns into reps_df.

    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame containing a `Name` column.
    data_dp:
        Path to the use-case data directory containing `checkm.tsv`.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Updated reps_df with CheckM (v1) stats merged, and a list of the added column names
    """
    # Legacy CheckM (v1) is kept only for marker-gene-based metrics not provided by CheckM2
    checkm_v1_df = pd.read_csv(data_dp / "checkm.tsv", sep="\t")
    checkm_v1_df["Bin Id"] = clean_genome_name(checkm_v1_df["Bin Id"])
    checkm_v1_df = checkm_v1_df.set_index("Bin Id")
    checkm_v1_cols = ["Strain heterogeneity", "# markers", "# marker sets"]
    checkm_v1_df = checkm_v1_df[checkm_v1_cols]
    reps_df = pd.merge(reps_df, checkm_v1_df, left_on="Name", right_index=True, how="left")
    return reps_df, checkm_v1_df.columns.tolist()


def _merge_coverm(reps_df: pd.DataFrame, data_dp: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load CoverM stats, compute mean coverage per genome, and merge into reps_df.

    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame containing a `Name` column.
    data_dp:
        Path to the use-case data directory containing `coverm.tsv`.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Updated reps_df with CoverM mean coverage merged, and a list of the added column names
    """
    # CoverM provides per-sample coverage columns; here we only need a single
    # summary value (mean across samples) for the representative table
    coverm_df = pd.read_csv(data_dp / "coverm.tsv", sep="\t")
    coverm_df["Genome"] = clean_genome_name(coverm_df["Genome"])
    coverm_df = coverm_df.set_index("Genome").mean(axis=1).rename("coverm_mean_coverage").to_frame()
    reps_df = pd.merge(reps_df, coverm_df, left_on="Name", right_index=True, how="left")
    return reps_df, coverm_df.columns.tolist()


def _merge_kegg(reps_df: pd.DataFrame, data_dp: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load KEGG pathway completeness table, pivot to wide format, and merge into reps_df.

    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame containing a `Name` column.
    data_dp:
        Path to the use-case data directory containing `kegg_pathway_completeness.tsv`.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Updated reps_df with KEGG pathway completeness merged, and a list of the added column names
    """
    # Raw KEGG table is long format (one row per contig/pathway pair);
    # pivot to wide format so each pathway becomes its own column
    kegg_df = pd.read_csv(data_dp / "kegg_pathway_completeness.tsv", sep="\t")
    kegg_df["contig"] = clean_genome_name(kegg_df["contig"])
    kegg_df = kegg_df.pivot_table(
        index="contig", columns="pathway_name", values="completeness", aggfunc="max"
    ).add_prefix("kegg_")
    reps_df = pd.merge(reps_df, kegg_df, left_on="Name", right_index=True, how="left")
    return reps_df, kegg_df.columns.tolist()


def _finalize_reps_df(reps_df: pd.DataFrame, merged_cols: list[str]) -> pd.DataFrame:
    """Rename base columns, select final column order, and reset the index.

    Parameters
    ----------
    reps_df:
        Representative MAGs DataFrame containing a `Name` column.
    merged_cols:
        List of column names that were merged into reps_df.

    Returns
    -------
    pd.DataFrame
        Finalized reps_df with renamed columns, selected order, and reset index.
    """
    # Rename internal/technical column names to the human-friendly names used in
    # notebooks and exported tables (capitalized, tool-specific suffixes in parentheses)
    reps_df = reps_df.rename(columns={
        "Name": "MAG",
        "domain": "Domain",
        "phylum": "Phylum",
        "class": "Class",
        "order": "Order",
        "family": "Family",
        "genus": "Genus",
        "species": "Species",
        "gtdb_domain": "Domain (GTDB)",
        "gtdb_phylum": "Phylum (GTDB)",
        "gtdb_class": "Class (GTDB)",
        "gtdb_order": "Order (GTDB)",
        "gtdb_family": "Family (GTDB)",
        "gtdb_genus": "Genus (GTDB)",
        "gtdb_species": "Species (GTDB)",
        "kmetashot_superkingdom": "Domain (kmetashot)",
        "kmetashot_phylum": "Phylum (kmetashot)",
        "kmetashot_class": "Class (kmetashot)",
        "kmetashot_order": "Order (kmetashot)",
        "kmetashot_family": "Family (kmetashot)",
        "kmetashot_genus": "Genus (kmetashot)",
        "kmetashot_species": "Species (kmetashot)",
    })

    # Fixed base columns always come first, followed by whatever was merged in
    # from QUAST/Bakta/CheckM v1/CoverM/KEGG (order defined by merge_fn call order)
    all_cols = [
        "MAG",
        "Domain",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "Domain (GTDB)",
        "Phylum (GTDB)",
        "Class (GTDB)",
        "Order (GTDB)",
        "Family (GTDB)",
        "Genus (GTDB)",
        "Species (GTDB)",
        "Domain (kmetashot)",
        "Phylum (kmetashot)",
        "Class (kmetashot)",
        "Order (kmetashot)",
        "Family (kmetashot)",
        "Genus (kmetashot)",
        "Species (kmetashot)",
        "Cluster members",
        "Completeness",
        "Contamination",
    ]
    all_cols += merged_cols
    reps_df = reps_df[all_cols]
    return reps_df.reset_index(drop=True)


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
    # Expects a DataFrame already produced by `.describe()` (rows: mean/std/50%/etc.)
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
    # Convert genome length from bp to Mb for more readable summary output
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
    # 100 acts as a sentinel meaning "no filtering", since contamination can't exceed 100%
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
    # Additional thresholds are disabled but kept here for quick re-enabling if needed
    #explore_species_level_clusters(df, 5)
    #explore_species_level_clusters(df, 10)


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

    # Case-insensitive match against the literal string "unclassified" used as the fallback value
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

    # Number of representative clusters (species-level groups) per taxon at this rank
    level_counts = level_group.size().sort_values(ascending=False).to_frame("Cluster")
    level_counts["Cluster %"] = 100 * level_counts["Cluster"] / level_counts["Cluster"].sum()

    # Total number of individual MAGs (summed cluster membership) per taxon at this rank
    level_mag_counts = level_group["Cluster members"].sum().sort_values(ascending=False).to_frame(
        "Total MAG count"
    )

    level_summary = pd.concat([level_counts, level_mag_counts], axis=1)
    level_summary.sort_values(by="Total MAG count", ascending=False, inplace=True)

    # Add a TOTAL row summing all numeric columns, useful as a sanity check in reports
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
    # Keep only taxonomy columns up to Species (drop all downstream annotation columns)
    species_idx = df.columns.get_loc("Species")
    cov_taxo_df = df.copy()
    cov_taxo_df = cov_taxo_df.iloc[:, : species_idx + 1].copy()

    # Attach per-sample raw coverage values to each MAG's taxonomy
    cov_taxo_df = cov_taxo_df.merge(
        coverage_df,
        left_on="MAG",
        right_on="Genome",
        how="left",
    )
    cov_taxo_df = cov_taxo_df.drop(columns=["Genome"], errors="ignore")

    # Sum coverage within each Family/Genus/Species group (multiple MAGs can share taxonomy)
    abund_df = cov_taxo_df.groupby(["Family", "Genus", "Species"]).sum(numeric_only=True)
    mapped = abund_df.sum(axis=0)
    # "Unmapped" here represents the fraction of reads not covered by mapped MAGs (100 - mapped %)
    unmapped = 100 - mapped
    print_stats(unmapped.describe().to_frame("Unmapped reads"))
    print_stats(mapped.describe().to_frame("Mapped reads"))
    # Normalize per-sample coverage into relative abundance percentages
    abund_df = abund_df.div(mapped, axis=1) * 100
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
    # Re-aggregate the fine-grained (Family/Genus/Species) abundance table at each
    # individual taxonomic level for separate reporting
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
    # Select only the Bakta-derived columns (added by _merge_bakta) and drop the prefix
    # for cleaner display in notebooks
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
    # Missing pathway completeness means "not detected", treat as 0% complete
    kegg_path_df = kegg_path_df.fillna(0)

    print("Before removing rows and columns with only zeros:")
    print(f"Clusters: {kegg_path_df.shape[0]}")
    print(f"KEGG modules: {kegg_path_df.shape[1]}")

    # Drop pathways/MAGs with absolutely no completeness signal, to declutter downstream plots
    kegg_path_df = kegg_path_df.loc[:, (kegg_path_df != 0).any(axis=0)]
    kegg_path_df = kegg_path_df.loc[(kegg_path_df != 0).any(axis=1), :]

    print("\nAfter removing rows and columns with only zeros:")
    print(f"Clusters: {kegg_path_df.shape[0]}")
    print(f"KEGG modules: {kegg_path_df.shape[1]}")
    print()

    # Report, per MAG, how many KEGG modules have non-zero completeness (rough functional richness proxy)
    non_zero_per_row = (kegg_path_df != 0).sum(axis=1)
    print_stats(non_zero_per_row.describe().to_frame("KEGG modules"))
    return kegg_path_df


def extract_rank(classification, rank):
    """Extract a GTDB rank from a classification string.
    
    Parameters
    ----------
    classification:
        GTDB classification string (semicolon-separated).
    rank:
        Taxonomic rank to extract (one of "domain", "phylum", "class",
                "order", "family", "genus", "species").

    Returns
    -------
    str or None
        Extracted taxonomic name for the specified rank, or "unclassified" if not found, or None if the classification is NaN or the rank is invalid.
    """
    # GTDB classification strings use these standard rank prefixes, e.g. "d__Bacteria;p__..."
    prefix_map = {
        "domain": "d__",
        "phylum": "p__",
        "class": "c__",
        "order": "o__",
        "family": "f__",
        "genus": "g__",
        "species": "s__",
    }
    prefix = prefix_map.get(rank)
    if pd.isna(classification) or not prefix:
        return None

    for part in str(classification).split(";"):
        part = part.strip()
        if part.startswith(prefix):
            name = part[len(prefix):].strip()
            return name if name else "unclassified"

    # Fallback specifically for domain: some classification strings mention the domain
    # name directly without the "d__" prefix (e.g. free-text placements)
    if rank == "domain":
        cls = str(classification).strip()
        for domain_name in ["Bacteria", "Archaea"]:
            if domain_name in cls:
                return domain_name

    return "unclassified"
