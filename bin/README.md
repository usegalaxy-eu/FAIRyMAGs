# Scripts and Notebooks

This folder contains the scripts and analysis notebooks used for benchmarking and use cases. 

## Usage

1. Install conda

   See the Conda installation guide: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2. Install dependencies

   ```bash
   conda create -n fairymags_env -c conda-forge -c bioconda --file requirements.txt -y
   conda activate fairymags_env
   ```

3. Run notebooks from the repository root

   ```bash
   jupyter notebook bin/use-case-analysis.ipynb
   ```

   or

   ```bash
   jupyter notebook bin/pipeline-benchmark.ipynb
   ```

4. Optional: execute notebooks non-interactively

   ```bash
   jupyter nbconvert --to notebook --execute bin/use-case-analysis.ipynb --output executed.ipynb
   ```

## Notes

- Most scripts and notebooks expect the repository root as the working directory.
- Input tables are organized under `../data/` in `benchmarking` or `use-cases` (with one subfolder per use case) folders.
- Summaries and plots are typically stored under `../results/` in `benchmarking` or `use-cases` folders.


## Benchmarking

### `pipeline-benchmark.ipynb`
Compares MAG recovery pipelines (for example metaspades individual vs. spades individual).

- compares completeness and contamination distributions across pipelines
- shows UpSet plots for MAG overlap between pipelines
- creates heatmaps and grouped bar plots of pipeline performance

| | |
|---|---|
| Input | `../data/cdb_clusters_95.tsv`, `../data/checkm2.tsv` |
| Output | `../results/heatmap_mags_benchmark.png`, `../results/heatmap_mags_benchmark.svg` |
| Output | `../results/upset_mags_benchmark.png`, `../results/upset_mags_benchmark.svg` |
| Output | `../results/mags_grouped_bar_with_values.png`, `../results/mags_grouped_bar_with_values.svg` |

## Use cases

| Use Case | Description | BioProject |
|----------|-------------|------------|
| bee-use-case | Bee gut microbiome | PRJNA977416 |
| cloud-use-case | Aeromicrobiome sampled from clouds and clear atmosphere | PRJEB54740 |
| marine-use-case | Macroalgal microbiome | PRJNA915238 |
| termite-use-case | Termite head microbiome (unpublished) | - |

### `lookup_history_ids.py`

Looks up the Galaxy history identifiers associated with the datasets used in each use case.

- resolves sample or dataset names to the corresponding Galaxy history IDs
- helps trace analyses back to archived histories for reproducibility
- supports cross-referencing metadata tables with downloadable datasets

| | |
|---|---|
| Input | `../data/use-cases/{use-case}/metadata.tsv`, `../data/use-cases/{use-case}/...` |
| Output | `../results/{use-case}/history_ids.tsv` |

### `download_datasets.py`

Downloads the raw datasets referenced by each use case from Galaxy or remote sources.

- resolves the dataset identifiers associated with each use case
- fetches the corresponding input files into the local data folders
- supports reproducible re-downloads for benchmarking and downstream analysis
- helps keep the repository data directory synchronized with the archived Galaxy histories

| | |
|---|---|
| Input | `../data/use-cases/{use-case}/metadata.tsv`, dataset identifiers or Galaxy records |
| Output | `../data/use-cases/{use-case}/...` |

### `compare_use-cases.ipynb`
Main MAG quality assessment and summary generation notebook.

- Part 1: Visualizes MAG quality distributions (completeness and contamination) across all use cases using stacked bar plots
- Part 2: Builds summary tables with:
  - total MAGs per use case
  - species-level clusters (dRep secondary clusters, >95% ANI)
  - CheckM2 quality metrics
  - GTDB taxonomy summaries (top 30 taxa, all phyla)
  - representative cluster tables with full annotations

| | |
|---|---|
| Input | `../data/use-cases/{use-case}/`: `drep.csv`, `checkm2.tsv`, `checkm.tsv`, `gtdb.tsv`, `quast.tsv`, `bakta.tsv`, `coverm.tsv`, `kegg_pathway_completeness.tsv`, `metadata.tsv` |
| Output | `../results/summary.tsv`, `../results/taxa_phyla.tsv` |
| Output | `../results/combined_bar_plot.png`, `../results/combined_bar_plot.svg` |

### `explore-<use-case>-use-case.ipynb`

Explores the <use-case> MAG use case with expert curation and dataset summaries. One notebook per use case.

| | |
|---|---|
| Input | `../data/use-cases/bee-microbiome/expert_evaluation.xlsx`, `../data/bee-use-case/metadata.tsv`, `../data/bee-use-case/coverm.tsv`, `../results/bee-use-case/reps_bee.tsv` |
| Output | summary tables, abundance barplots, PCA, quality comparisons |

Analyses:
- basic dataset overview
- taxonomic summaries by phylum
- relative abundance summaries
- origin-based summaries for MAGs listed vs. not listed in the paper
- completeness/contamination comparisons between concordant and non-concordant MAGs

#### Representative table columns

The notebooks also generate representative cluster table contains roughly 350 columns and combines information from several annotation sources.

| Category | Typical columns |
|----------|-----------------|
| Base | MAG, Domain, Phylum, Class, Order, Family, Genus, Species, Cluster members, Completeness, Contamination |
| QUAST | `# N's per 100 kbp`, `# contigs`, `# contigs (>= 1000 bp)`, `GC (%)`, `L50`, `L90`, `Largest contig`, `N50`, `N90`, `Total length`, `auN` |
| Bakta | `bakta_CDSs`, `bakta_CRISPR arrays`, `bakta_hypotheticals`, `bakta_ncRNA regions`, `bakta_rRNAs`, `bakta_sig_peptides`, `bakta_tmRNAs`, `bakta_tRNAs` |
| CheckM v1 | `Strain heterogeneity`, `# markers`, `# marker sets` |
| CoverM | mean coverage across samples |
| KEGG | one column per pathway completion value, prefixed with `kegg_` |

### Shared helper modules

`helpers.py` contains reusable notebook functions to avoid duplicating logic.

Provides:
- data loading utilities (`load_dfs`)
- quality summary utilities (`compute_print_stats`, `explore_species_level_clusters_all`)
- taxonomy summaries (`compute_taxo_classification_summary`, `get_all_taxo_levels`)
- relative abundance summaries (`get_relative_abund_taxo_levels`)
- functional summaries (`get_bakta_annot_df`, `get_kegg_path_df`)

`bee_microbiome_helpers.py` contains bee-specific helper functions for repeated summary logic.

#### Import pattern used in notebooks

```python
from pathlib import Path
import sys

sys.path.insert(0, str((Path.cwd() / "bin").resolve()))
sys.path.insert(0, str(Path.cwd().resolve()))

from helpers import ...
```


