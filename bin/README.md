# FAIRyMAGs Analysis Notebooks

## Notebooks

### `use-case-analysis.ipynb`
MAGs quality assessment and summary table generation:
- **Part 1**: Visualizes MAGs quality distribution (completeness/contamination) across all use cases using stacked bar plots
- **Part 2**: Generates summary tables with:
  - Total MAGs per use case
  - Species-level clusters (dRep secondary clusters, >95% ANI)
  - CheckM2 quality metrics (completeness/contamination)
  - GTDB taxonomy (top 30 taxa, all phyla)
  - Cluster representative tables with full annotations

| | |
|---|---|
| **Input** | `../data/{use-case}/`: `drep.csv`, `checkm2.tsv`, `checkm.tsv`, `gtdb.tsv`, `quast.tsv`, `bakta.tsv`, `coverm.tsv`, `kegg_pathway_completeness.tsv`, `metadata.tsv` |
| **Output** | `../results/{use-case}/reps_{use-case}.tsv` (~350 columns per use case) |
| **Output** | `../results/summary.tsv`, `../results/taxa_phyla.tsv` |
| **Output** | `../results/combined_bar_plot.png/svg` |

#### Cluster representative table columns (~350 total)

| Category | Columns |
|----------|---------|
| **Base** | MAG, Domain, Phylum, Class, Order, Family, Genus, Species, Cluster members, Completeness, Contamination |
| **QUAST** | # N's per 100 kbp, # contigs, # contigs (>= 0 bp), # contigs (>= 1000 bp), GC (%), L50, L90, Largest contig, N50, N90, Total length, Total length (>= 0 bp), Total length (>= 1000 bp), auN |
| **Bakta** | CDSs, CRISPR arrays, gaps, hypotheticals, ncRNA regions, rRNAs, sig_peptides, tmRNAs, tRNAs (prefixed with `bakta_`) |
| **CheckM v1** | Strain heterogeneity, # markers, # marker sets |
| **CoverM** | Mean coverage across samples |
| **KEGG** | One column per pathway with completeness values (prefixed with `kegg_`) |

---

### `pipeline-benchmark.ipynb`
Compares different MAGs recovery pipelines (metaspades individual, spades individual, etc.):
- Compares completeness/contamination distributions across pipelines
- UpSet plots showing MAG overlap between pipelines
- Heatmaps of pipeline performance

| | |
|---|---|
| **Input** | `../data/cdb_clusters_95.tsv`, `../data/checkm2.tsv` |
| **Output** | `../results/heatmap_mags_benchmark.png/svg` |
| **Output** | `../results/upset_mags_benchmark.png/svg` |
| **Output** | `../results/mags_grouped_bar_with_values.png/svg` |

---

### Use Case Plots

Plots for individual use cases are created in a separate repository:
- **GitHub**: https://github.com/usegalaxy-eu/MAGs-visualization/tree/main/use-cases
- These include detailed visualizations like heatmaps, Sankey plots, functional annotations, and more

---

### `use-case-bee-microbiome-expert-evaluation.ipynb`
Explores expert curation results for the bee microbiome MAG use case.

| | |
|---|---|
| **Input** | `../data/use-cases/bee-microbiome/expert_evaluation.xlsx` |
| **Output** | Summary tables, violin/strip plots comparing completeness/contamination distributions |

**Analyses:**
- Basic dataset overview (rows, columns, taxonomy assignment coverage)
- Taxonomic summaries by phylum (counts, percentages, species-level MAG totals)
- Origin-based summaries for MAGs listed vs not listed in the paper
- Quality metric comparisons (completeness and contamination) between concordant and non-concordant MAGs

---

### `use-case-aeromicrobiome-exploration.ipynb`
Explores the aeromicrobiome (cloud-use-case) dataset from MAG quality to taxonomy, abundance, and functional summaries.

| | |
|---|---|
| **Input** | `../data/cloud-use-case/`: `metadata.tsv`, `coverm.tsv` |
| **Input** | `../results/cloud-use-case/reps_cloud.tsv` |
| **Output** | In-notebook summary tables for HQ/MQ/LQ clusters, taxonomy classification, and abundance statistics |
| **Output** | Functional summary tables derived from Bakta and KEGG annotations |

**Analyses:**
- Data loading and preprocessing of representative cluster table (`reps_cloud.tsv`)
- MAG quality summaries and MIMAG-like partitioning into HQ, MQ, and LQ subsets
- Taxonomy classification summaries from Domain to Species, including classified/unclassified counts and percentages
- Taxonomic level summaries (cluster counts and MAG counts) across all ranks
- Relative abundance summaries aggregated at Family, Genus, and Species levels using CoverM coverage
- Functional annotation summaries using Bakta features and KEGG pathway completeness columns

## Files

| File | Description |
|------|-------------|
| `use-case-analysis.ipynb` | Main analysis notebook for MAGs quality and summary tables |
| `pipeline-benchmark.ipynb` | Pipeline comparison notebook |
| `use-case-bee-microbiome-expert-evaluation.ipynb` | Expert curation evaluation for bee microbiome |
| `use-case-aeromicrobiome-exploration.ipynb` | Aeromicrobiome exploratory analysis (quality, taxonomy, abundance, and function) |
| `README.md` | This file |

## Use Cases

| Use Case | Description | BioProject |
|----------|-------------|------------|
| bee-use-case | Bee gut microbiome | PRJNA977416 |
| cloud-use-case | Aeromicrobiome sampled from clouds and clear atmosphere | PRJEB54740 |
| marine-use-case | Macroalgal microbiome | PRJNA915238 |
| termite-use-case | Termite head microbiome (unpublished) | - |

---

## Run Analysis Notebooks

1. **Install conda**

   * [Conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. **Install dependencies**

   ```bash
   conda create -n fairymags_env -c conda-forge -c bioconda --file requirements.txt -y
   conda activate fairymags_env
   ```

3. **Run notebooks**

   From the repository root:

   ```bash
   jupyter notebook bin/use-case-analysis.ipynb
   ```

   or

   ```bash
   jupyter notebook bin/pipeline-benchmark.ipynb
   ```

4. **Run from command line (optional)**

   ```bash
   jupyter nbconvert --to notebook --execute bin/use-case-analysis.ipynb --output executed.ipynb
   ```
