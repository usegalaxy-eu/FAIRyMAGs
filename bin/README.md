# Python scripts and Notebooks

## Benchmarking

## Use cases

### Bee microbiome

The `use-case-bee-microbiome-expert-evaluation.ipynb` notebook explores expert curation results for the bee microbiome MAG use case.

- Input file: `../data/use-cases/bee-microbiome/expert_evaluation.xlsx`
- Main analyses:
	- Basic dataset overview (rows, columns, taxonomy assignment coverage)
	- Taxonomic summaries by phylum (counts, percentages, species-level MAG totals)
	- Origin-based summaries for MAGs listed vs not listed in the paper
	- Quality metric comparisons (completeness and contamination) between concordant and non-concordant MAGs
- Typical outputs:
	- Aggregated summary tables (for phylum and origin)
	- Violin/strip plots comparing completeness and contamination distributions

**Note:** Run this notebook from the `bin/` folder so relative paths to `../data/` and `../results/` resolve correctly.

