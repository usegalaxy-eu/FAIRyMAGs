# Bee gut overview figure (earlier run)

`overview_figure.png` / `.svg` is the mags-visualization overview figure
(`mags-visualization overview-figure`, version 0.1.0) drawn from the bee gut tables
of the run **before** the 2026-11-22 rerun campaign (the tables one folder up).
Metadata panels: infection by *Nosema ceranae*, chronic neonicotinoid exposure and
probiotic treatment; samples labelled by condition and replicate.
Panel C leaves out the 521 MAGs that are unclassified at phylum level
(`--flow_drop_unclassified`, MAGs-visualization PR #28); the note below the panel
gives their number.

It is a preview. The manuscript version will be drawn from the rerun tables with
`bin/make_use_case_plots.sh bee-gut-microbiome/single` once taxonomy and
annotation of that run are finished.

The panels of the overview figure are also here as single plots, drawn in the same
style with `bin/make_use_case_plots.sh`:

| Panel | File |
|---|---|
| A abundance | `heatmap_overview_phylum.png` |
| B quality | `comp_conta_overview_checkm2.png` |
| C taxonomy | `sankey_plot_overview.png` (unclassified MAGs left out) |
| D function | `drep_cluster_functional_core_overview_top20.png` |

The other files are the classic versions of the same plots
(`heatmap_with_bars_phylum.png`, `comp_conta_marginals_checkm2.png`,
`sankey_plot_*.html`, `drep_cluster_*_top30*.png`, `pathway_module_heatmap_*.png`).
