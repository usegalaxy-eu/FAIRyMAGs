# Bee gut microbiome use case — single-coverage binning

Histories of the `FAIRyMAGs-iwc-20261021` campaign run (single-coverage
binning, `Use original samples`):
* Input: https://usegalaxy.eu/u/paulzierep/h/bee-gut-microbiome-prjna977416
* Quality control of raw data: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33af030ef315feafee3
* Host and contaminant read removal: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33af3d172001747ec6f
* MAGs building and quality control: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33a1eca9c8457bb4afb
* Taxonomy classification: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33a0db6f257ddf43a76
* Genome annotation: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33ae1af6ea9733757d1
* Functional annotation: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33adc802449a1c3e0ed

The tables in this directory are downloaded from these histories with
`bin/download_datasets.py`; the same histories are listed with their names,
ids and links in `histories.yaml`, and the workflow links and parameters are in
`workflow.yaml`.

Open items in this run: the genome annotation history still has failed
ToolDistillator parser jobs (the published `bakta`, `intregon`, `isescan` and
`plasmidfinder` tables are complete), and the functional annotation history has
failed InterProScan jobs (the published `kegg_pathway_completeness` table is
complete). Neither failure changes a downloaded table.

`metadata.tsv` is not duplicated here: the campaign reuses the same input history,
so the curated sample table is shared with `../metadata.tsv`. The pre-campaign
publication tables are kept in `archive/data-publication/bee-gut/`, and the ENA
submission of the MAGs is documented in `../README.md`.
