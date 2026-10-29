# Bee gut microbiome use case — single-coverage binning

* Input: https://usegalaxy.eu/u/paulzierep/h/bee-gut-microbiome-prjna977416
* Quality control of raw data: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33af030ef315feafee3
* Host and contaminant read removal: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33af3d172001747ec6f
* MAGs building and quality control: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33a1eca9c8457bb4afb
* Taxonomy classification: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33a0db6f257ddf43a76
* Genome annotation: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33ae1af6ea9733757d1
* Functional annotation: https://usegalaxy.eu/histories/view?id=11ac94870d0bb33adc802449a1c3e0ed

The tables are downloaded from these histories with `bin/download_datasets.py`
(`histories.yaml` lists them with their tags and links, `workflow.yaml` the workflow
links and parameters). `metadata.tsv` (the curated sample table), 
`expert_evaluation.xlsx` and the ENA submission files (`ena_submission.yaml`,
`ena_submission_samplesheet.csv`) are inputs rather than pipeline outputs, so they are
kept here too.

The pre-campaign publication tables are archived in
`archive/use-cases/last-sucessfull-run-use-cases/bee-gut/`.
