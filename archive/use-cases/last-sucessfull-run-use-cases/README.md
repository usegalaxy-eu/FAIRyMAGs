# Last successful publication run (pre-campaign)

Complete snapshot of the publication runs: the tables, the manifests that produced
them, and the ENA submission artefacts. It is kept here unchanged because the
`FAIRyMAGs-iwc-20261021` campaign rerun replaced the same filenames in
`data/use-cases/`. The directory layout mirrors `data/use-cases/`:

| directory | use case |
|---|---|
| `air/megahit/`, `air/metaspades/`, `air/metadata.tsv`, `air/workflow.yaml` | air |
| `bee-gut/` | bee gut microbiome |
| `macroalgal-epiphytic/` | macroalgal epiphytic microbiome |
| `termite-head/` | termite head microbiome |

Each use case directory holds:

* the pipeline tables (`quast`, `checkm2`, `checkm`, `coverm`, `drep`, `gtdb`,
  `kmetashot`, `bakta`, `intregon`, `isescan`, `plasmidfinder`,
  `kegg_pathway_completeness`, `metadata`)
* `histories.yaml`, the Galaxy histories the tables were downloaded from
* `workflow.yaml`, the workflow links, inputs and parameters of that run
* `README.md`, listing those history links

`bee-gut/` additionally holds the ENA submission and evaluation artefacts
(`ena_submission_samplesheet.csv`, `ena_submission.yaml`,
`expert_evaluation.xlsx`).

`macroalgal-epiphytic/` was restored from git history (`7461ef3^`) because that use
case had already been migrated to the campaign run before this archive was created.
All restored files were verified byte-identical to the blobs they came from, and the
`workflow.yaml` of every use case is the manifest captured before the migration in
`data/campaigns/iwc-20261021/original-manifests/` (byte-identical to the current
file for air, bee gut and termite head).

Not archived:

* `macroalgal-epiphytic/eggnog_annotation.tsv` — a campaign-only output, 525 MB in a
  single table, above the 100 MB GitHub file limit; download it on demand
* the campaign's own exports (`data/campaigns/iwc-20261021/{exports,tables,table-exports}`),
  which document the rerun rather than the publication run
