# Publication-run tables (pre-campaign)

Tables downloaded from the Galaxy histories of the publication runs. They are kept
here unchanged because the `FAIRyMAGs-iwc-20261021` campaign rerun replaced the
same filenames in `data/use-cases/`. The directory layout mirrors
`data/use-cases/`:

| directory | use case |
|---|---|
| `air/megahit/`, `air/metaspades/`, `air/metadata.tsv` | air |
| `bee-gut/` | bee gut microbiome |
| `macroalgal-epiphytic/` | macroalgal epiphytic microbiome |
| `termite-head/` | termite head microbiome |

`macroalgal-epiphytic/` was restored from git history (`7461ef3^`) because that use
case had already been migrated to the campaign run before this archive was created.
All restored files were verified byte-identical to the blobs they came from.

Each per-use-case `README.md` next to the tables lists the Galaxy histories the
tables were downloaded from. The `histories.yaml`/`workflow.yaml` manifests of these
publication runs are **not** copied here, so no script can resolve a stale manifest by
accident: they remain in git history (`7461ef3^` for `macroalgal-epiphytic`; the
commit before each use case's campaign migration for the others).

Not archived, because they are not pipeline tables and did not change:

* `bee-gut/expert_evaluation.xlsx`, `bee-gut/ena_submission_samplesheet.csv`,
  `bee-gut/ena_submission.yaml` (still in `data/use-cases/bee-gut/`)
* `macroalgal-epiphytic/eggnog_annotation.tsv` (528 MB, campaign-only output)
