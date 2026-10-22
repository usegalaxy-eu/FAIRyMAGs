# FAIRyMAGs IWC rerun campaign — 2026-10-21

## Frozen design

- Six workflow snapshots, with Git source commits, releases and checksums in `campaign.yaml`.
- Parent from PR #1454; other workflows from merged IWC main.
- All five binners enabled; Binette refinement; historical quality thresholds retained.
- Bee: single-coverage and extra multi-coverage branches after shared QC and bee-host removal.
- Termite: QC → Csec reference removal → Cdom reference removal → grouped and individual assembly branches. Grouped branch reuses the historical 28-sample → 8-group metadata mapping, not historical grouped FASTQs.
- Macroalgal: QC → human removal using hg38full → MAGs and annotation.
- Air: shared QC and human removal; MEGAHIT retains 14 samples, metaSPAdes excludes ERR9966622 using Galaxy Apply Rules and retains 13 samples.
- Every Parent branch has independent taxonomy, genome and functional annotation histories.
- Merged genome annotation 0.6 has no ToolDistillator switch. It is unchanged; ToolDistillator HTML/report outputs are not required for biological-result acceptance.
- Original manifests are preserved in `original-manifests/`; removed manual CONCOCT read-length values and effective Parent parameters are in `parameter-migrations.json`.

## Operation

The scheduler is `bin/run_workflows.py ... --campaign`; the independent exporter uses the same entry point with `--export-only` and calls `bin/download_datasets.py` for allowlisted table exports. `download-policy.yaml` points to each existing use-case `histories.yaml`: only its listed datasets are downloaded from the corresponding NEW campaign history. Additional Bee/Termite branches use the same use-case table list. QC, host-removal, FASTQ, assembly, MAG FASTA and unlisted report outputs are never downloaded.

```bash
systemctl --user status fairymags-campaign fairymags-downloads
journalctl --user -u fairymags-campaign -u fairymags-downloads -n 100
systemctl --user restart fairymags-campaign fairymags-downloads
```

User lingering is enabled so both enabled services survive logout and restart after boot. Network reads have bounded retries. Submission POSTs are never blindly retried: an interrupted `creating`/`submitting` checkpoint is blocked for reconciliation to avoid duplicate expensive runs.

## State and results

- `STATUS.md`: scheduler status and EU history links.
- `state.json`: imported workflows, submitted histories/invocations, parameters and job states. Do not edit while the scheduler runs.
- `invocations/`: latest raw invocation responses.
- `exports/`: historical checkpoints from the superseded broad exporter; not used for acceptance.
- `table-exports/`: current per-stage table checkpoints and missing-table reports.
- `tables/<use-case>/<stage>/<filename>`: only tables listed in `histories.yaml`, with dataset IDs, sizes and SHA-256 checksums in adjacent provenance files. Collections concatenate only when explicitly requested.
- `download-cleanup.json`: records removal of the unrequested broad exports on 2026-10-22; original published results and remote data were untouched.

Old use-case results are not overwritten. Existing Galaxy inputs are reused, never re-uploaded. Cache reuse is requested for every invocation but is not guaranteed for changed tools/inputs.

## Acceptance and recovery

Completion requires every stage to reach a terminal state and every table selected by `download-policy.yaml` / `histories.yaml` to be downloaded successfully. Workflow `required_outputs` are scientific dependency declarations, not a download list. `complete-with-errors` is not scientific acceptance: inspect the recorded failed jobs and missing required tables. Independent branches can continue, but a dependent workflow only receives checked, populated, non-null input datasets.

No automatic scientific repairs, assertion relaxation, database substitutions or expensive failed-analysis reruns occur. Investigate `blocked` stages before resuming. To resume with an assistant, request: “Resume FAIRyMAGs campaign at data/campaigns/iwc-20261021; inspect STATUS.md, state.json, exports and service logs.”

The service provides durable execution, not guaranteed autonomous resolution of scientific/server failures. No external notification channel is configured.
