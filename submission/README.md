# Submitting FAIRyMAGs MAGs to ENA

One script turns a use case directory into an ENA samplesheet;
[**nf-core/seqsubmit**](https://nf-co.re/seqsubmit) (`--mode mags`) uploads the
MAGs to ENA as **Third Party Annotation (TPA) genomes**.

```
submission/
└── build_samplesheet.py             # samplesheet + MAG FASTAs, one command
data/use-cases/<use-case>/
├── ena_submission.yaml              # everything that is not data (you edit this)
└── ena_submission_samplesheet.csv   # the result (written for you)
```

## Install

- `python3` with `bioblend` and `pyyaml`: `pip install bioblend pyyaml`
- `docker` and `nextflow` (Java 17+) — to run the pipeline
- A Galaxy API key in `./.env` (one line, `GALAXY_API=...`) — the same file
  `bin/download_datasets.py` uses. Needed to read the MAGs from Galaxy;
  `.env` is gitignored.

## 1. Describe the submission

Everything that is not data lives in `<use-case-dir>/ena_submission.yaml`:

```yaml
ena:
  study: PRJEBXXXXXX            # the ENA study the MAGs belong to
  centre_name: XXX              # submitting centre / broker code

mags:
  history: <history name>       # a history in this use case's histories.yaml
  tag: Dereplicated-Bins        # tag of the collection holding the MAGs
  # id_prefix: v2_              # optional, see below

# Free text: copied into every row, so it lands in ENA as sample attributes.
assembly_software: MEGAHIT
binning_software: "MetaBAT2,MaxBin2,SemiBin2,CONCOCT,COMEBin,Binette"
binning_parameters: ""          # parameters used + a provenance sentence
stats_generation_software: CheckM2
metagenome: "<sample metagenome>"
broad_environment: ...
local_environment: ...
environmental_medium: ...
```

The same directory must already hold the tables the script reads: `checkm2.tsv`
(completeness/contamination), `coverm.tsv` (coverage), `kmetashot.tsv` (NCBI
lineage, with `gtdb.tsv` as fallback) and `histories.yaml` (Galaxy URL and the
ids of the use case's histories).

## 2. Build the samplesheet and download the MAGs

```bash
python3 submission/build_samplesheet.py data/use-cases/<use-case>
```

That one command:

1. finds the MAG FASTAs in Galaxy — the history named in the yaml is looked up
   in `histories.yaml`, and the collection tagged **`Dereplicated-Bins`** in it
   is used (the dRep cluster representatives: exactly the MAGs that get
   submitted);
2. downloads them to `mags/` (gzipped), with `mags/mag_map.json` recording which
   Galaxy dataset each file came from;
3. writes `samplesheet.csv` — completeness/contamination from `checkm2.tsv`,
   coverage from `coverm.tsv`, the NCBI lineage from `kmetashot.tsv`
   (`gtdb.tsv` for the not-classified fallback), and the free text from the yaml;
4. checks every lineage against ENA and repairs the ones ENA would reject.

Flags: `--no-download` (samplesheet only, no Galaxy), `--dry-run` (write
nothing), `--fasta-dir` (default `mags`), `--history` / `--tag` (override the
yaml), `--no-fix` (fail instead of repair), `--id-prefix` (see below).
The script prints the exact Nextflow commands to run when it is done.

### Why step 4 exists

ENA only accepts a *submittable* taxon, in practice species rank or below, and
rejects the entire batch if one MAG stops at genus level:

```
<ERROR>In sample, alias: "X". Taxon Bacteria id 2 is not submittable.</ERROR>
```

The script appends the deeper taxon that sibling MAGs of the same lineage use. It
never guesses: if no sibling MAG provides a submittable continuation it stops and
shows the candidates. Nothing is registered in ENA when a batch is rejected, so a
rejected attempt costs nothing but a retry.

### Telling submissions apart with `--id-prefix`

If the MAGs are regenerated later (a new pipeline version, new parameters), the
resulting ids would be the same — `<run>_<binner>_bin<N>` — and so would their
ENA aliases. `--id-prefix` (or `mags.id_prefix` in `ena_submission.yaml`) brands
the submission identity so the sets stay distinguishable:

```bash
python3 submission/build_samplesheet.py <use-case-dir> \
    --id-prefix v2_ --fasta-dir mags_v2 --out samplesheet_v2.csv
```

The prefix is prepended to the `id` column, which nf-core/seqsubmit passes
through as `genome_name`, so it reaches every identifier this side controls:

| ENA record | without prefix | with `--id-prefix v2_` |
| --- | --- | --- |
| sample alias / title | `SRR24759596_binette_bin20` | `v2_SRR24759596_binette_bin20` |
| assembly name (`ASSEMBLYNAME`) | `SRR24759604_binette_bin91` | `v2_SRR24759604_binette_bin91` |
| analysis alias (set by Webin-CLI) | `webin-genome-SRR24759604_binette_bin91` | `webin-genome-v2_SRR24759604_binette_bin91` |

The `webin-genome-` part is added by ENA's Webin-CLI and cannot be changed. The
**FASTA file names are left alone** — ENA keeps the assembly name and the FASTA
file name separate, so nothing has to be renamed or re-downloaded. Use only
`A-Za-z0-9_.-`; the assembly name is public once the record is released and an
alias can never be edited after registration, so pick the prefix once. For finer
provenance, the `binning_parameters` free text also lands in ENA as a sample
attribute.

## 3. Run the pipeline

The samplesheet's `fasta` column points at `mags/<id>.fasta.gz`, so run Nextflow
**from the directory holding `samplesheet.csv` and `mags/`** — that whole
directory can be copied anywhere, e.g. to the machine that has Docker.

```bash
git clone https://github.com/nf-core/seqsubmit.git /path/to/seqsubmit

# one-off: upstream seqsubmit treats contamination=0.0 as "no measurement" and
# re-runs CheckM2 (~1 GB per MAG) for those rows. Applies a 1-line fix.
python3 submission/build_samplesheet.py --patch-seqsubmit /path/to/seqsubmit
```

Credentials come from the Nextflow secret store, never from the samplesheet or a
command line:

```bash
nextflow secrets set ENA_WEBIN "Webin-XXXXXX"
nextflow secrets set ENA_WEBIN_PASSWORD "..."
```

**a. Validate** — registers the MAG samples in ENA, uploads no sequences
(always run this first):

```bash
nextflow run /path/to/seqsubmit -profile docker -resume \
    --input samplesheet.csv --outdir out_validate --mode mags \
    --submission_study PRJEBXXXXXX --centre_name XXX --upload_tpa true \
    --test_upload false --is_private false --webincli_mode validate
```

**b. Submit** — uploads the FASTAs and returns the accessions:

```bash
nextflow run /path/to/seqsubmit -profile docker -resume \
    --input samplesheet.csv --outdir out_submit --mode mags \
    --submission_study PRJEBXXXXXX --centre_name XXX --upload_tpa true \
    --test_upload false --is_private false --webincli_mode submit
```

Results land in `out/`: one `*_accessions.tsv` per MAG, plus the ENA receipt in
`work/*/results/MAG_upload/submission_receipt.xml`. The analyses then show up at
`https://www.ebi.ac.uk/ena/submit/webin/report/analyses/<study>`.

## Troubleshooting

- **`💥 Invalid credentials for Webin account` during submit, while validate
  worked.** Log in at https://www.ebi.ac.uk/ena/submit/webin/ and accept the
  Access & Benefit Sharing prompt (since 2026-10-14), then re-run step 3b;
  `-resume` retries only the failed tasks.
- **`Taxon X id N is not submittable`.** The batch was rejected, nothing was
  registered. Re-run step 2 (it repairs the lineage) and submit again.
- **Fewer MAGs than samplesheet rows.** seqsubmit's `FASTA_VALIDATION` drops
  MAGs with a single contig.
- **Re-submitting a MAG that is already in ENA.** `-resume` reuses the
  registration; the alias (`<run>_<binner>_bin<N>`) must stay unique, so do not
  rename MAGs between attempts.

## Example: bee-gut (PRJEB128432)

738 dRep representatives were deposited as TPA genomes
([PRJEB128432](https://www.ebi.ac.uk/ena/browser/view/PRJEB128432)):

```yaml
ena:
  study: PRJEB128432
  centre_name: UFR
mags:
  history: Metagenome-Assembled Genomes
  tag: Dereplicated-Bins
assembly_software: MEGAHIT
binning_software: "MetaBAT2,MaxBin2,SemiBin2,CONCOCT,COMEBin,Binette"
binning_parameters: "..."                # parameters + FAIRyMAGs provenance sentence
stats_generation_software: CheckM2
metagenome: honeybee metagenome
broad_environment: ...                   # see data/use-cases/bee-gut/ena_submission.yaml
local_environment: ...
environmental_medium: ...
```

```bash
python3 submission/build_samplesheet.py data/use-cases/bee-gut
```

738 samplesheet rows; one lineage (`SRR24759607_binette_bin44`, genus
*Providencia*) was auto-repaired; 2 MAGs with a single contig were dropped by
`FASTA_VALIDATION`, so **736 ENA samples** were registered. The result is
committed as `data/use-cases/bee-gut/ena_submission_samplesheet.csv`.
