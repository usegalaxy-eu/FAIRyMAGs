# Submitting FAIRyMAGs MAGs to ENA

`submission/build_samplesheet.py` turns a use case into an ENA submission and
[**nf-core/seqsubmit**](https://nf-co.re/seqsubmit) (`--mode mags`) uploads it as
**Third Party Annotation (TPA) genomes** — one ENA sample and one analysis per
MAG, all attached to a study you register first.

## Quick start

```bash
# 0. once per use case: fetch the input tables from Galaxy
python3 bin/download_datasets.py data/use-cases/<use-case>/histories.yaml

# 1. edit the submission metadata
${EDITOR:-vi} data/use-cases/<use-case>/ena_submission.yaml

# 2. build the samplesheet + download the MAG FASTAs, from the use-case dir
cd data/use-cases/<use-case>
python3 ../../../submission/build_samplesheet.py .

# 3. run the pipeline, from the same dir. Validate first (registers the samples,
#    uploads no sequences), then submit.
nextflow run /path/to/seqsubmit -profile docker -resume \
    --input samplesheet.csv --outdir out_validate --mode mags \
    --submission_study PRJEBXXXXXX --centre_name XXX --upload_tpa true \
    --test_upload false --is_private false --webincli_mode validate
nextflow run /path/to/seqsubmit -profile docker -resume \
    --input samplesheet.csv --outdir out_submit --mode mags \
    --submission_study PRJEBXXXXXX --centre_name XXX --upload_tpa true \
    --test_upload false --is_private false --webincli_mode submit
```

Details for each step are below.

## Files

```
submission/
└── build_samplesheet.py            # samplesheet + MAG FASTAs, one command
data/use-cases/<use-case>/
├── histories.yaml                  # Galaxy URL + history ids + input-table tags
├── ena_submission.yaml             # everything that is not data (you edit this)
├── checkm2.tsv coverm.tsv ...      # input tables, fetched from Galaxy (step 0)
├── samplesheet.csv                 # built by step 2            (gitignored)
└── mags/                           # MAG FASTAs, one .fasta.gz per MAG (gitignored)
```

`data/use-cases/bee-gut/single/ena_submission_samplesheet.csv` is a copy of that
`samplesheet.csv`, committed for provenance. Its `fasta` column points at absolute
paths on the submission host and the FASTAs are not in the repo, so it is not
directly re-runnable.

## Install

- `python3` with `bioblend` and `pyyaml`, e.g.
  `python3 -m venv .venv && .venv/bin/pip install bioblend pyyaml`
- `docker` and `nextflow` (≥ 22.10, for the secret store) to run the pipeline
- A Galaxy API key in `./.env` (one line, `GALAXY_API=...`) — the same file
  `bin/download_datasets.py` uses; `.env` is gitignored.

## 0. Prerequisites

**An ENA study.** `--submission_study` must be a study that already exists —
register it in the [Webin report](https://www.ebi.ac.uk/ena/submit/webin/) first.
TPA submissions are reviewed case-by-case, a study is private for at most two
years at a time (extendable), and a study releases all-or-nothing.

**The input tables.** `checkm2.tsv` (completeness/contamination), `coverm.tsv`
(coverage), `kmetashot.tsv` (NCBI lineage) and `gtdb.tsv` (lineage fallback) are
pulled from the use case's Galaxy histories:

```bash
python3 bin/download_datasets.py data/use-cases/<use-case>/histories.yaml
```

**`histories.yaml`** ties the use case to Galaxy: the URL, the history names/ids,
and which dataset tag goes to which file:

```yaml
galaxy-url: https://usegalaxy.eu
histories:
- name: Metagenome-Assembled Genomes        # matched by mags.history
  id: 11ac94870d0bb33a808dfa7d28b57a6a
  datasets:
  - tag: bin-checkM2-quality
    file: checkm2.tsv
  - tag: bin-coverm-abundance
    file: coverm.tsv
- name: MAGs taxonomy annotation
  id: 11ac94870d0bb33aee0e6dfbac468f3b
  datasets:
  - tag: kmetashot-summary
    file: kmetashot.tsv
  - tag: joined_gtdb_map
    file: gtdb.tsv
```

## 1. Describe the submission

Everything that is not data lives in `<use-case-dir>/ena_submission.yaml`:

```yaml
ena:
  study: PRJEBXXXXXX            # an ENA study that already exists (step 0)
  centre_name: XXX              # your registered submitting centre, e.g. UFR

mags:
  history: <history name>       # a history in this use case's histories.yaml
  tag: Dereplicated-Bins        # tag of the collection holding the MAGs
  # id_prefix: v2_              # optional, see "Optional: --id-prefix"

# Free text, copied into every row, so it lands in ENA as sample attributes.
# One line each - no tabs, no newlines.
assembly_software: MEGAHIT
binning_software: "MetaBAT2,MaxBin2,SemiBin2,CONCOCT,COMEBin,Binette"
binning_parameters: "<tools vX; parameters; provenance sentence>"
stats_generation_software: CheckM2
metagenome: "<ENVO term or plain text>"
broad_environment: "<ENVO term or plain text>"
local_environment: "<ENVO term or plain text>"
environmental_medium: "<ENVO term or plain text>"
```

`mags.history` is matched against a `name` in `histories.yaml`; `mags.tag` is the
tag on the collection inside that history that holds the (dereplicated) MAGs — the
dRep cluster representatives, exactly the set that is submitted.

Each row's `id` is the MAG's element identifier and must be
`<run_accession>_<binner>_bin<N>` (e.g. `SRR24759596_binette_bin20`): the leading
run accession becomes the samplesheet's `accession` (the MAG's link to the raw
reads it was assembled from) and must be a column in `coverm.tsv`, otherwise the
script stops. That same value drives the sample alias, the assembly name and the
ENA analysis alias — see `--id-prefix` below.

`co-assembly` and `RNA_presence` are filled in by the script; they are not in the
yaml.

## 2. Build the samplesheet and download the MAGs

Run it **from the use-case directory** so `samplesheet.csv` and `mags/` land next
to the yaml:

```bash
cd data/use-cases/<use-case>
python3 ../../../submission/build_samplesheet.py .
```

That one command:

1. finds the MAG FASTAs in Galaxy (the history above, `Dereplicated-Bins`
   collection);
2. downloads them to `mags/` (gzipped), with `mags/mag_map.json` recording which
   Galaxy dataset each file came from;
3. writes `samplesheet.csv` — completeness/contamination from `checkm2.tsv`,
   coverage from `coverm.tsv`, the NCBI lineage from `kmetashot.tsv` (`gtdb.tsv`
   fallback), and the free text from the yaml;
4. checks every lineage against ENA and repairs the ones ENA would reject.

Useful flags:

| flag | effect |
| --- | --- |
| `--no-download` | samplesheet only; no Galaxy, FASTAs must exist already |
| `--dry-run` | report what would happen, write nothing |
| `--fasta-dir DIR` | where the FASTAs are/land (default `mags`), written into the sheet verbatim |
| `--out FILE` | samplesheet to write (default `samplesheet.csv`) |
| `--history NAME`, `--tag TAG` | override the yaml |
| `--overwrite` | re-download FASTAs that exist already |
| `--no-fix` | fail on a lineage ENA would reject, instead of repairing it |
| `--id-prefix P` | brand the submission identity (see below) |
| `--workers N` | parallel ENA taxonomy lookups (default 8) |

Keep `--fasta-dir` relative (the default) to leave samplesheet + FASTAs as one
directory you can copy to the machine that runs Nextflow. When it finishes, the
script prints the exact Nextflow commands with your study and centre filled in.

### Why the lineage check matters

ENA only accepts a *submittable* taxon — in practice species rank or below — and
rejects the entire batch if one MAG stops at genus level:

```
<ERROR>In sample, alias: "X". Taxon Bacteria id 2 is not submittable.</ERROR>
```

The script appends the deeper taxon that sibling MAGs of the same lineage use. It
never guesses: if no sibling MAG provides a submittable continuation it stops and
lists the candidates for you to fix by hand. Nothing is registered in ENA when a
batch is rejected, so a rejected attempt costs nothing but a retry.

## 3. Run the pipeline

The samplesheet's `fasta` column is a relative path, resolved against the launch
directory, so run Nextflow **from the directory holding `samplesheet.csv` and
`mags/`** — that whole directory can be copied anywhere, e.g. to the machine that
has Docker.

### One-off set-up

```bash
git clone https://github.com/nf-core/seqsubmit.git /path/to/seqsubmit   # pin: -b <tag>

# from the FAIRyMAGs repo root:
# upstream seqsubmit treats contamination=0.0 as "no measurement" and re-runs
# CheckM2 (~1 GB per MAG) for those rows. Applies a 1-line fix; it errors if the
# line has changed upstream, and is not needed once the fix is released.
python3 submission/build_samplesheet.py --patch-seqsubmit /path/to/seqsubmit

# credentials live in the Nextflow secret store, never in the samplesheet or a
# command line:
nextflow secrets set ENA_WEBIN "Webin-XXXXXX"
nextflow secrets set ENA_WEBIN_PASSWORD "..."
```

### a. Validate

Registers the MAG samples in ENA, uploads no sequences — always run this first:

```bash
nextflow run /path/to/seqsubmit -profile docker -resume \
    --input samplesheet.csv --outdir out_validate --mode mags \
    --submission_study PRJEBXXXXXX --centre_name XXX --upload_tpa true \
    --test_upload false --is_private false --webincli_mode validate
```

### b. Submit

Uploads the FASTAs and registers the analyses:

```bash
nextflow run /path/to/seqsubmit -profile docker -resume \
    --input samplesheet.csv --outdir out_submit --mode mags \
    --submission_study PRJEBXXXXXX --centre_name XXX --upload_tpa true \
    --test_upload false --is_private false --webincli_mode submit
```

`--upload_tpa true` marks the submission as a third-party annotation and
`--test_upload false` uses the production Webin service. The rest are seqsubmit's
own flags — see its usage docs. On an HPC without Docker, use
`-profile singularity`.

### Check the result

```
out_submit/mags/<mag_id>/upload/                   # per MAG: the assembly + its ENA receipt
out_submit/mags/                                   # run level: metadata, manifests, registered MAGs
work/*/results/MAG_upload/submission_receipt.xml   # the ENA receipt
```

Count the `success="true"` receipts and compare with the number of samplesheet
rows — the difference is what `FASTA_VALIDATION` dropped (MAGs with a single
contig). The registered samples and analyses then show up at
`https://www.ebi.ac.uk/ena/browser/view/<study>`.

## Optional: `--id-prefix`

If the MAGs are regenerated later (a new pipeline version, new parameters), the
resulting ids would be the same — `<run>_<binner>_bin<N>` — and so would their
ENA aliases. `--id-prefix` (or `mags.id_prefix` in `ena_submission.yaml`) brands
the submission identity so the sets stay distinguishable:

```bash
python3 ../../../submission/build_samplesheet.py . \
    --id-prefix v2_ --fasta-dir mags_v2 --out samplesheet_v2.csv
```

The prefix is prepended to the `id` column, which nf-core/seqsubmit passes through
as `genome_name`, so it reaches every identifier this side controls:

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

## Troubleshooting

- **`💥 Invalid credentials for Webin account` during submit, while validate
  worked.** Log in at https://www.ebi.ac.uk/ena/submit/webin/ and accept the
  Access & Benefit Sharing prompt (since 2026-10-14), then re-run the submit run;
  `-resume` retries only the failed tasks.
- **`Taxon X id N is not submittable`.** The batch was rejected, nothing was
  registered. Re-run step 2 (it repairs the lineage) and submit again.
- **Fewer samples than samplesheet rows.** seqsubmit's `FASTA_VALIDATION` drops
  MAGs with a single contig — see "Check the result".
- **`X is not in coverm.tsv` / `coverm.tsv has no column for run Y`.** The MAG id
  is not `<run_accession>_...`, or that run is missing from `coverm.tsv`.
- **A registered MAG has to change.** ENA aliases are permanent. While the record
  is still private you can cancel it and re-register (then re-run), but you cannot
  rename it.

## Example: bee-gut (PRJEB128432)

738 dRep representatives were deposited as TPA genomes under
[PRJEB128432](https://www.ebi.ac.uk/ena/browser/view/PRJEB128432); their raw reads
are [PRJNA977416](https://www.ebi.ac.uk/ena/browser/view/PRJNA977416).

```yaml
ena: {study: PRJEB128432, centre_name: UFR}
mags: {history: Metagenome-Assembled Genomes, tag: Dereplicated-Bins}
metagenome: honeybee metagenome
broad_environment: apiary-biome
local_environment: pain-cages
environmental_medium: Honeybee
# assembly/binning software and parameters: see ena_submission.yaml
```

738 samplesheet rows; one lineage (`SRR24759607_binette_bin44`, genus
*Providencia*) was auto-repaired; 2 single-contig MAGs were dropped by
`FASTA_VALIDATION`, so **736 ENA samples** were registered. The result is
committed as `data/use-cases/bee-gut/single/ena_submission_samplesheet.csv`.
