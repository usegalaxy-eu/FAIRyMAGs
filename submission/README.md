# Submitting FAIRyMAGs MAGs to ENA

The 738 dereplicated MAGs of the bee-gut use case were deposited in ENA as
**Third Party Annotation (TPA) genomes** (study
[PRJEB128432](https://www.ebi.ac.uk/ena/browser/view/PRJEB128432)) with
[nf-core/seqsubmit](https://nf-co.re/seqsubmit). This directory holds the one
script that prepares that submission.

```
submission/
└── build_samplesheet.py        # samplesheet + MAG FASTAs, in one command
data/use-cases/bee-gut/
├── ena_submission.yaml         # everything that is not data (read by the script)
└── ena_submission_samplesheet.csv   # the result, as submitted
```

## Install

- `python3` with `bioblend` and `pyyaml`: `pip install bioblend pyyaml`
- `docker` and `nextflow` (Java 17+) to run the pipeline
- A Galaxy API key in `./.env` (one line, `GALAXY_API=...`) — the same file
  `bin/download_datasets.py` uses. It is needed to read the MAGs from Galaxy;
  `.env` is gitignored.

## 1. Build the samplesheet

```bash
python3 submission/build_samplesheet.py data/use-cases/bee-gut
```

That one command:

1. finds the MAG FASTAs in Galaxy — the history named in `ena_submission.yaml`
   (`Metagenome-Assembled Genomes`) is looked up in the use case's
   `histories.yaml`, and the collection tagged **`Dereplicated-Bins`** in it is
   used (the dRep cluster representatives: exactly the 738 MAGs that get
   submitted);
2. downloads them to `mags/` (gzipped), with `mags/mag_map.json` recording
   which Galaxy dataset each file came from;
3. writes `samplesheet.csv` — completeness/contamination from `checkm2.tsv`,
   coverage from `coverm.tsv`, the NCBI lineage from `kmetashot.tsv`
   (`gtdb.tsv` for the not-classified fallback), and the free text from
   `ena_submission.yaml`;
4. checks every lineage against ENA and repairs the ones ENA would reject.

Expected output:

```
use case    : data/use-cases/bee-gut (738 MAGs in checkm2.tsv)
history     : Metagenome-Assembled Genomes (11ac94870d0bb33a808dfa7d28b57a6a)
collection  : dereplicated_genomes (e1d57ec973deed26, tag Dereplicated-Bins)
MAG FASTA files in collection: 738
  738/738  downloaded=738 skipped=0 failed=0
FASTAs      : 738 downloaded, 0 already there, 0 failed -> mags/
lineages    : 507 numeric taxid chains, 231 placeholders {'Unclassified Bacteria': 229, 'Unclassified Archaea': 2}
              querying ENA for 75 distinct final taxids ...
              NOT submittable: 586 (Providencia, rank genus)
auto-fixed  : SRR24759607_binette_bin44
                2;1224;1236;91347;1903414;586
             -> 2;1224;1236;91347;1903414;586;589
wrote       : samplesheet.csv (738 rows, 19 columns)
```

**Why step 4 matters.** ENA only accepts a *submittable* taxon, in practice
species rank or below, and rejects the entire batch if one MAG stops at genus
level:

```
<ERROR>In sample, alias: "SRR24759607_binette_bin44". Taxon Bacteria id 2 is not submittable.</ERROR>
```

One bee-gut MAG was classified only to genus *Providencia* (`...;586`).
Sibling MAGs of that lineage continue to `589` (*Providencia sp.*), which the
script picks up and appends. It never guesses: if no sibling MAG provides a
submittable continuation it stops and shows you the candidates. Nothing is
registered in ENA when a batch is rejected, so a rejected attempt costs
nothing but a retry.

Useful flags: `--no-download` (samplesheet only, no Galaxy),
`--dry-run` (write nothing), `--fasta-dir` (default `mags`),
`--history` / `--tag` (override the yaml), `--no-fix` (fail instead of repair),
`--id-prefix` (see below).

### Telling submissions apart with `--id-prefix`

If the MAGs are regenerated later (a new pipeline version, new parameters), the
resulting ids would be the same — `<run>_<binner>_bin<N>` — and so would their
ENA aliases. `--id-prefix` (or `mags.id_prefix` in `ena_submission.yaml`) brands
the submission identity so the two sets stay distinguishable in ENA:

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
`A-Za-z0-9_.-`; the assembly name is public once the record is released, and an
alias can never be edited after registration, so choose the prefix once and
keep it. For finer provenance (and anything machine-readable), the study's
`binning_parameters` free text in `ena_submission.yaml` also lands in ENA as a
sample attribute — that is where the FAIRyMAGs provenance sentence already is.

## 2. Run the pipeline

The samplesheet's `fasta` column points at `mags/<id>.fasta.gz`, so run
Nextflow **from the directory holding `samplesheet.csv` and `mags/`** — that
whole directory can be copied anywhere, e.g. to the machine that has Docker.

```bash
git clone https://github.com/nf-core/seqsubmit.git /path/to/seqsubmit

# one-off: seqsubmit treats contamination=0.0 as "no measurement" and re-runs
# CheckM2 (~1 GB per MAG). 81 of the 738 MAGs are 0.0. This applies a 1-line fix.
python3 submission/build_samplesheet.py --patch-seqsubmit /path/to/seqsubmit

# 1. validate: registers the MAG samples in ENA, uploads no sequences
nextflow run /path/to/seqsubmit -profile docker -resume \
    --input samplesheet.csv --outdir out_validate --mode mags \
    --submission_study PRJEB128432 --centre_name UFR --upload_tpa true \
    --test_upload false --is_private false --webincli_mode validate

# 2. submit: uploads the FASTAs and returns the accessions
nextflow run /path/to/seqsubmit -profile docker -resume \
    --input samplesheet.csv --outdir out_submit --mode mags \
    --submission_study PRJEB128432 --centre_name UFR --upload_tpa true \
    --test_upload false --is_private false --webincli_mode submit
```

Credentials come from the Nextflow secret store, never from the samplesheet or
a command line:

```bash
nextflow secrets set ENA_WEBIN "Webin-XXXXXX"
nextflow secrets set ENA_WEBIN_PASSWORD "..."
```

Results land in `out/`: one `*_accessions.tsv` per MAG, plus the ENA receipt in
`work/*/results/MAG_upload/submission_receipt.xml`. The analyses then show up at
`https://www.ebi.ac.uk/ena/submit/webin/report/analyses/<study>`.

## Troubleshooting

- **`💥 Invalid credentials for Webin account` during submit, while validate
  worked.** Log in at https://www.ebi.ac.uk/ena/submit/webin/ and accept the
  Access & Benefit Sharing prompt (rolled out 2026-10-14), then re-run step 2;
  `-resume` retries only the failed tasks.
- **`Taxon X id N is not submittable`.** The batch was rejected, nothing was
  registered. Re-run step 1 (it repairs the lineage) and submit again.
- **Two MAGs fewer than expected.** seqsubmit's `FASTA_VALIDATION` drops MAGs
  with a single contig, so 738 rows produce 736 ENA samples.
- **Re-submitting a MAG that is already in ENA.** `-resume` reuses the
  registration; the alias (`<run>_<binner>_bin<N>`) must stay unique, so do not
  rename MAGs between attempts.
