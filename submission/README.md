# ENA submission — deposit FAIRyMAGs MAGs in ENA

Everything below was run for real for the bee-gut use case: **738 MAGs → ENA study
[PRJEB128432](https://www.ebi.ac.uk/ena/browser/view/PRJEB128432)** (Third Party
Annotation / TPA genome analyses), using [nf-core/seqsubmit](https://nf-co.re/seqsubmit).

## Install

| what | how |
|------|-----|
| Docker | works without sudo (`docker run hello-world`) |
| Nextflow + Java 17+ | `curl -s https://get.nextflow.io \| bash` |
| Python deps | `pip install bioblend pyyaml` |
| this repo | `git clone https://github.com/usegalaxy-eu/FAIRyMAGs.git` |

Credentials: never in files. Set the ENA account in the Nextflow secret store and
your Galaxy key in `.env`:

```bash
nextflow secrets set ENA_WEBIN "Webin-XXXXXX"
nextflow secrets set ENA_WEBIN_PASSWORD "your-password"
echo "GALAXY_API=your-galaxy-key" >> .env
```

## Where the samplesheet comes from

`samplesheet.csv` is the **only** input nf-core/seqsubmit needs, and it is *built*, not
hand-written. Every column comes from files already versioned in
`data/use-cases/<use-case>/`, plus a few free-text strings you choose once per use case:

| column | source |
|--------|--------|
| `id`, `fasta` | MAG name + path to the FASTA (step 4 downloads it) |
| `accession` | run accession from `metadata.tsv` |
| `completeness`, `contamination` | `checkm2.tsv` |
| `genome_coverage` | `coverm.tsv` (column = run accession) |
| `NCBI_lineage` | `kmetashot.tsv` rank columns → `;`-joined taxids (fallback `gtdb.tsv`) |
| `assembly_software` | your `--assembly-software` (MEGAHIT_v1.2.9) |
| `binning_software`, `binning_parameters`, `stats_generation_software` | `--binning-*` flags, then re-stamped from the FAIRyMAGs workflow in step 2 |
| `metagenome`, `*_environment`, `environmental_medium` | your `--metagenome` / `--broad-environment` / … |
| `RNA_presence` | left empty → seqsubmit runs tRNA/rRNA detection |

The committed
`data/use-cases/bee-gut/ena_submission_samplesheet.csv` **is** that output for the bee
use case (738 rows, one row = one dereplicated MAG). Steps 1–3 below regenerate it
byte-for-byte, including the one lineage repair ENA demanded.

---

## Steps (bee-gut example)

> If your use case already has a committed samplesheet (bee-gut does), you can skip
> steps 1–3 and go straight to step 4 — steps 1–3 are how that file was made.

### 1. Build the samplesheet

```bash
python3 submission/scripts/02_build_samplesheet.py \
  --use-case-dir data/use-cases/bee-gut \
  --fasta-dir /vol/data/ena_bee_mags/fasta \
  --out data/use-cases/bee-gut/ena_submission_samplesheet.csv \
  --expected-rows 738 \
  --binning-software "MetaBAT2_v2.18.23,MaxBin2_v2.2.7,SemiBin2_v2.1.0,CONCOCT_v1.1.0,COMEBin_v1.0.4,Binette_v1.2.1" \
  --binning-parameters "defaults except: SemiBin2 environment=global; CONCOCT read_length=150; Binette contamination_weight=2; MAG filters min_length=50000 bp, min_completeness=0%, max_contamination=100%; dRep dereplication ANI=0.95; MEGAHIT min_contig_length=200 bp. MAGs generated with the FAIRyMAGs workflows (https://github.com/usegalaxy-eu/FAIRyMAGs) on usegalaxy.eu" \
  --stats-generation-software CheckM2_v1.1.0 \
  --assembly-software MEGAHIT_v1.2.9 \
  --metagenome "honeybee metagenome" \
  --broad-environment apiary-biome \
  --local-environment pain-cages \
  --environmental-medium Honeybee
```

No FASTA files are needed yet — step 1 only writes *paths*. `--fasta-dir` is the
directory where step 4 will put them (on the VM that was `/vol/data/ena_bee_mags/fasta`).
Use an **absolute** path: seqsubmit resolves the `fasta` column from its own work dir.

### 2. Stamp the binning provenance

Step 1 already writes these three columns, but they are free text — step 2 re-stamps them
from the FAIRyMAGs workflow invocation (per-step tool ids + workflow inputs), so the
submitted metadata states exactly which tools produced the MAGs:

```bash
python3 submission/scripts/03_update_binning_metadata.py \
  data/use-cases/bee-gut/ena_submission_samplesheet.csv
```

### 3. Taxonomy pre-flight — every lineage must end at a *submittable* taxon

ENA only accepts species rank or below. This is the check that caught the one bad row of
the bee-gut sheet (`...;586` = genus *Providencia*, ENA replied
`Taxon Bacteria id 2 is not submittable`):

```bash
python3 submission/scripts/04_check_ena_taxa.py data/use-cases/bee-gut/ena_submission_samplesheet.csv
```

If it flags a row, fix it and re-check (bee-gut: append species taxid 589 → *Providencia sp.*):

```bash
python3 submission/scripts/05_fix_lineage.py data/use-cases/bee-gut/ena_submission_samplesheet.csv \
  --id SRR24759607_binette_bin44 --append 589
python3 submission/scripts/04_check_ena_taxa.py data/use-cases/bee-gut/ena_submission_samplesheet.csv
```

### 4. Fetch the MAG FASTAs from Galaxy

Now the samplesheet exists, so use it to select exactly the MAGs to download:

```bash
python3 submission/scripts/01_fetch_mags_from_galaxy.py \
  --history 11ac94870d0bb33a808dfa7d28b57a6a \
  --histories-yaml data/use-cases/bee-gut/histories.yaml \
  --ids-from data/use-cases/bee-gut/ena_submission_samplesheet.csv \
  --outdir /vol/data/ena_bee_mags/fasta
```

Writes `<mag-id>.fasta.gz` per MAG plus `mag_map.json` (provenance: MAG → Galaxy dataset
id). `--dry-run` reports only, `--overwrite` re-downloads.

### 5. Apply the seqsubmit fix (once per checkout)

Needed because upstream treats `contamination=0.0` as "missing":

```bash
python3 submission/scripts/patch_seqsubmit_zero_stats.py /vol/data/tools/seqsubmit
```

### 6. Validate — registers the samples in ENA, uploads nothing

Create a run dir holding the samplesheet the pipeline will read, then run it in
`validate` mode:

```bash
export NXF_HOME=/vol/data/tools/nxf_home
mkdir -p /vol/data/seqsubmit_runs/real2
cp data/use-cases/bee-gut/ena_submission_samplesheet.csv \
   /vol/data/seqsubmit_runs/real2/samplesheet.csv

submission/run/run_seqsubmit.sh validate /vol/data/tools/seqsubmit /vol/data/seqsubmit_runs/real2
```

`params.validate.json`: `webincli_mode=validate`, `test_upload=false`, `upload_tpa=true`,
`submission_study=PRJEB128432`, `centre_name=UFR`. The FASTAs fetched in step 4 must
already exist at the paths in the `fasta` column. Result: 738 manifests (736 after the
pipeline drops 2 single-contig MAGs), 0 failures.

### 7. Submit — upload the FASTAs, collect the accessions

```bash
submission/run/run_seqsubmit.sh submit /vol/data/tools/seqsubmit /vol/data/seqsubmit_runs/real2
```

The script runs Nextflow with `-resume`, so only the `SUBMIT` tasks re-run (everything
else is cached). Afterwards:

- `out/` → `*_accessions.tsv` per MAG
- `work/*/results/MAG_upload/submission_receipt.xml` → ENA receipt
- analyses appear at `https://www.ebi.ac.uk/ena/submit/webin/report/analyses/ERP207843`

## Troubleshooting

- **`💥 Invalid credentials for Webin account` during submit** (but validate passed):
  ENA rolled out an Access & Benefit Sharing acknowledgement on 2026-10-14 — log in to
  https://www.ebi.ac.uk/ena/submit/webin/, accept the prompt, wait ~10 min, re-run step 7.
- **`Taxon X id N is not submittable`**: ENA rejected the whole batch atomically → fix the
  lineage (step 3) and re-run; nothing was registered.
- **Test server instead of production**: set `test_upload=true` in `run/params.validate.json`
  and `run/params.submit.json` (the test account rejects `centre_name`, so production is
  what we used).
