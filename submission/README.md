# ENA submission (short version)

Prerequisites: Docker, Nextflow (Java 17+), git, Python 3. Install: `pip install bioblend pyyaml`.

## 1. Get the repo & checkout
```bash
git clone https://github.com/usegalaxy-eu/FAIRyMAGs.git
cd FAIRyMAGs
git checkout add-bee-gut-ena-samplesheet  # or your branch
```

## 2. Set credentials
```bash
export NXF_HOME=${NXF_HOME:-$HOME/.nextflow}
nextflow secrets set ENA_WEBIN "Webin-XXXXXX"
nextflow secrets set ENA_WEBIN_PASSWORD "YOUR-PASSWORD"
nextflow secrets list
```

Add your Galaxy API key to `.env`:
```bash
echo "GALAXY_API=YOUR-KEY" >> .env
```

## 3. Get MAG FASTAs from Galaxy (bee gut example)
```bash
python3 submission/scripts/01_fetch_mags_from_galaxy.py \
  --history 11ac94870d0bb33a808dfa7d28b57a6a \
  --histories-yaml data/use-cases/bee-gut/histories.yaml \
  --ids-from data/use-cases/bee-gut/ena_submission_samplesheet.csv \
  --outdir mags
```

## 4. Build the samplesheet (bee gut example)
```bash
python3 submission/scripts/02_build_samplesheet.py \
  --use-case-dir data/use-cases/bee-gut \
  --fasta-dir ./mags \
  --out samplesheet.csv \
  --expected-rows 738 \
  --binning-software "MetaBAT2_v2.18.23,MaxBin2_v2.2.7,SemiBin2_v2.1.0,CONCOCT_v1.1.0,COMEBin_v1.0.4,Binette_v1.2.1" \
  --binning-parameters "defaults except: SemiBin2 environment=global; CONCOCT read_length=150; Binette contamination_weight=2; MAG filters min_length=50000 bp, min_completeness=0%, max_contamination=100%; dRep dereplication ANI=0.95; MEGAHIT min_contig_length=200 bp. MAGs generated with the FAIRyMAGs workflows (https://github.com/usegalaxy-eu/FAIRyMAGs) on usegalaxy.eu" \
  --stats-generation-software CheckM2_v1.1.0 \
  --metagenome "honeybee metagenome" \
  --broad-environment apiary-biome \
  --local-environment pain-cages \
  --environmental-medium Honeybee
```

## 5. Add binning info + check taxonomy
```bash
python3 submission/scripts/03_update_binning_metadata.py samplesheet.csv
python3 submission/scripts/04_check_ena_taxa.py samplesheet.csv
```

If step 5 fails, fix the offending row (example):
```bash
python3 submission/scripts/05_fix_lineage.py samplesheet.csv --id SRR24759607_binette_bin44 --append 589
python3 submission/scripts/04_check_ena_taxa.py samplesheet.csv
```

## 6. Apply the seqsubmit fix (one-time per checkout)
```bash
python3 submission/scripts/patch_seqsubmit_zero_stats.py /path/to/seqsubmit
```

## 7. Validate (registers samples in ENA, no upload)
```bash
mkdir -p workdir
cp samplesheet.csv workdir/
cp -r mags workdir/
submission/run/run_seqsubmit.sh validate /path/to/seqsubmit workdir
```

## 8. Submit (upload FASTAs + get accessions)
```bash
submission/run/run_seqsubmit.sh submit /path/to/seqsubmit workdir
```

## Tips
- Use `-resume` automatically (the script does). If it fails, just re-run step 8.
- **Must accept ABS in the Webin portal** (https://www.ebi.ac.uk/ena/submit/webin/) if `Invalid credentials` appear during submit (common after ENA policy updates).
- Results: `workdir/out/` has manifests + `*_accessions.tsv` after step 8. Check `workdir/work/.../submission_receipt.xml` if anything fails.
- For test server: change `test_upload` to `true` in `submission/run/params.validate.json` and `.submit.json`.
