# Download use case data

The use-case datasets (dRep clustering, CheckM/CheckM2 quality, CoverM abundance,
GTDB/kMetaShot taxonomy, annotation tables, KEGG completeness, etc.) live in
tagged Galaxy histories. Histories and datasets are located by **tags**:

- histories tagged `publication` + a use-case tag (`bee-use-case`,
  `macroalgal-use-case`, `cloud-use-case`, `termite-use-case`) and excluding
  `fail`/`rerun`
- datasets tagged e.g. `bin-coverm-abundance`, `gtdb-tk-summary`, `kegg-contig-table`

Each use case has a `data/<use-case>-use-case/histories.yaml` config that lists the
histories, their tags, and which tagged datasets should be downloaded to which file.

## How it works

1. **`bin/lookup_history_ids.py`** reads a `histories.yaml`, narrows the Galaxy
   histories to those matching the configured tags, matches each entry by name
   (substring), and writes the resolved history `id` back into the YAML.
2. **`bin/download_datasets.py`** reads the same YAML, fetches all datasets per
   history, finds each dataset by its tag, and saves it to
   `data/<use-case>-use-case/<file>`. Collections are concatenated (header kept
   once) when `concat: true` is set; if several datasets carry the same tag, the
   first non-empty one is used (a warning is printed otherwise).

## Commands

### Bee use case

```bash
python3 bin/lookup_history_ids.py data/bee-use-case/histories.yaml
python3 bin/download_datasets.py data/bee-use-case/histories.yaml
```

### Marine (macroalgal) use case

```bash
python3 bin/lookup_history_ids.py data/marine-use-case/histories.yaml
python3 bin/download_datasets.py data/marine-use-case/histories.yaml
```

### Termite use case

```bash
python3 bin/lookup_history_ids.py data/termite-use-case/histories.yaml
python3 bin/download_datasets.py data/termite-use-case/histories.yaml
```

### Cloud use case

```bash
python3 bin/lookup_history_ids.py data/cloud-use-case/histories.yaml
python3 bin/download_datasets.py data/cloud-use-case/histories.yaml
```