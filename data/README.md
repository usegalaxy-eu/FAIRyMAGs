# Data

## Use cases

The use-case datasets (dRep clustering, CheckM/CheckM2 quality, CoverM abundance, GTDB/kMetaShot taxonomy, annotation tables, KEGG completeness, etc.) live in tagged Galaxy histories. Histories and datasets are located by **tags**:

- histories tagged `publication` + a use-case tag (`bee-use-case`,
  `macroalgal-use-case`, `cloud-use-case`, `termite-use-case`) and excluding
  `fail`/`rerun`
- datasets tagged e.g. `bin-coverm-abundance`, `gtdb-tk-summary`, `kegg-contig-table`

Each use case has a `data/use-cases/<use-case>/histories.yaml` config that lists the histories, their tags, and which tagged datasets should be downloaded to which file.

### How it works

1. **`bin/lookup_history_ids.py`** reads a `histories.yaml`, narrows the Galaxy histories to those matching the configured tags, matches each entry by name (substring), and writes the resolved history `id` back into the YAML.
2. **`bin/download_datasets.py`** reads the same YAML, fetches all datasets per history, finds each dataset by its tag, and saves it to `data/use-cases/<use-case>/<file>`. Collections are concatenated (header kept once) when `concat: true` is set; if several datasets carry the same tag, the first non-empty one is used (a warning is printed otherwise).


### Bee gut microbiome use case

```bash
python3 bin/lookup_history_ids.py data/use-cases/bee-gut/histories.yaml
python3 bin/download_datasets.py data/use-cases/bee-gut/histories.yaml
```

### Marine (macroalgal) use case

```bash
python3 bin/lookup_history_ids.py data/use-cases/macroalgal-epiphytic/histories.yaml
python3 bin/download_datasets.py data/use-cases/macroalgal-epiphytic/histories.yaml
```

### Termite use case

```bash
python3 bin/lookup_history_ids.py data/use-cases/termite-head/histories.yaml
python3 bin/download_datasets.py data/use-cases/termite-head/histories.yaml
```

### Air 

```bash
python3 bin/lookup_history_ids.py data/use-cases/air/histories.yaml
python3 bin/download_datasets.py data/use-cases/air/histories.yaml
```