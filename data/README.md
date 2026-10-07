# Data

## Use cases

The use-case datasets (dRep clustering, CheckM/CheckM2 quality, CoverM abundance, GTDB/kMetaShot taxonomy, annotation tables, KEGG completeness, etc.) live in tagged Galaxy histories. Histories and datasets are located by **tags**:

- histories tagged `publication` + a use-case tag (`bee-use-case`,
  `macroalgal-use-case`, `cloud-use-case`, `termite-use-case`) and excluding
  `fail`/`rerun`
- datasets tagged e.g. `bin-coverm-abundance`, `gtdb-tk-summary`, `kegg-contig-table`

Each use case has a `data/use-cases/<use-case>/histories.yaml` config that lists the histories, their tags, and which tagged datasets should be downloaded to which file.

Each use case also has a human-readable `workflow.yaml` describing its complete
workflow chain. Workflows, histories, datasets, and collections are selected by
names and tags; Galaxy IDs are resolved only while running. Parameters use the
labels shown in the Galaxy workflow form.

Validate a pipeline without creating histories:

```bash
python3 bin/run_workflows.py data/use-cases/bee-gut/workflow.yaml --dry-run
```

Run the complete pipeline and poll every invocation until its outputs finish:

```bash
python3 bin/run_workflows.py data/use-cases/bee-gut/workflow.yaml
```

Each step reuses an existing history that already carries the name and tags the
manifest requests, and only invokes the workflow when no such history exists. For
a fully published use case such as bee, nothing is run at all and every step is
reported as reused. Reused histories are only read, never written to.

Before running anything the script prints the full plan, showing every step as
`CREATE` or `reuse` along with the history it will use, so you can see the whole
run before spending quota. After a real run it prints a summary table with the
action, history ID, invocation ID, failed dataset count, and duration per step.

To force a step to run again, tag its existing history with `fail` or `rerun`.
Those tags are listed in `exclude_history_tags`, so the history stops matching
and the runner creates a replacement.

`execution.on_failure` controls what happens when a step produces failed
datasets:

- `continue` (default): failures are reported as warnings and downstream steps
  proceed as long as the specific dataset or collection they need is usable.
  This suits MAG-level tools that fail on individual bins.
- `stop`: the first failed dataset, or a failed or cancelled invocation, halts
  the run. Use this when a partial result would be misleading.

Set `GALAXY_API_KEY` (or `GALAXY_API`) in the environment or in the repository
`.env` file. The runner requires `execution.new_history_only: true`, meaning it
never writes into a history it did not create.

Stopping the script does not cancel work on Galaxy, so restarting it is safe:
steps that already have their history are reused, and a step whose history is
still being produced is waited for rather than failed (the invocation is listed
for context, but dataset states are what decides). That wait is bounded by
`execution.max_wait_minutes` or `--max-wait-minutes` (default 1440). If a
history is abandoned rather than running, tag it `fail` to drop it out of
matching.

### How it works

1. **`bin/lookup_history_ids.py`** reads a `histories.yaml`, narrows the Galaxy histories to those matching the configured tags, matches each entry by name (substring), and writes the resolved history `id` back into the YAML.
2. **`bin/download_datasets.py`** reads the same YAML, fetches all datasets per history, finds each dataset by its tag, and saves it next to the YAML (`<config dir>/<file>`). When several datasets carry the same tag:

   - `name:` narrows the candidates to datasets whose name contains that substring.
   - `concat: true` prefers a collection over individual datasets, so tags applied both to a combined collection and to every per-MAG file resolve to the collection. Collections are concatenated with the header kept once; otherwise the first non-empty candidate is used.

   Pass `--dry-run` to print which dataset each entry would be saved from, without downloading anything.

   Files that already exist on disk are skipped (they are not re-downloaded on
   every run); pass `--overwrite` to download them again:

   ```
   python3 bin/download_datasets.py data/use-cases/bee-gut/histories.yaml        # skip existing files
   python3 bin/download_datasets.py --overwrite data/use-cases/bee-gut/histories.yaml   # re-download everything
   ```

A collection that still has pending elements is reported and skipped, so an
unfinished run never writes a half-filled table:

```
SKIP eggnog_annotation.tsv: collection not finished: 714/730 element(s) still pending
```

To write a collection that contains failed elements anyway, set
`allow_error: true` on that entry. The failed elements are omitted and the
remaining ones are concatenated, and `--dry-run` reports how many that leaves:

```
would write bakta.tsv from collection hid=8296 'bakta_annotation_tabular' (551/552 elements usable, 1 failed omitted)
```



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

Air runs two assemblies of the same reads, so its outputs are split per branch
and each branch has its own manifest. The branch tag in `required_tags` is what
keeps the identically named branch histories apart, and it also decides where
downloads land.

```bash
python3 bin/lookup_history_ids.py data/use-cases/air/megahit/histories.yaml
python3 bin/download_datasets.py data/use-cases/air/megahit/histories.yaml

python3 bin/lookup_history_ids.py data/use-cases/air/metaspades/histories.yaml
python3 bin/download_datasets.py data/use-cases/air/metaspades/histories.yaml
```

Add `--dry-run` to the download command to check what would be saved first.
