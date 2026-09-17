#!/usr/bin/env python3
"""Download use-case datasets from Galaxy based on the tags in histories.yaml.

For each history entry that has a `datasets` list, fetch the datasets in that
history, match each entry by its `tag`, and save the resulting file next to
the config file (data/<use-case>-use-case/<file>). Warns for any tag with no
matching dataset.

Usage:
    python3 bin/download_datasets.py data/bee-use-case/histories.yaml
"""

import os
import sys
import time

import yaml
from bioblend.galaxy import GalaxyInstance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")

with open(os.path.join(REPO_ROOT, ".env")) as fh:
    API_KEY = [l.split("=", 1)[1].strip() for l in fh if l.startswith("GALAXY_API=")][0]


def get_datasets(gi, hid, max_retries=5, backoff=5.0):
    items, offset = [], 0
    while True:
        try:
            batch = gi.datasets.get_datasets(history_id=hid, limit=500, offset=offset)
        except Exception as exc:
            print(f"    error at offset {offset}: {exc}", flush=True)
            for a in range(max_retries):
                time.sleep(backoff * (a + 1))
                try:
                    batch = gi.datasets.get_datasets(history_id=hid, limit=500, offset=offset)
                    break
                except Exception as e:
                    print(f"    retry {a+1}/{max_retries}: {e}", flush=True)
            else:
                raise
        if not batch:
            break
        items.extend(batch)
        offset += len(batch)
        if len(batch) < 500:
            break
    return items


def download(gi, did, retries=3):
    for a in range(retries):
        try:
            return gi.datasets.download_dataset(did, use_default_filename=False)
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(5 * (a + 1))


def download_combined(gi, d, hid, concat, retries=3):
    """Download a dataset, or if it is a collection, concat its element files."""
    if d["type"] != "collection":
        return download(gi, d["id"]), d["name"]

    info = gi.histories.show_dataset_collection(hid, d["id"])
    elements = [(e["element_identifier"], e["object"]["id"]) for e in info["elements"]]
    parts = [download(gi, eid) for _, eid in elements]
    if not concat:
        content = "\n".join(p.decode() for p in parts).encode()
        return content, d["name"]

    bodies = []
    names = []
    for el_name, content in zip([n for n, _ in elements], parts):
        text = content.decode()
        lines = text.splitlines(keepends=True)
        if bodies and lines and bodies[0].splitlines()[0] == lines[0].rstrip("\n"):
            lines = lines[1:]
        bodies.append("".join(lines))
        names.append(el_name)
    return "".join(bodies).encode(), " + ".join(names)


def main():
    default = os.path.join(DATA_ROOT, "bee-use-case", "histories.yaml")
    config_path = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(config_path):
        sys.exit(f"config not found: {config_path}")

    with open(config_path) as fh:
        doc = yaml.safe_load(fh)

    galaxy_dir = os.path.dirname(config_path)
    gi = GalaxyInstance(url=doc["galaxy-url"], key=API_KEY)

    for history in doc["histories"]:
        datasets = history.get("datasets") or []
        if not datasets:
            continue

        print(f"HISTORY: {history['name']}")
        items = get_datasets(gi, history["id"])
        print(f"  {len(items)} datasets fetched")

        for entry in datasets:
            tag = entry["tag"].lower()
            matches = [
                d for d in items
                if any(t.replace("name:", "").lower() == tag for t in (d.get("tags") or []))
            ]
            if not matches:
                print(f"  WARNING: no dataset with tag '{entry['tag']}' found -> skipped ({entry['file']})")
                continue

            content = source = None
            for d in matches:
                candidate, candidate_source = download_combined(gi, d, history["id"], entry.get("concat", False))
                if candidate:
                    content, source = candidate, candidate_source
                    picked = d
                    break

            if content is None:
                print(f"  WARNING: no non-empty dataset with tag '{entry['tag']}' found -> skipped ({entry['file']})")
                continue
            if len(matches) > 1:
                print(f"  WARNING: multiple datasets with tag '{entry['tag']}'; using first non-empty")

            target = os.path.join(galaxy_dir, entry["file"])
            with open(target, "wb") as fh:
                fh.write(content)
            print(f"  {entry['file']} <- {source} ({len(content)} B)")
        print()


if __name__ == "__main__":
    main()