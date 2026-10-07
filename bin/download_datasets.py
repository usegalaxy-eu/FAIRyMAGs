#!/usr/bin/env python3
"""Download use-case datasets from Galaxy based on the tags in histories.yaml.

For each history entry that has a `datasets` list, fetch the datasets in that
history, match each entry by its `tag`, and save the resulting file next to
the config file (data/<use-case>-use-case/<file>). Warns for any tag with no
matching dataset.

Usage:
    python3 bin/download_datasets.py data/bee-use-case/histories.yaml

Pass --dry-run to report which dataset each entry would be saved from,
without downloading anything.

Files that already exist on disk are skipped. Pass --overwrite to download
them again regardless of what is already there.
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


def select_matches(items, entry):
    """Return the datasets an entry refers to, in preference order.

    Optional keys on an entry:
      concat:  when true, prefer a collection over individual datasets. Tags
               such as `bakta_annotation_tabular` are also applied to every
               per-MAG file, so without this the first non-empty individual
               file would be saved instead of the combined table.
      name:    case-insensitive substring that must appear in the dataset
               name, used to pick among several datasets sharing one tag.
    """
    tag = entry["tag"].lower()
    matches = [
        d for d in items
        if any(t.replace("name:", "").lower() == tag for t in (d.get("tags") or []))
    ]
    name_filter = entry.get("name")
    if name_filter:
        wanted = name_filter.lower()
        matches = [d for d in matches if wanted in (d.get("name") or "").lower()]
    if entry.get("concat", False):
        collections = [d for d in matches if d.get("type") == "collection"]
        if collections:
            matches = collections
    return matches


def download(gi, did, retries=3):
    for a in range(retries):
        try:
            return gi.datasets.download_dataset(did, use_default_filename=False)
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(5 * (a + 1))


PENDING_STATES = {"new", "upload", "queued", "running", "setting_metadata", "deferred", "paused"}
FAILED_STATES = {"error", "failed_metadata", "discarded"}
ALLOW_ERROR_HINT = "; set allow_error: true to download it anyway"


class SkipEntry(Exception):
    """A candidate cannot be used. The caller prints the reason and moves on."""

    def __init__(self, message):
        super().__init__(message)
        self.reason = message


def state_of(obj):
    return (obj.get("state") or "").lower()


def collection_elements(gi, hid, did):
    info = gi.histories.show_dataset_collection(hid, did)
    return [(e["element_identifier"], e["object"]) for e in info["elements"]]


def usable_elements(elements, allow_error, label):
    """Return the elements that may be written, refusing unfinished collections.

    Raises SkipEntry when the collection is still being produced, or when it
    has failed elements and the entry does not set allow_error: true. Failed
    elements are dropped from the result, so the table is written without them.
    """
    pending = [n for n, o in elements if state_of(o) in PENDING_STATES]
    if pending:
        raise SkipEntry(
            f"{label} not finished: {len(pending)}/{len(elements)} element(s) still pending"
        )
    failed = [n for n, o in elements if state_of(o) in FAILED_STATES]
    if failed and not allow_error:
        raise SkipEntry(
            f"{label}: {len(failed)}/{len(elements)} element(s) failed"
            f"{ALLOW_ERROR_HINT}"
        )
    return [e for e in elements if state_of(e[1]) not in FAILED_STATES], failed


def download_combined(gi, d, hid, concat, allow_error=False):
    """Download a dataset, or if it is a collection, concat its element files."""
    if d["type"] != "collection":
        state = state_of(d)
        if state in PENDING_STATES:
            raise SkipEntry(f"dataset still {state}")
        if state in FAILED_STATES and not allow_error:
            raise SkipEntry(f"dataset is {state}{ALLOW_ERROR_HINT}")
        return download(gi, d["id"]), d["name"]

    elements = collection_elements(gi, hid, d["id"])
    usable, failed = usable_elements(elements, allow_error, "collection")
    if failed:
        print(f"    omitting {len(failed)}/{len(elements)} failed element(s)", flush=True)
    parts = [download(gi, o["id"]) for _, o in usable]
    if not concat:
        content = "\n".join(p.decode() for p in parts).encode()
        return content, d["name"]

    bodies = []
    names = []
    for (el_name, _), content in zip(usable, parts):
        text = content.decode()
        lines = text.splitlines(keepends=True)
        if bodies and lines and bodies[0].splitlines()[0] == lines[0].rstrip("\n"):
            lines = lines[1:]
        bodies.append("".join(lines))
        names.append(el_name)
    return "".join(bodies).encode(), " + ".join(names)


def preview_state(gi, d, hid, allow_error):
    """Report how usable a candidate is, without downloading it."""
    if d.get("type") != "collection":
        state = state_of(d)
        if state in PENDING_STATES:
            raise SkipEntry(f"dataset still {state}")
        if state in FAILED_STATES and not allow_error:
            raise SkipEntry(f"dataset is {state}{ALLOW_ERROR_HINT}")
        return " (errored, allow_error set)" if state in FAILED_STATES else ""
    elements = collection_elements(gi, hid, d["id"])
    usable, failed = usable_elements(elements, allow_error, "collection")
    note = f" ({len(usable)}/{len(elements)} elements usable"
    if failed:
        note += f", {len(failed)} failed omitted"
    return note + ")"


DRY_RUN = False
OVERWRITE = False


def main():
    global DRY_RUN
    global OVERWRITE
    args = [a for a in sys.argv[1:] if a not in ("--dry-run", "--overwrite")]
    DRY_RUN = "--dry-run" in sys.argv
    OVERWRITE = "--overwrite" in sys.argv
    default = os.path.join(DATA_ROOT, "bee-use-case", "histories.yaml")
    config_path = args[0] if args else default
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
            target = os.path.join(galaxy_dir, entry["file"])
            if not OVERWRITE and os.path.exists(target):
                print(f"  SKIP {entry['file']}: already downloaded (use --overwrite to replace)")
                continue

            matches = select_matches(items, entry)
            if not matches:
                print(f"  WARNING: no dataset with tag '{entry['tag']}' found -> skipped ({entry['file']})")
                continue

            if DRY_RUN:
                allow_error = entry.get("allow_error", False)
                if len(matches) == 1:
                    d = matches[0]
                    label = f"{d.get('type')} hid={d.get('hid')} {d.get('name')!r}"
                    try:
                        note = preview_state(gi, d, history["id"], allow_error)
                    except SkipEntry as skip:
                        print(f"  SKIP {entry['file']}: {skip.reason}")
                    else:
                        print(f"  would write {entry['file']} from {label}{note}")
                else:
                    print(f"  {entry['file']}: {len(matches)} candidates "
                          f"(the first non-empty one is saved):")
                    for d in matches[:6]:
                        line = f"      {d.get('type')} hid={d.get('hid')} {d.get('name')!r}"
                        try:
                            note = preview_state(gi, d, history["id"], allow_error)
                        except SkipEntry as skip:
                            note = f"  -> SKIP: {skip.reason}"
                        print(f"{line}{note}")
                    if len(matches) > 6:
                        print(f"      ... +{len(matches) - 6} more")
                continue

            content = source = None
            skipped = False
            for d in matches:
                try:
                    candidate, candidate_source = download_combined(
                        gi, d, history["id"], entry.get("concat", False),
                        entry.get("allow_error", False),
                    )
                except SkipEntry as skip:
                    print(f"  SKIP {entry['file']}: {skip.reason}")
                    skipped = True
                    continue
                if candidate:
                    content, source = candidate, candidate_source
                    break

            if content is None:
                if not skipped:
                    print(f"  WARNING: no non-empty dataset with tag "
                          f"'{entry['tag']}' found -> skipped ({entry['file']})")
                continue
            if len(matches) > 1:
                print(f"  WARNING: multiple datasets with tag '{entry['tag']}'; using first non-empty")

            with open(target, "wb") as fh:
                fh.write(content)
            print(f"  {entry['file']} <- {source} ({len(content)} B)")
        print()


if __name__ == "__main__":
    main()