#!/usr/bin/env python3
"""Download use-case datasets from Galaxy based on the tags in histories.yaml.

For each history entry that has a `datasets` list, fetch the datasets in that
history, match each entry by its `tag` (or by the exact `id` of an output the
producing workflow leaves untagged), and save the resulting file next to the
config file (data/<use-case>-use-case/<file>). Warns for any entry with no
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
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import quote

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


def entry_label(entry):
    """Describe an entry's selector, for log and error messages."""
    if entry.get("tag"):
        label = f"tag {entry['tag']!r}"
    elif entry.get("id"):
        label = f"id {entry['id']!r}"
    else:
        label = "entry without tag or id"
    if entry.get("name"):
        label += f" named like {entry['name']!r}"
    return label


def select_matches(items, entry):
    """Return the datasets an entry refers to, in preference order.

    An entry is selected either by `tag` or, when the producing workflow does
    not tag the output, by the exact `id` of the dataset or collection. `id`
    takes precedence when both are present.

    Optional keys on an entry:
      concat:  when true, prefer a collection over individual datasets. Tags
               such as `bakta_annotation_tabular` are also applied to every
               per-MAG file, so without this the first non-empty individual
               file would be saved instead of the combined table.
      name:    case-insensitive substring that must appear in the dataset
               name, used to pick among several datasets sharing one tag.
    """
    if entry.get("id"):
        matches = [d for d in items if d.get("id") == entry["id"]]
    elif entry.get("tag"):
        tag = entry["tag"].lower()
        matches = [
            d for d in items
            if any(t.replace("name:", "").lower() == tag for t in (d.get("tags") or []))
        ]
    else:
        matches = []
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


def download_campaign_output(session, base, reference, target):
    """Export one exact invocation output, preserving nested element paths.

    Unlike the legacy tag-based interface, this never guesses among datasets or
    concatenates unrelated files. A manifest records every leaf and checksum.
    Failed/skipped/pending leaves prevent acceptance of this required output.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        if recorded["reference"] != reference:
            raise RuntimeError("Existing export refers to another dataset")
        for entry in recorded["files"]:
            path = target / entry["file"]
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Export checksum mismatch: {path}")
        return recorded

    def get(path):
        response = session.get(base + path, timeout=(30, 180))
        response.raise_for_status()
        return response.json()

    def leaves(obj, names=()):
        if "elements" in obj:
            if obj.get("populated_state", "ok") != "ok":
                raise RuntimeError("Output collection is not populated")
            for element in obj["elements"]:
                yield from leaves(element["object"], names + (element["element_identifier"],))
        else:
            yield names, obj

    if reference["src"] == "hdca":
        obj = get(f"histories/{reference['history_id']}/contents/dataset_collections/{reference['id']}")
    else:
        obj = get("datasets/" + reference["id"])
    records = []
    for names, dataset in leaves(obj):
        if dataset.get("state") != "ok" or dataset.get("deleted") or dataset.get("purged"):
            raise RuntimeError(f"Output leaf {names} is not usable")
        if dataset.get("extension") == "expression.json":
            raise RuntimeError(f"Output leaf {names} is a skipped placeholder")
        # Percent-encode separators and '.' so identifiers cannot escape target.
        parts = [quote(name, safe="").replace(".", "%2E") for name in names]
        ext = dataset.get("extension", "dat").replace("/", "_")
        relative = Path(*parts, "dataset." + ext)
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        response = session.get(base + f"datasets/{dataset['id']}/display", stream=True, timeout=(30, 300))
        response.raise_for_status()
        expected_size = int(response.headers.get("Content-Length", dataset.get("file_size", 0)) or 0)
        if shutil.disk_usage(target).free < expected_size + 5 * 1024**3:
            response.close()
            raise RuntimeError("Insufficient free disk space for output plus 5 GiB reserve")
        temporary = path.with_suffix(path.suffix + ".partial")
        checksum = hashlib.sha256()
        size = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if shutil.disk_usage(target).free < len(chunk) + 5 * 1024**3:
                    response.close()
                    raise RuntimeError("Download stopped to preserve 5 GiB free disk space")
                checksum.update(chunk)
                size += len(chunk)
                handle.write(chunk)
        temporary.replace(path)
        records.append({"identifiers": list(names), "id": dataset["id"],
                        "file": str(relative), "sha256": checksum.hexdigest(), "bytes": size})
    result = {"reference": reference, "files": records}
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(manifest)
    return result


def download_campaign_table(session, base, history_id, entry, target, max_bytes=1024**3):
    """Download only an explicitly allowlisted table; fail on ambiguity.

    Collection tables are concatenated only when requested in histories.yaml.
    Sequence/alignment datatypes are refused even if incorrectly tagged.
    """
    def get(path, **kwargs):
        response = session.get(base + path, timeout=(30, 180), **kwargs)
        response.raise_for_status()
        return response.json()

    items, offset = [], 0
    while True:
        batch = get(f"histories/{history_id}/contents", params={
            "details": "all", "limit": 500, "offset": offset,
        })
        items.extend(d for d in batch if not d.get("deleted") and not d.get("purged"))
        if len(batch) < 500:
            break
        offset += len(batch)
    # Some API views call the content type history_content_type, not type.
    for item in items:
        if item.get("history_content_type") == "dataset_collection":
            item["type"] = "collection"
    candidates = select_matches(items, entry)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one table for {entry_label(entry)}; found {len(candidates)}")
    selected = candidates[0]
    if selected.get("type") == "collection":
        if not entry.get("concat"):
            raise RuntimeError(f"Collection for {entry_label(entry)} requires explicit concat: true")
        collection = get(f"histories/{history_id}/contents/dataset_collections/{selected['id']}")

        def leaves(obj):
            if obj.get("populated_state", "ok") != "ok":
                raise RuntimeError("Table collection is not populated")
            for element in obj.get("elements", []):
                if element["element_type"] == "dataset_collection":
                    yield from leaves(element["object"])
                else:
                    yield element["object"]

        datasets = list(leaves(collection))
    else:
        datasets = [get("datasets/" + selected["id"])]
    allowed = {"tabular", "tsv", "csv", "txt", "text", "gff", "gff3", "bed"}
    for dataset in datasets:
        if dataset.get("state") != "ok" or dataset.get("extension") not in allowed or dataset.get("deleted") or dataset.get("purged"):
            raise RuntimeError(f"Refusing non-ready/non-table dataset {dataset.get('id')} ({dataset.get('extension')})")
    if not datasets:
        raise RuntimeError("Selected table collection is empty")
    if sum(d.get("file_size", 0) or 0 for d in datasets) > max_bytes:
        raise RuntimeError("Selected table exceeds the 1 GiB per-file safety limit")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = target.with_suffix(target.suffix + ".provenance.json")
    if target.exists() and metadata_path.exists():
        record = json.loads(metadata_path.read_text())
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if record["history_id"] == history_id and record["selected_id"] == selected["id"] and record["sha256"] == digest.hexdigest():
            return record
        raise RuntimeError("Existing table provenance/checksum mismatch; refusing overwrite")
    temporary = target.with_suffix(target.suffix + ".partial")
    size, first_header = 0, None
    try:
        with temporary.open("wb") as handle:
            for dataset in datasets:
                with session.get(base + f"datasets/{dataset['id']}/display", stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    # Only table files enter this path. Stream by line to remove
                    # identical repeated headers when concatenation is explicit.
                    for index, line in enumerate(response.iter_lines()):
                        if index == 0:
                            if first_header is None:
                                first_header = line
                            elif line == first_header:
                                continue
                        content = line + b"\n"
                        size += len(content)
                        if size > max_bytes or shutil.disk_usage(target.parent).free < len(content) + 5 * 1024**3:
                            raise RuntimeError("Table download exceeds size/free-space safety limit")
                        handle.write(content)
        digest = hashlib.sha256()
        with temporary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        temporary.replace(target)
        record = {"history_id": history_id, "selected_id": selected["id"], "tag": entry.get("tag"),
                  "dataset_id": entry.get("id"),
                  "dataset_ids": [d["id"] for d in datasets], "bytes": size, "sha256": digest.hexdigest()}
        metadata_path.write_text(json.dumps(record, indent=2) + "\n")
        return record
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


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
                print(f"  WARNING: no dataset with {entry_label(entry)} found -> skipped ({entry['file']})")
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
                    print(f"  WARNING: no non-empty dataset with {entry_label(entry)} "
                          f"found -> skipped ({entry['file']})")
                continue
            if len(matches) > 1:
                print(f"  WARNING: multiple datasets for {entry_label(entry)}; using first non-empty")

            with open(target, "wb") as fh:
                fh.write(content)
            print(f"  {entry['file']} <- {source} ({len(content)} B)")
        print()


if __name__ == "__main__":
    main()
