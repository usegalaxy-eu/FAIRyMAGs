#!/usr/bin/env python3
"""Fetch the per-MAG FASTA files of a use case from Galaxy.

The ENA samplesheet needs two things that are not stored in the repository:

* one gzipped FASTA per MAG (``<outdir>/<mag-id>.fasta.gz``)
* a name -> Galaxy dataset id map (``<outdir>/mag_map.json``) so the exact
  same files can be re-downloaded later, e.g. when auditing a submission

Both are produced here.  Galaxy credentials are read from the repository
``.env`` file (``GALAXY_API=...``), the same way ``bin/download_datasets.py``
does it, so no key has to be passed on the command line.

Usage
-----
    python3 submission/scripts/01_fetch_mags_from_galaxy.py \
        --history 11ac94870d0bb33a808dfa7d28b57a6a \
        --outdir  mags \
        --ids-from data/use-cases/bee-gut/ena_submission_samplesheet.csv

    # inspect what would be downloaded, download nothing
    python3 submission/scripts/01_fetch_mags_from_galaxy.py \
        --history 11ac94870d0bb33a808dfa7d28b57a6a --dry-run

    # only (re)write the name -> dataset id map
    python3 submission/scripts/01_fetch_mags_from_galaxy.py \
        --history 11ac94870d0bb33a808dfa7d28b57a6a --map-only

The history may be given as an id or as a name (the name is matched against
the ``histories.yaml`` of the use case when ``--histories-yaml`` is passed).

Only finished datasets are downloaded; ``--dry-run`` reports their state.
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time

import yaml
from bioblend.galaxy import GalaxyInstance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FASTA_SUFFIXES = (".fasta", ".fna", ".fa", ".fasta.gz", ".fna.gz", ".fa.gz")

# Galaxy dataset states
PENDING_STATES = {"new", "upload", "queued", "running", "setting_metadata", "deferred", "paused"}
FAILED_STATES = {"error", "failed_metadata", "discarded"}


def read_api_key() -> str:
    """Read GALAXY_API from the repository .env file (or the environment)."""
    if os.environ.get("GALAXY_API"):
        return os.environ["GALAXY_API"]
    env = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(env):
        with open(env) as fh:
            for line in fh:
                if line.startswith("GALAXY_API="):
                    return line.split("=", 1)[1].strip()
    sys.exit("GALAXY_API not found: set it in the environment or in ./.env")


def resolve_history(gi, history: str) -> str:
    """Accept either a history id or a (partial) history name."""
    if len(history) == 32 and all(c in "0123456789abcdef" for c in history.lower()):
        return history
    wanted = history.lower()
    for h in gi.histories.get_histories():
        if h["name"].lower() == wanted:
            return h["id"]
    for h in gi.histories.get_histories():
        if wanted in h["name"].lower():
            print(f"using history matching name: {h['name']!r} ({h['id']})")
            return h["id"]
    sys.exit(f"no history matching {history!r}")


def mag_id(name: str) -> str:
    """Normalise a dataset name to a MAG id (strip fasta suffixes)."""
    n = name.strip()
    changed = True
    while changed:
        changed = False
        for suf in FASTA_SUFFIXES:
            if n.endswith(suf):
                n = n[: -len(suf)]
                changed = True
    return n


def collect_fastas(gi, hid: str) -> dict:
    """Return {mag_id: dataset_id} for every FASTA-like HDA in the history.

    History contents are walked recursively so MAGs that live inside a
    collection (list, list:list, ...) are found as well.
    """
    found: dict = {}
    pending = [h for h in gi.histories.show_history(hid, contents=True) if h["type"] == "collection"]
    while pending:
        hdca = pending.pop()
        info = gi.histories.show_dataset_collection(hid, hdca["id"])
        for el in info["elements"]:
            obj = el["object"]
            if "elements" in obj:  # nested collection
                pending.append(obj)
                continue
            name = obj.get("name") or el.get("element_identifier") or ""
            mid = mag_id(name)
            if not mid:
                continue
            if obj.get("state") in FAILED_STATES:
                print(f"  WARNING: {mid} is {obj['state']} -> skipped")
                continue
            found.setdefault(mid, obj["id"])
    return found


def download_one(gi, did: str, target: str, retries: int = 3) -> int:
    """Download a dataset and write it gzipped. Returns bytes written."""
    for attempt in range(retries):
        try:
            data = gi.datasets.download_dataset(did, use_default_filename=False)
            break
        except Exception as exc:  # noqa: BLE001 - retried below
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}/{retries}: {exc}")
            time.sleep(3 * (attempt + 1))

    tmp = target + ".part"
    if data[:2] == b"\x1f\x8b":  # already gzipped
        with open(tmp, "wb") as fh:
            fh.write(data)
    else:
        with gzip.GzipFile(tmp, "wb") as fh:
            fh.write(data)
    os.replace(tmp, target)
    return os.path.getsize(target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True, help="Galaxy history id or name")
    ap.add_argument("--url", default=None, help="Galaxy URL (default: from --histories-yaml)")
    ap.add_argument("--histories-yaml", default=None, help="use-case histories.yaml (for the URL)")
    ap.add_argument("--outdir", default="mags", help="where to write <mag-id>.fasta.gz (default: mags)")
    ap.add_argument("--ids-from", default=None, help="samplesheet.csv; only fetch these MAG ids")
    ap.add_argument("--map-only", action="store_true", help="write mag_map.json, download nothing")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    ap.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    url = args.url
    if url is None and args.histories_yaml:
        with open(args.histories_yaml) as fh:
            url = yaml.safe_load(fh).get("galaxy-url")
    if url is None:
        sys.exit("pass --url or --histories-yaml (to read galaxy-url from it)")

    gi = GalaxyInstance(url=url, key=read_api_key())
    hid = resolve_history(gi, args.history)
    print(f"history: {hid} @ {url}")

    found = collect_fastas(gi, hid)
    print(f"FASTA-like datasets found: {len(found)}")

    wanted = None
    if args.ids_from:
        with open(args.ids_from, newline="") as fh:
            wanted = [mag_id(r["id"]) for r in csv.DictReader(fh) if r.get("id")]
        print(f"MAG ids in samplesheet  : {len(wanted)}")
        missing = [w for w in wanted if w not in found]
        if missing:
            print(f"WARNING: {len(missing)} samplesheet MAG(s) not in the history, e.g. {missing[:5]}")
        found = {k: v for k, v in found.items() if k in set(wanted)}
        print(f"to fetch                : {len(found)}")

    os.makedirs(args.outdir, exist_ok=True)
    map_path = os.path.join(args.outdir, "mag_map.json")
    # keys carry the .fasta suffix to match the samplesheet `id` -> file relation
    with open(map_path, "w") as fh:
        json.dump({f"{k}.fasta": v for k, v in sorted(found.items())}, fh, indent=2)
        fh.write("\n")
    print(f"wrote {map_path} ({len(found)} entries)")

    if args.map_only or args.dry_run:
        if args.dry_run:
            for k, v in sorted(found.items())[:5]:
                print(f"  would download {k}.fasta.gz  <- dataset {v}")
            if len(found) > 5:
                print(f"  ... +{len(found) - 5} more")
        return

    ok = skip = err = 0
    for i, (mid, did) in enumerate(sorted(found.items()), 1):
        target = os.path.join(args.outdir, f"{mid}.fasta.gz")
        if not args.overwrite and os.path.exists(target) and os.path.getsize(target) > 0:
            skip += 1
            continue
        try:
            size = download_one(gi, did, target)
            ok += 1
            if i % 50 == 0 or i == len(found):
                print(f"  {i}/{len(found)}  ok={ok} skip={skip} err={err}")
        except Exception as exc:  # noqa: BLE001 - reported, run continues
            err += 1
            print(f"  FAILED {mid}: {exc}")
    print(f"DONE ok={ok} skip={skip} err={err} -> {args.outdir}")


if __name__ == "__main__":
    main()
