#!/usr/bin/env python3
"""Resolve Galaxy history IDs for a use-case histories YAML file.

Reads a histories YAML file (data/<use-case>-use-case/histories.yaml by
default), narrows the user's histories to those matching `required_tags` (all
present) and excluding `exclude_tags`, matches each entry by its `name`
substring, and writes the resolved history id back into the `id` field.

Usage:
    python3 bin/lookup_history_ids.py data/bee-use-case/histories.yaml
"""

import os
import sys

import yaml
from bioblend.galaxy import GalaxyInstance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")

with open(os.path.join(REPO_ROOT, ".env")) as fh:
    API_KEY = [l.split("=", 1)[1].strip() for l in fh if l.startswith("GALAXY_API=")][0]


def _norm(tag):
    t = str(tag).strip().lstrip("#")
    if t.startswith("name:"):
        t = t[5:]
    return t.lower()


def load(config_path):
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def tagged_histories(gi, doc):
    required = {_norm(t) for t in doc.get("required_tags", [])}
    exclude = {_norm(t) for t in doc.get("exclude_tags", [])}
    histories = gi.histories.get_histories(deleted=False)
    return [
        h for h in histories
        if required <= {_norm(x) for x in h.get("tags", [])}
        and not {_norm(x) for x in h.get("tags", [])} & exclude
    ]


def lookup_ids(gi, doc):
    candidates = tagged_histories(gi, doc)
    warnings = []
    for entry in doc["histories"]:
        kw = (entry.get("name") or "").lower()
        matches = [h for h in candidates if kw in (h.get("name") or "").lower()]
        if len(matches) == 1:
            entry["id"] = matches[0]["id"]
        elif len(matches) > 1:
            entry["id"] = None
            names = ", ".join(f"'{m['name']}' ({m['id']})" for m in matches)
            warnings.append(f"multiple histories match '{entry['name']}': {names}")
        else:
            entry["id"] = None
    return doc, len(candidates), warnings


def main():
    default = os.path.join(DATA_ROOT, "bee-use-case", "histories.yaml")
    config_path = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(config_path):
        sys.exit(f"config not found: {config_path}")

    doc = load(config_path)
    gi = GalaxyInstance(url=doc["galaxy-url"], key=API_KEY)

    before = {e["name"]: e.get("id") for e in doc["histories"]}
    doc, n_candidates, warnings = lookup_ids(gi, doc)

    with open(config_path, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"{n_candidates} tagged histories matched\n")
    for entry in doc["histories"]:
        old = before[entry["name"]]
        new = entry["id"]
        status = "OK" if new else "AMBIGUOUS/MISSING"
        print(f"[{status}] {entry['name']}: {old} -> {new}")

    for warning in warnings:
        print(f"\nWARNING: {warning}")


if __name__ == "__main__":
    main()