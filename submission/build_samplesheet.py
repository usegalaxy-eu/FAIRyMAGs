#!/usr/bin/env python3
"""Build the ENA samplesheet for a FAIRyMAGs use case, and fetch its MAG FASTAs.

One command turns a use case directory into a submission-ready samplesheet for
nf-core/seqsubmit (``--mode mags``).  Everything that is not data is read from
``<use-case-dir>/ena_submission.yaml``:

  1. find the MAG FASTA collection in Galaxy: the history named in that yaml is
     looked up in ``<use-case-dir>/histories.yaml`` (for its id and the Galaxy
     URL), and the collection carrying the ``Dereplicated-Bins`` tag inside it
     is used - that is the dRep cluster representatives, the set that is
     submitted;
  2. download one gzipped FASTA per MAG into ``--fasta-dir`` (plus a
     ``mag_map.json`` recording MAG -> Galaxy dataset id, for provenance);
  3. build the samplesheet: completeness/contamination from ``checkm2.tsv``,
     coverage from ``coverm.tsv``, the NCBI lineage from ``kmetashot.tsv``
     (``gtdb.tsv`` for the not-classified fallback), the rest from the yaml;
  4. repair lineages ENA would reject and verify every final taxon: ENA only
     accepts a *submittable* taxon, in practice species rank or below, and
     rejects the whole batch if one MAG stops at genus level.

Usage
-----
    python3 submission/build_samplesheet.py data/use-cases/bee-gut

    # samplesheet only: no Galaxy, no downloads (FASTAs must already exist)
    python3 submission/build_samplesheet.py data/use-cases/bee-gut --no-download

    # show what would happen, write nothing
    python3 submission/build_samplesheet.py data/use-cases/bee-gut --dry-run

    # brand a second submission so it is not confused with the first one
    python3 submission/build_samplesheet.py data/use-cases/bee-gut \
        --id-prefix FAIRyMAGs2_ --fasta-dir mags_v2

    # give a seqsubmit checkout the two-line fix for contamination=0.0
    python3 submission/build_samplesheet.py --patch-seqsubmit /path/to/seqsubmit

Galaxy credentials are read from ``./.env`` (``GALAXY_API=...``), like
``bin/download_datasets.py``, so no key is passed on the command line.

The ``fasta`` column holds ``<--fasta-dir>/<mag_id>.fasta.gz`` verbatim, so a
relative ``--fasta-dir`` (the default, ``mags``) keeps samplesheet + FASTAs in
one self-contained directory you can move to the machine that runs Nextflow.

``--id-prefix`` brands the *submission identity*: the samplesheet ``id`` becomes
``<prefix><mag_id>``, and because nf-core/seqsubmit passes that value through as
``genome_name``, it ends up in every ENA identifier you can control - the sample
alias and title, the assembly name (``ASSEMBLYNAME``), and therefore the
analysis alias ``webin-genome-<id>`` (the ``webin-genome-`` part is added by
ENA's Webin-CLI).  It is how two submissions of the same MAGs, made with
different pipeline versions, are told apart.  The FASTA *file names* are left
alone: ENA keeps the assembly name and the FASTA file name separate (its own
example pairs ``ASSEMBLYNAME TODO`` with ``FASTA genome.fasta.gz``), so nothing
has to be re-downloaded or renamed.  ENA aliases are permanent, so pick the
prefix once - and remember it is public as the assembly name.
"""

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

import yaml
from bioblend.galaxy import GalaxyInstance

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLUMNS = [
    "id",
    "fasta",
    "accession",
    "fastq_1",
    "fastq_2",
    "assembly_software",
    "binning_software",
    "binning_parameters",
    "stats_generation_software",
    "completeness",
    "contamination",
    "genome_coverage",
    "metagenome",
    "co-assembly",
    "broad_environment",
    "local_environment",
    "environmental_medium",
    "RNA_presence",
    "NCBI_lineage",
]

# free-text fields taken from ena_submission.yaml, and their samplesheet column
YAML_FIELDS = [
    "assembly_software",
    "binning_software",
    "binning_parameters",
    "stats_generation_software",
    "metagenome",
    "broad_environment",
    "local_environment",
    "environmental_medium",
]

RANKS = ("superkingdom", "phylum", "class", "order", "family", "genus", "species")

FASTA_SUFFIXES = (".fasta", ".fna", ".fa", ".fasta.gz", ".fna.gz", ".fa.gz")

NUMERIC_LINEAGE = re.compile(r"^[0-9]+(;[0-9]+)*$")

ENA_TAXONOMY = "https://www.ebi.ac.uk/ena/taxonomy/rest/tax-id/{}"

# taxa genome_uploader falls back to for a not-classified MAG
FALLBACK_TAXA = {"77133": "uncultured bacterium", "115547": "uncultured archaeon"}

# Galaxy dataset states that mean "do not use this element"
FAILED_STATES = {"error", "failed_metadata", "discarded"}

SEQSUBMIT_NF = "workflows/genomesubmit.nf"
SEQSUBMIT_OLD = (
    "genome_evaluation_input: !(meta.completeness) || !(meta.contamination) || "
    "!(meta.stats_generation_software)"
)
SEQSUBMIT_NEW = (
    "genome_evaluation_input: meta.completeness == null || meta.contamination == null || "
    "meta.stats_generation_software == null"
)


# --------------------------------------------------------------------------- #
# repository files
# --------------------------------------------------------------------------- #
def strip_fasta(name: str) -> str:
    """'SRR1_binette_bin4.fasta' -> 'SRR1_binette_bin4'."""
    out = (name or "").strip()
    changed = True
    while changed:
        changed = False
        for suffix in FASTA_SUFFIXES:
            if out.endswith(suffix):
                out = out[: -len(suffix)]
                changed = True
    return out


def read_table(path: str, key: str):
    """Read a table into {key: row}, without FASTA suffixes.

    The delimiter is taken from the header line rather than the file name:
    the downloaded tables are not named consistently (``kmetashot.tsv`` is
    comma separated, the others are not).
    """
    if not os.path.exists(path):
        sys.exit(f"missing file: {path}")
    with open(path, newline="") as fh:
        header = fh.readline()
        delimiter = "\t" if header.count("\t") >= header.count(",") else ","
        fh.seek(0)
        rows = {strip_fasta(row[key]): row for row in csv.DictReader(fh, delimiter=delimiter)}
    if not rows:
        sys.exit(f"{path}: no rows")
    return rows


def read_coverm(path: str) -> dict:
    """{mag_id: {run_accession: mean coverage}} from coverm.tsv.

    The first column is the genome, every other column is named after a run
    accession, so the run a MAG came from has to be picked out of the header.
    """
    if not os.path.exists(path):
        sys.exit(f"missing file: {path}")
    out = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        for row in reader:
            if row:
                out[strip_fasta(row[0])] = dict(zip(header[1:], row[1:]))
    return out


def load_use_case(use_case_dir: str) -> dict:
    """Read the use case directory: its config and the four per-MAG tables."""
    config_path = os.path.join(use_case_dir, "ena_submission.yaml")
    if not os.path.exists(config_path):
        sys.exit(f"missing config: {config_path}")
    with open(config_path) as fh:
        config = yaml.safe_load(fh) or {}

    for field in YAML_FIELDS:
        if not config.get(field):
            sys.exit(f"{config_path}: '{field}' is required")
    if not config.get("mags", {}).get("history") or not config.get("mags", {}).get("tag"):
        sys.exit(f"{config_path}: 'mags: {{history: ..., tag: ...}}' is required")
    for name in ("binning_software", "binning_parameters", "stats_generation_software"):
        if "\t" in config[name] or "\n" in config[name]:
            sys.exit(f"{config_path}: '{name}' must not contain tabs or newlines")

    return {
        "config": config,
        "checkm2": read_table(os.path.join(use_case_dir, "checkm2.tsv"), "Name"),
        "kmetashot": read_table(os.path.join(use_case_dir, "kmetashot.tsv"), "bin"),
        "gtdb": read_table(os.path.join(use_case_dir, "gtdb.tsv"), "user_genome"),
        "coverm": read_coverm(os.path.join(use_case_dir, "coverm.tsv")),
    }


def build_lineage(mag_id: str, kmetashot: dict, gtdb: dict) -> str:
    """NCBI taxid lineage from kmetashot, or an 'Unclassified X' placeholder.

    kmetashot.tsv stores NCBI *taxids* in its rank columns; ENA resolves them
    to names itself.  When kmetashot has no superkingdom the MAG was not
    classified, and genome_uploader falls back to 'uncultured
    bacterium'/'uncultured archaeon' when it sees these placeholders.
    """
    row = kmetashot.get(mag_id, {})
    if row.get("superkingdom", "0") not in ("", "0", None):
        return ";".join(row[rank] for rank in RANKS if row.get(rank, "0") not in ("", "0"))

    classification = (gtdb.get(mag_id, {}).get("classification") or "").strip()
    for kingdom in ("Archaea", "Eukaryota", "Bacteria"):
        if kingdom in classification:
            return f"Unclassified {kingdom}"
    return "Unclassified Bacteria"


# --------------------------------------------------------------------------- #
# Galaxy: find the dereplicated bins collection and download the FASTAs
# --------------------------------------------------------------------------- #
def read_api_key() -> str:
    """GALAXY_API from the environment or from the repository .env file."""
    if os.environ.get("GALAXY_API"):
        return os.environ["GALAXY_API"]
    dotenv = os.path.join(REPO_ROOT, ".env")
    if os.path.exists(dotenv):
        with open(dotenv) as fh:
            for line in fh:
                if line.startswith("GALAXY_API="):
                    return line.split("=", 1)[1].strip()
    sys.exit("GALAXY_API not found: set it in the environment or in ./.env")


def find_collection(gi, histories_yaml: str, history_name: str, tag: str) -> tuple:
    """Return (history_id, collection_id, collection_name) for the tagged collection."""
    if not os.path.exists(histories_yaml):
        sys.exit(f"missing file: {histories_yaml}")
    with open(histories_yaml) as fh:
        conf = yaml.safe_load(fh)

    histories = conf.get("histories", [])
    matches = [h for h in histories if h.get("name") == history_name]
    if not matches:
        matches = [h for h in histories if history_name.lower() in (h.get("name") or "").lower()]
    if not matches:
        sys.exit(f"no history named {history_name!r} in {histories_yaml}")
    if len(matches) > 1:
        sys.exit(f"{history_name!r} matches {len(matches)} histories in {histories_yaml}")

    history_id = matches[0]["id"]
    collections = [c for c in gi.histories.show_history(history_id, contents=True) if c["type"] == "collection"]
    tagged = [c for c in collections if tag in (c.get("tags") or [])]

    if not tagged:
        seen = sorted({t for c in collections for t in (c.get("tags") or [])})
        sys.exit(
            f"no collection tagged {tag!r} in history {history_name!r} ({history_id}).\n"
            f"tags present on its {len(collections)} collections: {seen}"
        )
    if len(tagged) > 1:
        listing = "\n".join(f"  {c['id']}  {c.get('name')}" for c in tagged)
        sys.exit(f"{len(tagged)} collections are tagged {tag!r} in {history_id}:\n{listing}")

    return history_id, tagged[0]["id"], tagged[0].get("name")


def collect_mags(gi, collection_id: str) -> dict:
    """{mag_id: dataset_id} for every FASTA of a collection (lists are walked)."""
    found = {}
    pending = [collection_id]
    while pending:
        info = gi.dataset_collections.show_dataset_collection(pending.pop())
        for element in info["elements"]:
            obj = element["object"]
            if "elements" in obj:  # nested collection
                pending.append(obj["id"])
                continue
            name = element.get("element_identifier") or obj.get("name") or ""
            mag_id = strip_fasta(name)
            if not mag_id:
                continue
            if obj.get("state") in FAILED_STATES:
                print(f"  WARNING: {mag_id} is {obj['state']} -> skipped")
                continue
            found.setdefault(mag_id, obj["id"])
    return found


def download_one(gi, dataset_id: str, target: str, retries: int = 3) -> None:
    """Download a dataset to target, gzipped (as the samplesheet expects)."""
    data = None
    for attempt in range(retries):
        try:
            data = gi.datasets.download_dataset(dataset_id, use_default_filename=False)
            break
        except Exception as exc:  # noqa: BLE001 - retried below
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1}/{retries}: {exc}")
            time.sleep(3 * (attempt + 1))

    part = target + ".part"
    if data[:2] == b"\x1f\x8b":  # already gzipped
        with open(part, "wb") as fh:
            fh.write(data)
    else:
        with gzip.GzipFile(part, "wb") as fh:
            fh.write(data)
    os.replace(part, target)


def fetch_mags(use_case_dir: str, fasta_dir: str, history_name: str, tag: str,
               overwrite: bool, dry_run: bool) -> dict:
    """Resolve the collection and (unless dry run) download every MAG FASTA."""
    gi = GalaxyInstance(url=yaml.safe_load(open(os.path.join(use_case_dir, "histories.yaml")))["galaxy-url"],
                        key=read_api_key())
    history_id, collection_id, collection_name = find_collection(
        gi, os.path.join(use_case_dir, "histories.yaml"), history_name, tag
    )
    print(f"history     : {history_name} ({history_id})")
    print(f"collection  : {collection_name} ({collection_id}, tag {tag})")

    found = collect_mags(gi, collection_id)
    print(f"MAG FASTA files in collection: {len(found)}")

    if dry_run:
        for mag_id in sorted(found)[:5]:
            print(f"  would download {mag_id}.fasta.gz  <- dataset {found[mag_id]}")
        if len(found) > 5:
            print(f"  ... +{len(found) - 5} more")
        return found

    os.makedirs(fasta_dir, exist_ok=True)
    map_path = os.path.join(fasta_dir, "mag_map.json")
    with open(map_path, "w") as fh:
        json.dump({f"{k}.fasta": v for k, v in sorted(found.items())}, fh, indent=2)
        fh.write("\n")

    done = skipped = failed = 0
    for i, (mag_id, dataset_id) in enumerate(sorted(found.items()), 1):
        target = os.path.join(fasta_dir, f"{mag_id}.fasta.gz")
        if not overwrite and os.path.exists(target) and os.path.getsize(target) > 0:
            skipped += 1
            continue
        try:
            download_one(gi, dataset_id, target)
            done += 1
        except Exception as exc:  # noqa: BLE001 - reported, run continues
            failed += 1
            print(f"  FAILED {mag_id}: {exc}")
        if i % 50 == 0 or i == len(found):
            print(f"  {i}/{len(found)}  downloaded={done} skipped={skipped} failed={failed}")
    print(f"FASTAs      : {done} downloaded, {skipped} already there, {failed} failed -> {fasta_dir}/")
    if failed:
        sys.exit(f"{failed} FASTA file(s) could not be downloaded")
    return found


# --------------------------------------------------------------------------- #
# ENA taxonomy: a lineage must end at a submittable taxon
# --------------------------------------------------------------------------- #
def fetch_taxon(taxid: str) -> dict:
    with urlopen(ENA_TAXONOMY.format(taxid), timeout=30) as response:
        data = json.loads(response.read().decode())
    if isinstance(data, list):
        data = data[0] if data else {}
    return data


def is_submittable(taxon: dict) -> bool:
    # the ENA API returns the flag as the string "true" / "false"
    return str(taxon.get("submittable", "")).strip().lower() == "true"


def audit_lineages(rows: list, fix: bool = True, workers: int = 8) -> dict:
    """Check every lineage and repair those ENA would reject.

    A lineage that stops at genus level makes ENA reject the *whole* batch
    ("In sample, alias: X. Taxon Bacteria id 2 is not submittable."), so this
    runs before anything is submitted.  Sibling MAGs of the same lineage
    usually continue one level further; that taxon is the repair, preferring a
    generic "<Genus> sp." placeholder.
    """
    numeric = [(row["id"], row["NCBI_lineage"].strip()) for row in rows
               if NUMERIC_LINEAGE.match(row["NCBI_lineage"].strip())]
    placeholders = Counter(row["NCBI_lineage"].strip() for row in rows
                           if not NUMERIC_LINEAGE.match(row["NCBI_lineage"].strip()))
    print(f"lineages    : {len(numeric)} numeric taxid chains, "
          f"{sum(placeholders.values())} placeholders {dict(placeholders)}")

    if not numeric:
        sys.exit("no numeric lineage to check - is kmetashot.tsv the right file?")

    final_of = {lineage: lineage.split(";")[-1] for _, lineage in numeric}
    taxids = sorted(set(final_of.values()))
    print(f"              querying ENA for {len(taxids)} distinct final taxids ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        taxa = dict(zip(taxids, pool.map(fetch_taxon, taxids)))

    for taxid, taxon in sorted(taxa.items()):
        if not is_submittable(taxon):
            print(f"              NOT submittable: {taxid} ({taxon.get('scientificName')}, "
                  f"rank {taxon.get('rank')})")

    # children used by sibling MAGs, keyed by the lineage prefix
    by_prefix = defaultdict(set)
    for _, lineage in numeric:
        parts = lineage.split(";")
        for i in range(1, len(parts)):
            by_prefix[";".join(parts[:i])].add(parts[i])

    fixed, unresolved = [], []
    for row in rows:
        lineage = row["NCBI_lineage"].strip()
        if not NUMERIC_LINEAGE.match(lineage):
            continue
        if is_submittable(taxa.get(lineage.split(";")[-1], {})):
            continue

        deeper = sorted(by_prefix.get(lineage, set()) - {lineage.split(";")[-1]}, key=int)
        candidates = []
        for taxid in deeper[:8]:
            taxon = taxa.get(taxid) or fetch_taxon(taxid)
            taxa[taxid] = taxon
            candidates.append((taxid, taxon.get("scientificName", "?"), is_submittable(taxon)))

        pick = next((t for t, name, ok in candidates if ok and name.endswith(" sp.")), None)
        if pick is None:
            pick = next((t for t, _, ok in candidates if ok), None)

        if pick and fix:
            row["NCBI_lineage"] = f"{lineage};{pick}"
            fixed.append((row["id"], lineage, row["NCBI_lineage"]))
        else:
            unresolved.append((row["id"], lineage, candidates))

    for mag_id, before, after in fixed:
        print(f"auto-fixed  : {mag_id}\n                {before}\n             -> {after}")

    if unresolved:
        for mag_id, lineage, candidates in unresolved:
            print(f"\nUNRESOLVED: {mag_id}\n  lineage: {lineage}")
            for taxid, name, ok in candidates:
                print(f"    candidate {taxid:<10} {name:<40} submittable={ok}")
        sys.exit(f"\n{len(unresolved)} lineage(s) end at a taxon ENA does not accept and no "
                 f"sibling MAG provides a submittable continuation.\n"
                 f"Fix NCBI_lineage in the samplesheet by hand, or add the missing rank to kmetashot.tsv.")

    # the ENA fallbacks for not-classified MAGs must be submittable too
    for taxid, name in FALLBACK_TAXA.items():
        if placeholders and not is_submittable(fetch_taxon(taxid)):
            sys.exit(f"ENA fallback taxon {taxid} ({name}) is not submittable")

    lengths = Counter(len(row["NCBI_lineage"].split(";")) for row in rows
                      if NUMERIC_LINEAGE.match(row["NCBI_lineage"].strip()))
    if len(lengths) > 1:
        print(f"WARNING     : lineages have mixed depth {dict(sorted(lengths.items()))} - "
              f"every chain must still end at a submittable taxon, which it does")

    if not fixed:
        print("              every lineage ends at a submittable taxon")
    return {"fixed": fixed, "numeric": len(numeric), "placeholders": sum(placeholders.values())}


# --------------------------------------------------------------------------- #
# seqsubmit: 0.0 is a measurement, not a missing value
# --------------------------------------------------------------------------- #
def patch_seqsubmit(checkout: str, revert: bool = False) -> None:
    """Fix (or revert) the truthiness bug that re-runs CheckM2 for contamination=0.0.

    workflows/genomesubmit.nf asks for genome evaluation with
    ``!(meta.completeness) || !(meta.contamination) || !(meta.stats_generation_software)``.
    In Groovy ``0.0`` is falsy, so a MAG that is legitimately perfect
    (contamination = 0.0) is treated as "no measurement" and CheckM2 is run on
    it again - a ~1 GB database download per MAG.  A null check is correct:
    the input schema allows ``>= 0`` for both statistics.
    """
    path = os.path.join(checkout, SEQSUBMIT_NF)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")
    text = open(path).read()

    if revert:
        if SEQSUBMIT_NEW not in text:
            print("seqsubmit   : already unpatched")
            return
        open(path, "w").write(text.replace(SEQSUBMIT_NEW, SEQSUBMIT_OLD))
        print(f"seqsubmit   : reverted {path}")
        return

    if SEQSUBMIT_NEW in text:
        print("seqsubmit   : already patched")
        return
    if SEQSUBMIT_OLD not in text:
        sys.exit(f"expected line not found in {path} - has upstream changed it already?")
    if text.count(SEQSUBMIT_OLD) != 1:
        sys.exit(f"expected exactly 1 occurrence in {path}, found {text.count(SEQSUBMIT_OLD)}")
    open(path, "w").write(text.replace(SEQSUBMIT_OLD, SEQSUBMIT_NEW))
    print(f"seqsubmit   : patched {path} (contamination=0.0 no longer means 'missing')")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("use_case_dir", nargs="?", help="e.g. data/use-cases/bee-gut")
    ap.add_argument("--out", default="samplesheet.csv", help="samplesheet to write (default: samplesheet.csv)")
    ap.add_argument("--fasta-dir", default="mags",
                    help="where the MAG FASTAs are/land; written into the sheet verbatim (default: mags)")
    ap.add_argument("--id-prefix", default="",
                    help="brand every submitted id with this prefix (e.g. FAIRyMAGs2_); the "
                         "sample alias, assembly name and ENA analysis alias then carry it. "
                         "Allowed: A-Za-z0-9_.-  (default: empty, ids are used as they are)")
    ap.add_argument("--history", help="override mags.history from ena_submission.yaml")
    ap.add_argument("--tag", help="override mags.tag from ena_submission.yaml")
    ap.add_argument("--no-download", action="store_true",
                    help="skip Galaxy entirely (FASTAs must already be in --fasta-dir)")
    ap.add_argument("--dry-run", action="store_true", help="report what would happen, write nothing")
    ap.add_argument("--overwrite", action="store_true", help="re-download FASTAs that exist already")
    ap.add_argument("--no-fix", action="store_true", help="fail on a rejected lineage instead of repairing it")
    ap.add_argument("--patch-seqsubmit", metavar="CHECKOUT", help="apply the contamination=0.0 fix and stop")
    ap.add_argument("--revert-seqsubmit", metavar="CHECKOUT", help="undo that fix and stop")
    ap.add_argument("--workers", type=int, default=8, help="parallel ENA taxonomy lookups (default: 8)")
    args = ap.parse_args()

    if args.patch_seqsubmit:
        return patch_seqsubmit(args.patch_seqsubmit)
    if args.revert_seqsubmit:
        return patch_seqsubmit(args.revert_seqsubmit, revert=True)
    if not args.use_case_dir:
        ap.error("a use case directory is required (e.g. data/use-cases/bee-gut)")

    prefix = args.id_prefix or ""
    if prefix and not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        ap.error(f"--id-prefix {prefix!r} is not usable: ENA aliases and assembly names "
                 f"allow only A-Za-z0-9_.- (no spaces, slashes or commas)")

    use_case = load_use_case(args.use_case_dir)
    config = use_case["config"]
    checkm2, coverm = use_case["checkm2"], use_case["coverm"]
    print(f"use case    : {args.use_case_dir} ({len(checkm2)} MAGs in checkm2.tsv)")
    if prefix:
        print(f"id prefix   : {prefix!r} -> ENA sample alias, assembly name and analysis alias")

    # 1. + 2. the MAG FASTAs -------------------------------------------------
    mags = {}
    if args.no_download:
        print("Galaxy      : skipped (--no-download)")
    else:
        mags = fetch_mags(
            args.use_case_dir,
            args.fasta_dir,
            args.history or config["mags"]["history"],
            args.tag or config["mags"]["tag"],
            args.overwrite,
            args.dry_run,
        )
        if args.dry_run:
            return

    # 3. the samplesheet rows ------------------------------------------------
    rows = []
    for mag_id, checkm in checkm2.items():  # checkm2.tsv row order is kept
        run = mag_id.split("_")[0]
        if mag_id not in coverm:
            sys.exit(f"{mag_id}: not in coverm.tsv")
        if run not in coverm[mag_id]:
            sys.exit(f"{mag_id}: coverm.tsv has no column for run {run}")
        if mags and mag_id not in mags:
            sys.exit(f"{mag_id}: not in the Galaxy collection")
        rows.append({
            "id": mag_id,
            "fasta": os.path.join(args.fasta_dir, f"{mag_id}.fasta.gz"),
            "accession": run,
            "fastq_1": "",
            "fastq_2": "",
            "completeness": checkm["Completeness"].strip(),
            "contamination": checkm["Contamination"].strip(),
            "genome_coverage": coverm[mag_id][run].strip(),
            "co-assembly": "false",
            "RNA_presence": "",
            "NCBI_lineage": build_lineage(mag_id, use_case["kmetashot"], use_case["gtdb"]),
            **{field: config[field] for field in YAML_FIELDS},
        })

    if mags and set(mags) != set(checkm2):
        only_galaxy = sorted(set(mags) - set(checkm2))
        only_table = sorted(set(checkm2) - set(mags))
        sys.exit(f"the collection and checkm2.tsv disagree: {len(only_galaxy)} only in Galaxy "
                 f"{only_galaxy[:5]}, {len(only_table)} only in checkm2.tsv {only_table[:5]}")

    # 4. taxonomy ------------------------------------------------------------
    audit_lineages(rows, fix=not args.no_fix, workers=args.workers)

    # 5. write ---------------------------------------------------------------
    missing = [row for row in rows if not os.path.exists(row["fasta"])]

    # brand the submission identity: the sample alias, the assembly name and
    # the ENA analysis alias (webin-genome-<id>) all come from this column
    for row in rows:
        row["id"] = prefix + row["id"]

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with open(args.out, newline="") as fh:
        check = list(csv.DictReader(fh))
    assert len(check) == len(rows), f"row count changed: {len(rows)} -> {len(check)}"
    assert all(row["id"] and row["fasta"] and row["accession"] for row in check)
    assert all(row["id"].startswith(prefix) for row in check)
    assert all(os.path.basename(row["fasta"]) == row["id"][len(prefix):] + ".fasta.gz"
               for row in check)
    assert len({row["id"] for row in check}) == len(check), "ids must be unique"

    print(f"wrote       : {args.out} ({len(check)} rows, {len(COLUMNS)} columns)")
    if missing:
        print(f"WARNING     : {len(missing)} FASTA file(s) not in {args.fasta_dir} yet, e.g. "
              f"{[os.path.basename(row['fasta']) for row in missing[:3]]}")
        print("              run without --no-download to fetch them")

    study = (config.get("ena") or {}).get("study")
    centre = (config.get("ena") or {}).get("centre_name")
    print(f"\nrun the pipeline, from the directory holding {args.out} and {args.fasta_dir}/:")
    for mode in ("validate", "submit"):
        print(f"  nextflow run /path/to/seqsubmit -profile docker -resume \\\n"
              f"      --input {args.out} --outdir out_{mode} --mode mags \\\n"
              f"      --submission_study {study} --centre_name {centre} --upload_tpa true \\\n"
              f"      --test_upload false --is_private false --webincli_mode {mode}")
    print("  # validate registers the samples in ENA without uploading anything -"
          " always run it first")


if __name__ == "__main__":
    main()
