#!/usr/bin/env python3
"""Freeze IWC definitions and migrate the four FAIRyMAGs use-case manifests.

Preparation performs only remote reads. Workflow import and execution belong
to run_workflows.py --campaign, so the resulting plan can be reviewed first.
"""

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from bioblend.galaxy import GalaxyInstance
import yaml

import run_workflows as runner
from workflow_campaign import atomic_json


ROOT = Path(__file__).resolve().parents[1]
GH = str(Path.home() / ".local/bin/gh")
PATHS = {
    "qc": "workflows/read-preprocessing/short-read-qc-trimming/short-read-quality-control-and-trimming.ga",
    "host": "workflows/microbiome/host-contamination-removal/host-contamination-removal-short-reads/host-or-contamination-removal-on-short-reads.ga",
    "parent": "workflows/microbiome/mags-building/MAGs-generation.ga",
    "taxonomy": "workflows/microbiome/mags-taxonomy-annotation/MAGs-taxonomy-annotation.ga",
    "genome": "workflows/microbiome/mag-genome-annotation-parallel/MAG-Genome-Annotation-Parallel.ga",
    "functional": "workflows/genome_annotation/functional-annotation/functional-annotation-of-sequences/Functional_annotation_of_sequences.ga",
}


def github(path):
    return json.loads(subprocess.check_output([GH, "api", "repos/galaxyproject/iwc/" + path]))


def prepare(directory, campaign):
    directory = Path(directory).resolve()
    if (directory / "campaign.yaml").exists():
        raise RuntimeError("Campaign already prepared; do not replace a frozen campaign")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workflows").mkdir(exist_ok=True)
    main = github("commits/main")["sha"]
    parent = json.loads(subprocess.check_output([GH, "pr", "view", "1454", "--repo", "galaxyproject/iwc", "--json", "headRefOid"]))["headRefOid"]
    workflows, definitions = {}, {}
    for key, path in PATHS.items():
        commit = parent if key == "parent" else main
        content = base64.b64decode(github(f"contents/{path}?ref={commit}")["content"])
        definition = json.loads(content)
        file = f"workflows/{key}.ga"
        (directory / file).write_bytes(content)
        workflows[key] = {"name": definition["name"], "release": definition.get("release"),
                          "commit": commit, "file": file, "sha256": hashlib.sha256(content).hexdigest(),
                          "source": "PR #1454" if key == "parent" else "merged IWC main"}
        definitions[key] = definition
    gi = GalaxyInstance(url="https://usegalaxy.eu", key=runner.api_key_from_env())
    steps, migrations = [], []

    def output_ref(step, output):
        return {"step": step, "output": output}

    def typed(workflow, parameters):
        states = {s["label"]: json.loads(s["tool_state"]) for s in definitions[workflow]["steps"].values()
                  if s["type"] == "parameter_input"}
        result = {}
        for key, value in parameters.items():
            typ = states[key].get("parameter_type")
            if value is not None and typ in {"integer", "float"}:
                value = (int if typ == "integer" else float)(value)
            result[key] = value
        return result

    def add(case, name, workflow, inputs, parameters, required):
        key = f"{case}.{name}"
        steps.append({"name": key, "use_case": case, "workflow": workflow,
                      "inputs": inputs, "parameters": typed(workflow, parameters), "required_outputs": required})
        return key

    for case in ["bee-gut", "termite-head", "macroalgal-epiphytic", "air"]:
        source_path = ROOT / "data/use-cases" / case / "workflow.yaml"
        old = json.loads(json.dumps(yaml.safe_load(source_path.read_text()), default=str))
        atomic_json(directory / "original-manifests" / (case + ".json"), old)
        stages = {s["name"]: s for s in old["steps"]}

        def source(key):
            selector = old["sources"][key]
            hist = runner.resolve_history(gi, selector["history"])
            value = runner.resolve_content(gi, hist["id"], selector["content"])
            return {**value, "history_id": hist["id"]}

        qc = add(case, "quality_control", "qc", {"Raw reads": source("raw_reads")},
                 stages["quality_control"]["parameters"], ["fastp trimmed reads"])
        reads = output_ref(qc, "fastp trimmed reads")
        if case == "termite-head":
            for stage, ref in [("remove_cryptotermes", "cryptotermes_reference"), ("remove_cdom", "cdom_reference")]:
                params = copy.deepcopy(stages[stage]["parameters"])
                # The built-in selector is ignored in custom-reference mode,
                # but remains a required workflow input on this merged release.
                host = add(case, stage, "host", {"Short-reads": reads, "Reference genome file": source(ref)},
                           params, ["Contamination Filtered Reads from Reference Input"])
                reads = output_ref(host, "Contamination Filtered Reads from Reference Input")
        else:
            params = {"Choose Reference Genome": "Use a built-in genome index",
                      "Host/Contaminant Reference Genome": "apiMel3" if case == "bee-gut" else "hg38full"}
            host = add(case, "contamination_removal", "host", {"Short-reads": reads}, params,
                       ["Contamination Filtered Reads from Local Index"])
            reads = output_ref(host, "Contamination Filtered Reads from Local Index")

        branches = {"bee-gut": [("single", "mags"), ("multi", "mags")],
                    "termite-head": [("grouped", "mags"), ("individual", "mags")],
                    "macroalgal-epiphytic": [("single", "mags")],
                    "air": [("megahit", "mags_megahit"), ("metaspades", "mags_metaspades")]}[case]
        for branch, old_stage in branches:
            branch_reads = reads
            if case == "air" and branch == "metaspades":
                filter_key = case + ".exclude_metaspades_memory_sample"
                steps.append({"name": filter_key, "use_case": case, "operation": "exclude-sample",
                              "inputs": {"input": reads}, "exclude": "ERR9966622", "expected_count": 13,
                              "required_outputs": ["Filtered reads"]})
                branch_reads = output_ref(filter_key, "Filtered reads")
            params = copy.deepcopy(stages[old_stage]["parameters"])
            removed = params.pop("Read length (CONCOCT)", None)
            params.update({"Use grouping?": "Use grouping" if branch == "grouped" else "Use original samples",
                           "Binning strategy": "Multi-coverage binning (all samples)" if branch == "multi" else "Single-coverage binning (own/merged reads)",
                           "Bin refinement tool": "Binette", "Run COMEBin": True,
                           "Run SemiBin?": True, "Run CONCOCT?": True, "Run MetaBAT2?": True, "Run MaxBin2?": True})
            inputs = {"Trimmed reads": branch_reads}
            if branch == "grouped":
                inputs["Metadata"] = {"src": "hda", "id": "26c75dcccb616ac86b50a9fcbbc32cd4",
                                      "history_id": "11ac94870d0bb33aa2437ee428efd50a"}
                params["Sample grouping column"] = "group"
            labels = [o["label"] for s in definitions["parent"]["steps"].values() for o in s.get("workflow_outputs", [])]
            bins = [label for label in labels if "dereplicated" in label.lower() and "bin" in label.lower()]
            if len(bins) != 1:
                raise RuntimeError(f"Cannot uniquely resolve Parent representative bins: {labels}")
            mag = add(case, "mags_" + branch, "parent", inputs, params, bins)
            migrations.append({"case": case, "branch": branch, "removed_manual_read_length": removed,
                               "parameters": typed("parent", params), "inputs": inputs})
            for stage, workflow, input_name in [("taxonomy", "taxonomy", "Sequence collection"),
                                                 ("genome_annotation", "genome", "Bacterial Genomes"),
                                                 ("functional_annotation", "functional", "Sequence collection")]:
                source_stage = stage + ("_" + branch if case == "air" else "")
                parameters = copy.deepcopy(stages[source_stage]["parameters"])
                if workflow == "functional":
                    parameters["Generate eggNOG annotation abundance tables"] = True
                required = {"taxonomy": ["GTDB-Tk summary files", "full GTDB to NCBI mapping"],
                            "genome": ["PlasmidFinder Merged Summary", "bakta cut annotation summary", "Integron Finder Merged Output", "ISEScan Merged Output"],
                            "functional": ["interproscan tabular", "kegg contig table", "eggNOG-mapper annotation abundance table"]}[workflow]
                add(case, stage + "_" + branch, workflow, {input_name: output_ref(mag, bins[0])}, parameters, required)
    doc = {"campaign": campaign, "galaxy_url": "https://usegalaxy.eu", "poll_seconds": 300,
           "workflows": workflows, "steps": steps,
           "notes": {"air_exclusion": "ERR9966622", "termite_groups": "existing 28-sample / 8-group mapping",
                     "tooldistillator": "No disable control in merged 0.6; retain unchanged, collect biological outputs independently"}}
    (directory / "campaign.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    atomic_json(directory / "parameter-migrations.json", migrations)
    print(f"Prepared {len(steps)} stages in {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    prepare(args.directory, args.name)
