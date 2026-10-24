"""Restartable campaign execution used by run_workflows.py --campaign.

Submission is intentionally never retried automatically: a lost POST response
is an ambiguous submission, not permission to launch an expensive duplicate.
All state and exports are written atomically. Credentials never enter state.
"""

import datetime
import fcntl
import hashlib
import json
import re
from pathlib import Path
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml


class Blocked(RuntimeError):
    pass


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Campaign:
    def __init__(self, config, key):
        self.path = Path(config).resolve()
        self.doc = yaml.safe_load(self.path.read_text())
        self.root = self.path.parent
        self.state_path = self.root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "campaign": self.doc["campaign"], "workflows": {}, "runs": {}, "created": now(),
        }
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.state.get("config_sha256", digest) != digest:
            raise Blocked("Campaign config changed after initialization; reconcile state explicitly")
        self.state["config_sha256"] = digest
        self.session = requests.Session()
        self.session.headers["x-api-key"] = key
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=2, status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )))
        self.base = self.doc["galaxy_url"].rstrip("/") + "/api/"

    def api(self, method, path, **kwargs):
        response = self.session.request(method, self.base + path, timeout=(30, 180), **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    def save(self):
        self.state["updated"] = now()
        atomic_json(self.state_path, self.state)

    def contents(self, hid):
        items, offset = [], 0
        while True:
            batch = self.api("GET", f"histories/{hid}/contents", params={
                "limit": 500, "offset": offset,
            })
            items.extend(batch)
            if len(batch) < 500:
                return items
            offset += len(batch)

    def collection(self, hid, cid):
        return self.api("GET", f"histories/{hid}/contents/dataset_collections/{cid}")

    def check_data(self, ref, hid=None):
        if ref["src"] == "hda":
            obj = self.api("GET", "datasets/" + ref["id"])
            if obj.get("state") != "ok" or obj.get("deleted") or obj.get("purged"):
                raise Blocked(f"Required dataset {ref['id']} is not usable")
            if obj.get("extension") == "expression.json":
                raise Blocked(f"Required dataset {ref['id']} is a skipped placeholder")
            return obj
        if not hid:
            hid = self.api("GET", "dataset_collections/" + ref["id"])["history_id"]
        obj = self.collection(hid, ref["id"])
        if obj.get("populated_state") != "ok" or obj.get("deleted"):
            raise Blocked(f"Required collection {ref['id']} is not populated")

        def leaves(collection):
            for e in collection.get("elements", []):
                if e["element_type"] == "dataset_collection":
                    yield from leaves(e["object"])
                else:
                    yield e["object"]

        datasets = list(leaves(obj))
        if not datasets:
            raise Blocked(f"Required collection {ref['id']} has no datasets")
        for dataset in datasets:
            if dataset.get("state") != "ok" or dataset.get("extension") == "expression.json" or dataset.get("deleted"):
                raise Blocked(f"Collection {ref['id']} contains failed, pending or skipped data")
        return obj

    def install_workflows(self):
        for key, spec in self.doc["workflows"].items():
            path = self.root / spec["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise Blocked(f"Workflow snapshot checksum mismatch: {key}")
            previous = self.state["workflows"].get(key)
            if previous:
                if previous.get("status") != "imported":
                    raise Blocked(f"Ambiguous workflow import {key}; inspect before retrying")
                continue
            definition = json.loads(path.read_text())
            definition["tags"] = [self.doc["campaign"], "FAIRyMAGs", "agentic",
                                  f"workflow:{key}", f"workflow-version:{spec['release']}",
                                  f"workflow-commit:{spec['commit'][:12]}"]
            self.state["workflows"][key] = {"status": "importing", **spec}
            self.save()
            result = self.api("POST", "workflows/upload", json={"workflow": definition})
            wid = result["id"]
            self.state["workflows"][key].update(id=wid, status="verifying")
            self.save()
            self.api("PUT", "workflows/" + wid, json={"tags": definition["tags"]})
            exported = self.api("GET", f"workflows/{wid}/download")
            # Imports normalize UI state, so verify runtime pins and interface,
            # not byte equality of the normalized export.
            def pins(w):
                result = []
                for step in w["steps"].values():
                    if step["type"] == "tool":
                        result.append((step.get("tool_id"), step.get("tool_version")))
                    if step.get("subworkflow"):
                        result.extend(pins(step["subworkflow"]))
                return sorted(result)
            if pins(exported) != pins(definition):
                self.state["workflows"][key]["status"] = "pin-mismatch"
                self.save()
                raise Blocked(f"Imported workflow tool pins differ: {key}")
            atomic_json(self.root / "workflows" / f"{key}.eu-export.ga", exported)
            self.state["workflows"][key]["status"] = "imported"
            self.save()

    def workflow_inputs(self, key):
        definition = json.loads((self.root / self.doc["workflows"][key]["file"]).read_text())
        return {s["label"]: s for s in definition["steps"].values()
                if s["type"] in {"parameter_input", "data_input", "data_collection_input"}}

    def preflight(self):
        known = set()
        for step in self.doc["steps"]:
            if step["name"] in known:
                raise Blocked(f"Duplicate stage {step['name']}")
            if step.get("workflow"):
                inputs = self.workflow_inputs(step["workflow"])
                unknown = (set(step.get("inputs", {})) | set(step.get("parameters", {}))) - set(inputs)
                if unknown:
                    raise Blocked(f"Unknown inputs in {step['name']}: {sorted(unknown)}")
                for label, value in step.get("parameters", {}).items():
                    state = inputs[label]["tool_state"]
                    state = json.loads(state) if isinstance(state, str) else state
                    typ = state.get("parameter_type")
                    if typ == "boolean" and type(value) is not bool:
                        raise Blocked(f"{step['name']}: {label} must be boolean")
                    if typ == "integer" and type(value) is not int:
                        raise Blocked(f"{step['name']}: {label} must be integer")
            for ref in step.get("inputs", {}).values():
                if "step" in ref and ref["step"] not in known:
                    raise Blocked(f"Invalid dependency in {step['name']}")
            known.add(step["name"])

    def resolve_inputs(self, step):
        resolved = {}
        for label, ref in step.get("inputs", {}).items():
            if "step" in ref:
                upstream = self.state["runs"].get(ref["step"], {})
                if upstream.get("status") not in {"complete", "complete-with-errors"}:
                    return None
                if ref["output"] not in upstream.get("outputs", {}):
                    raise Blocked(f"Upstream output missing: {ref['step']} / {ref['output']}")
                value = upstream["outputs"][ref["output"]]
                self.check_data(value, upstream["history_id"])
            else:
                value = {k: ref[k] for k in ("id", "src")}
                self.check_data(value, ref.get("history_id"))
            resolved[label] = value
        return resolved

    def submit(self, step, values):
        key = step["name"]
        run = self.state["runs"].setdefault(key, {})
        if run.get("status") in {"creating", "submitting"}:
            raise Blocked(f"Ambiguous POST for {key}; reconcile existing history/invocation first")
        spec = self.doc["workflows"].get(step.get("workflow"), {})
        tags = ["FAIRyMAGs", "agentic", self.doc["campaign"], f"use-case:{step['use_case']}",
                f"stage:{key}", f"workflow:{step.get('workflow', 'filter-air-metaspades')}",
                f"workflow-version:{spec.get('release', '1')}"]
        if spec:
            tags.append(f"workflow-commit:{spec['commit'][:12]}")
            tags.append("workflow-name:" + re.sub(r"[^a-z0-9]+", "-", spec["name"].lower()).strip("-"))
        if not run.get("history_id"):
            run.update(status="creating", parameters=step.get("parameters", {}), tags=tags)
            self.save()
            name = f"{self.doc['campaign']} | {step['use_case']} | {key} | {spec.get('name', 'Air sample filter')} v{spec.get('release', '1')}"
            history = self.api("POST", "histories", json={"name": name})
            run.update(history_id=history["id"], status="created")
            self.save()
        self.api("PUT", "histories/" + run["history_id"], json={"tags": tags})
        run.update(status="submitting", resolved_inputs=values, submitted=now())
        self.save()
        if step.get("operation") == "exclude-sample":
            source = values["input"]
            rules = {"rules": [
                {"type": "add_column_metadata", "value": "identifier0"},
                {"type": "add_column_metadata", "value": "identifier1"},
                {"type": "add_filter_matches", "target_column": 0,
                 "value": step["exclude"], "invert": True},
            ], "mapping": [{"type": "list_identifiers", "columns": [0]},
                            {"type": "paired_identifier", "columns": [1]}]}
            result = self.api("POST", "tools", json={"history_id": run["history_id"],
                "tool_id": "__APPLY_RULES__", "inputs": {"input": source, "rules": rules},
                "use_cached_job": True})
            run.update(status="running", jobs=[j["id"] for j in result.get("jobs", [])],
                       outputs={"Filtered reads": {"id": result["output_collections"][0]["id"], "src": "hdca"}})
        else:
            mapping = self.workflow_inputs(step["workflow"])
            inputs = {mapping[label]["uuid"]: value for label, value in values.items()}
            inputs.update({mapping[label]["uuid"]: value for label, value in step.get("parameters", {}).items()})
            result = self.api("POST", f"workflows/{self.state['workflows'][step['workflow']]['id']}/invocations", json={
                "history_id": run["history_id"], "inputs": inputs,
                "inputs_by": "step_uuid", "use_cached_job": True,
            })
            if isinstance(result, list):
                result = result[0]
            run.update(status="running", invocation_id=result["id"])
        self.save()

    def refresh(self, step, run):
        if step.get("operation"):
            jobs = [self.api("GET", f"jobs/{jid}") for jid in run["jobs"]]
            if any(j["state"] in {"error", "failed", "deleted"} for j in jobs):
                raise Blocked("Air collection filtering failed")
            if not all(j["state"] == "ok" for j in jobs):
                return
            collection = self.check_data(run["outputs"]["Filtered reads"], run["history_id"])
            names = [e["element_identifier"] for e in collection["elements"]]
            if step["exclude"] in names or len(names) != step["expected_count"]:
                raise Blocked(f"Air filtered collection has unexpected identifiers: {names}")
            run.update(status="complete", completed=now(), identifiers=names)
            self.save()
            return
        inv = self.api("GET", "invocations/" + run["invocation_id"])
        atomic_json(self.root / "invocations" / f"{step['name']}.json", inv)
        run["invocation_state"] = inv["state"]
        if inv["state"] in {"failed", "cancelled"}:
            run.update(status="blocked", error=str(inv.get("messages", [])))
            self.save()
            return
        summary = self.api("GET", f"invocations/{run['invocation_id']}/jobs_summary")
        summaries = summary if isinstance(summary, list) else [summary]
        counts = {}
        for entry in summaries:
            for state, count in entry.get("states", {}).items():
                counts[state] = counts.get(state, 0) + count
        run["job_states"] = counts
        run["progress_checked"] = now()
        # Scheduling invocations already have running/error jobs. Record those
        # without treating a temporarily terminal job set as stage completion.
        if inv["state"] not in {"scheduled", "completed"}:
            self.save()
            return
        if not counts or any(count for state, count in counts.items() if state not in {"ok", "skipped", "error", "deleted"}):
            self.save()
            return
        # Also inspect history jobs: protects against unfinished nested invocations.
        jobs, offset = [], 0
        while True:
            batch = self.api("GET", "jobs", params={"history_id": run["history_id"], "limit": 500, "offset": offset})
            jobs.extend(batch)
            if len(batch) < 500:
                break
            offset += len(batch)
        if any(j["state"] not in {"ok", "skipped", "error", "deleted"} for j in jobs):
            self.save()
            return
        outputs = {}
        for label, output in inv.get("outputs", {}).items():
            outputs[label] = {"id": output["id"], "src": "hda"}
        for label, output in inv.get("output_collections", {}).items():
            outputs[label] = {"id": output["id"], "src": "hdca"}
        # Invocation outputs may be keyed by output UUID; output label metadata
        # is the authoritative bridge, never a history-name heuristic.
        for source in (inv.get("outputs", {}), inv.get("output_collections", {})):
            for key, output in source.items():
                label = output.get("workflow_output", {}).get("label")
                if label and key in outputs:
                    outputs[label] = outputs.pop(key)
        if not outputs:
            raise Blocked("Scheduling finished but invocation has no declared outputs")
        run["outputs"] = outputs
        errors = [j["id"] for j in jobs if j["state"] in {"error", "deleted"}]
        run.update(status="complete-with-errors" if errors else "complete", errors=errors, completed=now())
        self.save()

    def report(self):
        lines = [f"# {self.doc['campaign']}", "", f"Updated: {now()}", "",
                 "| Stage | Status | History | Details |", "|---|---|---|---|"]
        for step in self.doc["steps"]:
            run = self.state["runs"].get(step["name"], {})
            hid = run.get("history_id")
            link = f"[EU]({self.doc['galaxy_url']}/histories/view?id={hid})" if hid else "—"
            details = run.get("error", "")[:250]
            if run.get("job_states"):
                details += " " + ", ".join(f"{state}: {count}" for state, count in sorted(run["job_states"].items()))
            lines.append(f"| {step['name']} | {run.get('status', 'waiting')} | {link} | {details.strip()} |")
        (self.root / "STATUS.md").write_text("\n".join(lines) + "\n")

    def download_entries(self, step):
        policy_path = self.root / "download-policy.yaml"
        if not policy_path.exists():
            return []
        policy = yaml.safe_load(policy_path.read_text()) or {}
        workflow = step.get("workflow")
        selector = policy.get("histories", {}).get(workflow)
        if not selector:
            return []
        case = step["use_case"]
        key = case + ("/metaspades" if step["name"].endswith("_metaspades") else "/megahit") if case == "air" else case
        manifest = policy["manifests"][key]
        document = yaml.safe_load((self.root / manifest).read_text())
        histories = [h for h in document["histories"] if selector.lower() in h["name"].lower()]
        if len(histories) != 1:
            raise RuntimeError(f"Ambiguous download manifest history: {manifest}: {selector}")
        return histories[0].get("datasets", [])

    def export(self, step, run):
        from download_datasets import download_campaign_table

        # Fail closed: no whitelist means no downloads. Policy is separate from
        # the immutable scientific-run config, so it cannot resubmit workflows.
        entries = self.download_entries(step)
        # Exporter has its own checkpoint file, never writes scheduler state.
        export_path = self.root / "table-exports" / (step["name"] + ".json")
        export_state = json.loads(export_path.read_text()) if export_path.exists() else {}
        exports = export_state.setdefault("exports", {})
        required = {entry["file"] for entry in entries}
        for entry in entries:
            label = entry["file"]
            if Path(label).name != label or not label.endswith((".tsv", ".csv", ".txt")):
                raise RuntimeError(f"Invalid allowlisted table filename: {label}")
            target = self.root / "tables" / step["use_case"] / step["name"] / label
            try:
                record = download_campaign_table(self.session, self.base, run["history_id"], entry, target)
            except RuntimeError as exc:
                export_state.setdefault("export_problems", {})[label] = str(exc)
                atomic_json(export_path, export_state)
                continue
            exports[label] = str(target.relative_to(self.root))
            export_state.get("export_problems", {}).pop(label, None)
            atomic_json(export_path, export_state)
        missing = required - set(exports)
        export_state["missing_required_exports"] = sorted(missing)
        export_state["collection_complete"] = not missing
        atomic_json(export_path, export_state)

    def tick(self):
        self.install_workflows()
        for step in self.doc["steps"]:
            run = self.state["runs"].get(step["name"], {})
            try:
                if run.get("status") == "running":
                    self.refresh(step, run)
                elif run.get("status") not in {"complete", "complete-with-errors", "blocked"}:
                    values = self.resolve_inputs(step)
                    if values is not None:
                        self.submit(step, values)
            except Blocked as exc:
                self.state["runs"].setdefault(step["name"], {}).update(status="blocked", error=str(exc))
                self.save()
            except requests.RequestException as exc:
                # Leave submitting/creating intact on a POST ambiguity.
                self.state["runs"].setdefault(step["name"], {})["last_network_error"] = f"{now()}: {type(exc).__name__}"
                self.save()
                print(step["name"], type(exc).__name__, flush=True)
        self.report()
    def export_tick(self):
        # Independent process: downloads cannot block workflow scheduling.
        for step in self.doc["steps"]:
            run = self.state["runs"].get(step["name"], {})
            if run.get("status") in {"complete", "complete-with-errors"}:
                try:
                    self.export(step, run)
                except requests.RequestException as exc:
                    print(step["name"], "export", type(exc).__name__, flush=True)
        acceptance = {}
        for step in self.doc["steps"]:
            run = self.state["runs"].get(step["name"], {})
            path = self.root / "table-exports" / (step["name"] + ".json")
            export = json.loads(path.read_text()) if path.exists() else {}
            required = {entry["file"] for entry in self.download_entries(step)}
            missing = required - set(export.get("exports", {}))
            acceptance[step["name"]] = {
                "status": run.get("status", "waiting"),
                "required_downloads_complete": not missing,
                "missing_required_exports": sorted(missing),
                "failed_jobs": run.get("errors", []),
            }
        atomic_json(self.root / "ACCEPTANCE.json", {
            "updated": now(), "stages": acceptance,
            "all_required_downloads_complete": all(v["required_downloads_complete"] for v in acceptance.values()),
            "requires_failure_review": any(v["failed_jobs"] or v["status"] == "blocked" for v in acceptance.values()),
        })


def run_campaign(config, key, once=False, dry_run=False, export_only=False):
    campaign = Campaign(config, key)
    campaign.preflight()
    if dry_run:
        print(f"Validated {len(campaign.doc['steps'])} campaign stages; no remote changes")
        return
    lock_name = "exporter.lock" if export_only else "controller.lock"
    with (campaign.root / lock_name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if export_only:
                campaign.state = json.loads(campaign.state_path.read_text())
                campaign.export_tick()
            else:
                campaign.tick()
            if once:
                return
            time.sleep(campaign.doc.get("poll_seconds", 300))
