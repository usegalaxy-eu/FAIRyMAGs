#!/usr/bin/env python3
"""Run a use-case workflow pipeline declared in workflow.yaml.

The manifest identifies workflows, histories, datasets, and collections by
human-readable names and tags. Galaxy IDs are resolved only in memory.

Each step reuses an existing history that already has the requested name and
tags, and invokes the workflow only when no such history exists. Tag a history
with 'fail' or 'rerun' to make the runner replace it with a fresh run.
"""

import argparse
import os
import sys
import time
from urllib.parse import parse_qs, urlparse

import yaml
from bioblend.galaxy import GalaxyInstance


TERMINAL_INVOCATION_STATES = {"cancelled", "completed", "failed", "ready"}
# Galaxy publishes a skipped conditional branch's outputs as null placeholders.
PLACEHOLDER_EXTENSION = "expression.json"
PENDING_DATASET_STATES = {
    "new", "upload", "queued", "running", "setting_metadata", "deferred", "paused",
}
FAILED_DATASET_STATES = {"error", "failed_metadata", "discarded"}


def norm(value):
    value = str(value).strip().lstrip("#")
    if value.lower().startswith("name:"):
        value = value[5:]
    return value.casefold()


def tags_match(item, required=(), excluded=()):
    tags = {norm(tag) for tag in item.get("tags") or []}
    return {norm(tag) for tag in required} <= tags and not tags & {norm(tag) for tag in excluded}


def exactly_one(items, description):
    if len(items) != 1:
        choices = ", ".join(f"{item.get('name')!r} ({item.get('id')})" for item in items) or "none"
        raise RuntimeError(f"expected exactly one {description}; found {len(items)}: {choices}")
    return items[0]


def workflow_id(link):
    values = parse_qs(urlparse(link).query).get("id", [])
    if len(values) != 1:
        raise ValueError(f"workflow link must contain exactly one id parameter: {link}")
    return values[0]


def resolve_workflow(gi, selector):
    wid = workflow_id(selector["link"])
    workflow = gi.workflows.show_workflow(wid)
    if workflow.get("name") != selector["name"]:
        raise RuntimeError(
            f"workflow link resolves to {workflow.get('name')!r}, expected {selector['name']!r}"
        )
    if not tags_match(workflow, selector.get("tags", [])):
        raise RuntimeError(f"workflow {selector['name']!r} lacks required tags {selector.get('tags', [])}")
    return workflow


def resolve_history(gi, selector, excluded=()):
    matches = [
        history
        for history in gi.histories.get_histories(deleted=False)
        if history.get("name") == selector["name"]
        and tags_match(history, selector.get("tags", []), excluded)
    ]
    return exactly_one(matches, f"history named {selector['name']!r} with tags {selector.get('tags', [])}")


def history_tag_set(history):
    return {norm(tag) for tag in history.get("tags") or []}


def find_output_histories(gi, name, tags, excluded=()):
    return [
        history
        for history in gi.histories.get_histories(deleted=False)
        if history.get("name") == name and tags_match(history, tags, excluded)
    ]


def select_reusable(matches, required_tags):
    """Pick the single existing history to reuse, or fail if ambiguous.

    Extra tags do not prevent a match, so several histories can match one
    selector. Prefer the history whose tags are exactly the requested ones and
    treat anything else as ambiguous rather than guessing.
    """
    if len(matches) == 1:
        return matches[0]
    exact = [h for h in matches if history_tag_set(h) == {norm(tag) for tag in required_tags}]
    if len(exact) == 1:
        return exact[0]
    choices = ", ".join(
        f"{h.get('name')!r} ({h['id']}) tags={sorted(history_tag_set(h))}" for h in matches
    )
    raise RuntimeError(
        f"cannot pick a single history to reuse; found {len(matches)}: {choices}. "
        "Tag the ones you want replaced with 'fail' or 'rerun'."
    )


def describe(items):
    return ", ".join(f"{i.get('name')!r} state={dataset_state(i)}" for i in items)


def check_reusable(gi, history, on_failure):
    failed = failed_datasets(gi, history["id"])
    if failed and on_failure == "stop":
        raise RuntimeError(
            f"reused history {history['id']} has {len(failed)} failed dataset(s) and "
            f"on_failure is 'stop': {describe(failed)}. Add the 'fail' tag to re-run the step."
        )
    report_failed(f"reused history {history['id']}", failed)
    return failed


def history_contents(gi, history_id):
    items = []
    offset = 0
    while True:
        batch = gi.datasets.get_datasets(history_id=history_id, limit=500, offset=offset)
        if not batch:
            break
        items.extend(batch)
        offset += len(batch)
        if len(batch) < 500:
            break
    return items


def dataset_state(item):
    return item.get("state") or "unknown"


def unusable(item):
    return dataset_state(item) in PENDING_DATASET_STATES | FAILED_DATASET_STATES


def failed_datasets(gi, history_id):
    return [item for item in history_contents(gi, history_id)
            if dataset_state(item) in FAILED_DATASET_STATES]


def pending_datasets(gi, history_id):
    return [item for item in history_contents(gi, history_id)
            if dataset_state(item) in PENDING_DATASET_STATES]


def history_invocations(gi, history_id):
    """List workflow invocations in a history, for context only.

    Galaxy's invocation `state` is not a reliable running signal: finished
    histories have been observed reporting 'scheduled' and 'ready'. Dataset
    states are the authoritative signal, so this is used for reporting.
    """
    try:
        response = gi.make_get_request(f"{gi.url}/invocations", params={"history_id": history_id})
        return response.json()
    except Exception as exc:
        print(f"  warning: could not list invocations for {history_id}: {exc}", flush=True)
        return []


def wait_for_history_ready(gi, history_id, interval, max_seconds):
    """Attach to a history that is still being produced and wait it out."""
    deadline = time.monotonic() + max_seconds
    while True:
        pending = pending_datasets(gi, history_id)
        if not pending:
            return
        states = ", ".join(
            f"{i.get('id')}={i.get('state')}" for i in history_invocations(gi, history_id)
        ) or "none reported"
        print(
            f"  history {history_id}: {len(pending)} dataset(s) still pending "
            f"(invocations: {states})",
            flush=True,
        )
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"history {history_id} still has {len(pending)} pending dataset(s) after "
                f"{max_seconds}s; giving up. It is probably paused on Galaxy."
            )
        time.sleep(interval)


def resolve_or_wait(gi, history_id, selector, interval, max_seconds):
    """Resolve content, attaching to a still-running history if necessary.

    Used when a step reuses a history whose workflow was left running, for
    example because a previous invocation of this script was interrupted.
    """
    try:
        return resolve_content(gi, history_id, selector)
    except RuntimeError:
        pending = pending_datasets(gi, history_id)
        if not pending:
            raise
        print(
            f"  content {selector['name']!r} is not usable yet and history {history_id} has "
            f"{len(pending)} pending dataset(s): waiting for that run to finish",
            flush=True,
        )
        wait_for_history_ready(gi, history_id, interval, max_seconds)
        return resolve_content(gi, history_id, selector)


def leaf_datasets(collection):
    for element in collection.get("elements") or []:
        obj = element.get("object") or {}
        if element.get("element_type") == "dataset_collection" or obj.get("elements") is not None:
            yield from leaf_datasets(obj)
        else:
            yield obj


def is_placeholder_dataset(item):
    """A skipped step writes a 4-byte expression.json dataset with a null body."""
    extension = item.get("file_ext") or item.get("extension")
    return extension == PLACEHOLDER_EXTENSION


def is_placeholder_content(gi, item):
    """True when every dataset behind an output is a skipped-branch placeholder."""
    if item.get("type") != "collection":
        return is_placeholder_dataset(item)
    collection = gi.dataset_collections.show_dataset_collection(item["id"])
    leaves = list(leaf_datasets(collection))
    return bool(leaves) and all(is_placeholder_dataset(leaf) for leaf in leaves)


def drop_placeholder_copies(gi, history_id, selector, candidates):
    """Ignore skipped-branch outputs when several copies share one name.

    A workflow that offers two branches for the same input still publishes both
    output collections: the branch that did not run keeps the same name but holds
    only placeholder datasets. Name, type and tags cannot separate the two, so the
    placeholders are dropped instead of reporting a false ambiguity.
    """
    real = [item for item in candidates if not is_placeholder_content(gi, item)]
    if real and len(real) != len(candidates):
        print(
            f"  content {selector['name']!r} in history {history_id}: ignoring "
            f"{len(candidates) - len(real)} skipped-branch placeholder copy/copies",
            flush=True,
        )
    return real or candidates


def resolve_content(gi, history_id, selector):
    """Resolve one dataset or collection by name, type and tags.

    A step is allowed to proceed when the content it needs is usable, even if
    other datasets in the same history failed. Only the requested content is
    judged, and only usable states are accepted.
    """
    expected_type = selector.get("type")
    named = []
    for item in history_contents(gi, history_id):
        actual_type = "collection" if item.get("type") == "collection" else "dataset"
        if item.get("name") != selector["name"] or (expected_type and actual_type != expected_type):
            continue
        if item.get("deleted") or not tags_match(item, selector.get("tags", [])):
            continue
        named.append(item)
    usable = [item for item in named if not unusable(item)]
    if len(usable) > 1:
        usable = drop_placeholder_copies(gi, history_id, selector, usable)
    if len(usable) == 1:
        item = usable[0]
        if is_placeholder_content(gi, item):
            raise RuntimeError(
                f"content {selector['name']!r} in history {history_id} is a skipped-branch placeholder"
            )
        src = "hdca" if item.get("type") == "collection" else "hda"
        return {"id": item["id"], "src": src}
    if len(usable) > 1:
        raise RuntimeError(
            f"content {selector['name']!r} in history {history_id} is ambiguous; "
            f"found {len(usable)} usable copies"
        )
    if named:
        states = ", ".join(f"{i.get('name')!r} state={dataset_state(i)}" for i in named)
        raise RuntimeError(
            f"content {selector['name']!r} in history {history_id} exists but is not usable: {states}"
        )
    raise RuntimeError(f"content {selector['name']!r} not found in history {history_id}")


def report_failed(label, items):
    if not items:
        return
    print(f"  warning: {label}: {len(items)} failed dataset(s); continuing", flush=True)
    for item in items[:10]:
        print(f"    - {item.get('name')!r} state={dataset_state(item)}", flush=True)
    if len(items) > 10:
        print(f"    ... and {len(items) - 10} more", flush=True)


def workflow_inputs(workflow):
    return {entry["label"]: entry["uuid"] for entry in workflow.get("inputs", {}).values()}


def wait_for_invocation(gi, history_id, invocation_id, interval, on_failure):
    """Wait until the invocation is done and nothing is still running.

    With on_failure 'continue' (the default) failed datasets do not block: they
    are reported as warnings, and a dependent step stops only if the content it
    actually needs is missing or unusable. With on_failure 'stop' the first
    observed failure, or a failed or cancelled invocation, raises immediately.
    """
    while True:
        invocation = gi.workflows.show_invocation(history_id, invocation_id)
        contents = history_contents(gi, history_id)
        pending = [i for i in contents if dataset_state(i) in PENDING_DATASET_STATES]
        failed = [i for i in contents if dataset_state(i) in FAILED_DATASET_STATES]
        state = invocation.get("state")
        print(
            f"  invocation {invocation_id}: {state} "
            f"pending={len(pending)} failed={len(failed)}",
            flush=True,
        )
        if on_failure == "stop":
            if failed:
                raise RuntimeError(
                    f"invocation {invocation_id} produced {len(failed)} failed dataset(s) "
                    f"and on_failure is 'stop': {describe(failed)}"
                )
            if state in {"failed", "cancelled"}:
                raise RuntimeError(
                    f"invocation {invocation_id} ended as {state} and on_failure is 'stop'"
                )
        if state in TERMINAL_INVOCATION_STATES and not pending:
            report_failed(f"history {history_id}", failed)
            return
        time.sleep(interval)


def source_value(gi, doc, sources, reference, interval, max_seconds):
    if isinstance(reference, str):
        if not reference.startswith("source:"):
            raise ValueError(f"invalid input reference: {reference}")
        source_name = reference.split(":", 1)[1]
        if source_name not in sources:
            selector = doc["sources"][source_name]
            history = resolve_history(gi, selector["history"], doc.get("exclude_history_tags", []))
            sources[source_name] = resolve_or_wait(
                gi, history["id"], selector["content"], interval, max_seconds
            )
        return sources[source_name]
    raise ValueError("step references are resolved separately")


def validate_manifest(doc):
    if doc.get("schema_version") != 2:
        raise ValueError("workflow.yaml must use schema_version: 2")
    execution = doc.get("execution", {})
    if not execution.get("new_history_only"):
        raise ValueError("execution.new_history_only must be true")
    on_failure = execution.get("on_failure", "continue")
    if on_failure not in {"continue", "stop"}:
        raise ValueError("execution.on_failure must be 'continue' or 'stop'")
    names = [step["name"] for step in doc.get("steps", [])]
    if len(names) != len(set(names)):
        raise ValueError("step names must be unique")
    known = set()
    for step in doc.get("steps", []):
        for reference in step.get("inputs", {}).values():
            if isinstance(reference, dict) and reference.get("step") not in known:
                raise ValueError(f"step {step['name']} depends on unknown or later step {reference.get('step')}")
        known.add(step["name"])


def short(value):
    """Abbreviate a Galaxy id. Ids share a long common prefix, so keep the tail."""
    if value is None:
        return "-"
    value = str(value)
    return f"{value[:4]}..{value[-8:]}" if len(value) > 14 else value


def build_plan(gi, doc):
    """Resolve every step before anything runs, so the whole plan is visible first."""
    base_tags = doc.get("history_tags", [])
    excluded = doc.get("exclude_history_tags", [])
    plan = []
    for step in doc["steps"]:
        workflow = resolve_workflow(gi, step["workflow"])
        labels = workflow_inputs(workflow)
        unknown = set(step.get("inputs", {})) | set(step.get("parameters", {}))
        unknown -= set(labels)
        if unknown:
            raise RuntimeError(
                f"unknown input/parameter labels for {workflow['name']}: {sorted(unknown)}"
            )
        tags = [*base_tags, *step.get("tags", [])]
        target = step["output_history"]["name"]
        existing = find_output_histories(gi, target, tags, excluded)
        reusable = select_reusable(existing, tags) if existing else None
        plan.append({
            "step": step,
            "workflow": workflow,
            "labels": labels,
            "tags": tags,
            "target": target,
            "existing": existing,
            "reusable": reusable,
        })
    return plan


def print_plan(doc, plan, dry_run):
    creates = [p for p in plan if p["reusable"] is None]
    print(
        f"PLAN use case {doc.get('use_case', '?')} on {doc['galaxy_url']}: "
        f"{len(plan)} steps, will create {len(creates)}, reuse {len(plan) - len(creates)}"
    )
    print(f"  {'#':>2}  {'step':<32} {'action':<7} {'history':<14} workflow")
    for number, p in enumerate(plan, 1):
        action = "reuse" if p["reusable"] is not None else "CREATE"
        history_id = short(p["reusable"]["id"]) if p["reusable"] is not None else "(new)"
        print(
            f"  {number:>2}  {p['step']['name']:<32} {action:<7} {history_id:<14} "
            f"{p['workflow']['name']}"
        )
        if p["reusable"] is None:
            for history in p["existing"]:
                print(
                    f"      note: history {history['id']} exists but is not reusable "
                    f"(tags={sorted(history_tag_set(history))})"
                )
    if dry_run:
        print("\n--dry-run: nothing will be created or invoked.")
    print(flush=True)


def execute_plan(gi, doc, plan, interval, max_seconds, on_failure, dry_run):
    execution = doc["execution"]
    sources = {}
    runs = {}
    results = []
    for number, p in enumerate(plan, 1):
        step, workflow, labels = p["step"], p["workflow"], p["labels"]
        print(f"\nSTEP {number}/{len(plan)} {step['name']}: {workflow['name']}")
        print(f"  tags: {', '.join(p['tags'])}")
        if p["reusable"] is not None:
            print(f"  reusing existing history {p['reusable']['id']}: workflow will NOT be invoked")
        else:
            print("  creating a new history and invoking the workflow")
        if dry_run:
            continue

        started = time.monotonic()
        if p["reusable"] is not None:
            failed = check_reusable(gi, p["reusable"], on_failure)
            history_id = p["reusable"]["id"]
            runs[step["name"]] = {"history_id": history_id, "invocation_id": None, "reused": True}
            results.append({
                "step": step["name"], "action": "reuse", "history": history_id,
                "invocation": None, "failed": len(failed),
                "seconds": time.monotonic() - started,
            })
            continue

        resolved = {}
        for label, reference in step.get("inputs", {}).items():
            if isinstance(reference, str):
                value = source_value(gi, doc, sources, reference, interval, max_seconds)
            else:
                parent = reference["step"]
                if parent not in runs:
                    raise RuntimeError(f"step {parent} has not run")
                value = resolve_or_wait(
                    gi, runs[parent]["history_id"], reference["content"], interval, max_seconds
                )
            resolved[labels[label]] = value
        for label, value in step.get("parameters", {}).items():
            resolved[labels[label]] = value

        history = gi.histories.create_history(name=p["target"])
        gi.histories.update_history(history["id"], tags=[f"name:{tag}" for tag in p["tags"]])
        invocation = gi.workflows.invoke_workflow(
            workflow["id"],
            inputs=resolved,
            history_id=history["id"],
            inputs_by="step_uuid",
            use_cached_job=execution.get("use_cached_job", False),
        )
        runs[step["name"]] = {
            "history_id": history["id"],
            "invocation_id": invocation["id"],
            "reused": False,
        }
        print(f"  history={history['id']} invocation={invocation['id']}", flush=True)
        wait_for_invocation(gi, history["id"], invocation["id"], interval, on_failure)
        results.append({
            "step": step["name"], "action": "create", "history": history["id"],
            "invocation": invocation["id"], "failed": len(failed_datasets(gi, history["id"])),
            "seconds": time.monotonic() - started,
        })
    return runs, results


def print_summary(results):
    print("\nSUMMARY")
    header = (f"  {'step':<32} {'action':<7} {'history':<14} {'invocation':<18} "
              f"{'failed':>7} {'time':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(
            f"  {r['step']:<32} {r['action']:<7} {short(r['history']):<14} "
            f"{short(r['invocation']):<18} {r['failed']:>7} {r['seconds']:>8.1f}s"
        )
    created = sum(1 for r in results if r["action"] == "create")
    reused = sum(1 for r in results if r["action"] == "reuse")
    print(f"\n  created {created}, reused {reused}")


def run(config_path, api_key, dry_run=False, poll_interval=None, max_wait_minutes=None):
    with open(config_path) as handle:
        doc = yaml.safe_load(handle)
    validate_manifest(doc)
    gi = GalaxyInstance(url=doc["galaxy_url"], key=api_key)
    execution = doc["execution"]
    interval = poll_interval or execution.get("poll_interval_seconds", 30)
    on_failure = execution.get("on_failure", "continue")
    max_seconds = (max_wait_minutes or execution.get("max_wait_minutes", 1440)) * 60

    plan = build_plan(gi, doc)
    print_plan(doc, plan, dry_run)
    runs, results = execute_plan(gi, doc, plan, interval, max_seconds, on_failure, dry_run)
    if not dry_run:
        print_summary(results)
    return runs


def api_key_from_env():
    key = os.environ.get("GALAXY_API_KEY") or os.environ.get("GALAXY_API")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as handle:
            for line in handle:
                name, separator, value = line.strip().partition("=")
                if separator and name in {"GALAXY_API_KEY", "GALAXY_API"}:
                    return value
    raise RuntimeError("set GALAXY_API_KEY (or GALAXY_API) or add it to .env")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a use-case workflow.yaml")
    # Campaign mode is local-only: it delegates to bin/workflow_campaign.py, which
    # is untracked (.gitignore) while the fairymags-campaign and
    # fairymags-downloads user services still run from this checkout.
    parser.add_argument("--campaign", action="store_true", help="run a persistent, version-pinned campaign")
    parser.add_argument("--once", action="store_true", help="perform one campaign scheduling cycle")
    parser.add_argument("--export-only", action="store_true", help="collect campaign outputs without scheduling")
    parser.add_argument("--dry-run", action="store_true", help="resolve and validate without creating histories")
    parser.add_argument("--poll-interval", type=int, help="override polling interval in seconds")
    parser.add_argument(
        "--max-wait-minutes", type=int,
        help="give up waiting on a still-running history after this many minutes (default 1440)",
    )
    args = parser.parse_args()
    try:
        if args.campaign:
            from workflow_campaign import run_campaign

            run_campaign(args.config, api_key_from_env(), once=args.once, dry_run=args.dry_run,
                         export_only=args.export_only)
            return
        run(
            args.config, api_key_from_env(), args.dry_run, args.poll_interval,
            args.max_wait_minutes,
        )
    except (KeyError, RuntimeError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
