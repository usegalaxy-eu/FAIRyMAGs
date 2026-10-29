"""Offline checks for the use-case publication runner."""

import unittest

from run_workflows import leaf_datasets, resolve_content


def dataset(name, extension="fastqsanger.gz", state="ok", deleted=False, hid=1):
    return {
        "id": f"hda{hid}", "name": name, "type": "file", "state": state,
        "deleted": deleted, "file_ext": extension, "file_size": 1024, "hid": hid,
    }


def placeholder(name="Bowtie2 on dataset 11", hid=30):
    return dataset(name, extension="expression.json", state="ok", deleted=False, hid=hid)


def collection(cid, name, elements, hid=1, deleted=False):
    return {
        "id": cid, "name": name, "type": "collection", "collection_type": "list:paired",
        "state": "ok", "deleted": deleted, "hid": hid, "element_count": len(elements),
        "elements": elements,
    }


def paired_collection_element(identifier, forward, reverse):
    return {
        "element_type": "dataset_collection",
        "object": {
            "collection_type": "paired",
            "elements": [
                {"element_type": "hda", "object": forward},
                {"element_type": "hda", "object": reverse},
            ],
        },
        "element_identifier": identifier,
    }


class FakeDatasets:
    def __init__(self, items):
        self.items = items

    def get_datasets(self, history_id=None, limit=500, offset=0):
        return self.items[offset:offset + limit]


class FakeCollections:
    def __init__(self, by_id):
        self.by_id = by_id

    def show_dataset_collection(self, collection_id, instance_type="history"):
        return self.by_id[collection_id]


class FakeGalaxy:
    def __init__(self, items, collections=None):
        self.datasets = FakeDatasets(items)
        self.dataset_collections = FakeCollections(collections or {})


class ResolveContentTests(unittest.TestCase):
    def test_leaf_datasets_flattens_nested_collections(self):
        nested = {
            "collection_type": "list:paired",
            "elements": [
                paired_collection_element("s1", dataset("f1"), dataset("r1")),
                paired_collection_element("s2", dataset("f2"), dataset("r2")),
            ],
        }
        self.assertEqual(len(list(leaf_datasets(nested))), 4)

    def test_skipped_branch_copy_is_ignored(self):
        # The host workflow publishes two collections with the same name: the
        # branch that did not run holds only 4-byte expression.json placeholders.
        skipped = collection("c_skipped", "Reads without host or contaminant reads", [
            paired_collection_element("s1", placeholder(), placeholder()),
        ])
        real = collection("c_real", "Reads without host or contaminant reads", [
            paired_collection_element("s1", dataset("Bowtie2"), dataset("Bowtie2")),
        ])
        gi = FakeGalaxy([skipped, real], {"c_skipped": skipped, "c_real": real})
        resolved = resolve_content(gi, "history", {"name": "Reads without host or contaminant reads"})
        self.assertEqual(resolved, {"id": "c_real", "src": "hdca"})

    def test_only_placeholder_copies_stay_ambiguous(self):
        first = collection("c1", "output", [paired_collection_element("s1", placeholder(), placeholder())])
        second = collection("c2", "output", [paired_collection_element("s1", placeholder(), placeholder())])
        gi = FakeGalaxy([first, second], {"c1": first, "c2": second})
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            resolve_content(gi, "history", {"name": "output"})

    def test_lone_placeholder_dataset_is_rejected(self):
        gi = FakeGalaxy([placeholder()])
        with self.assertRaisesRegex(RuntimeError, "skipped-branch placeholder"):
            resolve_content(gi, "history", {"name": "Bowtie2 on dataset 11"})

    def test_real_collection_is_accepted(self):
        real = collection("c_real", "Trimmed reads", [
            paired_collection_element("s1", dataset("fastp"), dataset("fastp")),
        ])
        gi = FakeGalaxy([real], {"c_real": real})
        resolved = resolve_content(gi, "history", {"name": "Trimmed reads", "type": "collection"})
        self.assertEqual(resolved, {"id": "c_real", "src": "hdca"})

    def test_tags_still_filter_copies(self):
        wanted = collection("c_wanted", "output", [
            paired_collection_element("s1", dataset("Bowtie2"), dataset("Bowtie2")),
        ])
        wanted["tags"] = ["name:PE"]
        other = collection("c_other", "output", [
            paired_collection_element("s1", dataset("Bowtie2"), dataset("Bowtie2")),
        ])
        gi = FakeGalaxy([wanted, other], {"c_wanted": wanted, "c_other": other})
        resolved = resolve_content(gi, "history", {"name": "output", "tags": ["PE"]})
        self.assertEqual(resolved, {"id": "c_wanted", "src": "hdca"})


if __name__ == "__main__":
    unittest.main()
