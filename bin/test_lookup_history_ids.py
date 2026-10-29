"""Offline checks for resolving history ids in a use-case manifest."""

import unittest
from unittest.mock import Mock

from lookup_history_ids import lookup_ids


def history(hid, name, tags):
    return {"id": hid, "name": name, "tags": tags}


BASE = ["FAIRyMAGs", "FAIRyMAGs-iwc-20261021", "use-case-bee-gut"]


class LookupTests(unittest.TestCase):
    def resolve(self, histories, entries, exclude=("fail", "rerun", "superseded")):
        gi = Mock()
        gi.histories.get_histories.return_value = histories
        doc = {
            "required_tags": ["FAIRyMAGs-iwc-20261021", "use-case-bee-gut"],
            "exclude_tags": list(exclude),
            "histories": entries,
        }
        return lookup_ids(gi, doc)

    def test_name_substring_selects_single_history(self):
        histories = [
            history("qc", "FAIRyMAGs-iwc-20261021 | bee-gut | bee-gut.quality_control | Short-read quality control and trimming v0.6", BASE),
            history("mags", "FAIRyMAGs-iwc-20261021 | bee-gut | bee-gut.mags_single | MAGs generation v1.0", BASE),
        ]
        doc, count, warnings = self.resolve(histories, [{"name": "quality control", "id": None}])
        self.assertEqual(count, 2)
        self.assertEqual(doc["histories"][0]["id"], "qc")
        self.assertEqual(warnings, [])

    def test_entry_tags_disambiguate_single_from_multi_coverage(self):
        # Both binning runs of one workflow carry the same workflow name, so the
        # manifest narrows them with the stage tag.
        histories = [
            history("multi", "... | bee-gut.mags_multi | Metagenome-Assembled Genomes (MAGs) generation v1.0",
                    BASE + ["stage-bee-gut.mags_multi"]),
            history("single", "... | bee-gut.mags_single | Metagenome-Assembled Genomes (MAGs) generation v1.0",
                    BASE + ["stage-bee-gut.mags_single"]),
        ]
        doc, count, warnings = self.resolve(
            histories,
            [{"name": "Metagenome-Assembled Genomes (MAGs) generation",
              "tags": ["stage-bee-gut.mags_single"], "id": None}],
        )
        self.assertEqual(count, 2)
        self.assertEqual(doc["histories"][0]["id"], "single")
        self.assertEqual(warnings, [])

    def test_ambiguous_name_without_tags_is_reported(self):
        histories = [
            history("multi", "... | bee-gut.mags_multi | MAGs generation v1.0", BASE),
            history("single", "... | bee-gut.mags_single | MAGs generation v1.0", BASE),
        ]
        doc, count, warnings = self.resolve(histories, [{"name": "MAGs generation", "id": None}])
        self.assertIsNone(doc["histories"][0]["id"])
        self.assertEqual(len(warnings), 1)

    def test_superseded_run_is_never_selected(self):
        # Re-runs rename and re-tag the old history, so its stage tag no longer
        # matches the live run even before exclude_tags is considered.
        histories = [
            history("old", "SUPERSEDED ... | bee-gut.mags_single | MAGs generation v1.0",
                    BASE + ["stage-bee-gut.mags_single-superseded"]),
            history("new", "... | bee-gut.mags_single | MAGs generation v1.0",
                    BASE + ["stage-bee-gut.mags_single"]),
        ]
        doc, count, warnings = self.resolve(
            histories,
            [{"name": "MAGs generation", "tags": ["stage-bee-gut.mags_single"], "id": None}],
        )
        self.assertEqual(count, 1)
        self.assertEqual(doc["histories"][0]["id"], "new")
        self.assertEqual(warnings, [])

    def test_history_without_required_tags_is_ignored(self):
        histories = [history("publication", "... | MAGs generation v1.0", ["FAIRyMAGs", "publication"])]
        doc, count, warnings = self.resolve(histories, [{"name": "MAGs generation", "id": None}])
        self.assertEqual(count, 0)
        self.assertIsNone(doc["histories"][0]["id"])


if __name__ == "__main__":
    unittest.main()
