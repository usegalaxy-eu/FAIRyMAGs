"""Offline safety checks for the long-running campaign controller."""

import unittest
from pathlib import Path
from unittest.mock import Mock

import yaml

from download_datasets import download_campaign_table
from workflow_campaign import Blocked, Campaign


class CampaignTests(unittest.TestCase):
    def test_download_policy_uses_only_manifest_tables(self):
        campaign = self.campaign()
        campaign.root = Path(__file__).resolve().parents[1] / "data/campaigns/iwc-20261021"
        config = yaml.safe_load((campaign.root / "campaign.yaml").read_text())
        counts = {}
        for step in config["steps"]:
            entries = campaign.download_entries(step)
            expected = {"parent": 5, "taxonomy": 2, "genome": 4, "functional": 1}.get(step.get("workflow"), 0)
            self.assertEqual(len(entries), expected, step["name"])
            counts[step["name"]] = len(entries)
        self.assertEqual(sum(counts.values()), 84)

    def test_download_rejects_fastq_even_if_tagged_as_table(self):
        session = Mock()
        response = session.get.return_value
        response.json.side_effect = [
            [{"id": "reads", "tags": ["wanted"], "type": "file"}],
            {"id": "reads", "state": "ok", "extension": "fastqsanger.gz"},
        ]
        with self.assertRaisesRegex(RuntimeError, "non-ready/non-table"):
            download_campaign_table(session, "https://example/api/", "h", {"tag": "wanted"}, "/unused.tsv")
        self.assertEqual(session.get.call_count, 2)

    def test_download_rejects_ambiguous_tag(self):
        session = Mock()
        session.get.return_value.json.return_value = [
            {"id": str(i), "tags": ["wanted"]} for i in range(2)
        ]
        with self.assertRaisesRegex(RuntimeError, "found 2"):
            download_campaign_table(session, "https://example/api/", "h", {"tag": "wanted"}, "/unused.tsv")
        self.assertEqual(session.get.call_count, 1)

    def campaign(self):
        campaign = Campaign.__new__(Campaign)
        campaign.state = {"runs": {}}
        campaign.save = Mock()
        campaign.root = Mock()
        return campaign

    def test_dependency_waits_for_upstream(self):
        campaign = self.campaign()
        campaign.state["runs"]["up"] = {"status": "running"}
        step = {"inputs": {"Reads": {"step": "up", "output": "Reads"}}}
        self.assertIsNone(campaign.resolve_inputs(step))

    def test_required_missing_output_blocks(self):
        campaign = self.campaign()
        campaign.state["runs"]["up"] = {"status": "complete", "outputs": {}}
        step = {"inputs": {"Reads": {"step": "up", "output": "Reads"}}}
        with self.assertRaises(Blocked):
            campaign.resolve_inputs(step)

    def test_optional_failure_does_not_destroy_valid_dependency(self):
        campaign = self.campaign()
        value = {"src": "hdca", "id": "reads"}
        campaign.state["runs"]["up"] = {"status": "complete-with-errors", "history_id": "h", "outputs": {"Reads": value}}
        campaign.check_data = Mock()
        step = {"inputs": {"Reads": {"step": "up", "output": "Reads"}}}
        self.assertEqual(campaign.resolve_inputs(step), {"Reads": value})
        campaign.check_data.assert_called_once_with(value, "h")

    def test_ambiguous_submission_is_never_repeated(self):
        campaign = self.campaign()
        campaign.state["runs"]["stage"] = {"status": "submitting"}
        campaign.api = Mock()
        with self.assertRaises(Blocked):
            campaign.submit({"name": "stage"}, {})
        campaign.api.assert_not_called()

    def test_skipped_collection_leaf_is_not_valid_input(self):
        campaign = self.campaign()
        collection = {"populated_state": "ok", "elements": [{"element_type": "hda", "object": {
            "id": "null", "state": "ok", "extension": "expression.json"}}]}
        campaign.collection = Mock(return_value=collection)
        with self.assertRaises(Blocked):
            campaign.check_data({"src": "hdca", "id": "c"}, "h")

    def test_empty_required_collection_is_rejected(self):
        campaign = self.campaign()
        campaign.collection = Mock(return_value={"populated_state": "ok", "elements": []})
        with self.assertRaises(Blocked):
            campaign.check_data({"src": "hdca", "id": "c"}, "h")


if __name__ == "__main__":
    unittest.main()
