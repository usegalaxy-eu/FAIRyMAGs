"""Offline safety checks for the long-running campaign controller."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from download_datasets import download_campaign_table, select_matches
from workflow_campaign import Blocked, Campaign, tag


class CampaignTests(unittest.TestCase):
    def test_tags_stay_one_opaque_token(self):
        # Galaxy splits tags on "=" or ":" into a name/value pair and reserves
        # "." as its hierarchy separator, so key:value tags cannot be used as
        # plain labels. A dash keeps the key/value shape as one token.
        tags = [tag("stage", "bee-gut.mags_multi"), tag("workflow-version", "0.6")]
        self.assertEqual(tags, ["stage-bee-gut.mags_multi", "workflow-version-0.6"])
        for value in tags:
            self.assertNotIn(":", value)
            self.assertNotIn("=", value)
            self.assertNotIn(",", value)
            self.assertNotIn(";", value)
            self.assertNotIn(" ", value)

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

    def test_id_selector_picks_untagged_output(self):
        # CheckM2/CheckM reports are published untagged by the parent workflow,
        # so the manifest addresses them by dataset id instead of by tag.
        items = [
            {"id": "checkm2-quality", "tags": []},
            {"id": "checkm-results", "tags": ["Reads"]},
        ]
        self.assertEqual(
            [d["id"] for d in select_matches(items, {"id": "checkm2-quality"})],
            ["checkm2-quality"],
        )
        self.assertEqual(
            [d["id"] for d in select_matches(items, {"id": "checkm2-quality", "tag": "Reads"})],
            ["checkm2-quality"],
        )

    def test_entry_without_tag_or_id_matches_nothing(self):
        items = [{"id": "a", "tags": ["Reads"]}]
        self.assertEqual(select_matches(items, {"file": "x.tsv"}), [])
        self.assertEqual(select_matches(items, {}), [])

    def campaign(self):
        campaign = Campaign.__new__(Campaign)
        campaign.state = {"runs": {}}
        campaign.save = Mock()
        campaign.root = Path("/unused-campaign")
        return campaign

    def test_dependency_waits_for_upstream(self):
        campaign = self.campaign()
        campaign.state["runs"]["up"] = {"status": "running"}
        step = {"inputs": {"Reads": {"step": "up", "output": "Reads"}}}
        self.assertIsNone(campaign.resolve_inputs(step))

    @patch("workflow_campaign.atomic_json")
    def test_scheduling_progress_is_visible_but_not_complete(self, atomic_json):
        campaign = self.campaign()
        campaign.api = Mock(side_effect=[
            {"state": "ready"},
            [{"states": {"ok": 12, "error": 2}}, {"states": {"running": 3}}],
        ])
        run = {"status": "running", "invocation_id": "i", "history_id": "h"}
        campaign.refresh({"name": "stage"}, run)
        self.assertEqual(run["job_states"], {"ok": 12, "error": 2, "running": 3})
        self.assertEqual(run["status"], "running")
        self.assertIn("progress_checked", run)
        self.assertNotIn("completed", run)
        self.assertEqual(campaign.api.call_count, 2)

    @patch("workflow_campaign.atomic_json")
    def test_ready_invocation_with_only_finished_jobs_still_waits(self, atomic_json):
        campaign = self.campaign()
        campaign.api = Mock(side_effect=[{"state": "ready"}, {"states": {"ok": 12}}])
        run = {"status": "running", "invocation_id": "i", "history_id": "h"}
        campaign.refresh({"name": "stage"}, run)
        self.assertEqual(run["status"], "running")
        self.assertEqual(run["outputs"], {})

    @patch("workflow_campaign.atomic_json")
    def test_ready_output_releases_dependency_before_jobs_finish(self, atomic_json):
        campaign = self.campaign()
        value = {"src": "hdca", "id": "bins"}
        campaign.api = Mock(side_effect=[
            {"state": "ready", "output_collections": {
                "uuid": {"id": "bins", "workflow_output": {"label": "Dereplicated Bins"}}}},
            {"states": {"ok": 12, "new": 2}},
        ])
        campaign.check_data = Mock(return_value={"element_count": 708})
        run = {"status": "running", "invocation_id": "i", "history_id": "h"}
        campaign.state["runs"]["up"] = run
        campaign.refresh({"name": "up", "required_outputs": ["Dereplicated Bins"]}, run)
        self.assertEqual(run["status"], "running")
        self.assertNotIn("completed", run)
        self.assertEqual(run["ready_outputs"]["Dereplicated Bins"]["element_count"], 708)
        step = {"inputs": {"Genomes": {"step": "up", "output": "Dereplicated Bins"}}}
        self.assertEqual(campaign.resolve_inputs(step), {"Genomes": value})
        self.assertEqual(campaign.check_data.call_count, 2)

    @patch("workflow_campaign.atomic_json")
    def test_pending_output_does_not_release_dependency(self, atomic_json):
        campaign = self.campaign()
        campaign.api = Mock(side_effect=[
            {"state": "scheduled", "output_collections": {"Bins": {"id": "bins"}}},
            {"states": {"ok": 12, "new": 2}},
        ])
        campaign.check_data = Mock(side_effect=Blocked("Pending collection leaf"))
        run = {"status": "running", "invocation_id": "i", "history_id": "h"}
        campaign.state["runs"]["up"] = run
        campaign.refresh({"name": "up", "required_outputs": ["Bins"]}, run)
        self.assertEqual(run["ready_outputs"], {})
        self.assertIn("Bins", run["pending_outputs"])
        self.assertIsNone(campaign.resolve_inputs({"inputs": {"Genomes": {"step": "up", "output": "Bins"}}}))

    def test_failed_upstream_cannot_release_previously_ready_output(self):
        campaign = self.campaign()
        campaign.state["runs"]["up"] = {"status": "blocked", "ready_outputs": {"Bins": {}}}
        self.assertIsNone(campaign.resolve_inputs({"inputs": {"Genomes": {"step": "up", "output": "Bins"}}}))

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
