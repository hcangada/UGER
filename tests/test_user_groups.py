import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from uger.data import load_interaction_stream
from uger.models import build_model
from uger.user_groups import evaluate_user_groups, main, parse_args


class UserGroupDriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "interactions.tsv"
        records = []
        for index in range(60):
            records.append((index % 4, (index // 4 + index % 4) % 12, index))
        for index in range(20):
            records.append((index % 5, 12 + index % 6, 60 + index))
        for index in range(20):
            records.append((index % 5, 18 + index % 6, 80 + index))
        self.source.write_text("user_id\titem_id\ttimestamp\n" + "\n".join(
            f"u{user}\ti{item}\t{timestamp}" for user, item, timestamp in records
        ) + "\n", encoding="utf-8")
        self.stream = load_interaction_stream(self.source, num_incremental=2)
        self.config = {"initial_fraction": 0.6, "num_snapshots": 2, "validation_fraction": 0.1,
                       "split_mode": "count", "delimiter": "\t", "seed": 5,
                       "embedding_dim": 2, "num_layers": 0, "keep_prob": 1., "message_dropout": 0.}
        self.reference = self.root / "reference"
        self.comparison = self.root / "comparison"
        self.reference.mkdir()
        self.comparison.mkdir()
        before = build_model("MF", self.stream.num_users, self.stream.num_items, embedding_dim=2, n_layers=0)
        after = build_model("MF", self.stream.num_users, self.stream.num_items, embedding_dim=2, n_layers=0)
        with torch.no_grad():
            before.u_embeddings.weight[:] = torch.tensor([2., 0.])
            before.i_embeddings.weight[::2] = torch.tensor([1., 0.])
            before.i_embeddings.weight[1::2] = torch.tensor([0., 1.])
            after.load_state_dict(before.state_dict())
            after.u_embeddings.weight[1] = torch.tensor([0., 2.])
            after.u_embeddings.weight[2] = torch.tensor([-1., 3.])
            after.u_embeddings.weight[3] = torch.tensor([1., 1.])
        self.write(self.reference / "stage_000.pt", before, "Fine-tuning", 0)
        self.write(self.reference / "stage_001.pt", after, "Fine-tuning", 1)
        self.write(self.comparison / "stage_001.pt", after, "UGER", 1)

    def write(self, path, model, method, stage, config=None, graph_edges=None):
        torch.save({"format_version": 2, "method": method, "backbone": "MF", "stage": stage,
                    "checkpoint_kind": "continual", "config": self.config if config is None else config,
                    "data_fingerprint": self.stream.fingerprint, "num_users": self.stream.num_users,
                    "num_items": self.stream.num_items, "model_state_dict": model.state_dict(),
                    "graph_edges": self.stream.snapshots[stage] if graph_edges is None else graph_edges}, path)

    def test_reference_groups_are_shared_and_use_the_next_window(self):
        result = evaluate_user_groups(self.reference, {"UGER": self.comparison}, self.source, 2, 2)
        self.assertEqual(result["seed"], 5)
        self.assertEqual(result["skipped_trained_stages"][0]["stage"], 0)
        self.assertEqual(len(result["stages"]), 1)
        record = result["stages"][0]
        self.assertEqual(record["trained_through_stage"], 1)
        self.assertEqual(record["evaluated_on_stage"], 2)
        groups = record["groups"]
        self.assertEqual(len(groups["stable"]), 2)
        self.assertEqual(len(groups["dynamic"]), 2)
        self.assertNotIn(4, groups["stable"] + groups["dynamic"])
        self.assertEqual(groups["excluded_new_users"], 1)
        self.assertEqual(groups["stable"][0], 0)
        for group in ("stable", "dynamic"):
            baseline = record["methods"]["Fine-tuning"][group]
            comparison = record["methods"]["UGER"][group]
            self.assertEqual(baseline, comparison)
            self.assertEqual(baseline["validation"]["target_events"], 10)
            self.assertEqual(baseline["test"]["target_events"], 10)
            self.assertEqual(baseline["test"]["eligible_items"], self.stream.num_items)

    def test_method_embeddings_do_not_change_the_common_cohorts(self):
        before = evaluate_user_groups(self.reference, {"UGER": self.comparison}, self.source, 2, 2)
        payload = torch.load(self.comparison / "stage_001.pt", weights_only=False)
        payload["model_state_dict"]["u_embeddings.weight"].mul_(-1)
        torch.save(payload, self.comparison / "stage_001.pt")
        after = evaluate_user_groups(self.reference, {"UGER": self.comparison}, self.source, 2, 2)
        self.assertEqual(before["stages"][0]["groups"], after["stages"][0]["groups"])

    def test_comparison_seeds_must_match(self):
        payload = torch.load(self.comparison / "stage_001.pt", weights_only=False)
        payload["config"]["seed"] = 99
        torch.save(payload, self.comparison / "stage_001.pt")
        with self.assertRaisesRegex(ValueError, "seed"):
            evaluate_user_groups(self.reference, {"UGER": self.comparison}, self.source, 2, 2)

    def test_reference_requires_the_incoming_only_graph(self):
        payload = torch.load(self.reference / "stage_001.pt", weights_only=False)
        payload["graph_edges"] = np.concatenate(self.stream.snapshots[:2])
        torch.save(payload, self.reference / "stage_001.pt")
        with self.assertRaisesRegex(ValueError, "incoming-only"):
            evaluate_user_groups(self.reference, {}, self.source, 2, 2)

    def test_checkpoint_stage_and_data_are_checked(self):
        path = self.comparison / "stage_001.pt"
        payload = torch.load(path, weights_only=False)
        payload["stage"] = 0
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "stage"):
            evaluate_user_groups(self.reference, {"UGER": self.comparison}, self.source, 2, 2)
        payload["stage"] = 1
        payload["data_fingerprint"] = "different"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            evaluate_user_groups(self.reference, {"UGER": self.comparison}, self.source, 2, 2)

    def test_driver_writes_only_requested_results(self):
        output = self.root / "groups.json"
        arguments = ["--reference-run", str(self.reference), "--method-run", f"UGER={self.comparison}",
                     "--data-file", str(self.source), "--clusters", "2", "--minimum-history", "2", "--output", str(output)]
        self.assertEqual(main(arguments), output)
        self.assertTrue(output.is_file())
        with self.assertRaises(FileExistsError):
            main(arguments)

    def test_cli_requires_explicit_grouping_hyperparameters(self):
        with self.assertRaises(SystemExit):
            parse_args(["--reference-run", str(self.reference), "--data-file", str(self.source), "--output", "groups.json"])


if __name__ == "__main__":
    unittest.main()
