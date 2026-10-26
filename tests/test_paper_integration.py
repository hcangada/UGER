import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import ttest_rel

from uger.config import parse_args
from uger.data import load_interaction_stream, merge_edges
from uger.reporting import aggregate_runs
from uger.trainer import run
from uger.user_groups import evaluate_user_groups


class PaperTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "events.tsv"
        rows = ["user_id\titem_id\ttimestamp"]
        for cycle in range(10):
            for user in range(10):
                rows.append(f"u{user}\ti{user + cycle}\t{10 * cycle + user}")
        self.data.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.stream = load_interaction_stream(self.data)

    def arguments(self, name, *extra):
        return parse_args([
            "--data-file", str(self.data), "--output-dir", str(self.root / "runs"),
            "--run-name", name, "--device", "cpu", "--num-threads", "1", "--seed", "31",
            "--pretrain-epochs", "1", "--incremental-epochs", "1", "--embedding-dim", "4",
            "--num-layers", "2", "--batch-size", "32", "--selection-batch-size", "64",
            "--proposal-size", "6", "--clean-quota", "2", "--probe-quota", "2",
            "--num-negatives", "2", "--keep-prob", "0.8", *extra,
        ])

    def run_quietly(self, args):
        with patch("uger.trainer.logging.basicConfig"), patch(
            "uger.trainer.logging.FileHandler", return_value=logging.NullHandler()
        ), patch("uger.trainer.logging.info"):
            return run(args)

    def assert_stage_protocol(self, directory, capacity):
        previous = None
        for stage in range(6):
            payload = torch.load(directory / f"stage_{stage:03d}.pt", weights_only=False)
            self.assertEqual(payload["format_version"], 2)
            self.assertEqual(payload["stage"], stage)
            self.assertEqual(payload["buffer_capacity"], capacity)
            pool = self.stream.snapshots[0] if stage == 0 else merge_edges(self.stream.snapshots[stage], previous["buffer"])
            self.assertEqual(set(map(tuple, payload["graph_edges"])), set(map(tuple, pool)))
            self.assertEqual(len(payload["buffer"]), min(capacity, len(pool)))
            self.assertTrue(set(map(tuple, payload["buffer"])).issubset(set(map(tuple, pool))))
            self.assertEqual(len(payload["results"]), stage)
            if stage:
                result = payload["results"][-1]
                window = self.stream.snapshots[stage]
                self.assertEqual(result["trained_through_stage"], stage - 1)
                self.assertEqual(result["evaluated_on_stage"], stage)
                self.assertEqual(result["validation"]["target_events"], len(window) // 2)
                self.assertEqual(result["target_events"], len(window) - len(window) // 2)
                self.assertEqual(result["eligible_items"], self.stream.num_items)
                self.assertTrue(all(np.isfinite(value) for value in result["metrics"].values()))
            previous = payload

    def test_complete_uger_runs_with_all_three_paper_backbones(self):
        for backbone in ("LightGCN", "NGCF", "MGCCF"):
            with self.subTest(backbone=backbone):
                directory = self.run_quietly(self.arguments(backbone, "--backbone", backbone))
                self.assert_stage_protocol(directory, len(self.stream.snapshots[1]))
                first = torch.load(directory / "stage_000.pt", weights_only=False)
                final = torch.load(directory / "stage_005.pt", weights_only=False)
                self.assertTrue(any(not torch.equal(value, final["model_state_dict"][name])
                                    for name, value in first["model_state_dict"].items()))

    def test_cli_controls_preserve_the_shared_buffer_schedule(self):
        source = self.run_quietly(self.arguments("source"))
        variants = ("STAR-Rec", "STAR-Rec-CE", "w/o-CE", "random-probe", "Probe-c", "KL-probe", "w/o-refinement", "w/o-stab")
        with np.load(source / "buffers.npz", allow_pickle=False) as buffers:
            schedule = {key: buffers[key].copy() for key in buffers.files if key.startswith("stage_")}
        for variant in variants:
            with self.subTest(variant=variant):
                directory = self.run_quietly(self.arguments(
                    variant, "--variant", variant, "--pretrained", str(source / "pretrained.pt"),
                    "--replay-schedule", str(source / "buffers.npz"),
                ))
                self.assert_stage_protocol(directory, len(self.stream.snapshots[1]))
                for stage in range(6):
                    payload = torch.load(directory / f"stage_{stage:03d}.pt", weights_only=False)
                    np.testing.assert_array_equal(payload["buffer"], schedule[f"stage_{stage:03d}"])

    def test_fixed_replay_schedule_resumes_without_repeating_the_flag(self):
        source = self.run_quietly(self.arguments("source"))
        complete = self.run_quietly(self.arguments(
            "scheduled", "--variant", "STAR-Rec", "--pretrained", str(source / "pretrained.pt"),
            "--replay-schedule", str(source / "buffers.npz"),
        ))
        resumed = self.run_quietly(self.arguments(
            "resumed", "--variant", "STAR-Rec", "--resume", str(complete / "stage_001.pt"),
        ))
        expected = torch.load(complete / "stage_005.pt", weights_only=False)
        actual = torch.load(resumed / "stage_005.pt", weights_only=False)
        self.assertEqual(expected["results"], actual["results"])
        for name, value in expected["model_state_dict"].items():
            torch.testing.assert_close(value, actual["model_state_dict"][name], rtol=0, atol=0)
        for key, value in expected["buffer_schedule"].items():
            np.testing.assert_array_equal(value, actual["buffer_schedule"][key])

    def test_resume_rejects_a_different_supplied_future_schedule(self):
        source = self.run_quietly(self.arguments("source"))
        scheduled = self.run_quietly(self.arguments(
            "scheduled", "--variant", "STAR-Rec", "--pretrained", str(source / "pretrained.pt"),
            "--replay-schedule", str(source / "buffers.npz"),
        ))
        with np.load(source / "buffers.npz", allow_pickle=False) as buffers:
            changed = {key: buffers[key].copy() for key in buffers.files}
        changed["stage_003"] = changed["stage_003"][::-1].copy()
        path = self.root / "different.npz"
        np.savez_compressed(path, **changed)
        with self.assertRaisesRegex(ValueError, "schedule"):
            self.run_quietly(self.arguments(
                "changed", "--variant", "STAR-Rec", "--resume", str(scheduled / "stage_001.pt"),
                "--replay-schedule", str(path),
            ))

    def test_user_group_driver_consumes_real_stage_checkpoints(self):
        reference = self.run_quietly(self.arguments("reference", "--method", "Fine-tuning"))
        comparison = self.run_quietly(self.arguments("comparison", "--pretrained", str(reference / "pretrained.pt")))
        report = evaluate_user_groups(reference, {"UGER": comparison}, self.data, clusters=2, minimum_history=2)
        self.assertEqual(len(report["stages"]), 4)
        self.assertEqual([row["evaluated_on_stage"] for row in report["stages"]], [2, 3, 4, 5])
        for row in report["stages"]:
            for group in ("stable", "dynamic"):
                left = row["methods"]["Fine-tuning"][group]
                right = row["methods"]["UGER"][group]
                self.assertEqual(left["test"]["evaluated_users"], right["test"]["evaluated_users"])


class PaperReportingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.samples = {
            "UGER": [0.21, 0.20, 0.22, 0.19, 0.23],
            "INFER": [0.18, 0.169, 0.188, 0.157, 0.195],
            "PISA": [0.207, 0.202, 0.217, 0.192, 0.227],
        }
        self.directories = []
        for method, values in self.samples.items():
            for seed, value in enumerate(values, 1):
                directory = self.root / f"{method}_{seed}"
                directory.mkdir()
                config = {"dataset": "Unit", "backbone": "LightGCN", "method": method,
                          "variant": "full", "seed": seed, "data_fingerprint": "unit-data",
                          "initial_fraction": 0.6, "validation_fraction": 0.1,
                          "num_snapshots": 2, "split_mode": "count", "buffer_size": 8}
                stages = [{"evaluated_on_stage": stage, "trained_through_stage": stage - 1,
                           "evaluated_users": 1, "validation": {},
                           "metrics": {"Recall@10": value + offset, "NDCG@10": (value + offset) / 2}}
                          for stage, offset in ((1, -0.01), (2, 0.01))]
                (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
                (directory / "metrics.json").write_text(json.dumps(stages), encoding="utf-8")
                (directory / "summary.json").write_text("{}", encoding="utf-8")
                self.directories.append(directory)

    def test_paired_tests_use_five_stage_averaged_run_observations(self):
        result = aggregate_runs(self.directories)[0]
        self.assertEqual(result["seeds"], [1, 2, 3, 4, 5])
        row = result["metrics"]["Recall@10"]
        reference = np.asarray(self.samples["UGER"])
        self.assertAlmostEqual(row["estimates"]["UGER"]["mean_percent"], reference.mean() * 100)
        self.assertAlmostEqual(row["estimates"]["UGER"]["sd_percent"], reference.std(ddof=1) * 100)
        self.assertEqual(len(row["estimates"]["UGER"]["run_means_percent"]), 5)
        raw = [ttest_rel(reference, self.samples[name]).pvalue for name in ("INFER", "PISA")]
        for name, pvalue in zip(("INFER", "PISA"), raw):
            self.assertAlmostEqual(row["comparisons"][name]["p_raw"], pvalue)
        self.assertAlmostEqual(row["comparisons"]["INFER"]["p_holm"], min(1, 2 * raw[0]))
        self.assertAlmostEqual(row["comparisons"]["PISA"]["p_holm"], max(min(1, 2 * raw[0]), raw[1]))
        self.assertFalse(row["significant_against_all_provided_comparators"])

    def test_different_seed_sets_are_rejected(self):
        incomplete = [directory for directory in self.directories if directory.name != "INFER_5"]
        with self.assertRaisesRegex(ValueError, "seed"):
            aggregate_runs(incomplete)

    def test_comparisons_require_matching_replay_capacities(self):
        path = self.root / "INFER_1" / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["buffer_size"] = 80
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identical"):
            aggregate_runs(self.directories)

    def test_comparisons_require_the_same_initial_validation_split(self):
        path = self.root / "INFER_1" / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["validation_fraction"] = 0.2
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identical"):
            aggregate_runs(self.directories)


if __name__ == "__main__":
    unittest.main()
