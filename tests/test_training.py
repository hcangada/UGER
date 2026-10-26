import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from uger.config import parse_args
from uger.data import ObservedHistory, load_interaction_stream, merge_edges, sample_recommendation_batch
from uger.replay import ReplaySelection
from uger.trainer import UGERTrainer, run, seed_everything


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "interactions.tsv"
        rows = ["user_id\titem_id\ttimestamp"]
        for cycle in range(9):
            for user in range(20):
                timestamp = cycle * 20 + user
                rows.append(f"u{user}\ti{(user + 3 * cycle) % 30}\t{timestamp}")
        self.data.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.addCleanup(self.temporary.cleanup)

    def arguments(self, name, *extra):
        return parse_args([
            "--data-file", str(self.data), "--output-dir", str(self.root / "runs"),
            "--run-name", name, "--device", "cpu", "--num-threads", "1", "--seed", "31",
            "--num-snapshots", "2", "--pretrain-epochs", "1", "--incremental-epochs", "1",
            "--embedding-dim", "4", "--num-layers", "1", "--batch-size", "64",
            "--selection-batch-size", "64", "--proposal-size", "4",
            "--clean-quota", "1", "--probe-quota", "1", "--num-negatives", "2",
            "--top-k", "5", "10", *extra,
        ])

    def run_quietly(self, arguments):
        with patch("uger.trainer.logging.basicConfig"), patch(
            "uger.trainer.logging.FileHandler", return_value=logging.NullHandler()
        ), patch("uger.trainer.logging.info"):
            return run(arguments)

    def test_end_to_end_resume_preserves_parameters_metrics_and_history(self):
        args = self.arguments("complete")
        directory = self.run_quietly(args)
        stream = load_interaction_stream(self.data, num_incremental=2)
        checkpoints = [torch.load(directory / f"stage_{stage:03d}.pt", weights_only=False) for stage in range(3)]
        for stage, payload in enumerate(checkpoints):
            observed = merge_edges(*stream.snapshots[:stage + 1])
            pool = stream.snapshots[0] if stage == 0 else merge_edges(stream.snapshots[stage], checkpoints[stage - 1]["buffer"])
            self.assertEqual(set(map(tuple, payload["graph_edges"])), set(map(tuple, pool)))
            self.assertTrue(set(map(tuple, payload["buffer"])).issubset(set(map(tuple, pool))))
            expected_size = min(args.buffer_size, len(pool))
            self.assertEqual(len(payload["buffer"]), expected_size)
            self.assertTrue(np.all(payload["buffer"][:, 2] <= observed[:, 2].max()))
            self.assertEqual(len(payload["results"]), stage)
        resumed_directory = self.run_quietly(self.arguments("resumed", "--resume", str(directory / "stage_000.pt")))
        resumed = torch.load(resumed_directory / "stage_002.pt", weights_only=False)
        for name, value in checkpoints[-1]["model_state_dict"].items():
            torch.testing.assert_close(value, resumed["model_state_dict"][name], rtol=0, atol=0)
        np.testing.assert_array_equal(checkpoints[-1]["buffer"], resumed["buffer"])
        self.assertEqual(checkpoints[-1]["results"], resumed["results"])
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["evaluated_stages"], 2)
        self.assertTrue(all(np.isfinite(value) for value in summary["stage_mean"].values()))

    def test_resuming_the_terminal_stage_preserves_completed_outputs(self):
        directory = self.run_quietly(self.arguments("terminal-source"))
        resumed_directory = self.run_quietly(self.arguments(
            "terminal-resumed", "--resume", str(directory / "stage_002.pt")
        ))
        source = torch.load(directory / "stage_002.pt", weights_only=False)
        resumed = torch.load(resumed_directory / "stage_002.pt", weights_only=False)
        for name, value in source["model_state_dict"].items():
            torch.testing.assert_close(value, resumed["model_state_dict"][name], rtol=0, atol=0)
        self.assertEqual(json.loads((directory / "metrics.json").read_text()),
                         json.loads((resumed_directory / "metrics.json").read_text()))
        np.testing.assert_array_equal(source["buffer"], resumed["buffer"])

    def test_pretraining_outputs_only_a_stage_zero_checkpoint(self):
        directory = self.run_quietly(self.arguments("pretrain", "--mode", "pretrain"))
        payload = torch.load(directory / "pretrained.pt", weights_only=False)
        self.assertEqual(payload["stage"], 0)
        self.assertEqual(payload["checkpoint_kind"], "pretrained")
        self.assertEqual(payload["selected_pretraining_epochs"], 1)
        self.assertEqual(payload["results"], [])
        self.assertEqual(len(payload["buffer"]), 0)
        self.assertEqual(list(directory.glob("stage_*.pt")), [])
        stream = load_interaction_stream(self.data, num_incremental=2)
        np.testing.assert_array_equal(payload["graph_edges"], stream.snapshots[0])

    def test_separate_pretraining_preserves_continual_rng_and_results(self):
        complete_directory = self.run_quietly(self.arguments("single-command", "--keep-prob", "0.8"))
        pretrain_directory = self.run_quietly(self.arguments(
            "separate-pretrain", "--mode", "pretrain", "--keep-prob", "0.8"
        ))
        reused_directory = self.run_quietly(self.arguments(
            "reused-pretrain", "--pretrained", str(pretrain_directory / "pretrained.pt"),
            "--keep-prob", "0.8",
        ))
        complete = torch.load(complete_directory / "stage_002.pt", weights_only=False)
        reused = torch.load(reused_directory / "stage_002.pt", weights_only=False)
        for name, value in complete["model_state_dict"].items():
            torch.testing.assert_close(value, reused["model_state_dict"][name], rtol=0, atol=0)
        np.testing.assert_array_equal(complete["buffer"], reused["buffer"])
        self.assertEqual(complete["results"], reused["results"])
        self.assertTrue(torch.equal(complete["rng_state"]["generator"], reused["rng_state"]["generator"]))
        self.assertEqual(complete["rng_state"]["numpy"], reused["rng_state"]["numpy"])
        self.assertEqual(complete["rng_state"]["selection_numpy"], reused["rng_state"]["selection_numpy"])

    def test_fixed_batch_update_reduces_its_recommendation_loss(self):
        args = self.arguments("controlled")
        seed_everything(args.seed)
        stream = load_interaction_stream(self.data, num_incremental=2)
        trainer = UGERTrainer(args, stream, self.root, torch.device("cpu"))
        incoming = stream.snapshots[0][:8]
        history = ObservedHistory(stream.snapshots[0])
        graph = trainer.graph(stream.snapshots[0])
        batch = sample_recommendation_batch(incoming, history, 2, np.random.default_rng(19))
        before = trainer.model.recommendation_loss(batch, graph).item()
        with patch("uger.trainer.sample_recommendation_batch", return_value=batch):
            statistics = trainer.train_epoch(incoming, np.empty((0, 3), dtype=np.int64), graph, history, False)
        after = trainer.model.recommendation_loss(batch, graph).item()
        self.assertLess(after, before)
        self.assertEqual(statistics["steps"], 1)
        self.assertTrue(np.isfinite(statistics["recommendation_loss"]))

    def test_replay_selection_follows_updates_and_buffer_is_fixed_per_stage(self):
        args = self.arguments("ordering", "--incremental-epochs", "2")
        stream = load_interaction_stream(self.data, num_incremental=2)
        trainer = UGERTrainer(args, stream, self.root, torch.device("cpu"))
        selections = []
        training_buffers = []

        def pretrain():
            with torch.no_grad():
                next(trainer.model.parameters()).zero_()
            initial = stream.snapshots[0]
            return ObservedHistory(initial), trainer.graph(initial), initial

        def select(model, pool, graph, history, capacity, beta, num_negatives, batch_size, rng, method):
            capacity = min(capacity, len(pool))
            selections.append((float(next(model.parameters()).detach().flatten()[0]), pool.copy()))
            zeros = np.zeros(len(pool))
            return ReplaySelection(pool[:capacity].copy(), zeros, zeros, zeros, capacity)

        def epoch(incoming, buffer, graph, history, consistency):
            training_buffers.append(buffer.copy())
            with torch.no_grad():
                next(trainer.model.parameters()).add_(1)
            return {"recommendation_loss": 0.0, "consistency_loss": 0.0, "steps": 1,
                    "skipped_rows": 0, "probe_fallbacks": 0}

        with patch.object(trainer, "pretrain", side_effect=pretrain), patch.object(
            trainer, "train_epoch", side_effect=epoch
        ), patch("uger.trainer.select_replay_buffer", side_effect=select), patch("uger.trainer.logging.info"):
            trainer.continual()
        self.assertEqual([value for value, _ in selections], [0.0, 2.0, 4.0])
        self.assertEqual(len(training_buffers), 4)
        np.testing.assert_array_equal(training_buffers[0], training_buffers[1])
        np.testing.assert_array_equal(training_buffers[2], training_buffers[3])
        np.testing.assert_array_equal(selections[1][1], merge_edges(stream.snapshots[1], training_buffers[0]))
        np.testing.assert_array_equal(selections[2][1], merge_edges(stream.snapshots[2], training_buffers[2]))

    def test_replay_stability_keeps_singleton_rows_without_bpr_negatives(self):
        args = self.arguments("singleton")
        stream = load_interaction_stream(self.data, num_incremental=2)
        trainer = UGERTrainer(args, stream, self.root, torch.device("cpu"))
        observed = np.asarray([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [2, 1, 4]])
        incoming = observed[-1:]
        buffer = observed[[0, 3]]
        history = ObservedHistory(observed)
        graph = trainer.graph(observed)
        replay_users = []

        def consistency(model, incoming_batch, replay_batch, *extra, **kwargs):
            replay_users.extend(replay_batch["users"].tolist())
            return next(model.parameters()).sum() * 0, {"fallback_used": False}

        with patch.object(trainer.stability, "compute", side_effect=consistency):
            trainer.train_epoch(incoming, buffer, graph, history, True)
        self.assertEqual(sorted(replay_users), [0, 1])


if __name__ == "__main__":
    unittest.main()
