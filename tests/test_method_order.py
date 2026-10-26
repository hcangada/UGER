import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from uger.config import parse_args
from uger.data import ObservedHistory, sample_recommendation_batch
from uger.stability import build_candidates
from uger.trainer import UGERTrainer, seed_everything


class MethodOrderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.args = parse_args([
            "--embedding-dim", "4", "--num-layers", "1", "--batch-size", "2",
            "--proposal-size", "4", "--clean-quota", "1", "--probe-quota", "1",
            "--num-negatives", "2", "--seed", "13", "--weight-decay", "0",
        ])
        seed_everything(self.args.seed)
        self.trainer = UGERTrainer(
            self.args, SimpleNamespace(num_users=5, num_items=5),
            Path(self.temporary.name), torch.device("cpu"),
        )
        self.observed = np.asarray([[0, 0, 10], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
        self.history = ObservedHistory(self.observed)
        self.graph = self.trainer.graph(self.observed)

    def test_each_paired_step_reuses_joint_recommendation_negatives(self):
        incoming = np.asarray([[0, 0, timestamp] for timestamp in range(10, 15)])
        buffer = np.asarray([[1, 1, timestamp] for timestamp in range(2)])
        pending = []
        probe_calls = []
        joint_calls = []
        original_loss = self.trainer.model.recommendation_loss

        def consistency(model, new, replay, *extra, **kwargs):
            self.assertGreater(len(new["users"]), 0)
            self.assertGreater(len(replay["users"]), 0)
            self.assertTrue(torch.all(new["users"] == 0))
            self.assertTrue(torch.all(replay["users"] == 1))
            pending.append({key: value.detach().clone() for key, value in new.items()})
            probe_calls.append(len(new["users"]))
            return next(model.parameters()).sum() * 0, {"fallback_used": False}

        def recommendation(batch, graph, masks=None):
            self.assertTrue(pending)
            new = pending.pop(0)
            for user, items, timestamp in zip(new["users"], new["items"], new["timestamps"]):
                same_row = (batch["users"] == user) & (batch["timestamps"] == timestamp)
                self.assertEqual(int(same_row.sum()), 1)
                torch.testing.assert_close(batch["items"][same_row][0], items)
            self.assertTrue(torch.any(batch["users"] == 1))
            joint_calls.append(len(batch["users"]))
            return original_loss(batch, graph, masks=masks)

        with patch.object(self.trainer.stability, "compute", side_effect=consistency), patch.object(
            self.trainer.model, "recommendation_loss", side_effect=recommendation
        ), patch("uger.trainer.sample_recommendation_batch", wraps=sample_recommendation_batch) as sampler:
            statistics = self.trainer.train_epoch(incoming, buffer, self.graph, self.history, True)
        self.assertEqual(len(probe_calls), 5)
        self.assertEqual(len(joint_calls), statistics["steps"])
        self.assertEqual(sampler.call_count, statistics["steps"])
        self.assertEqual(pending, [])
        self.assertTrue(all(size == 2 for size in joint_calls))

    def test_probe_candidates_and_refinement_precede_one_clean_update(self):
        self.trainer.args.batch_size = 8
        initial = {name: value.detach().clone() for name, value in self.trainer.model.named_parameters()}
        events = []
        contexts = []
        original_probe = self.trainer.stability._probe
        original_discrepancy = self.trainer.stability._discrepancy
        original_step = self.trainer.optimizer.step

        def assert_clean():
            for name, value in self.trainer.model.named_parameters():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)

        def probe(*args, **kwargs):
            assert_clean()
            events.append("probe")
            output = original_probe(*args, **kwargs)
            assert_clean()
            return output

        def candidates(*args, **kwargs):
            assert_clean()
            events.append("candidates")
            return build_candidates(*args, **kwargs)

        def discrepancy(model, parameters, perturbation, users, items, mask, reference, graph, masks):
            assert_clean()
            events.append("discrepancy")
            context = (items.detach().clone(), mask.detach().clone(), reference.detach().clone())
            if contexts:
                for actual, expected in zip(context, contexts[0]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            contexts.append(context)
            output = original_discrepancy(model, parameters, perturbation, users, items, mask, reference, graph, masks)
            assert_clean()
            return output

        def step(*args, **kwargs):
            assert_clean()
            events.append("update")
            return original_step(*args, **kwargs)

        with patch.object(self.trainer.stability, "_probe", side_effect=probe), patch.object(
            self.trainer.stability, "_discrepancy", side_effect=discrepancy
        ), patch("uger.stability.build_candidates", side_effect=candidates), patch.object(
            self.trainer.optimizer, "step", side_effect=step
        ):
            self.trainer.train_epoch(self.observed[:1], self.observed[1:2], self.graph, self.history, True)
        self.assertEqual(events[0], "probe")
        self.assertEqual(events[1], "candidates")
        self.assertEqual(events[-1], "update")
        self.assertEqual(events.count("probe"), 1)
        self.assertEqual(events.count("candidates"), 1)
        self.assertEqual(events.count("update"), 1)
        self.assertGreaterEqual(events.count("discrepancy"), 5)
        self.assertTrue(any(not torch.equal(value, initial[name]) for name, value in self.trainer.model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
