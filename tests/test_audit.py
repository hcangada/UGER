import copy
from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from uger.audit import audit_candidate_policies, cache_stage_supports, main
from uger.data import ObservedHistory, load_interaction_stream
from uger.models import build_model
from uger.stability import UGERStability


class RankingModel(nn.Module):
    def __init__(self, value=-0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float64))
        self.n_users = 2
        self.n_items = 20
        self.register_buffer("features", torch.arange(20, dtype=torch.float64))

    def encode(self, graph, masks=None):
        return self.weight.expand(self.n_users, 1), self.features[:, None]

    def forward(self, users, items, graph, masks=None):
        return self.weight * self.features[items]

    def recommendation_loss(self, batch, graph, masks=None):
        scores = self(batch["users"], batch["items"], graph, masks)
        return functional.softplus(scores[:, 1:] - scores[:, :1]).mean()

    def sample_masks(self, graph, generator=None):
        return {}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.model = RankingModel()
        self.history = ObservedHistory(np.asarray([[0, 0, 0], [0, 18, 1], [0, 19, 2], *[[1, item, item + 3] for item in range(1, 18)]], dtype=np.int64))
        self.replay = np.asarray([[0, 0, 0]], dtype=np.int64)
        self.mining = np.asarray([[0, 18, 1]], dtype=np.int64)
        self.reference = np.asarray([[0, 19, 2]], dtype=np.int64)
        self.stability = UGERStability(epsilon=0.1, proposal_size=14, clean_quota=2, probe_quota=2)
        self.cache = cache_stage_supports(self.model, None, self.replay, self.history, self.stability, torch.Generator().manual_seed(7))

    def test_reference_update_is_shared_and_cloned_optimizer_state_is_preserved(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad()
        self.model.weight.square().backward()
        optimizer.step()
        with torch.no_grad():
            self.model.weight.fill_(-0.02)
        before = self.model.weight.detach().clone()
        state = copy.deepcopy(optimizer.state_dict())
        torch_state = torch.get_rng_state().clone()
        report = audit_candidate_policies(self.model, optimizer, None, self.history, self.mining, self.reference,
                                          self.cache, self.stability, num_negatives=2, k=10, seed=23)
        torch.testing.assert_close(self.model.weight.detach(), before)
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        torch.testing.assert_close(optimizer.state_dict()["state"][0]["momentum_buffer"], state["state"][0]["momentum_buffer"])
        self.assertEqual(optimizer.state_dict()["param_groups"], state["param_groups"])
        entries = []
        for policy, result in report["policies"].items():
            self.assertEqual(len(result["candidate_sets"][0]), 5)
            self.assertIn(0, result["candidate_sets"][0])
            self.assertLessEqual(result["selected_perturbation_norm"], 0.1 + 1e-12)
            coverage = result["coverage"]
            self.assertEqual(coverage["evaluated_interactions"], 1, policy)
            self.assertEqual(coverage["excluded_empty_entrants"], 0)
            self.assertGreater(coverage["per_interaction"][0]["entering_count"], 0)
            entries.append(coverage["per_interaction"][0]["entering_items"])
            self.assertEqual(result["turnover"]["per_interaction"][0]["eligible_items"], 18)
        self.assertTrue(all(value == entries[0] for value in entries))
        self.assertTrue(report["incoming_batches_disjoint"])
        self.assertEqual(report["coverage_ranking_domain"], "shared_stage_cached_proposal_pool")
        self.assertEqual(report["turnover_ranking_domain"], "full_catalogue_with_anchor_and_without_other_history_positives")

    def test_fixed_support_is_cached_but_clean_support_is_reselected(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        frozen = self.cache.fixed_candidates[0].copy()
        with torch.no_grad():
            self.model.weight.fill_(0.02)
        report = audit_candidate_policies(self.model, optimizer, None, self.history, self.mining, self.reference,
                                          self.cache, self.stability, num_negatives=2, seed=23,
                                          policies=("Fixed", "Clean"))
        self.assertEqual(report["policies"]["Fixed"]["candidate_sets"][0], frozen.tolist())
        self.assertNotEqual(report["policies"]["Clean"]["candidate_sets"][0], frozen.tolist())
        self.assertEqual(self.cache.fixed_candidates[0].tolist(), frozen.tolist())
        for result in report["policies"].values():
            self.assertIsNone(result["coverage"]["mean_coverage"])
            self.assertEqual(result["coverage"]["excluded_empty_entrants"], 1)
            self.assertEqual(result["coverage"]["evaluated_interactions"], 0)

    def test_disjoint_batches_and_frozen_eligibility_are_enforced(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        with self.assertRaisesRegex(ValueError, "disjoint"):
            audit_candidate_policies(self.model, optimizer, None, self.history, self.mining, self.mining,
                                      self.cache, self.stability)
        self.history.add([[0, 10, 30]])
        with self.assertRaisesRegex(ValueError, "eligibility"):
            audit_candidate_policies(self.model, optimizer, None, self.history, self.mining, self.reference,
                                      self.cache, self.stability)

    def test_short_proposal_pools_are_excluded_from_coverage_only(self):
        small = UGERStability(epsilon=0.1, proposal_size=7, clean_quota=2, probe_quota=2)
        cache = cache_stage_supports(self.model, None, self.replay, self.history, small, torch.Generator().manual_seed(7))
        report = audit_candidate_policies(self.model, torch.optim.Adam(self.model.parameters()), None, self.history,
                                          self.mining, self.reference, cache, small, k=10, policies=("UGER",))
        result = report["policies"]["UGER"]
        self.assertIsNone(result["coverage"]["mean_coverage"])
        self.assertEqual(result["coverage"]["excluded_insufficient_items"], 1)
        self.assertEqual(result["turnover"]["evaluated_interactions"], 1)
        self.assertEqual(result["turnover"]["excluded_insufficient_items"], 0)

    def test_checkpoint_cli_is_opt_in_and_preserves_its_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "interactions.tsv"
            rows = [f"{index % 4}\t{(index % 4) * 6 + (index // 4) % 6}\t{index}" for index in range(80)]
            source.write_text("user_id\titem_id\ttimestamp\n" + "\n".join(rows) + "\n")
            stream = load_interaction_stream(source, num_incremental=2)
            config = {
                "initial_fraction": 0.6, "num_snapshots": 2, "validation_fraction": 0.1,
                "split_mode": "count", "embedding_dim": 4, "num_layers": 0,
                "keep_prob": 1.0, "message_dropout": 0.0, "learning_rate": 0.01,
                "weight_decay": 0.0, "epsilon": 0.1, "proposal_size": 14,
                "clean_quota": 2, "probe_quota": 2, "normalization_floor": 1e-12,
                "fallback_scale": 0.01, "random_scale": 0.01, "consistency_weight": 1.0,
                "num_negatives": 2,
            }
            model = build_model("MF", stream.num_users, stream.num_items, embedding_dim=4, n_layers=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            checkpoint = directory / "stage_000.pt"
            torch.save({
                "format_version": 2, "checkpoint_kind": "continual", "config": config,
                "data_fingerprint": stream.fingerprint, "num_users": stream.num_users,
                "num_items": stream.num_items, "stage": 0, "backbone": "MF",
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "buffer": stream.snapshots[0][:2],
            }, checkpoint)
            before_checkpoint = checkpoint.read_bytes()
            before_source = source.read_bytes()
            output = directory / "audit.json"
            main(["--checkpoint", str(checkpoint), "--data-file", str(source), "--output", str(output),
                  "--incoming-size", "2", "--replay-size", "2", "--warmup-steps", "1"])
            report = json.loads(output.read_text())
            self.assertEqual(report["training_stage"], 1)
            self.assertEqual(report["stage_start_checkpoint"], 0)
            self.assertEqual(report["warmup_steps"], 1)
            self.assertEqual(set(report["policies"]), {"Fixed", "Clean", "Probe", "UGER"})
            self.assertEqual(checkpoint.read_bytes(), before_checkpoint)
            self.assertEqual(source.read_bytes(), before_source)
            with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                main(["--checkpoint", str(checkpoint), "--data-file", str(source), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
