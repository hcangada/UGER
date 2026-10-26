import copy
import math
import unittest

import numpy as np
import torch

from uger.data import ObservedHistory, build_normalized_graph
from uger.diagnostics import (
    assign_user_groups,
    deterministic_topk,
    entering_competitor_coverage,
    evaluate_competitor_coverage,
    evaluate_replay_turnover,
    full_catalogue_eligible_items,
    item_cluster_centroids,
    preference_shift_scores,
    ranking_turnover,
)
from uger.models import build_model


class GroupingTests(unittest.TestCase):
    def test_centroids_are_seeded_and_detached(self):
        embeddings = torch.tensor([[0., 0.], [0., 1.], [9., 10.], [10., 10.]], requires_grad=True)
        first = item_cluster_centroids(embeddings, 2, torch.Generator().manual_seed(0))
        second = item_cluster_centroids(embeddings, 2, torch.Generator().manual_seed(0))
        torch.testing.assert_close(first, second)
        ordered = first[torch.argsort(first[:, 0])]
        torch.testing.assert_close(ordered, torch.tensor([[0., 0.5], [9.5, 10.]]))
        self.assertFalse(first.requires_grad)

    def test_empty_clusters_remain_finite(self):
        values = item_cluster_centroids(torch.ones(4, 2), 3, torch.Generator().manual_seed(0))
        self.assertEqual(tuple(values.shape), (3, 2))
        self.assertTrue(torch.isfinite(values).all())

    def test_groups_use_only_shared_reference_training_histories(self):
        before = torch.tensor([[3., 0.]] * 5)
        after = torch.tensor([[3., 0.], [0., 3.], [-3., 3.], [0., 3.], [0., 3.]])
        centroids = torch.eye(2)
        previous = ObservedHistory([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        current = ObservedHistory([[0, 0, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1],
                                   [2, 0, 0], [2, 1, 1], [3, 0, 0], [4, 0, 1], [4, 1, 2]])
        groups = assign_user_groups(before, after, centroids, previous, current, minimum_history=2)
        self.assertEqual(groups["stable"], [0])
        self.assertEqual(set(groups["dynamic"]), {1, 2})
        self.assertEqual(groups["excluded_new_users"], 1)
        self.assertEqual(groups["excluded_insufficient_history"], 1)
        self.assertEqual(set(groups["drift"]), {0, 1, 2})
        self.assertEqual(groups["drift"][0], 0)
        self.assertGreater(groups["drift"][2], groups["drift"][1])

    def test_jensen_shannon_is_symmetric_bounded_and_zero_without_drift(self):
        before = torch.tensor([[2., 0.], [0., 2.]])
        after = torch.tensor([[0., 2.], [2., 0.]])
        centroids = torch.eye(2)
        values = preference_shift_scores(before, after, centroids)
        torch.testing.assert_close(values, preference_shift_scores(after, before, centroids))
        self.assertTrue(torch.all(values > 0))
        self.assertTrue(torch.all(values <= math.log(2)))
        torch.testing.assert_close(preference_shift_scores(before, before, centroids), torch.zeros(2))


class RankingDiagnosticTests(unittest.TestCase):
    def test_ties_use_item_ids(self):
        np.testing.assert_array_equal(deterministic_topk([0, 0, 0], [5, 1, 3], 2), [1, 3])

    def test_coverage_excludes_the_anchor_from_entrants(self):
        items = np.arange(12)
        before = -items.astype(float)
        after = before.copy()
        after[10], after[11] = 100, 200
        result = entering_competitor_coverage(before, after, items, [11, 10, 1], 11)
        self.assertEqual(result["entering_items"], [10])
        self.assertEqual(result["coverage"], 1)
        missing = entering_competitor_coverage(before, after, items, [11, 1], 11)
        self.assertEqual(missing["coverage"], 0)

    def test_empty_entrants_are_not_assigned_perfect_coverage(self):
        items = np.arange(12)
        before = -items.astype(float)
        after = before.copy()
        after[11] = 100
        result = entering_competitor_coverage(before, after, items, [11, 1], 11)
        self.assertIsNone(result["coverage"])
        self.assertEqual(result["entering_count"], 0)

    def test_turnover_is_set_based(self):
        items = np.arange(12)
        before = -items.astype(float)
        after = before.copy()
        after[:10] = after[:10][::-1]
        self.assertEqual(ranking_turnover(before, after, items), 0)
        after[10:] = [100, 200]
        self.assertAlmostEqual(ranking_turnover(before, after, items), 0.2)

    def test_full_catalogue_mask_restores_only_the_anchor(self):
        history = ObservedHistory([[0, 0, 0], [0, 1, 1]])
        ordinary = full_catalogue_eligible_items(14, history, 0)
        diagnostic = full_catalogue_eligible_items(14, history, 0, anchor=0)
        self.assertNotIn(0, ordinary)
        self.assertNotIn(1, diagnostic)
        self.assertIn(0, diagnostic)
        self.assertIn(13, diagnostic)
        self.assertEqual(len(diagnostic), 13)

    def make_pair(self, num_items=14):
        before = build_model("MF", 2, num_items, embedding_dim=1)
        with torch.no_grad():
            before.u_embeddings.weight.fill_(1)
            before.i_embeddings.weight[:, 0] = torch.arange(num_items, 0, -1)
            before.i_embeddings.weight[1, 0] = 1000
        after = copy.deepcopy(before)
        with torch.no_grad():
            after.i_embeddings.weight[-1, 0] = 2000
        return before, after

    def test_complete_update_turnover_uses_the_full_catalogue(self):
        before, after = self.make_pair()
        history = ObservedHistory([[0, 0, 0], [0, 1, 1]])
        graph = build_normalized_graph([[0, 0, 0], [0, 1, 1]], 2, 14)
        result = evaluate_replay_turnover(before, after, graph, [[0, 0, 0]], history)
        self.assertEqual(result["evaluated_interactions"], 1)
        self.assertEqual(result["per_interaction"][0]["eligible_items"], 13)
        self.assertAlmostEqual(result["mean_turnover"], 0.1)
        self.assertTrue(before.training)
        self.assertTrue(after.training)

    def test_cases_with_fewer_than_ten_items_are_excluded(self):
        before, after = self.make_pair(10)
        history = ObservedHistory([[0, 0, 0], [0, 1, 1]])
        graph = build_normalized_graph([[0, 0, 0], [0, 1, 1]], 2, 10)
        result = evaluate_replay_turnover(before, after, graph, [[0, 0, 0]], history)
        self.assertIsNone(result["mean_turnover"])
        self.assertEqual(result["excluded_insufficient_items"], 1)

    def test_model_coverage_excludes_empty_cases_and_preserves_modes(self):
        before, after = self.make_pair(12)
        with torch.no_grad():
            before.i_embeddings.weight[1, 0] = 11
            after.i_embeddings.weight[1, 0] = 11
            after.i_embeddings.weight[10, 0] = 100
        after.eval()
        graph = build_normalized_graph([[0, 0, 0]], 2, 12)
        result = evaluate_competitor_coverage(before, after, graph, [[0, 11, 0]],
                                             [np.arange(12)], [[11, 10]], k=10)
        self.assertEqual(result["mean_coverage"], 1)
        self.assertEqual(result["evaluated_interactions"], 1)
        self.assertTrue(before.training)
        self.assertFalse(after.training)


if __name__ == "__main__":
    unittest.main()
