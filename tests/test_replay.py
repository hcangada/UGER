import unittest

import numpy as np
import torch

from uger.data import ObservedHistory, build_normalized_graph, sample_recommendation_batch
from uger.models import LightGCN
from uger.replay import (
    gradient_alignment_scores,
    minmax_normalize,
    select_replay_buffer,
    structural_scores,
)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.pool = np.asarray(
            [[0, 0, 1], [0, 1, 2], [1, 1, 3], [1, 2, 4], [2, 0, 5], [2, 3, 6]],
            dtype=np.int64,
        )
        self.history = ObservedHistory(self.pool)
        self.graph = build_normalized_graph(self.pool, 3, 4)
        self.model = LightGCN(3, 4, embedding_dim=3, n_layers=1)

    def test_gradient_alignment_matches_serial_manual_gradients(self):
        seed = 29
        sampled = sample_recommendation_batch(
            self.pool, self.history, 2, np.random.default_rng(seed)
        )
        parameters = tuple(self.model.parameters())
        gradients = []
        for index in range(len(sampled["users"])):
            batch = {key: value[index:index + 1] for key, value in sampled.items()}
            loss = self.model.recommendation_loss(batch, self.graph)
            gradients.append(torch.autograd.grad(loss, parameters))
        aggregate = [sum(gradient[position] for gradient in gradients) for position in range(len(parameters))]
        expected = np.zeros(len(self.pool))
        for row, gradient in zip(sampled["row_indices"].tolist(), gradients):
            expected[row] = sum((individual * total).sum().item() for individual, total in zip(gradient, aggregate))
        self.model.train()
        actual = gradient_alignment_scores(
            self.model, self.pool, self.graph, self.history, 2, 2, np.random.default_rng(seed)
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertTrue(self.model.training)

    def test_capacity_is_fixed_and_capped_by_pool(self):
        for capacity, expected in ((2, 2), (100, 6), (0, 0)):
            with self.subTest(capacity=capacity):
                result = select_replay_buffer(
                    self.model, self.pool, self.graph, self.history, capacity, 0,
                    1, 2, np.random.default_rng(9), method="UGER",
                )
                self.assertEqual(result.capacity, expected)
                self.assertEqual(len(result.edges), expected)
                self.assertTrue(set(map(tuple, result.edges)).issubset(set(map(tuple, self.pool))))

    def test_fusion_selects_highest_scores_with_deterministic_ties(self):
        pool = np.asarray([[2, 2, 9], [0, 1, 4], [1, 0, 6]], dtype=np.int64)
        graph = build_normalized_graph(pool, 3, 4)
        result = select_replay_buffer(
            self.model, pool, graph, ObservedHistory(pool), 2, 0,
            1, 2, np.random.default_rng(8), method="UGER",
        )
        np.testing.assert_array_equal(result.fusion_scores, np.zeros(3))
        np.testing.assert_array_equal(result.edges, np.asarray([[0, 1, 4], [1, 0, 6]]))

    def test_fusion_scores_match_separately_normalized_components(self):
        result = select_replay_buffer(
            self.model, self.pool, self.graph, self.history, 3, 0.35,
            2, 2, np.random.default_rng(13), method="UGER",
        )
        expected = 0.35 * minmax_normalize(result.model_scores) + 0.65 * minmax_normalize(result.structural_scores)
        np.testing.assert_allclose(result.fusion_scores, expected)
        selected = np.lexsort((self.pool[:, 2], self.pool[:, 1], self.pool[:, 0], -expected))[:3]
        self.assertEqual(set(map(tuple, result.edges)), set(map(tuple, self.pool[selected])))

    def test_normalization_handles_constant_and_empty_scores(self):
        np.testing.assert_array_equal(minmax_normalize([5, 5, 5]), np.zeros(3))
        np.testing.assert_array_equal(minmax_normalize([-2, 0, 2]), [0, 0.5, 1])
        self.assertEqual(minmax_normalize([]).size, 0)
        with self.assertRaises(FloatingPointError):
            minmax_normalize([0, np.nan])

    def test_structural_degrees_count_unique_graph_edges(self):
        pool = np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 2], [1, 0, 3]])
        np.testing.assert_allclose(structural_scores(pool), [0.5, 0.5, 0.75, 0.75])


if __name__ == "__main__":
    unittest.main()
