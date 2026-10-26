import unittest

import numpy as np
import torch

from uger.data import ObservedHistory, build_normalized_graph
from uger.evaluation import evaluate
from uger.models import build_model


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.edges = np.asarray([[0, 0, 1], [1, 2, 2], [1, 1, 3]], dtype=np.int64)
        self.history = ObservedHistory(self.edges)
        self.graph = build_normalized_graph(self.edges, 3, 4)
        self.model = build_model("MF", 3, 4, embedding_dim=1)
        with torch.no_grad():
            self.model.u_embeddings.weight.fill_(1)
            self.model.i_embeddings.weight.zero_()
            self.model.i_embeddings.weight[1].fill_(3)
            self.model.i_embeddings.weight[2].fill_(2)
            self.model.i_embeddings.weight[3].fill_(1000)

    def test_full_catalogue_and_new_users_are_evaluated(self):
        targets = np.asarray([[0, 1, 4], [0, 3, 5], [2, 1, 6], [0, 0, 7]])
        result = evaluate(self.model, self.graph, targets, self.history, top_k=(1, 2))
        self.assertEqual(result["evaluated_users"], 2)
        self.assertEqual(result["eligible_items"], 4)
        self.assertEqual(result["cold_events_excluded"], 0)
        self.assertEqual(result["cold_events_included"], 2)
        self.assertEqual(result["repeated_events_excluded"], 1)
        self.assertEqual(result["metrics"]["Recall@1"], 0.25)
        self.assertEqual(result["metrics"]["NDCG@1"], 0.5)
        self.assertEqual(result["metrics"]["Recall@2"], 1)
        self.assertAlmostEqual(result["metrics"]["NDCG@2"], (1 + 1 / np.log2(3)) / 2)
        self.assertTrue(self.model.training)

    def test_score_ties_follow_item_ids(self):
        with torch.no_grad():
            self.model.i_embeddings.weight.zero_()
        targets = np.asarray([[0, 2, 4]])
        result = evaluate(self.model, self.graph, targets, self.history, top_k=(1, 2))
        self.assertEqual(result["metrics"]["Recall@1"], 0)
        self.assertEqual(result["metrics"]["Recall@2"], 1)
        self.assertAlmostEqual(result["metrics"]["NDCG@2"], 1 / np.log2(3))

    def test_metrics_use_distinct_relevant_items_per_user(self):
        with torch.no_grad():
            self.model.i_embeddings.weight.zero_()
        targets = np.asarray([[0, 1, 4], [0, 1, 5], [0, 2, 6]])
        result = evaluate(self.model, self.graph, targets, self.history, top_k=(1, 2))
        self.assertEqual(result["metrics"]["Recall@1"], 0.5)
        self.assertEqual(result["metrics"]["Recall@2"], 1)
        self.assertEqual(result["metrics"]["NDCG@1"], 1)
        self.assertEqual(result["metrics"]["NDCG@2"], 1)

    def test_empty_targets_return_finite_zero_metrics(self):
        result = evaluate(
            self.model, self.graph, np.empty((0, 3), dtype=np.int64), ObservedHistory(), top_k=(1, 2)
        )
        self.assertEqual(result["evaluated_users"], 0)
        self.assertTrue(all(value == 0 for value in result["metrics"].values()))

    def test_unseen_catalogue_items_remain_ranking_competitors(self):
        targets = np.asarray([[0, 1, 4]])
        result = evaluate(self.model, self.graph, targets, self.history, top_k=(1, 2))
        self.assertEqual(result["metrics"]["Recall@1"], 0)
        self.assertEqual(result["metrics"]["Recall@2"], 1)

    def test_validation_and_test_share_the_pre_window_mask(self):
        with torch.no_grad():
            self.model.i_embeddings.weight.zero_()
        validation = np.asarray([[0, 1, 4]])
        test = np.asarray([[0, 1, 5]])
        before = {user: items.copy() for user, items in self.history.user_items.items()}
        first = evaluate(self.model, self.graph, validation, self.history, top_k=(1,))
        second = evaluate(self.model, self.graph, test, self.history, top_k=(1,))
        self.assertEqual(first["metrics"]["Recall@1"], 1)
        self.assertEqual(second["metrics"]["Recall@1"], 1)
        self.assertEqual(self.history.user_items, before)

    def test_group_cohorts_do_not_change_the_catalogue(self):
        targets = np.asarray([[0, 3, 4], [2, 1, 5]])
        result = evaluate(self.model, self.graph, targets, self.history, top_k=(1,), users=[0], return_per_user=True)
        self.assertEqual(result["evaluated_users"], 1)
        self.assertEqual(result["cohort_events_excluded"], 1)
        self.assertEqual(result["eligible_items"], 4)
        self.assertEqual(result["metrics"]["Recall@1"], 1)
        self.assertEqual(set(result["per_user"]), {0})

    def test_invalid_global_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.model, self.graph, np.asarray([[3, 1, 4]]), self.history)

    def test_repeated_cutoffs_do_not_double_count_metrics(self):
        targets = np.asarray([[0, 3, 4]])
        once = evaluate(self.model, self.graph, targets, self.history, top_k=(1,))
        repeated = evaluate(self.model, self.graph, targets, self.history, top_k=(1, 1))
        self.assertEqual(once["metrics"], repeated["metrics"])


if __name__ == "__main__":
    unittest.main()
