import unittest

import torch
from torch.func import functional_call
from torch.nn import functional as F

from uger.models import (
    MODEL_REGISTRY,
    BaseGraphRecommender,
    LightGCN,
    MGCCF,
    NGCF,
    build_model,
    register_model,
)


def make_graph():
    adjacency = torch.zeros(7, 7)
    for user, item in ((0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2), (2, 3)):
        adjacency[user, 3 + item] = 1
        adjacency[3 + item, user] = 1
    inverse = adjacency.sum(dim=1).clamp_min(1).rsqrt()
    return (inverse[:, None] * adjacency * inverse[None, :]).to_sparse().coalesce()


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.graph = make_graph()
        self.users = torch.tensor([0, 1, 2], dtype=torch.long)
        self.items = torch.tensor([[0, 2, 3], [1, 0, 3], [2, 1, 3]], dtype=torch.long)
        self.batch = {"users": self.users, "items": self.items}

    def test_lightgcn_matches_layer_average(self):
        model = LightGCN(3, 4, embedding_dim=3, n_layers=2)
        initial = torch.cat((model.u_embeddings.weight, model.i_embeddings.weight))
        dense = self.graph.to_dense()
        expected = (initial + dense @ initial + dense @ dense @ initial) / 3
        users, items = model.encode(self.graph)
        torch.testing.assert_close(users, expected[:3])
        torch.testing.assert_close(items, expected[3:])

    def test_zero_layer_lightgcn_is_matrix_factorization(self):
        model = LightGCN(3, 4, embedding_dim=3, n_layers=0)
        scores = model(self.users, self.items, self.graph)
        expected = (
            model.u_embeddings(self.users).unsqueeze(1)
            * model.i_embeddings(self.items)
        ).sum(dim=-1)
        torch.testing.assert_close(scores, expected)

    def test_bpr_is_stable_for_extreme_negative_margins(self):
        model = build_model("mf", 1, 2, embedding_dim=1)
        with torch.no_grad():
            model.u_embeddings.weight.fill_(1)
            model.i_embeddings.weight.copy_(torch.tensor([[-1000.0], [1000.0]]))
        graph = torch.eye(3).to_sparse()
        loss = model.recommendation_loss(
            {"users": torch.tensor([0]), "items": torch.tensor([[0, 1]])}, graph
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 2000.0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.i_embeddings.weight.grad).all())

    def test_per_interaction_bpr_and_mean_agree(self):
        model = LightGCN(3, 4, embedding_dim=4, n_layers=1)
        scores = model(self.users, self.items, self.graph)
        expected = F.softplus(scores[:, 1:] - scores[:, :1]).mean(dim=1)
        actual = model.per_interaction_loss(self.batch, self.graph)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            model.recommendation_loss(self.batch, self.graph), expected.mean()
        )

    def test_ngcf_matches_paper_propagation_with_output_normalization_disabled(self):
        model = NGCF(3, 4, embedding_dim=4, n_layers=1, normalize_output=False)
        initial = torch.cat((model.u_embeddings.weight, model.i_embeddings.weight))
        neighbors = self.graph.to_dense() @ initial
        propagated = F.leaky_relu(
            F.linear(initial + neighbors, model.sum_layers[0].weight)
            + F.linear(initial * neighbors, model.interaction_layers[0].weight),
            negative_slope=0.2,
        )
        expected = torch.cat((initial, propagated), dim=1)
        users, items = model.encode(self.graph)
        torch.testing.assert_close(users, expected[:3])
        torch.testing.assert_close(items, expected[3:])

    def test_ngcf_all_parameters_receive_finite_gradients(self):
        model = NGCF(3, 4, embedding_dim=4, n_layers=2, layer_sizes=(5, 3))
        loss = model.recommendation_loss(self.batch, self.graph)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.norm().item(), 0)

    def test_functional_perturbation_recomputes_graph_predictions(self):
        for name in ("lightgcn", "ngcf", "mgccf", "mf"):
            with self.subTest(name=name):
                model = build_model(name, 3, 4, embedding_dim=4, n_layers=2)
                parameters = dict(model.named_parameters())
                before = {name: tensor.detach().clone() for name, tensor in parameters.items()}
                perturbations = {
                    name: torch.randn_like(tensor).mul(0.01).requires_grad_()
                    for name, tensor in parameters.items()
                }
                shifted = {name: tensor + perturbations[name] for name, tensor in parameters.items()}
                clean = model(self.users, self.items, self.graph, {})
                perturbed = functional_call(
                    model, shifted, (self.users, self.items, self.graph, {})
                )
                self.assertFalse(torch.allclose(clean, perturbed))
                gradients = torch.autograd.grad(perturbed.square().sum(), tuple(perturbations.values()))
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                for name, tensor in model.named_parameters():
                    torch.testing.assert_close(tensor, before[name])

    def test_cached_masks_make_stochastic_forwards_repeatable(self):
        for model in (
            LightGCN(3, 4, embedding_dim=6, n_layers=2, keep_prob=0.5),
            NGCF(3, 4, embedding_dim=6, n_layers=2, keep_prob=0.6, message_dropout=0.5),
            MGCCF(3, 4, embedding_dim=6, n_layers=2, keep_prob=0.6, message_dropout=0.5),
        ):
            with self.subTest(model=type(model).__name__):
                first = model.sample_masks(self.graph, torch.Generator().manual_seed(2))
                duplicate = model.sample_masks(self.graph, torch.Generator().manual_seed(2))
                second = model.sample_masks(self.graph, torch.Generator().manual_seed(3))
                scores = model(self.users, self.items, self.graph, first)
                torch.testing.assert_close(scores, model(self.users, self.items, self.graph, first))
                torch.testing.assert_close(scores, model(self.users, self.items, self.graph, duplicate))
                self.assertFalse(torch.equal(first["edge"], second["edge"]))

    def test_evaluation_disables_random_masks(self):
        model = NGCF(3, 4, embedding_dim=4, keep_prob=0.5, message_dropout=0.5)
        model.eval()
        self.assertEqual(model.sample_masks(self.graph), {})
        torch.testing.assert_close(
            model(self.users, self.items, self.graph),
            model(self.users, self.items, self.graph),
        )

    def test_graph_is_not_checkpoint_state(self):
        model = LightGCN(3, 4, embedding_dim=4)
        model(self.users, self.items, self.graph)
        self.assertEqual(set(model.state_dict()), {"u_embeddings.weight", "i_embeddings.weight"})
        restored = LightGCN(3, 4, embedding_dim=4)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            model(self.users, self.items, self.graph),
            restored(self.users, self.items, self.graph),
        )

    def test_invalid_graph_and_batch_are_rejected(self):
        model = LightGCN(3, 4, embedding_dim=4)
        with self.assertRaises(TypeError):
            model(self.users, self.items, self.graph.to_dense())
        with self.assertRaises(ValueError):
            model(self.users, self.items[:, 0], self.graph)
        with self.assertRaises(ValueError):
            model.recommendation_loss({"users": self.users, "items": self.items[:, :1]}, self.graph)
        with self.assertRaises(ValueError):
            model(self.users, self.items, self.graph, {"edge": torch.ones(1, dtype=torch.bool)})
        with self.assertRaises(IndexError):
            model(torch.tensor([-1]), self.items[:1], self.graph)
        with self.assertRaises(IndexError):
            model(self.users[:1], torch.tensor([[0, -1]]), self.graph)

    def test_external_adapters_are_registered_explicitly(self):
        class Adapter(LightGCN):
            pass

        register_model("test-adapter", Adapter)
        try:
            model = build_model("test_adapter", 3, 4, embedding_dim=4)
            self.assertIsInstance(model, BaseGraphRecommender)
            self.assertEqual(model(self.users, self.items, self.graph).shape, (3, 3))
        finally:
            MODEL_REGISTRY.pop("testadapter")
        self.assertIsInstance(build_model("MGCCF", 3, 4), MGCCF)

    def test_mgccf_matches_three_graph_sum_fusion(self):
        model = MGCCF(3, 4, embedding_dim=4, n_layers=1)
        model.eval()
        users, items = model.u_embeddings.weight, model.i_embeddings.weight
        adjacency = (self.graph.to_dense() > 0).float()
        mean = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1)
        neighborhood = mean @ torch.cat((users, items), dim=0)
        expected_users = torch.tanh(model.user_layers[0](torch.cat((
            users, torch.tanh(model.user_aggregators[0](neighborhood[:3]))
        ), dim=1)))
        expected_items = torch.tanh(model.item_layers[0](torch.cat((
            items, torch.tanh(model.item_aggregators[0](neighborhood[3:]))
        ), dim=1)))
        interactions = adjacency[:3, 3:]
        user_similarity = (interactions @ interactions.T > 0).float()
        item_similarity = (interactions.T @ interactions > 0).float()
        user_similarity.fill_diagonal_(0)
        item_similarity.fill_diagonal_(0)
        expected_users = expected_users + torch.tanh(model.user_similarity(user_similarity @ users)) + model.user_skip(users)
        expected_items = expected_items + torch.tanh(model.item_similarity(item_similarity @ items)) + model.item_skip(items)
        actual_users, actual_items = model.encode(self.graph)
        torch.testing.assert_close(actual_users, expected_users)
        torch.testing.assert_close(actual_items, expected_items)

    def test_mgccf_all_parameters_receive_finite_gradients(self):
        model = MGCCF(3, 4, embedding_dim=4, n_layers=2)
        gradients = torch.autograd.grad(
            model.recommendation_loss(self.batch, self.graph, {}), tuple(model.parameters())
        )
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(all(gradient.norm().item() > 0 for gradient in gradients))

    def test_mgccf_cosine_graph_is_symmetric_and_uses_only_training_edges(self):
        model = MGCCF(3, 4, embedding_dim=4, similarity_degree=1)
        context = model._context(self.graph)
        for graph in context.values():
            dense = graph.to_dense()
            torch.testing.assert_close(dense, dense.T)
            torch.testing.assert_close(dense.diagonal(), torch.zeros_like(dense.diagonal()))
            self.assertLessEqual(graph._nnz(), dense.shape[0])
        changed = torch.zeros_like(self.graph.to_dense())
        changed[0, 3] = changed[3, 0] = 1
        changed[1, 3] = changed[3, 1] = 1
        changed[2, 6] = changed[6, 2] = 1
        changed_context = model._context(changed.to_sparse().coalesce())
        self.assertEqual(changed_context["user_similarity"].to_dense()[0, 1], 1)
        self.assertEqual(changed_context["item_similarity"]._nnz(), 0)

    def test_mgccf_sampled_neighbors_are_shared_and_quota_bounded(self):
        model = MGCCF(3, 4, embedding_dim=4, n_layers=2, neighbor_samples=(1, 2))
        first = model.sample_masks(self.graph, torch.Generator().manual_seed(9))
        repeated = model.sample_masks(self.graph, torch.Generator().manual_seed(9))
        for quota, mask, duplicate in zip((1, 2), first["neighbors"], repeated["neighbors"]):
            torch.testing.assert_close(mask, duplicate)
            degrees = torch.bincount(self.graph.indices()[0, mask], minlength=7)
            self.assertTrue(torch.all(degrees <= quota))
        torch.testing.assert_close(
            model(self.users, self.items, self.graph, first),
            model(self.users, self.items, self.graph, first),
        )

    def test_mgccf_graph_caches_are_not_checkpoint_parameters(self):
        model = MGCCF(3, 4, embedding_dim=4)
        before = set(model.state_dict())
        model(self.users, self.items, self.graph, {})
        self.assertEqual(set(model.state_dict()), before)
        model.eval()
        self.assertEqual(model.sample_masks(self.graph), {})
        restored = MGCCF(3, 4, embedding_dim=4)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            model(self.users, self.items, self.graph, {}),
            restored(self.users, self.items, self.graph, {}),
        )


if __name__ == "__main__":
    unittest.main()
