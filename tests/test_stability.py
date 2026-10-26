import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as functional
from torch import nn

from uger.stability import UGERStability, build_candidates, build_ranked_candidates, global_norm, project_global


class ScalarModel(nn.Module):
    def __init__(self, value=0.7):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float64))

    def forward(self, users, items, graph, masks=None):
        scale = 1.0 if masks is None else masks
        return self.weight * items.to(self.weight.dtype) * scale

    def recommendation_loss(self, batch, graph, masks=None):
        scores = self(batch["users"], batch["items"], graph, masks=masks)
        return functional.softplus(scores[:, 1:] - scores[:, :1]).mean()


class ZeroGradientModel(ScalarModel):
    def recommendation_loss(self, batch, graph, masks=None):
        return self.weight * 0.0


class VectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Parameter(torch.tensor([0.7, -0.09], dtype=torch.float64))
        self.offset = nn.Parameter(torch.tensor([0.12], dtype=torch.float64))

    def forward(self, users, items, graph, masks=None):
        values = items.to(self.linear.dtype)
        return self.linear[0] * values + self.linear[1] * values.square() + self.offset * values.sin()

    def recommendation_loss(self, batch, graph, masks=None):
        scores = self(batch["users"], batch["items"], graph, masks=masks)
        return functional.softplus(scores[:, 1:] - scores[:, :1]).mean()


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.incoming = {"users": torch.tensor([0]), "items": torch.tensor([[0, 1]])}
        self.replay = {"users": torch.tensor([0]), "items": torch.tensor([[0, 1]])}
        self.empty = {"users": torch.empty(0, dtype=torch.long), "items": torch.empty((0, 2), dtype=torch.long)}

    def generator(self, seed=17):
        return torch.Generator().manual_seed(seed)

    def test_global_projection_bounds_joint_vector_without_mutation(self):
        original = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([4.0])}
        projected = project_global(original, 2.0)
        self.assertAlmostEqual(global_norm(projected).item(), 2.0, places=6)
        torch.testing.assert_close(projected["a"], torch.tensor([1.2, 0.0]))
        torch.testing.assert_close(projected["b"], torch.tensor([1.6]))
        torch.testing.assert_close(original["a"], torch.tensor([3.0, 0.0]))
        torch.testing.assert_close(original["b"], torch.tensor([4.0]))
        inside = project_global(projected, 3.0)
        torch.testing.assert_close(inside["a"], projected["a"])

    def test_ties_anchor_overlap_and_single_filler(self):
        items = torch.tensor([4, 2, 1, 3, 0])
        tied = torch.ones(5)
        selected = build_candidates(items, tied, tied, 0, 1, 1)
        torch.testing.assert_close(selected, torch.tensor([0, 1, 2]))
        selected = build_candidates(items, tied, tied, 3, 1, 1)
        torch.testing.assert_close(selected, torch.tensor([0, 1, 3]))
        items = torch.arange(6)
        clean = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        probe = torch.tensor([0.0, 1.0, 9.0, 8.0, 2.0, 3.0])
        selected = build_candidates(items, clean, probe, 0, 1, 1)
        torch.testing.assert_close(selected, torch.tensor([0, 2, 3]))

    def test_short_pool_and_zero_quotas_keep_anchor(self):
        items = torch.tensor([9, 1])
        scores = torch.tensor([0.1, 0.9])
        selected = build_candidates(items, scores, scores, 9, 2, 2)
        torch.testing.assert_close(selected, torch.tensor([1, 9]))
        selected = build_candidates(items, scores, scores, 9, 0, 0)
        torch.testing.assert_close(selected, torch.tensor([9]))

    def test_clean_control_is_not_the_zero_probe_quota_endpoint(self):
        items = torch.arange(6)
        clean = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        probe = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 9.0])
        clean_control = build_ranked_candidates(items, clean, 0, 3)
        zero_probe_quota = build_candidates(items, clean, probe, 0, 2, 0)
        torch.testing.assert_close(clean_control, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(zero_probe_quota, torch.tensor([0, 1, 5]))

    def test_quota_condition_covers_clean_and_probe_leaders(self):
        generator = self.generator(101)
        for clean_quota in (1, 2, 3):
            for probe_quota in (clean_quota, clean_quota + 1):
                for _ in range(12):
                    items = torch.randperm(20, generator=generator)
                    clean = torch.randint(0, 5, (20,), generator=generator).double()
                    probe = torch.randint(0, 5, (20,), generator=generator).double()
                    selected = build_candidates(items, clean, probe, 19, clean_quota, probe_quota)
                    clean_top = sorted(zip(items.tolist(), clean.tolist()), key=lambda row: (-row[1], row[0]))[:clean_quota]
                    probe_top = sorted(zip(items.tolist(), probe.tolist()), key=lambda row: (-row[1], row[0]))[:clean_quota]
                    selected_set = set(selected.tolist())
                    self.assertEqual(len(selected_set), clean_quota + probe_quota + 1)
                    self.assertIn(19, selected_set)
                    self.assertTrue({row[0] for row in clean_top}.issubset(selected_set))
                    self.assertTrue({row[0] for row in probe_top}.issubset(selected_set))

    def test_random_initialization_controls_share_their_seed_and_proposals(self):
        model = VectorModel()
        results = []
        for policy in ("clean", "enhanced"):
            module = UGERStability(epsilon=0.1, proposal_size=7, clean_quota=1, probe_quota=1, initialization="random", candidate_policy=policy)
            loss, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0}}, range(12), self.generator(67))
            self.assertTrue(torch.isfinite(loss))
            self.assertAlmostEqual(global_norm(diagnostics["initial_perturbation"]).item(), 0.001, places=12)
            self.assertEqual(diagnostics["candidate_mask"].sum().item(), 3)
            results.append(diagnostics)
        torch.testing.assert_close(results[0]["proposal_items"], results[1]["proposal_items"])
        for name in results[0]["initial_perturbation"]:
            torch.testing.assert_close(results[0]["initial_perturbation"][name], results[1]["initial_perturbation"][name])
        self.assertFalse(results[0]["incoming_probe_computed"])
        self.assertTrue(results[1]["incoming_probe_computed"])

    def test_random_candidate_probe_changes_only_mining_direction(self):
        model = VectorModel()
        results = []
        for direction in ("incoming", "random"):
            module = UGERStability(epsilon=0.1, proposal_size=7, clean_quota=1, probe_quota=1, candidate_probe=direction)
            _, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0}}, range(12), self.generator(93))
            results.append(diagnostics)
            self.assertAlmostEqual(global_norm(diagnostics["candidate_perturbation"]).item(), global_norm(diagnostics["probe_perturbation"]).item(), places=12)
        torch.testing.assert_close(results[0]["proposal_items"], results[1]["proposal_items"])
        for name in results[0]["initial_perturbation"]:
            torch.testing.assert_close(results[0]["initial_perturbation"][name], results[1]["initial_perturbation"][name])
        cosine_numerator = sum((results[1]["candidate_perturbation"][name] * results[1]["probe_perturbation"][name]).sum() for name in results[1]["candidate_perturbation"])
        self.assertLess(abs(cosine_numerator.item()), global_norm(results[1]["candidate_perturbation"]).item() ** 2 * 0.99)

    def test_kl_probe_starts_on_proposal_support_before_candidate_refinement(self):
        model = VectorModel()
        module = UGERStability(epsilon=0.1, proposal_size=7, clean_quota=1, probe_quota=1, initialization="kl", candidate_probe="initial")
        widths = []
        original = module._kl_gradient

        def capture(*args, **kwargs):
            widths.append(args[4].shape[1])
            return original(*args, **kwargs)

        with patch.object(module, "_kl_gradient", side_effect=capture):
            loss, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0}}, range(12), self.generator(43))
        self.assertEqual(widths, [7, 3])
        self.assertFalse(diagnostics["incoming_probe_computed"])
        self.assertGreater(global_norm(diagnostics["initial_perturbation"]).item(), 0)
        self.assertLessEqual(global_norm(diagnostics["initial_perturbation"]).item(), 0.1)
        for name in diagnostics["initial_perturbation"]:
            torch.testing.assert_close(diagnostics["candidate_perturbation"][name], diagnostics["initial_perturbation"][name])
        self.assertTrue(torch.isfinite(loss))

    def test_no_refinement_uses_initial_probe_and_joint_radius_is_global(self):
        model = VectorModel()
        module = UGERStability(epsilon=0.1, proposal_size=7, clean_quota=1, probe_quota=1, refine=False)
        _, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0}}, range(12), self.generator(43))
        self.assertEqual(diagnostics["selected_index"], 1)
        self.assertEqual(len(diagnostics["trial_kl"]), 2)
        self.assertAlmostEqual(global_norm(diagnostics["chosen_perturbation"]).item(), 0.1, places=10)
        for name in diagnostics["chosen_perturbation"]:
            torch.testing.assert_close(diagnostics["chosen_perturbation"][name], diagnostics["probe_perturbation"][name])
            self.assertFalse(diagnostics["chosen_perturbation"][name].requires_grad)
        self.assertLess(global_norm({"linear": diagnostics["chosen_perturbation"]["linear"]}).item(), 0.1)
        self.assertLess(global_norm({"offset": diagnostics["chosen_perturbation"]["offset"]}).item(), 0.1)

    def test_best_of_three_includes_probe_and_does_not_change_clean_parameters(self):
        model = VectorModel()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        module = UGERStability(epsilon=0.1, proposal_size=7, clean_quota=1, probe_quota=1)
        loss, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0}}, range(12), self.generator(43))
        trials = diagnostics["trial_kl"]
        self.assertEqual(len(trials), 3)
        self.assertEqual(diagnostics["selected_index"], int(trials.argmax()))
        self.assertGreaterEqual(diagnostics["selected_kl"], trials[1].item())
        self.assertLessEqual(diagnostics["perturbation_norm"], 0.1 + 1e-12)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name])
            self.assertIsNone(parameter.grad)
        actual = torch.autograd.grad(loss, list(model.parameters()))
        chosen = diagnostics["chosen_perturbation"]
        items = diagnostics["candidate_items"]
        scores = (model.linear[0] + chosen["linear"][0]) * items + (model.linear[1] + chosen["linear"][1]) * items.square() + (model.offset + chosen["offset"]) * items.double().sin()
        expected_loss = functional.kl_div(functional.log_softmax(scores.double(), dim=-1), diagnostics["clean_reference"], reduction="none").sum(-1).mean()
        expected = torch.autograd.grad(expected_loss, list(model.parameters()))
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, rtol=1e-9, atol=1e-12)

    def test_all_controls_keep_short_pools_and_cache_restricted_references(self):
        model = VectorModel()
        for policy, initialization, direction in (("clean", "random", "incoming"), ("enhanced", "random", "incoming"), ("clean", "incoming", "incoming"), ("enhanced", "incoming", "random"), ("enhanced", "incoming", "incoming"), ("probe", "incoming", "incoming"), ("enhanced", "kl", "initial")):
            with self.subTest(policy=policy, initialization=initialization, direction=direction):
                module = UGERStability(epsilon=0.1, proposal_size=7, clean_quota=1, probe_quota=1, candidate_policy=policy, initialization=initialization, candidate_probe=direction)
                loss, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0, 1}}, [0, 1, 2], self.generator(56))
                selected = diagnostics["candidate_items"][0, diagnostics["candidate_mask"][0]]
                torch.testing.assert_close(selected, torch.tensor([0, 2]))
                reference = functional.softmax(model(self.replay["users"], selected[None], None).detach(), dim=-1)
                torch.testing.assert_close(reference, diagnostics["clean_reference"])
                self.assertTrue(torch.isfinite(loss))

    def test_explicit_proposal_pools_support_matched_controls(self):
        module = UGERStability(epsilon=0.1, proposal_size=7, clean_quota=1, probe_quota=1)
        supplied = [torch.tensor([0, 5, 2, 9])]
        _, diagnostics = module.compute(VectorModel(), self.incoming, self.replay, None, {0: {0, 1}}, range(12), self.generator(3), proposal_pools=supplied)
        torch.testing.assert_close(diagnostics["proposal_items"][0], supplied[0])
        with self.assertRaises(ValueError):
            module.compute(VectorModel(), self.incoming, self.replay, None, {0: {0, 1}}, range(12), self.generator(3), proposal_pools=[torch.tensor([0, 1, 2])])

    def test_outer_gradient_matches_frozen_target_analytic_gradient(self):
        model = ScalarModel()
        module = UGERStability(epsilon=0.1, proposal_size=3, clean_quota=1, probe_quota=1, refine=False)
        before = model.weight.detach().clone()
        loss, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0}}, [0, 1, 2], self.generator())
        self.assertEqual(diagnostics["selected_index"], 1)
        self.assertIsNone(model.weight.grad)
        torch.testing.assert_close(model.weight.detach(), before)
        self.assertFalse(diagnostics["clean_reference"].requires_grad)
        items = diagnostics["candidate_items"][0].to(torch.float64)
        target = diagnostics["clean_reference"][0]
        incoming_gradient = functional.sigmoid(before)
        shift = -module.epsilon * incoming_gradient / (incoming_gradient + module.nu)
        student = functional.softmax((before + shift) * items, dim=0)
        expected = ((student - target) * items).sum()
        gradient = torch.autograd.grad(loss, model.weight)[0]
        torch.testing.assert_close(gradient, expected, rtol=1e-10, atol=1e-12)
        self.assertGreater(abs(gradient.item()), 1e-4)

    def test_zero_gradient_fallback_is_seeded_and_feasible(self):
        model = ZeroGradientModel()
        module = UGERStability(epsilon=0.1, proposal_size=3, clean_quota=1, probe_quota=1)
        result = []
        for _ in range(2):
            loss, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0}}, [0, 1, 2], self.generator(8))
            self.assertTrue(diagnostics["fallback_used"])
            self.assertEqual(diagnostics["incoming_gradient_norm"], 0.0)
            self.assertAlmostEqual(diagnostics["probe_norm"], 0.001, places=12)
            self.assertLessEqual(diagnostics["perturbation_norm"], 0.1 + 1e-12)
            self.assertTrue(torch.isfinite(loss))
            result.append((loss.detach(), diagnostics["trial_kl"]))
        torch.testing.assert_close(result[0][0], result[1][0])
        torch.testing.assert_close(result[0][1], result[1][1])

    def test_empty_incoming_uses_fallback_without_nan(self):
        module = UGERStability(epsilon=0.1, proposal_size=3, clean_quota=1, probe_quota=1)
        loss, diagnostics = module.compute(ScalarModel(), self.empty, self.replay, None, {0: {0}}, [0, 1, 2], self.generator())
        self.assertTrue(diagnostics["fallback_used"])
        self.assertTrue(torch.isfinite(loss))

    def test_singleton_padding_contributes_zero_and_mean_uses_all_replay_rows(self):
        model = ScalarModel()
        module = UGERStability(epsilon=0.1, proposal_size=4, clean_quota=1, probe_quota=1, refine=False)
        replay = {"users": torch.tensor([0, 1]), "items": torch.tensor([[0, 1], [1, 0]])}
        incoming = {"users": torch.tensor([1]), "items": torch.tensor([[1, 2]])}
        history = {0: {0, 1, 2, 3}, 1: {1}}
        both, diagnostics = module.compute(model, incoming, replay, None, history, [0, 1, 2, 3], self.generator())
        one_replay = {key: value[1:] for key, value in replay.items()}
        one, _ = module.compute(model, incoming, one_replay, None, history, [0, 1, 2, 3], self.generator())
        self.assertEqual(diagnostics["candidate_mask"][0].sum().item(), 1)
        self.assertEqual(diagnostics["clean_reference"][0, 0].item(), 1.0)
        self.assertTrue(torch.isfinite(both))
        torch.testing.assert_close(both, one / 2.0, rtol=1e-9, atol=1e-12)

    def test_empty_replay_is_differentiable_zero(self):
        model = ScalarModel()
        module = UGERStability(proposal_size=3, clean_quota=1, probe_quota=1)
        loss, diagnostics = module.compute(model, self.incoming, self.empty, None, {}, [], self.generator())
        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(diagnostics["fallback_used"])
        self.assertEqual(diagnostics["candidate_items"].shape, (0, 0))
        loss.backward()
        self.assertEqual(model.weight.grad.item(), 0.0)

    def test_proposals_exclude_observed_nonanchors_and_duplicates(self):
        model = ScalarModel()
        module = UGERStability(proposal_size=5, clean_quota=2, probe_quota=2)
        loss, diagnostics = module.compute(model, self.incoming, self.replay, None, {0: {0, 1, 3}}, list(range(9)), self.generator())
        items = diagnostics["candidate_items"][0, diagnostics["candidate_mask"][0]].tolist()
        self.assertIn(0, items)
        self.assertNotIn(1, items)
        self.assertNotIn(3, items)
        self.assertEqual(len(items), len(set(items)))
        self.assertTrue(torch.isfinite(loss))

    def test_graph_dropout_masks_are_sampled_once_and_shared(self):
        from uger.models import LightGCN

        model = LightGCN(2, 4, embedding_dim=3, n_layers=1, keep_prob=0.5)
        indices = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]])
        graph = torch.sparse_coo_tensor(indices, torch.ones(4), (6, 6)).coalesce()
        module = UGERStability(epsilon=0.1, proposal_size=4, clean_quota=1, probe_quota=1)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        with patch.object(model, "sample_masks", wraps=model.sample_masks) as sample_masks:
            loss, diagnostics = module.compute(model, self.incoming, self.replay, graph, {0: {0}}, [0, 1, 2, 3], self.generator())
        self.assertEqual(sample_masks.call_count, 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertLessEqual(diagnostics["perturbation_norm"], 0.1 + 1e-6)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.detach(), before[name])
        loss.backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_graph_backbones_small_global_radius_have_finite_consistency_gradients(self):
        from uger.models import LightGCN, NGCF

        indices = torch.tensor([[0, 2, 0, 3, 1, 3, 1, 4], [2, 0, 3, 0, 3, 1, 4, 1]])
        graph = torch.sparse_coo_tensor(indices, torch.ones(8) / 2, (6, 6)).coalesce()
        incoming = {"users": torch.tensor([0, 1]), "items": torch.tensor([[1, 2], [2, 3]])}
        replay = {"users": torch.tensor([0, 1]), "items": torch.tensor([[0, 3], [1, 0]])}
        for model_type in (LightGCN, NGCF):
            for radius in (0.001, 0.01):
                with self.subTest(backbone=model_type.__name__, epsilon=radius):
                    torch.manual_seed(53)
                    model = model_type(2, 4, embedding_dim=8, n_layers=2, keep_prob=0.75)
                    module = UGERStability(epsilon=radius, proposal_size=4, clean_quota=1, probe_quota=1)
                    generator = self.generator(51)
                    masks = model.sample_masks(graph, generator)
                    loss, diagnostics = module.compute(model, incoming, replay, graph, {0: {0, 1}, 1: {1, 2}}, [0, 1, 2, 3], generator, masks=masks)
                    self.assertEqual(loss.dtype, torch.float64)
                    self.assertEqual(diagnostics["clean_reference"].dtype, torch.float64)
                    self.assertGreaterEqual(loss.item(), 0.0)
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(diagnostics["trial_kl"]).all())
                    self.assertGreaterEqual(diagnostics["trial_kl"].min().item(), 0.0)
                    self.assertLessEqual(diagnostics["perturbation_norm"], radius * (1 + 1e-6))
                    loss.backward()
                    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
