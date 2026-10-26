import contextlib
import io
import unittest

from uger.config import parse_args


class ConfigurationTests(unittest.TestCase):
    def test_mgccf_neighbor_defaults_follow_the_layer_count(self):
        self.assertEqual(parse_args(["--backbone", "MGCCF", "--num-layers", "1"]).mgccf_neighbors, [15])
        self.assertEqual(parse_args(["--backbone", "MGCCF", "--num-layers", "3"]).mgccf_neighbors, [15, 10, 10])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--backbone", "MGCCF", "--num-layers", "1", "--mgccf-neighbors", "15", "10"])

    def test_probe_ratio_preserves_total_budget(self):
        for ratio, clean, probe in ((0, 40, 0), (0.25, 30, 10), (0.5, 20, 20), (0.75, 10, 30), (1, 0, 40)):
            args = parse_args(["--probe-ratio", str(ratio)])
            self.assertEqual((args.clean_quota, args.probe_quota), (clean, probe))
            self.assertEqual(clean + probe + 1, 41)

    def test_paper_controls_map_to_distinct_candidate_and_search_choices(self):
        expected = {
            "STAR-Rec": ("clean", "random", "incoming"),
            "STAR-Rec-CE": ("enhanced", "random", "incoming"),
            "w/o-CE": ("clean", "incoming", "incoming"),
            "random-probe": ("enhanced", "incoming", "random"),
            "KL-probe": ("enhanced", "kl", "initial"),
        }
        for name, choices in expected.items():
            args = parse_args(["--variant", name])
            self.assertEqual((args.candidate_policy, args.perturbation_init, args.candidate_probe), choices)

    def test_framework_ablations_remove_only_the_requested_component(self):
        self.assertEqual(parse_args(["--variant", "w/o-model"]).beta, 0)
        self.assertEqual(parse_args(["--variant", "w/o-structure"]).beta, 1)
        self.assertEqual(parse_args(["--variant", "w/o-stab"]).consistency_weight, 0)

    def test_invalid_budgets_and_nonfinite_settings_are_rejected(self):
        cases = (["--buffer-size", "-1"], ["--probe-ratio", "0.333"],
                 ["--epsilon", "nan"], ["--learning-rate", "inf"],
                 ["--method", "Uniform", "--variant", "w/o-CE"])
        for argv in cases:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
