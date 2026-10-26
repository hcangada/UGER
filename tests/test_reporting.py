import unittest

from uger.reporting import holm_adjust


class ReportingTests(unittest.TestCase):
    def test_holm_uses_monotone_familywise_adjustment(self):
        actual = holm_adjust([0.01, 0.04, 0.03])
        for left, right in zip(actual, [0.03, 0.06, 0.06]):
            self.assertAlmostEqual(left, right)

    def test_invalid_pvalues_are_rejected(self):
        for values in ([float("nan")], [-0.1], [1.1]):
            with self.assertRaises(ValueError):
                holm_adjust(values)


if __name__ == "__main__":
    unittest.main()
