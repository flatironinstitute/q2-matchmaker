from q2_differential._stats import hotelling_ttest, convex_hull_test
import numpy as np
import unittest


class TestStats(unittest.TestCase):
    def setUp(self):
        n, p = 100, 5
        self.Xrand = np.random.randn(n, p)
        self.Xreal = np.random.randn(n, p)
        self.Xreal[:, 0] += 10

    def test_hotelling_ttest(self):
        _, pval = hotelling_ttest(self.Xrand)
        self.assertGreater(pval, 0.01)
        _, pval = hotelling_ttest(self.Xreal)
        self.assertLess(pval, 0.01)

    def test_convex_hull_test(self):
        ans = convex_hull_test(self.Xrand)
        self.assertTrue(ans)
        ans = convex_hull_test(self.Xreal)
        self.assertFalse(ans)


if __name__ == '__main__':
    unittest.main()
