import unittest
from q2_differential._stan import _case_control_sim, _case_control_func


class TestBatch(unittest.TestCase):

    def setUp(self):
        self.table, self.metadata = _case_control_sim(n=100, d=5, depth=100)

    def test_batch(self):
        res = _case_control_func(self.table.values[:, 0],
                                 case_ctrl_ids=self.metadata['reps'].values,
                                 case_member=self.metadata['diff'].values,
                                 depth=self.table.sum(axis=1),
                                 mc_samples=2000)
        self.assertEqual(res.shape, (2000 * 4, 4))


if __name__ == '__main__':
    unittest.main()
