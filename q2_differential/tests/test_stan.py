import unittest
from q2_differential._stan import (
    _case_control_sim, _case_control_full)


class TestCaseControl(unittest.TestCase):

    def setUp(self):
        self.table, self.metadata = _case_control_sim(n=50, d=5, depth=100)

    def test_case_control_full(self):
        posterior, prior = _case_control_full(
            self.table.values,
            case_ctrl_ids=self.metadata['reps'].values,
            case_member=self.metadata['diff'].values,
            depth=self.table.sum(axis=1),
            mc_samples=200)
        self.assertEqual(res.shape, (200 * 4, 4))

# def test():
#     from q2_differential._stan import (
#         _case_control_sim, _case_control_full)
#
#     table, metadata = _case_control_sim(n=100, d=5, depth=100)
#     posterior, prior = _case_control_full(
#         table.values,
#         case_ctrl_ids=metadata['reps'].values,
#         case_member=metadata['diff'].values,
#         depth=table.sum(axis=1),
#         mc_samples=2000)
#

if __name__ == '__main__':
    unittest.main()
