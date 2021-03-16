import unittest
from q2_differential._stan import (
    _case_control_sim, _case_control_full)


class TestCaseControl(unittest.TestCase):

    def setUp(self):
        self.table, self.metadata, self.diff = _case_control_sim(
            n=50, d=3, depth=100)

    def test_case_control_full(self):
        posterior, prior = _case_control_full(
            self.table.values,
            case_ctrl_ids=self.metadata['reps'].values,
            case_member=self.metadata['diff'].values,
            depth=self.table.sum(axis=1),
            reference='0',
            mc_samples=500)
        res_diff = posterior.stan_variable('diff')
        print(self.diff)
        print(res_diff.mean(0))

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
