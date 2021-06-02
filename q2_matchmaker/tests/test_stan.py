import unittest
import numpy as np
from q2_matchmaker._stan import (
    _case_control_sim, _case_control_full,
    _case_control_data, _case_control_single,
)
from biom import Table
from birdman.diagnostics import r2_score
import arviz as az

try:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    import dask
    import dask.array as da
    no_dask = False
except:
    no_dask = True


class TestCaseControl(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.table, self.metadata, self.diff = _case_control_sim(
            n=50, d=4, depth=100)

    def test_case_control_full(self):
        # fit once
        sm, posterior, prior = _case_control_full(
            self.table.values,
            case_ctrl_ids=self.metadata['reps'].values,
            case_member=self.metadata['diff'].values,
            depth=self.table.sum(axis=1),
            mc_samples=100)
        dat = _case_control_data(self.table.values,
                                 case_ctrl_ids=self.metadata['reps'].values,
                                 case_member=self.metadata['diff'].values,
                                 depth=self.table.sum(axis=1),
                                 reference='0')
        gen = sm.generate_quantities(
            data=dat, mcmc_sample=posterior)
        gen_table = gen.generated_quantities[0].reshape((50, 4)) + 1
        # refit to see if the parameters can be recovered
        # from the generated data
        _, re_posterior, re_prior = _case_control_full(
            gen_table,
            case_ctrl_ids=self.metadata['reps'].values,
            case_member=self.metadata['diff'].values,
            depth=self.table.sum(axis=1),
            reference='0',
            mc_samples=1000)

        # TODO: test with random initialization
        res_diff = re_posterior.stan_variable('diff')
        exp_diff = posterior.stan_variable('diff').mean(0)
        rm = res_diff.mean(0)
        rs = res_diff.std(0)
        for i in range(len(self.diff)):
            self.assertTrue(
                (rm[i] - 3 * rs[i]) <= exp_diff[i] and
                (exp_diff[i] <= (rm[i] + 3 * rs[i]))
            )


class TestCaseControlSingle(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.table, self.metadata, self.diff = _case_control_sim(
            n=50, d=4, depth=100)

    def test_cc(self):
        res = _case_control_single(
            self.table.values[:, 0],
            case_ctrl_ids=self.metadata['reps'].values,
            case_member=self.metadata['diff'].values,
            test_counts=self.table.values[:, 0],
            test_case_ctrl_ids=self.metadata['reps'].values,
            test_case_member=self.metadata['diff'].values,
            depth=self.table.sum(axis=1),
            mc_samples=2000)
        print(res[0])
        print(res[1])



if __name__ == '__main__':
    unittest.main()
