import unittest
import numpy as np
from q2_differential._stan import (
    _case_control_sim, _case_control_full,
    _case_control_data, _case_control_single,
    NegativeBinomialCaseControl
)
from biom import Table
from birdman.diagnostics import r2_score
import arviz as az

try:
    from dask_jobqueue import SLURMCluster
    import dask
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
            depth=self.table.sum(axis=1),
            mc_samples=2000)


class TestNegativeBinomialCaseControl(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.table, self.metadata, self.diff = _case_control_sim(
            n=50, d=3, depth=100)

    def test_cc(self):
        biom_table = Table(self.table.values.T,
                           list(self.table.columns),
                           list(self.table.index))

        nb = NegativeBinomialCaseControl(
            table=biom_table,
            matching_column="reps",
            status_column="diff",
            metadata=self.metadata,
            reference_status='1',
            chains=4,
            seed=42)
        nb.compile_model()
        nb.fit_model()
        # inf = nb.to_inference_object()
        # loo = az.loo(inf)
        # bfmi = az.bfmi(inf)
        # rhat = az.rhat(inf, var_names=nb.param_names)
        # ess = az.ess(inf, var_names=nb.param_names)
        # r2 = r2_score(inf)
        # print('loo', loo)
        # print('bfmi', bfmi.mean(), bfmi.std())
        # print('rhat', rhat)
        # print('r2', r2)
        # summary_stats = loo
        # summary_stats.loc['bfmi'] = [bfmi.mean().values, bfmi.std().values]
        # summary_stats.loc['r2'] = r2.values


    @unittest.skipIf(no_dask, 'Dask-jobqueue is not installed')
    def test_cc_slurm(self):

        biom_table = Table(self.table.values.T,
                           list(self.table.columns),
                           list(self.table.index))
        cluster = SLURMCluster(cores=4,
                               processes=4,
                               memory='16GB',
                               walltime='01:00:00',
                               interface='ib0',
                               nanny=True,
                               death_timeout='300s',
                               local_directory='/scratch',
                               shebang='#!/usr/bin/env bash',
                               env_extra=["export TBB_CXX_TYPE=gcc"],
                               queue='ccb')

        nb = NegativeBinomialCaseControl(
            table=biom_table,
            matching_column="reps",
            status_column="diff",
            metadata=self.metadata,
            reference_status='1',
            chains=1,
            seed=42)
        nb.compile_model()
        nb.fit_model(dask_cluster=cluster)
        print(nb.fit)
        print(nb.fit[0])
        # most hacky solution ever
        for n in nb.fit:
            try:
                az.from_cmdstanpy(posterior=n)
            except:
                continue

        inf = nb.to_inference_object(dask_cluster=cluster)
        loo = az.loo(inf)
        bfmi = az.bfmi(inf)
        rhat = az.rhat(inf, var_names=nb.param_names)
        ess = az.ess(inf, var_names=nb.param_names)
        r2 = r2_score(inf)
        print('loo', loo)
        print('bfmi', bfmi)
        print('rhat', rhat)
        print('r2', r2)


if __name__ == '__main__':
    unittest.main()
