import unittest
import numpy as np
from q2_matchmaker._stan import (
    _case_control_sim,
    NegativeBinomialCaseControl
)
from biom import Table
from birdman.diagnostics import r2_score
import arviz as az
from dask.distributed import Client, LocalCluster
from scipy.stats import pearsonr


class TestNegativeBinomialCaseControl(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.table, self.metadata, self.diff = _case_control_sim(
            n=100, d=20, depth=1000)

    def test_cc_diagnostics(self):
        biom_table = Table(self.table.values.T,
                           list(self.table.columns),
                           list(self.table.index))

        nb = NegativeBinomialCaseControl(
            table=biom_table,
            matching_column="reps",
            status_column="diff",
            metadata=self.metadata,
            reference_status='1',
            chains=1,
            seed=42)
        nb.compile_model()
        dask_args = {'n_workers': 1, 'threads_per_worker': 1}
        cluster = LocalCluster(**dask_args)
        cluster.scale(dask_args['n_workers'])
        Client(cluster)
        nb.fit_model()
        inf = nb.to_inference_object()
        self.assertEqual(inf['posterior']['mu'].shape, (10, 1, 1000))

        res = r2_score(inf)
        self.assertGreater(res['r2'], 0.3)
        az.loo(inf)
        az.bfmi(inf)
        az.rhat(inf, var_names=nb.param_names)
        az.ess(inf, var_names=nb.param_names)

    def test_cc_fit(self):
        biom_table = Table(self.table.values.T,
                           list(self.table.columns),
                           list(self.table.index))
        nb = NegativeBinomialCaseControl(
            table=biom_table,
            matching_column="reps",
            status_column="diff",
            metadata=self.metadata,
            reference_status='1',
            mu_scale=1,
            sigma_scale=.1,
            disp_scale=0.01,
            # priors specific to the simulation
            control_loc=-6,
            control_scale=3,
            chains=1,
            seed=42)
        nb.compile_model()
        dask_args = {'n_workers': 1, 'threads_per_worker': 1}
        cluster = LocalCluster(**dask_args)
        cluster.scale(dask_args['n_workers'])
        Client(cluster)
        nb.fit_model()
        samples = nb.to_inference_object()
        exp = np.array([0] + list(self.diff))
        res = samples['posterior']['diff'].mean(dim=['chain', 'draw'])
        r, p = pearsonr(res, exp)
        self.assertGreater(r, 0.3)
        self.assertLess(p, 0.05)


if __name__ == '__main__':
    unittest.main()
