data {
  int<lower=0> C;            // number of controls
  int<lower=0> N;            // number of samples (2 * C)
  int<lower=0> Z;            // number of nonzero entries
  int y[Z];                  // observed microbe abundances
  int S[Z];                  // sample ids
  int F[Z];                  // feature ids
  int Nc[N];                 // number of censored values per sample
  real U;                    // lower limit
  int cc_bool[N];            // case-control true/false
  int cc_ids[N];             // control ids

  // priors
  real mu_scale;
  real sigma_scale;
  real disp_scale;
  real control_loc;
  real control_scale;
}

parameters {
  vector<upper=0>[C, D] control; // Mean of the control samples
  vector[D] diff;                // Difference between case and control
  vector[D] mu;                  // mean prior for diff
  vector<lower=0>[D] sigma;      // variance of batch random effects
  vector<lower=0>[2, D] disp;    // per microbe dispersion for both case-controls
}

model {
  real lam;

  // setting priors ...
  disp ~ normal(0., disp_scale);
  sigma ~ normal(0., sigma_scale);
  mu ~ normal(0, mu_scale);
  diff ~ normal(mu, sigma);
  // vague normal prior for controls
  control ~ normal(control_loc, control_scale);
  // generating counts
  for (z in 1:Z){
    n = S[z];
    i = F[z];
    lam = control[cc_ids[n]] + diff * cc_bool[n];
    y ~ lognormal(lam, disp[cc_bool[n] + 1]);
    target += (Nc[n] / N) * lognormal_lccdf(U | lam, disp[cc_bool[n] + 1]);
  }
}

generated quantities {
  vector[N, D] y_predict;
  vector[N, D] log_lhood;
  for (n in 1:N){
    vector[D] lam = control[cc_ids[n]] + diff * cc_bool[n];
    vector[D] s = disp[cc_bool[n] + 1];
    y_predict[n,] = lognormal_rng(lam, s);
    log_lhood[n,] = lognormal_lpdf(y[n,] | lam, s);
  }
}
