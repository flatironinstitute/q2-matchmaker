data {
  int<lower=0> C;            // number of controls
  int<lower=0> N;            // number of samples (2 * C)
  real depth[N];             // log sequencing depths of microbes
  int y[N];                  // observed microbe abundances
  int cc_bool[N];            // case-control true/false
  int cc_ids[N];             // control ids
}

parameters {
  vector[C] control;         // Mean of the control samples
  real diff;                 // Difference between case and control
  real mu;                   // mean prior for diff
  real<lower=0.001> sigma;   // variance of batch random effects
  real<lower=0.001> disp[2]; // per microbe dispersion for both case-controls
}

transformed parameters {
  vector[N] lam;
  for (n in 1:N){
    lam[n] = control[cc_ids[n]] + diff * cc_bool[n];
  }
}

model {
  // setting priors ...
  disp ~ inv_gamma(1., 1.);
  sigma ~ inv_gamma(1., 1.);
  mu ~ normal(0, 10);
  diff ~ normal(mu, sigma);
  control ~ normal(0, 10); // vague normal prior for controls

  // generating counts
  for (n in 1:N){
    target += neg_binomial_2_log_lpmf(y[n] | lam[n] + depth[n],
                                      disp[cc_bool[n] + 1]);
  }
}
