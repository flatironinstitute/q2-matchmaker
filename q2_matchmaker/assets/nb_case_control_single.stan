 data {
  int<lower=0> C;             // number of controls
  int<lower=0> N;             // number of samples (2 * C)
  real depth[N];              // log sequencing depths of microbes
  int y[N];                   // observed microbe abundances
  int cc_bool[N];             // case-control true/false
  int cc_ids[N];              // control ids
  // priors
  real diff_scale;
  real disp_scale;
  real control_loc;
  real control_scale;
}

parameters {
  vector[C] control;          // Mean of the control samples
  real diff;                  // Difference between case and control
  real mu;                    // mean prior for diff
  real<lower=0> disp[2];      // per microbe dispersion for both case-controls
  real control_mu;
  real control_sigma;
}

transformed parameters {
  vector[C] logit_control;    // Mean of the control samples
  vector[N] lam;
  vector[N] phi;
  logit_control = log(inv_logit(control));
  for (n in 1:N) {
    lam[n] = logit_control[cc_ids[n]];
    if (cc_bool[n]) lam[n] += diff;
    phi[n] = 1 ./ disp[cc_bool[n] + 1];
  }
}

model {
  // setting priors ...
  disp ~ lognormal(0., disp_scale);
  diff ~ normal(0, diff_scale);
  // vague normal prior for controls
  control_mu ~ normal(control_loc, 5);
  control_sigma ~ lognormal(0, control_scale);
  control ~ normal(control_mu, control_sigma);
  // generating counts
  y ~ neg_binomial_2_log(lam, phi);
}

generated quantities {
  vector[N] y_predict;
  vector[N] log_lhood;
  for (n in 1:N){
    y_predict[n] = neg_binomial_2_log_rng(lam[n], phi[n]);
    log_lhood[n] = neg_binomial_2_log_lpmf(y[n] | lam[n], phi[n]);
  }
}
