 data {
  int<lower=0> C;             // number of controls
  int<lower=0> N;             // number of samples (2 * C)
  int<lower=0> B;             // number of batches
  real depth[N];              // log sequencing depths of microbes
  int y[N];                   // observed microbe abundances
  int cc_bool[N];             // case-control (true if case, false if control)
  int cc_ids[N];              // control ids
  int batch_ids[N];           // batch ids
  // priors
  real diff_scale;
  real disp_scale;
  real control_loc;
  real control_scale;
  real batch_scale;
}

parameters {
  real diff;                   // Difference between case and control
  real<lower=0> disp[2];       // per microbe dispersion for both case-controls
  real batch_mu[B];            // per batch bias
  real<lower=0> batch_disp[B]; // per batch dispersion
  real<offset=control_loc, multiplier=3> control_mu;
  real control_sigma;
  vector<offset=control_mu, multiplier=control_sigma>[C] control;
  real a1;
}

transformed parameters {
  vector[N] lam;
  vector[N] phi;

  for (n in 1:N) {

    lam[n] = depth[n] + batch_mu[batch_ids[n]] + control[cc_ids[n]];
    if (cc_bool[n]) lam[n] += diff;

    phi[n] = inv(exp(a1 - lam[n]) + disp[cc_bool[n] + 1] + batch_disp[batch_ids[n]]);
    //phi[n] = inv(disp[cc_bool[n] + 1]);
  }
}

model {
  // setting priors ...
  a1 ~ normal(1, 1);
  disp ~ lognormal(log(0.1), disp_scale);
  batch_mu ~ normal(0, 3);
  batch_disp ~ lognormal(log(0.1), batch_scale);
  //disp ~ lognormal(log(10), disp_scale);
  diff ~ normal(0, diff_scale);
  // vague normal prior for controls
  control_mu ~ normal(control_loc, 3);
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
