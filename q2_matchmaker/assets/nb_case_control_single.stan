 data {
  int<lower=0> C;             // number of controls
  int<lower=0> N;             // number of samples (2 * C)
  int<lower=0> B;             // number of batches
  int<lower=0> D;             // number of diseases (which *doesn't* include controls)
  real depth[N];              // log sequencing depths of microbes
  int y[N];                   // observed microbe abundances
  int cc_ids[N];              // control ids
  int batch_ids[N];           // batch ids
  int disease_ids[N];         // disease ids
  // priors
  real diff_scale;
  real disp_scale;
  real control_loc;
  real control_scale;
  real batch_scale;
}

parameters {
  real diff[D];                 // Difference between each disease and control
  real a0;                      // per microbe constant dispersion
  real a1;                      // per microbe linear dispersion
  
  real<lower=0> disease_disp[D+1];    // per microbe quadratic dispersion for all diseases
  real batch_mu[B];                   // per batch bias
  real<lower=0> batch_disp[B];        // per batch dispersion
  real<offset=control_loc, multiplier=3> control_mu;               // log control proportions (prior mean)
  real control_sigma;                                              // log control proportions (prior std)
  vector<offset=control_mu, multiplier=control_sigma>[C] control;  // log control proportions
  
}

transformed parameters {
  vector[N] lam;
  vector[N] phi;
  vector[N] alpha;

  for (n in 1:N) {
    real delta = 0;
    if (disease_ids[n] > 0) // if not control
        delta = diff[disease_ids[n]]
    
    lam[n] = depth[n] + control[cc_ids[n]] + delta + batch_mu[batch_ids[n]]
  
    phi[n] = inv(exp(a1 - lam[n]) + disease_disp[disease_id[n] + 1] + batch_disp[batch_ids[n]]);
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
