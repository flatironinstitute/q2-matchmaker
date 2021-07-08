 data {
  int<lower=0> C;                // number of controls
  int<lower=0> N;                // number of samples (2 * C)
  int<lower=0> D;                // number of features
  real<lower=0> y[N, D];         // observed chemical concentrations
  int cc_bool[N];                // case-control true/false
  int cc_ids[N];                 // control ids
  // priors
  real mu_scale;
  real sigma_scale;
  real disp_scale;
  real control_loc;
  real control_scale;
}

parameters {
  matrix<lower=0>[C, D] control;  // Mean of the control samples
  vector[D] diff;                 // Difference between case and control
  vector[D] mu;                   // mean prior for diff
  vector<lower=0>[D] sigma;       // variance of batch random effects
  matrix<lower=0>[2, D] disp;     // per microbe dispersion for both case-controls
}

transformed parameters{
  // not super memory efficent, but we are only looking at
  // small datasets here.
  matrix<lower=0>[N, D] lam;
  for (n in 1:N){
    lam[n] = control[cc_ids[n]] + to_row_vector(diff * cc_bool[n]);
  }
}

model {
  // setting priors ...
  to_vector(disp) ~ normal(0., disp_scale);
  sigma ~ normal(0., sigma_scale);
  mu ~ normal(0., mu_scale);
  diff ~ normal(mu, sigma);
  to_vector(control) ~ normal(control_loc, control_scale);
  // generating concentrations
  for (n in 1:N){
    y[n] ~ normal(lam[n], disp[cc_bool[n] + 1]);
  }
}

generated quantities {
  matrix[N, D] y_predict;
  vector[N] log_lhood;
  for (n in 1:N){
    real ll = 0;
    for (d in 1:D){
      real m = lam[n, d];
      real s = disp[cc_bool[n] + 1, d];
      y_predict[n, d] = normal_rng(m, s);
      ll += normal_lpdf(y[n, d] | m, s);
    }
    log_lhood[n] = ll;
  }
}
