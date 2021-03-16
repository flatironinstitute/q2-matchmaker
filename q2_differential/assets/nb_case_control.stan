functions{
  vector alr_inv_lg(vector x){
    int d = size(x);
    vector[d + 1] y = to_vector(append_col(0., to_row_vector(x)));
    return y;
  }
}


data {
  int<lower=0> C;               // number of controls
  int<lower=0> N;               // number of samples (2 * C)
  int<lower=0> D;               // number of samples (2 * C)
  real depth[N];                // log sequencing depths of microbes
  int y[N, D];                  // observed microbe abundances
  int cc_bool[N];               // case-control true/false
  int cc_ids[N];                // control ids
}

parameters {
  matrix[C, D - 1] control;     // Mean of the control samples
  vector[D - 1] diff;           // Difference between case and control
  vector[D - 1] mu;             // mean prior for diff
  vector<lower=0>[D - 1] sigma; // variance of batch random effects
  matrix<lower=0>[2, D] disp;   // per microbe dispersion for both case-controls
}

model {
  vector[D] lam;
  // setting priors ...
  to_vector(disp) ~ normal(0., 1.);
  sigma ~ normal(0., 1.);
  mu ~ normal(0., 1.);
  diff ~ normal(mu, sigma);
  to_vector(control) ~ normal(0, 10);  // vague normal prior for controls
  // generating counts
  for (n in 1:N){
    lam = alr_inv_lg(to_vector(control[cc_ids[n]]) + diff * cc_bool[n]);
    y[n] ~ neg_binomial_2_log(lam + depth[n],
                              disp[cc_bool[n] + 1]);
  }
}

generated quantities {
  matrix[N, D] y_predict;
  for (n in 1:N){
    vector[D] lam;
    lam = alr_inv_lg(to_vector(control[cc_ids[n]]) + diff * cc_bool[n]);
    for (d in 1:D){
      real m = lam[d] + depth[n];
      real s = disp[cc_bool[n] + 1, d];
      y_predict[n, d] = neg_binomial_2_log_rng(m, s);
    }
  }
}
