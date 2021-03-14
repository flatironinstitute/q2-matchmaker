
functions {
  vector nbcc(vector beta, vector theta, real[] x, int[] y){
    int D = dims(y)[2];
    int N = dims(x)[1];
    int C = dims(beta)[1] / D - 3;
    // parameters
    matrix[C, D] control;
    for (c in 1:C){
      control[c] = to_row_vector(beta[(c - 1) * D : c * D]);
    }
    vector[D] diff = beta[C*D: (C+1)*D];
    vector[2] disp[D];
    disp[1] = beta[(C+1)*D: (C+2)*D];
    disp[2] = beta[(C+2)*D: (C+3)*D];
    // data
    real depth = x[1];
    int cc_bool = y[1];
    int cc_id = y[2];
    // get log-likelihood
    vector[D] lam = to_vector(control[cc_id]) + diff * cc_bool;
    real lp = neg_binomial_2_log_lpmf(y[2:] | lam + depth, disp[cc_bool + 1]);
    return [lp]';
  }
}

data {
  int<lower=0> C;            // number of controls
  int<lower=0> N;            // number of samples (2 * C)
  int<lower=0> D;            // number of microbes
  real depth[N];             // log sequencing depths of microbes
  int y[N, D];               // observed microbe abundances
  int cc_bool[N];            // case-control true/false
  int cc_ids[N];             // control ids
}

transformed data{
  int ys[N, 3 + D];
  real xs[N, 1];
  for (n in 1:N){
    ys[n, 1] = cc_bool[n];
    ys[n, 2] = cc_ids[n];
    ys[n, 2:2+D] = y[n];
    xs[n] = {depth[n]};
  }

  vector[0] theta[N];

}

parameters {
  matrix[C, D] control;        // Mean of the control samples
  vector[D] diff;              // Difference between case and control
  vector[D] mu;                // mean prior for diff
  vector<lower=0>[D] sigma;    // variance of batch random effects
  matrix<lower=0>[2, D] disp;  // per microbe dispersion for both case-controls
}

transformed parameters{
  vector[(C + 3) * D] beta;
  for (c in 1:C){
    beta[(c - 1) * D : c * D] = to_vector(control[c]);
  }
  beta[C*D: (C+1)*D] = diff;
  beta[(C+1)*D: (C+2)*D] = to_vector(disp[1]);
  beta[(C+2)*D: (C+3)*D] = to_vector(disp[2]);
}
model {
  // setting priors ...
  to_vector(disp) ~ inv_gamma(1., 1.);
  sigma ~ inv_gamma(1., 1.);
  mu ~ normal(0, 10);
  diff ~ normal(mu, sigma);
  to_vector(control) ~ normal(0, 10); // vague normal prior for controls
  // generating counts
  target += sum(map_rect(nbcc, beta, theta, xs, ys));
}
