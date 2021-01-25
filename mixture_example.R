library(cmdstanr)
library(tidyverse)
library(tidybayes)
library(posterior)
model_code = '
functions{
    real F_cdf(real x, real p, real mu_1, real sigma_1, real mu_2, real sigma_2){
        real mixture_1 = p*normal_cdf(x, mu_1, sigma_1);
        real mixture_2 = (1-p)*normal_cdf(x, mu_2, sigma_2);

        return mixture_1 + mixture_2;
    }
}
data {
  int<lower=1> n;
  vector[n+1] edges; // always one more edge than counts
  int counts[n];

  real prior_mu_mean;
  real prior_mu_sigma;

  real prior_p_a;
  real prior_p_b;
}
parameters {
  ordered[2] mu_k;
  real<lower=0> sigma[2];

  real<lower=0, upper=1> p;
}
transformed parameters {
  simplex[n] theta;

  theta[1] = F_cdf(edges[2], p, mu_k[1], sigma[1], mu_k[2], sigma[2]);
  for (i in 2:(n-1)) {
    theta[i] = F_cdf(edges[i+1], p, mu_k[1], sigma[1], mu_k[2], sigma[2]) - F_cdf(edges[i], p, mu_k[1], sigma[1], mu_k[2], sigma[2]);
  }

  theta[n] = 1 - F_cdf(edges[n], p, mu_k[1], sigma[1], mu_k[2], sigma[2]);
}
model {
    mu_k ~ normal(prior_mu_mean, prior_mu_sigma);
    sigma ~ cauchy(0, 5);


    p ~ beta(prior_p_a, prior_p_b);

    counts ~ multinomial(theta);
}
generated quantities{
    int yppc[size(counts)] = multinomial_rng(theta, sum(counts));
}
'
ndraws = 1000
samples_from_latent_dist_1 = rnorm(1000, -2, 1)
samples_from_latent_dist_2 = rnorm(1000, 2, 2)
p = rbinom(1000, 1, 0.5)
samples_from_latent_dist = p*samples_from_latent_dist_1 + (1-p)*samples_from_latent_dist_2
hist(samples_from_latent_dist)

mixture_bins = tibble(x = samples_from_latent_dist) %>%
    mutate(x2 = cut_width(x, 1)) %>%
    count(x2) %>%
    mutate(
        binned_samples = str_replace_all(x2, "\\[|\\]", "") %>%
            str_replace_all("\\(|\\)", '')
    ) %>%
    separate(binned_samples, c('left', 'right'), sep = ",")

model_data <- list(
    n = length(mixture_bins$n),
    edges = sort(
        as.numeric(unique(c(mixture_bins$left, mixture_bins$right)))
    ),
    counts = mixture_bins$n,
    prior_mu_mean = 0,
    prior_mu_sigma = 1,
    prior_p_a = 1,
    prior_p_b = 1
)


tmpfile = write_stan_file(model_code)
model = cmdstan_model(tmpfile)
f = model$sample(model_data, chains =12)

r = rstan::read_stan_csv(f$output_files())


np = bayesplot::nuts_params(r)
bayesplot::mcmc_parcoord(r, regex_pars = c('mu','sigma'), np = np)
