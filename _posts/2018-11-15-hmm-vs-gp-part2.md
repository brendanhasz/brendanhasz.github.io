---
layout: post
title: "Multilevel Gaussian Processes and Hidden Markov Models with Stan"
date: 2018-11-15
description: "Multilevel and multitrial Gaussian Processes and hidden Markov models in R, using Stan and bridge sampling."
github_url: https://github.com/brendanhasz/hmm-vs-gp
img_url: /assets/img/hmm-vs-gp-part2/unnamed-chunk-5-1.svg
tags: [bayesian, stan]
language: [r]
comments: true
---

In [a previous post](/2018/10/10/hmm-vs-gp.html),
we built Bayesian models of Gaussian processes and hidden Markov models
in R using [Stan](http://mc-stan.org/).
However, there were a few things we left undone!
First, the Stan models in that post assumed a single
process evolving over a short period of time. However, if we have data
over repeated trials, we have to alter the
models to sum the log probability over all trials. Also, with real
data we may have multiple subjects or groups, which means we have to build a
multilevel model which allows the parameters to vary slightly between
subjects or groups.

**Outline**

-   [Setup](#setup)
-   [Repeated Measures](#repeated-measures)
-   [Multilevel Models](#multilevel-models)
    -   [Gaussian Process](#gaussian-process)
    -   [Hidden Markov Model](#hidden-markov-model)
    -   [Non-centered parameterizations](#non-centered-parameterizations)
    -   [Validation](#validation)
-   [Original Computing Environment](#original-computing-environment)

## Setup

Let's set up our computing environment:

``` r
# Packages
library(rstan)
library(ggplot2)
library(bayesplot)
library(invgamma)
library(bridgesampling)
library(RColorBrewer)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
seed = 1234567

# Colors
c_light <- c("#DCBCBC")
c_mid <- c("#B97C7C")
c_dark <- c("#8F2727")
c_blue_light <- c("#b2c4df")
c_blue_mid <- c("#6689bf")
c_blue_dark <- c("#3d5272")
color_scheme_set("red")
```

## Repeated Measures

The Stan models we've created so far are only able to handle a single
time series. But we have multiple time series: one per trial. To handle multiple trials, we
have to create modified Stan models which accumulate the contribution of
each trial by summing the log likelihoods of each trial. Here's Stan
code for a Gaussian process which can handle multiple trials:

``` r
writeLines(readLines("lgp_multitrial.stan"))
```

``` stan
data {
  int<lower=1> N;  //number of datapoints per trial
  int<lower=1> Nt; //number of trials
  real x[N];       //x values (assume same for each trial)
  row_vector<lower=0, upper=1>[N] y[Nt]; //y values
}
 
transformed data {
  vector[N] mu;      //mean function (vector of 0s)
  real sum_ln_scale; //sum of scales for logit normal dists
  mu = rep_vector(0, N);
  sum_ln_scale = 0;
  for (i in 1:Nt) //pre-compute contribution of logit normal scales
    sum_ln_scale += -sum(log(y[i])+log(1-y[i]));
}
 
parameters {
  real<lower=0> rho;   //length scale
  real<lower=0> alpha; //marginal/output/signal standard deviation
  real<lower=0> sigma; //noise standard deviation
}
 
model {
  // Covariance matrix (assume x same for each trial)
  matrix[N, N] K = cov_exp_quad(x, alpha, rho) + 
                   diag_matrix(rep_vector(square(sigma), N));
 
  // Priors
  target += inv_gamma_lpdf(rho | 2, 0.5);
  target += normal_lpdf(alpha | 0, 2) + log(2); //half-normal dists
  target += normal_lpdf(sigma | 0, 1) + log(2); //mult density by 2
   
  // Accumulate evidence over trials
  for (i in 1:Nt)
    target += multi_normal_lpdf(logit(y[i]) | mu, K);
     
  // Add scales such that likelihood integrates to 1 over y
  target += sum_ln_scale;
   
}
```

And Stan code for a hidden Markov model which can handle multiple
trials:

``` r
writeLines(readLines("hmm_multitrial.stan"))
```

``` stan
data {
  int<lower=1> N; //number of observations per trial
  int<lower=1> Nt; //number of trials
  real<lower=0, upper=1> y[Nt, N]; //observations
}
 
parameters {
  simplex[2] phi[2];      //transition probabilities
  real<lower=1> theta[2]; //observation distribution params
}
 
model {
 
  // Priors
  target += beta_lpdf(phi[1,1] | 1.2, 1.2);
  target += beta_lpdf(phi[2,2] | 1.2, 1.2);
  target += gamma_lpdf(theta[1]-1 | 2, 2);
  target += gamma_lpdf(theta[2]-1 | 2, 2);
 
  // Compute the marginal probability over possible sequences
  {
    real acc[2];
    real gamma[N, 2];
    for (i in 1:Nt) { // accumulate evidence over trials
      gamma[1,1] = beta_lpdf(y[i,1] | 1, theta[1]);
      gamma[1,2] = beta_lpdf(y[i,2] | theta[2], 1);
      for (t in 2:N) {
        for (k in 1:2) {
          acc[1] = gamma[t-1, 1] + log(phi[1,k]);
          acc[2] = gamma[t-1, 2] + log(phi[2,k]);
          gamma[t,k] = log_sum_exp(acc);
        }
        gamma[t,1] += beta_lpdf(y[i,t] | 1, theta[1]);
        gamma[t,2] += beta_lpdf(y[i,t] | theta[2], 1);
      }
      target += log_sum_exp(gamma[N]);
    }
  }
 
}
```

These are relatively minor changes to the Stan routines - basically
we've simply accumulated the log posterior across trials. The rest of
this section is simply validating that these Stan models work correctly.

Let's generate some data from a Gaussian process which contiains
multiple time series (as if we collected data over multiple trials).

``` r
# Data
N = 100
Nt = 10 #number of trials
x = seq(0, 5, l=N)

# Parameters
rho = 0.5
alpha = 2
sigma = 0.3

# Arrays to store generated data
f = matrix(data=NA, nrow=Nt, ncol=N)
y = matrix(data=NA, nrow=Nt, ncol=N)
xs = matrix(x, nrow=Nt, ncol=N, byrow=TRUE)

# Simulate
sim_params = list(N=N, x=x, rho=rho, alpha=alpha, sigma=sigma)
for (trial in 1:Nt){
  sim_gp = stan(file='simulate_lgp.stan', data=sim_params, iter=1, 
                chains=1, seed=trial*100, algorithm="Fixed_param")
  f[trial,] = extract(sim_gp)$f
  y[trial,] = extract(sim_gp)$y
}

# Store data
gp_data = list(N=N, Nt=Nt, x=x, y=y)
```

And now we can take a look at all the latent functions we generated
(lines) and the corresponding observations (points).

``` r
# Set the palette to something as ungarish as possible...
palette(brewer.pal(n=10, name="Paired"))

# Plot the data
matplot(t(xs), t(f), type='l', lty=1, lwd=2, 
        ylim=c(0, 1), xlab='', ylab='')
par(new=T)
matplot(t(xs), t(y), type='p', pch=20, 
        ylim=c(0, 1), xlab='x', ylab='f')
title('Multiple draws from a Gaussian process')
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-5-1.svg)

Similarly, we'll generate multiple time series from a hidden Markov
model.

``` r
# Data
N = 100
Nt = 10 #number of trials

# Parameters
phi = array(c(0.9, 0.1, 0.1, 0.9), dim=c(2,2)) #transition probs
theta = c(5, 5) #observation distribution parameters

# Arrays to store generated data
s = matrix(data=NA, nrow=Nt, ncol=N)
y = matrix(data=NA, nrow=Nt, ncol=N)
xs = matrix(x, nrow=Nt, ncol=N, byrow=TRUE)

# Simulate
sim_params = list(N=N, phi=phi, theta=theta)
for (trial in 1:Nt){
  sim_hmm = stan(file='simulate_hmm.stan', data=sim_params, iter=1, 
                 chains=1, seed=trial*100, algorithm="Fixed_param")
  s[trial,] = extract(sim_hmm)$s-1
  y[trial,] = extract(sim_hmm)$y
}

# Store data
hmm_data = list(N=N, Nt=Nt, x=x, y=y)
```

And here's the data generated by the hidden Markov model. (Note that the
y values below aren't accurate, the traces are staggered so we can see
each trace)

``` r
# Plot the data
matplot(t(xs), t(s+0:9), type='l', lty=1, lwd=2, 
        ylim=c(0, 10), xlab='', ylab='')
par(new=T)
matplot(t(xs), t(y+0:9), type='p', pch=20, 
        ylim=c(0, 10), xlab='x', ylab='f')
title('Multiple draws from a hidden Markov model')
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-7-1.svg)

Now we can fit both models to both sets of data.

``` r
# Fit each model to each multitrial dataset
fit_gp_to_gp = stan(file='lgp_multitrial.stan', data=gp_data, seed=seed)
fit_hmm_to_gp = stan(file='hmm_multitrial.stan', data=gp_data, seed=seed)
fit_gp_to_hmm = stan(file='lgp_multitrial.stan', data=hmm_data, seed=seed)
fit_hmm_to_hmm = stan(file='hmm_multitrial.stan', data=hmm_data, seed=seed)
```

Let's check the MCMC diagnostics to ensure there weren't any major
problems with the fits.

``` r
# Check MCMC diagnostics for GP fit to GP data
check_hmc_diagnostics(fit_gp_to_gp)
```

    Divergences:
    0 of 4000 iterations ended with a divergence.
     
    Tree depth:
    0 of 4000 iterations saturated the maximum tree depth of 10.
     
    Energy:
    E-BFMI indicated no pathological behavior.

``` r
# Check MCMC diagnostics for HMM fit to GP data
check_hmc_diagnostics(fit_hmm_to_gp)
```

    Divergences:
    0 of 4000 iterations ended with a divergence.
     
    Tree depth:
    0 of 4000 iterations saturated the maximum tree depth of 10.
     
    Energy:
    E-BFMI indicated no pathological behavior.

``` r
# Check MCMC diagnostics for GP fit to HMM data
check_hmc_diagnostics(fit_gp_to_hmm)
```

    Divergences:
    0 of 4000 iterations ended with a divergence.
     
    Tree depth:
    0 of 4000 iterations saturated the maximum tree depth of 10.
     
    Energy:
    E-BFMI indicated no pathological behavior.

``` r
# Check MCMC diagnostics for HMM fit to HMM data
check_hmc_diagnostics(fit_hmm_to_hmm)
```

    Divergences:
    0 of 4000 iterations ended with a divergence.
     
    Tree depth:
    0 of 4000 iterations saturated the maximum tree depth of 10.
     
    Energy:
    E-BFMI indicated no pathological behavior.

The chains converged for all four fits (Rhat values look good -
i.e. they are near 1).

``` r
# Check GP fit to GP data
print(fit_gp_to_gp)
```

    Inference for Stan model: lgp_multitrial.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
     
             mean se_mean   sd    2.5%     50%   97.5% n_eff Rhat
    rho      0.52    0.00 0.02    0.48    0.52    0.56  2701    1
    alpha    2.10    0.00 0.17    1.80    2.09    2.47  2571    1
    sigma    0.30    0.00 0.01    0.29    0.30    0.32  3052    1
    lp__  1649.05    0.03 1.19 1645.99 1649.34 1650.39  1653    1

``` r
# Check HMM fit to GP data
print(fit_hmm_to_gp)
```

    Inference for Stan model: hmm_multitrial.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
     
               mean se_mean   sd   2.5%    50%  97.5% n_eff Rhat
    phi[1,1]   0.98    0.00 0.01   0.97   0.98   0.99  3568    1
    phi[1,2]   0.02    0.00 0.01   0.01   0.02   0.03  3568    1
    phi[2,1]   0.03    0.00 0.01   0.02   0.03   0.05  4000    1
    phi[2,2]   0.97    0.00 0.01   0.95   0.97   0.98  4000    1
    theta[1]   3.24    0.00 0.21   2.84   3.24   3.66  3023    1
    theta[2]   4.42    0.01 0.37   3.77   4.39   5.23  2920    1
    lp__     499.96    0.04 1.43 496.37 500.28 501.75  1503    1


``` r
# Check GP fit to HMM data
print(fit_gp_to_hmm)
```

    Inference for Stan model: lgp_multitrial.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
     
            mean se_mean   sd   2.5%    50%  97.5% n_eff Rhat
    rho     0.16    0.00 0.01   0.14   0.16   0.19  3190    1
    alpha   1.87    0.00 0.12   1.65   1.86   2.11  3165    1
    sigma   1.68    0.00 0.05   1.59   1.68   1.77  3471    1
    lp__  359.29    0.03 1.22 356.11 359.62 360.71  1793    1

``` r
# Check HMM fit to HMM data
print(fit_hmm_to_hmm)
```

    Inference for Stan model: hmm_multitrial.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
     
               mean se_mean   sd   2.5%    50%  97.5% n_eff Rhat
    phi[1,1]   0.90    0.00 0.01   0.88   0.91   0.93  4000    1
    phi[1,2]   0.10    0.00 0.01   0.07   0.09   0.12  4000    1
    phi[2,1]   0.11    0.00 0.02   0.08   0.11   0.14  4000    1
    phi[2,2]   0.89    0.00 0.02   0.86   0.89   0.92  4000    1
    theta[1]   4.91    0.00 0.23   4.46   4.90   5.38  4000    1
    theta[2]   4.87    0.00 0.26   4.38   4.87   5.40  4000    1
    lp__     494.90    0.03 1.37 491.59 495.19 496.66  2180    1

Were the models able to recover the parameters used to generate the
data? 

First let's make a function to plot the posterior distributions against the true value.

``` r
# Function to plot posterior distributions w/ 95% confidence intervals
interval_density = function(x, bot=0.025, top=0.975,
                            main="", xlab="", ylab="",
                            xlim=c(min(x),max(x)), lwd=1,
                            col1=c("#DCBCBC"), col2=c("#B97C7C"),
                            true=NA, true_col=c("#8F2727")) {
  dens = density(x[x>xlim[1] & x<xlim[2]])
  plot(dens, main=main, xlab=xlab, ylab=ylab, xlim=xlim,
       lwd=lwd, yaxt='n', bty='n', type='n')
  polygon(dens, col=col1, border=NA)
  qbot <- quantile(x, bot)
  qtop <- quantile(x, top)
  x1 = min(which(dens$x >= qbot))
  x2 = max(which(dens$x < qtop))
  with(dens, polygon(x=x[c(x1,x1:x2,x2)], y=c(0, y[x1:x2], 0), col=col2, border=NA))
  if (!is.na(true)) {
    abline(v=true, col=true_col, lwd=3)
  }
}
```

The Gaussian process fit was able to sucessfully recover the
parameters of the Gaussian process used to generate the data. The
vertical red line shows the true parameter value, and the distribution
is the posterior.

``` r
# Plot true vs posterior for GP
posterior = extract(fit_gp_to_gp)
par(mfrow=c(1, 3))
interval_density(posterior$rho, xlab="rho", true=rho)
interval_density(posterior$alpha, xlab="alpha", true=alpha)
interval_density(posterior$sigma, xlab="sigma", true=sigma)
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-12-1.svg)

Similarly, the hidden Markov model fit was able to sucessfully recover
the parameters of the hidden Markov model used to generate the data.

``` r
# Plot true vs posterior
posterior = extract(fit_hmm_to_hmm)
par(mfrow=c(2, 2))
interval_density(posterior$phi[,1,1], xlab="phi[,1,1]",
                 true=phi[1,1])
interval_density(posterior$phi[,2,2], xlab="phi[,2,2]",
                 true=phi[2,2])
interval_density(posterior$theta[,1], xlab="theta[1]",
                 true=theta[1])
interval_density(posterior$theta[,2], xlab="theta[2]",
                 true=theta[2])
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-13-1.svg)

Do the fits of one model to data generated by the other model at least
look reasonable? The Gaussian process fit to the data generated by the
hidden Markov model looks reasonable.

``` r
# Plot true vs posterior
posterior = extract(fit_gp_to_hmm)
par(mfrow=c(1, 3))
interval_density(posterior$rho, xlab="rho", xlim=c(0, 0.4))
interval_density(posterior$alpha, xlab="alpha")
interval_density(posterior$sigma, xlab="sigma")
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-14-1.svg)

As does the hidden Markov model fit to the data generated by the
Gaussian process.

``` r
# Plot true vs posterior
posterior = extract(fit_hmm_to_gp)
par(mfrow=c(2, 2))
interval_density(posterior$phi[,1,1], xlab="phi[,1,1]")
interval_density(posterior$phi[,2,2], xlab="phi[,2,2]")
interval_density(posterior$theta[,1], xlab="theta[1]")
interval_density(posterior$theta[,2], xlab="theta[2]")
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-15-1.svg)

The Bayes factor should favor a fit of the model which generated the
data over a fit of the other model. Let's use bridge sampling to
estimate the marginal probabilities of each of the four fits so that we
can estimate the Bayes factors.

``` r
# Perform bridge sampling for each model
bridge_gp_gp = bridge_sampler(fit_gp_to_gp)
bridge_hmm_gp = bridge_sampler(fit_hmm_to_gp)
bridge_gp_hmm = bridge_sampler(fit_gp_to_hmm)
bridge_hmm_hmm = bridge_sampler(fit_hmm_to_hmm)
```

The bridge-sampling-estimated Bayes factor favored the Gaussian process
fit to data generated by the Gaussian process.

``` r
lbf1 = bf(bridge_gp_gp, bridge_hmm_gp, log=TRUE)
cat(sprintf("LBF of GP over HMM on GP-generated data: %0.3g\n", 
            lbf1$bf))
```

    LBF of GP over HMM on GP-generated data: 1.15e+03

Conversely, the bridge-sampling-estimated Bayes factor favored the
hidden Markov model fit to data generated by the hidden Markov model.

``` r
lbf2 = bf(bridge_hmm_hmm, bridge_gp_hmm, log=TRUE)
cat(sprintf("LBF of HMM over GP on HMM-generated data: %0.3g\n", 
            lbf2$bf))
```

    LBF of HMM over GP on HMM-generated data: 137


## Multilevel Models

Another problem with the Stan models so far is they only handle a single
subject. We want our models to include random effects (that is, account
for inter-subject variability). So, we'll build multilevel versions of
both the Gaussian process and hidden Markov models which can handle
multiple trials from multiple subjects. This "multilevel" model will
have a subject level and a population level. At the subject level, each
subject has their own set of parameters, which are used to compute the
probability of the data for that subject. But, each subect's parameters
aren't completely independent: at the population level, each subject's
paramters are drawn from a population distribution.

### Gaussian Process

Our basic single-level Gaussian process has three parameters: the length scale
(\\( \rho \\)), the signal standard deviation (\\( \alpha \\), a.k.a. the marginal or output
standard deviation), and the noise standard deviation (\\( \sigma \\)).  All three
parameters are constrained to be greater than zero.  We'll use a
[log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)
to model the population distribution for all three parameters.  A log normal
distribution is just a normal distribution, but where the variable has been
passed through the \\( \log \\) function.  This causes the distribution to be defined
only \\( >0 \\). Furthermore, we'll put a prior on the medians of these
population-level distributions which is the same as the priors we were
previously using on the parameters themselves (in the single-subject versions
of the models).

Here's a diagram of the multilevel Gaussian process model, and below
we'll walk through it step by step.

![Multilevel Gaussian Process](/assets/img/hmm-vs-gp-part2/multilevel_gaussian_process.svg)

The length scale parameter (\\( \rho \\)) for each subject \\( i \\) is drawn from a
population log-normal distribution with median \\( \rho_m \\) and standard deviation
parameter \\( \rho_\sigma \\).

$$
\forall i, ~ \rho_i \sim \text{LogNormal}(\log (\rho_m), ~ \rho_\sigma)
$$

I say "standard deviation parameter" instead of "standard deviation", because
the second parameter of the log-normal distribution is not the standard
deviation of the distribution - it's the standard deviation of the *logarithm*
of the distribution.

The prior on the median of this population distribution is an inverse gamma
distribution with \\( \alpha=2 \\) and \\( \beta=0.5 \\) (which was the prior on the \\( \rho \\)
parameter in the single-subject models).

$$
\rho_m \sim \text{InvGamma}(2, 0.5)
$$

The prior on the standard deviation parameter (\\( \rho_\sigma \\)) is a half-normal
distribution with a standard deviation of \\( 0.5 \\).

$$
\rho_\sigma \sim \text{HalfNormal}(0, 0.5)
$$

The signal standard deviation parameter (\\( \alpha \\)) for each subject \\( i \\) is
drawn from a population log-normal distribution with median \\( \alpha_m \\) and
standard deviation parameter \\( \alpha_\sigma \\).

$$
\forall i, ~ \alpha_i \sim \text{LogNormal}(\log (\alpha_m), ~ \alpha_\sigma)
$$

The prior on the median of this population distribution is a half-normal
distribution with \\( \mu=0 \\) and \\( \sigma=2 \\) (which was the prior on the \\( \alpha \\)
parameter in the single-subject models).

$$
\alpha_m \sim \text{HalfNormal}(0, 2)
$$

The prior on the standard deviation parameter (\\( \alpha_\sigma \\)) is also a
half-normal distribution, but with a standard deviation of \\( 0.5 \\).

$$
\alpha_\sigma \sim \text{HalfNormal}(0, 0.5)
$$

Finally, the noise standard deviation parameter (\\( \sigma \\)) for each subject \\( i \\)
is drawn from a population log-normal distribution with median \\( \sigma_m \\) and
standard deviation parameter \\( \sigma_{sigma} \\).

$$
\forall i, ~ \sigma_i \sim \text{LogNormal}(\log (\sigma_m), \sigma_\sigma)
$$

The prior on the median of this population distribution is a half-normal
distribution with a standard deviation of \\( 1 \\) (which was the prior on the
\\( \alpha \\) parameter in the single-subject models).

$$
\sigma_m \sim \text{HalfNormal}(0, 1)
$$

And the prior on the standard deviation parameter (\\( \sigma_\sigma \\)) is a
half-normal distribution with a standard deviation of \\( 0.5 \\).


### Hidden Markov Model

The single-subject hidden Markov model has four parameters: the recurrent
transition probabilities for state 1 (\\( \phi_{1,1} \\)) and state 2 (\\( \phi_{2,2} \\)),
along with the observation parameters for state 1 (\\( \theta_1 \\)) and state 2
(\\( \theta_2 \\)).  The transition probabilities (\\( \phi_{i,i} \\)) are constrained
between 0 and 1, so we'll use a
[logit-normal distribution](https://en.wikipedia.org/wiki/Logit-normal_distribution)
to model their population distribution, which keeps the density between 0 and 1.
This is similar to what we did in order to bound the Gaussian process between 0
and 1.  On the other hand, the observation parameters (\\( \theta_i \\)) are
constrained to be greater than zero (with no upper bound), so we'll use a
log-normal distribution to model their population distribution (as we did with
the population distributions for the multilevel Gaussian process).  

Like with the multilevel Gaussian process, we'll put a prior on the medians of
the population distributions which are identical to the priors we used on the
raw parameters in the single-subject versions of the models.  Why the medians
instead of the means?  Just for simplicity and consistency, really.  Because of
the skew introduced by the log- and logit-transforms, the means of the log- and
logit-normal distributions are not equal to their \\( \mu \\) parameters.  The mean of
a log-normal distribution is relatively easy to compute
( \\( exp(\mu+\frac{\sigma^2}{2} ) \\) ), but the mean of a logit-normal distribution has
no analytical solution, and would have to be estimated numerically.  Which would
be a pain to do manually in Stan, would be *hideously* inelegant, and as of
version 2.17.0, Stan doesn't have a built-in function for doing this.  So, I
decided to just put the prior on the median (instead of the mean) for all
population-level distributions.

Here's a diagram of the multilevel hidden Markov model, and below we'll walk
through it step by step.

![Multilevel Hidden Markov Model](/assets/img/hmm-vs-gp-part2/multilevel_hidden_markov_model.svg)

The transition probability parameters (\\( \phi_{i,i} \\)) for each subject \\( s \\) are
drawn from population logit-normal distributions with medians \\( \phi_{i,m} \\) and
standard deviation parameters \\( \phi_{i,\sigma} \\).

$$
\forall s, ~ \phi_{i,i,s} \sim \text{LogitNormal}(\text{logit}^{-1}(\phi_{i,m}), ~ \phi_{i,\sigma})
$$

Where \\( \phi_{i,i,s} \\) is the \\( \phi_{i,i} \\) parameter for subject \\( s \\) and state
\\( i \\), \\( \phi_{i,m} \\) is the median of the population distribution for the recurrent
transition probability parameters of state \\( i \\), and \\( \phi_{i,\sigma} \\) is the
standard deviation parameter for that population distribution.

The prior on the medians of these population distributions are beta
distributions with \\( \alpha=1.2 \\) and \\( \beta=1.2 \\) (which was the prior on the
\\( \phi_{i,i} \\) parameters in the single-subject models).

$$
\forall i \in \{1,2\}, ~ \phi_{i,m} \sim \text{Beta}(1.2, 1.2)
$$

And the prior on the standard deviation parameters are half-normal
distributions with standard deviations of 1.

$$
\forall i \in \{1,2\}, ~ \phi_{i,\sigma} \sim \text{HalfNormal}(0, 1)
$$

The observation parameters (\\( \theta_i \\)) for each subject \\( s \\) are drawn from
population log-normal distributions with medians \\( \theta_{i,m} \\) and standard
deviation parameters \\( \theta_{i,\sigma} \\).  

$$
\forall s, ~ \theta_{i,s} \sim \text{LogNormal}(\log(\theta_{i,m}), ~ \theta_{i,\sigma})
$$

Where \\( \theta_{i,s} \\) is the \\( \theta_i \\) parameter for subject \\( s \\) and state \\( i \\),
\\( \theta_{i,m} \\) is the median of the population distribution for the observation
distribution parameter for state \\( i \\), and \\( \theta_{i,\sigma} \\) is the standard
deviation parameter for that population distribution.

The prior on the medians of these population distributions are gamma
distributions with \\( \alpha=2 \\) and \\( \beta=2 \\) (which was the prior on the
\\( \theta_i \\) parameters in the single-subject models).

$$
\forall i \in \{1, 2\}, ~ \theta_{i,m} \sim \text{Gamma}(2,2)
$$

And the prior on the standard deviation parameters are half-normal
distributions with standard deviations of 2.

$$
\forall i \in \{1, 2\}, ~ \theta_{i,\sigma} \sim \text{HalfNormal}(0,2)
$$

### Non-centered parameterizations

In theory, the models above are good as is.  However, in practice, the models
are difficult to sample from because of the geometry of the posterior
distribution.  With a hierarchical model, individuals' parameters are drawn
from a population distribution.  When the variance of the population
distribution is large, the individuals' parameters are able to approach their
non-pooled values (the value which that individual's parameter would have if we
fit the model to only data from that individual).  However, the variance of the
population distribution can shrink (up to a point) while maintaining a similar
posterior probability, because while the likelihood of any one individual's data
decreases, the likelihood due to individuals' parameters being drawn from the
population distribution increases (because the distribution is being
compressed, and so it gets "taller", because the distribution must integrate to 
1).  This can lead to a "funnel"-like geometry in the posterior.

![Hierarchical funnel](/assets/img/hmm-vs-gp-part2/hierarchical_funnel.svg)

Stan and other MCMC-based samplers use a sampling method which takes discrete
steps (check out this
[great interactive animation by Chi Feng](https://chi-feng.github.io/mcmc-demo/app.html#EfficientNUTS,banana)
showing how different MCMC samplers sample from the posterior).  So, at the
"neck" of the "funnel," the posterior is extremely thin, and the sampler can
quickly step out of a region of high posterior density.  When it does so, the
large gradient outside the funnel can cause the sampling transitions to
"diverge" (shoot out towards infinity).  Most of the transitions in the neck of
the funnel will end in this kind of divergence when the funnel is sharp enough,
and so the resulting MCMC samples won't accurately reflect the true posterior
because they didn't sample in the neck of the funnel!  This will cause our
parameter estimates to be incorrect, as they will be biased away from the region
of parameter-space corresponding to the neck of the funnel.  If we try to fit
the models as is, nearly *half* of the transitions end up diverging!

To correct for this problem, we can use a non-centered parameterization of our
models.  Instead of defining an individual \\( i \\)'s parameter (say, \\( \theta_i \\)) as
being drawn from a population distribution, we will instead draw a per-subject
scaling factor \\( \tilde{\theta}_i \\) from a standard normal distribution
(which is independent from the population distribution variance).

$$
\tilde{\theta}_i \sim \text{Normal}(0, 1)
$$

Then we can set each individual's parameter by multiplying the population
distribution standard deviation (\\( \theta_\sigma \\)) by the scaling factor, and
adding the population distribution's location parameter (\\( \theta_m \\)).

$$
\theta_i = \theta_m + \tilde{\theta}_i ~ \theta_\sigma
$$

This is mathematically equivalent to our previous model, but transforms the
posterior distribution's geometry into a much simpler form.  Now,
parameter-space only includes the value for \\( \tilde{\theta} \\) and not for
\\( \theta \\) itself, which is now a transformed parameter.  This means the posterior
is no longer funnel-shaped, and divergent transitions become much less of a
problem.  For more about this problem and how to fix it, check out this great
[case study by Michael Betancourt](http://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html).
Or [read their paper about it](https://arxiv.org/abs/1312.0906).

Here's a diagram of the non-centered parameterization of the Gaussian process
model.  Instead of drawing individuals' parameters from a population
distribution, we now construct them from per-subject scaling factors which are
drawn from a standard normal distribution.

![Multilevel Gaussian Process Non-centered Parameterization](/assets/img/hmm-vs-gp-part2/multilevel_gaussian_process_noncentered.svg)

Here is the Stan routine for the multilevel Gaussian process model, with
a non-centered parameterization. The main difference is that now we have
to compute a covariance matrix for each subject individually, and now we
have population distributions from which individual subject's parameters
are drawn.

``` r
writeLines(readLines("lgp_multilevel.stan"))
```

``` stan
data {
  int<lower=1> N;  //number of datapoints per trial
  int<lower=1> Nt; //number of trials (total across all subjects)
  int<lower=1> Ns; //number of subjects
  real x[N];       //independent var (same across trials/subjects)
  int<lower=1,upper=Ns> S[Nt]; //subject ID for each trial
  row_vector<lower=0, upper=1>[N] y[Nt]; //dependent variable
}
 
transformed data {
  vector[N] mu;      //mean function (vector of 0s)
  real sum_ln_scale; //sum of scales for logit normal dists
  mu = rep_vector(0, N);
  sum_ln_scale = 0;
  for (i in 1:Nt) //pre-compute contribution of logit normal scales
    sum_ln_scale += -sum(log(y[i])+log(1-y[i]));
}
 
parameters {
  // Per-subject parameters (non-centered parameterization)
  real rho_tilde[Ns];   //non-centered std of length scale
  real alpha_tilde[Ns]; //non-centered std of signal std dev
  real sigma_tilde[Ns]; //non-centered std of noise std dev
   
  // Population-level parameters
  real<lower=0> rho_m;   //median of rho population distribution
  real<lower=0> rho_s;   //std of rho population distribution
  real<lower=0> alpha_m; //median of alpha population distribution
  real<lower=0> alpha_s; //std of alpha population distribution
  real<lower=0> sigma_m; //median of sigma population distribution
  real<lower=0> sigma_s; //std of sigma population distribution
}
 
transformed parameters {
  // Per-subject parameters
  real<lower=0> rho[Ns];   //length scale
  real<lower=0> alpha[Ns]; //signal standard deviation
  real<lower=0> sigma[Ns]; //noise standard deviation
   
  // Non-centered parameterization of per-subject parameters
  for (s in 1:Ns) {
    rho[s] = exp(log(rho_m) + rho_s * rho_tilde[s]);
    alpha[s] = exp(log(alpha_m) + alpha_s * alpha_tilde[s]);
    sigma[s] = exp(log(sigma_m) + sigma_s * sigma_tilde[s]);
  }
}
 
model {
   
  // Covariance matrix for each subject (x same for each trial)
  matrix[N, N] K[Ns];
  for (s in 1:Ns) {
    K[s] = cov_exp_quad(x, alpha[s], rho[s]) + 
           diag_matrix(rep_vector(square(sigma[s]), N));
  }
 
  // Priors (on population-level params)
  target += inv_gamma_lpdf(rho_m | 2, 0.5);
  target += normal_lpdf(alpha_m | 0, 2)   + log(2);
  target += normal_lpdf(sigma_m | 0, 1)   + log(2);
  target += normal_lpdf(rho_s   | 0, 0.5) + log(2);
  target += normal_lpdf(alpha_s | 0, 0.5) + log(2);
  target += normal_lpdf(sigma_s | 0, 0.5) + log(2);
   
  // Subject-level parameters drawn from pop-level distributions
  // (non-centered parameterizations)
  target += normal_lpdf(rho_tilde   | 0, 1); //log(rho) ~ normal(exp(rho_m), rho_s)
  target += normal_lpdf(alpha_tilde | 0, 1); //log(alpha) ~ normal(exp(alpha_m), alpha_s)
  target += normal_lpdf(sigma_tilde | 0, 1); //log(sigma) ~ normal(exp(sigma_m), sigma_s)
  
  // Jacobian adjustments for GLM parts of model
  target += -sum(log(rho));
  target += -sum(log(alpha));
  target += -sum(log(sigma));
   
  // Accumulate evidence over trials
  for (i in 1:Nt)
    target += multi_normal_lpdf(logit(y[i]) | mu, K[S[i]]);
     
  // Add logit-normal scale terms to log posterior
  target += sum_ln_scale;
   
}
```

I tried implementing several optimizations, including pre-computing
logit(y) (in the transformed data block), writing a custom user-defined
Stan function to compute the covariance matrix which assumed
linearly-spaced x values, etc. However, none of the optimizations
actually ended up making the sampling run any faster! Stan's pretty fast
as is. 

Well. Maybe not "fast". *Optimized*.

We'll do the same thing for the hidden Markov model. Here's a diagram of
the non-centered parameterization of the hidden Markov model.

![Multilevel Hidden Markov Model Non-centered Parameterization](/assets/img/hmm-vs-gp-part2/multilevel_hidden_markov_model_noncentered.svg)

And here's the Stan routine for the multilevel hidden Markov model with
a non-centered parameterization.

``` r
writeLines(readLines("hmm_multilevel.stan"))
```

``` stan
data {
  int<lower=1> N;     //number of observations per trial
  int<lower=1> Nt;    //number of trials
  int<lower=1> Ns;    //number of subjects
  int<lower=1> S[Nt]; //subject id
  real<lower=0, upper=1> y[Nt, N]; //observations
}
 
parameters {
  // Per-subject parameters (non-centered parameterization)
  real phi_tilde[Ns,2];   //transition probabilities
  real theta_tilde[Ns,2]; //observation distribution params
   
  // Population-level parameters
  real<lower=0,upper=1> phi_m[2]; //median of phi population dists
  real<lower=0> phi_s[2];   //std of phi population dists
  real<lower=1> theta_m[2]; //median of theta population dists
  real<lower=0> theta_s[2]; //std of theta population dists
}
 
transformed parameters {
  // Per-subject parameters
  simplex[2] phi[Ns,2];      //transition probabilities
  real<lower=1> theta[Ns,2]; //observation distribution params
 
  // Non-centered parameterization of per-subject parameters
  for (s in 1:Ns) {
    phi[s,1,1] = inv_logit(logit(phi_m[1])+phi_s[1]*phi_tilde[s,1]);
    phi[s,2,2] = inv_logit(logit(phi_m[2])+phi_s[2]*phi_tilde[s,2]);
    phi[s,1,2] = 1 - phi[s,1,1];
    phi[s,2,1] = 1 - phi[s,2,2];
    theta[s,1] = 1+exp(log(theta_m[1]-1)+theta_s[1]*theta_tilde[s,1]);
    theta[s,2] = 1+exp(log(theta_m[2]-1)+theta_s[2]*theta_tilde[s,2]);
  }
}
 
model {
   
  // Priors for each of the 2 states
  for (i in 1:2) {
    // Priors (on population-level params)
    target += beta_lpdf(phi_m[i] | 1.2, 1.2);
    target += gamma_lpdf(theta_m[i]-1 | 2, 2);
    target += normal_lpdf(phi_s[i] | 0, 1) + log(2);
    target += normal_lpdf(theta_s[i] | 0, 2) + log(2);
   
    // Subject-level parameters drawn from pop-level distributions
    // (non-centered parameterizations)
    target += normal_lpdf(phi_tilde[,i] | 0, 1);   //logit(phi) ~ normal(inv_logit(phi_m), phi_s)
    target += normal_lpdf(theta_tilde[,i] | 0, 1); //log(theta) ~ normal(exp(theta_m), theta_s)
   
    // Jacobian adjustments for GLM parts of model
    for (s in 1:Ns)
      target += -log(phi[s,i,i]*(1-phi[s,i,i]));
    target += -sum(log(theta[,i]));
  }
 
  // Compute the marginal probability over possible sequences
  {
    real acc[2];
    real gamma[N, 2];
    for (i in 1:Nt) { // accumulate evidence over trials
      gamma[1,1] = beta_lpdf(y[i,1] | 1, theta[S[i],1]);
      gamma[1,2] = beta_lpdf(y[i,2] | theta[S[i],2], 1);
      for (t in 2:N) {
        for (k in 1:2) {
          acc[1] = gamma[t-1, 1] + log(phi[S[i],1,k]);
          acc[2] = gamma[t-1, 2] + log(phi[S[i],2,k]);
          gamma[t,k] = log_sum_exp(acc);
        }
        gamma[t,1] += beta_lpdf(y[i,t] | 1, theta[S[i],1]);
        gamma[t,2] += beta_lpdf(y[i,t] | theta[S[i],2], 1);
      }
      target += log_sum_exp(gamma[N]);
    }
  }
 
}
```


### Validation

Let's generate some data from a Gaussian process which contiains
simulated data from multiple trials and multiple subjects.

``` r
# Data
N = 50 #datapoints per trial
Ns = 5 #number of subjects
Nts = 5 #number of trials per subject
Nt = Ns*Nts #total number of trials
x = seq(0, 2.5, l=N)

# Population Distribution Means and variances
rho_mu = 0.5
rho_var = 0.025
alpha_mu = 2
alpha_var = 0.1
sigma_mu = 0.3
sigma_var = 0.01

# Compute gamma distribution parameters from mean + variance
rho_a = rho_mu*rho_mu/rho_var
rho_b = rho_mu/rho_var
alpha_a = alpha_mu*alpha_mu/alpha_var
alpha_b = alpha_mu/alpha_var
sigma_a = sigma_mu*sigma_mu/sigma_var
sigma_b = sigma_mu/sigma_var

# Arrays to store generated data
f = matrix(data=NA, nrow=Nt, ncol=N) #latent function
y = matrix(data=NA, nrow=Nt, ncol=N) #observation values
sid = matrix(data=NA, Nt) #subject id
xs = matrix(x, nrow=Nts, ncol=N, byrow=TRUE)

# Generate parameters for each subject
rho_true = rgamma(Ns, rho_a, rho_b)
alpha_true = rgamma(Ns, alpha_a, alpha_b)
sigma_true = rgamma(Ns, sigma_a, sigma_b)

# Simulate
for (sub in 1:Ns){
  sim_params = list(N=N, x=x, rho=rho_true[sub], 
                    alpha=alpha_true[sub], sigma=sigma_true[sub])
  for (trial in 1:Nts){
    ix = (sub-1)*Nts+trial #index in array of all trials
    sim_gp = stan(file='simulate_lgp.stan', 
                  data=sim_params, iter=1, 
                  chains=1, seed=ix, algorithm="Fixed_param")
    f[ix,] = extract(sim_gp)$f
    y[ix,] = extract(sim_gp)$y
    sid[ix,] = sub
  }
}

# Store data
gp_data = list(N=N, Nt=Nt, Ns=Ns, x=x, S=as.vector(sid), y=y)
```

And now we can take a look at all the latent functions we generated
(lines) and the corresponding observations (points) for each subject
(stacked vertically).

``` r
# Plot the data
plot.new()
par(mfrow=c(Ns,1))
for (sub in 1:Ns){
  ix = 1:Nts+(sub-1)*Nts #indexes of this subject
  par(mfg=c(sub,1), mar=c(1,4,1,1))
  matplot(t(xs), t(f[ix,]), type='l', lty=1, lwd=2, 
          ylim=c(0, 1), xlab='', ylab='')
  par(new=T)
  matplot(t(xs), t(y[ix,]), type='p', pch=20, 
          ylim=c(0, 1), xlab='x', ylab=sprintf("Sub %d",sub))
}
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-24-1.svg)

We can see that different subjects have different parameters in this data.  For
example, subjects 1 and 2 have much faster length-scales than subject 3, as the latent
function moves around much slower for subject 3.  However, none of the subjects
have wildly different parameter values.  This is what the multilevel model 
allows for - each subject can have a different parameter value, but not in a 
totally unconstrained way.

Similarly, we'll generate multiple trials from multiple subjects using a
hidden Markov model.

``` r
# Population Distribution Means and variances
phi_mu = 0.9
phi_var = 0.043
theta_mu = 4
theta_var = 0.3

# Compute distribution parameters from mean + variance
phi_a = 18
phi_b = 2
theta_a = (theta_mu-1)*(theta_mu-1)/theta_var
theta_b = (theta_mu-1)/theta_var

# Arrays to store generated data
s = matrix(data=NA, nrow=Nt, ncol=N) #hidden state
y = matrix(data=NA, nrow=Nt, ncol=N) #observed values
sid = matrix(data=NA, Nt)          #subject id

# Parameters for each subject
phi_true = matrix(data=NA, Ns, 2)   #phi for each subject
theta_true = matrix(data=NA, Ns, 2) #theta for each subject

# Simulate
for (sub in 1:Ns){
  p1 = rbeta(1, phi_a, phi_b) #phi parameters for this subject
  p2 = rbeta(1, phi_a, phi_b)
  t1 = 1+rgamma(1, theta_a, theta_b) #thetas for this subj
  t2 = 1+rgamma(1, theta_a, theta_b)
  phi_true[sub,] = c(p1,p2)
  theta_true[sub,] = c(t1,t2)
  phi = t(array(c(p1, 1-p1, 1-p2, p2), dim=c(2,2)))
  theta = c(t1, t2)
  sim_params = list(N=N, phi=phi, theta=theta)
  for (trial in 1:Nts){
    ix = (sub-1)*Nts+trial #index in array of all trials
    sim_hmm = stan(file='simulate_hmm.stan', 
                   data=sim_params, iter=1, 
                   chains=1, seed=ix, algorithm="Fixed_param")
    s[ix,] = extract(sim_hmm)$s-1
    y[ix,] = extract(sim_hmm)$y
    sid[ix,] = sub
  }
}

# Store data
hmm_data = list(N=N, Nt=Nt, Ns=Ns, x=x, S=as.vector(sid), y=y)
```

And here's the data generated by the hidden Markov model.

``` r
# Plot the data
plot.new()
par(mfrow=c(Ns,1))
for (sub in 1:Ns){
  ix = 1:Nts+(sub-1)*Nts #indexes of this subject
  par(mfg=c(sub,1), mar=c(1,4,1,1))
  matplot(t(xs), t(s[ix,]), type='l', lty=1, lwd=2, 
          ylim=c(0, 1), xlab='', ylab='')
  par(new=T)
  matplot(t(xs), t(y[ix,]), type='p', pch=20, 
          ylim=c(0, 1), xlab='x', ylab=sprintf("Sub %d",sub))
}
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-26-1.svg)

Now we can fit both models to both sets of data.

``` r
# Fit each model to each multisubject dataset
adt = 0.9 #use slightly higher adapt_delta parameter
fit_gp_to_gp = stan(file='lgp_multilevel.stan', data=gp_data, 
                    seed=seed, control=list(adapt_delta=adt))
fit_hmm_to_gp = stan(file='hmm_multilevel.stan', data=gp_data, 
                     seed=seed, control=list(adapt_delta=adt))
fit_gp_to_hmm = stan(file='lgp_multilevel.stan', data=hmm_data, 
                     seed=seed, control=list(adapt_delta=adt))
fit_hmm_to_hmm = stan(file='hmm_multilevel.stan', data=hmm_data, 
                     seed=seed, control=list(adapt_delta=adt))
```

The chains converged for all four fits (Rhat values look good - i.e. they are near 1), and the number of effective MCMC samples (`n_eff`) look reasonable for all parameters (we want `n_eff` to be greater than at least 100).

``` r
# Check GP fit to GP data
print(fit_gp_to_gp)

# Check HMM fit to GP data
print(fit_hmm_to_gp)

# Check GP fit to HMM data
print(fit_gp_to_hmm)

# Check HMM fit to HMM data
print(fit_hmm_to_hmm)
```

<div class="highlighter-rouge" style="width:100%; height:400px; overflow-y:scroll;">
  <div class="highlight">
    <pre class="highlight">
    <code>Inference for Stan model: lgp_multilevel.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
     
                      mean    sd    2.5%     50%  97.5% n_eff Rhat
    rho_tilde[1]     -0.94  0.57   -2.05   -0.93   0.15  1683 1.00
    rho_tilde[2]     -0.84  0.55   -1.94   -0.82   0.23  1700 1.00
    rho_tilde[3]      0.60  0.49   -0.32    0.59   1.60  1942 1.00
    rho_tilde[4]      0.38  0.47   -0.51    0.37   1.32  1919 1.00
    rho_tilde[5]      1.17  0.57    0.14    1.15   2.40  1971 1.00
    alpha_tilde[1]   -0.25  0.76   -1.78   -0.25   1.25  3363 1.00
    alpha_tilde[2]    0.06  0.77   -1.56    0.06   1.56  4000 1.00
    alpha_tilde[3]    0.69  0.80   -0.90    0.68   2.29  2846 1.00
    alpha_tilde[4]    0.11  0.77   -1.40    0.10   1.65  4000 1.00
    alpha_tilde[5]   -0.64  0.87   -2.30   -0.67   1.21  4000 1.00
    sigma_tilde[1]   -0.64  0.55   -1.79   -0.62   0.39  1575 1.00
    sigma_tilde[2]    0.55  0.55   -0.48    0.53   1.66  1742 1.00
    sigma_tilde[3]   -1.10  0.62   -2.39   -1.06   0.04  1351 1.00
    sigma_tilde[4]    0.26  0.51   -0.71    0.25   1.29  1580 1.00
    sigma_tilde[5]    0.71  0.57   -0.32    0.69   1.87  1710 1.00
    rho_m             0.46  0.09    0.28    0.45   0.66  1607 1.00
    rho_s             0.43  0.15    0.22    0.40   0.81  1838 1.00
    alpha_m           2.09  0.23    1.64    2.08   2.61  2598 1.00
    alpha_s           0.18  0.14    0.01    0.15   0.54  1566 1.00
    sigma_m           0.24  0.03    0.20    0.24   0.30  1393 1.00
    sigma_s           0.21  0.11    0.08    0.18   0.49  1186 1.00
    rho[1]            0.31  0.02    0.28    0.31   0.35  4000 1.00
    rho[2]            0.33  0.02    0.29    0.33   0.36  4000 1.00
    rho[3]            0.57  0.04    0.50    0.57   0.65  4000 1.00
    rho[4]            0.52  0.04    0.45    0.52   0.60  4000 1.00
    rho[5]            0.72  0.07    0.58    0.72   0.85  4000 1.00
    alpha[1]          2.00  0.22    1.58    2.00   2.47  4000 1.00
    alpha[2]          2.12  0.24    1.69    2.10   2.64  4000 1.00
    alpha[3]          2.39  0.35    1.90    2.32   3.24  4000 1.00
    alpha[4]          2.14  0.26    1.70    2.12   2.70  4000 1.00
    alpha[5]          1.85  0.29    1.27    1.86   2.40  2268 1.00
    sigma[1]          0.22  0.01    0.20    0.22   0.24  4000 1.00
    sigma[2]          0.27  0.01    0.24    0.27   0.30  4000 1.00
    sigma[3]          0.20  0.01    0.18    0.20   0.22  4000 1.00
    sigma[4]          0.26  0.01    0.23    0.25   0.28  4000 1.00
    sigma[5]          0.28  0.01    0.25    0.28   0.30  4000 1.00
    lp__           2057.86  4.10 2048.94 2058.172064.98   948 1.01
     
    Inference for Stan model: hmm_multilevel.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
     
                       mean    sd   2.5%    50%  97.5% n_eff Rhat
    phi_tilde[1,1]     0.19  0.83  -1.48   0.19   1.80  3414    1
    phi_tilde[1,2]     0.00  0.77  -1.58   0.02   1.50  3492    1
    phi_tilde[2,1]    -0.44  0.86  -2.04  -0.47   1.36  2886    1
    phi_tilde[2,2]    -0.51  0.76  -2.04  -0.49   0.95  2892    1
    phi_tilde[3,1]     0.08  0.90  -1.72   0.07   1.86  4000    1
    phi_tilde[3,2]     0.93  0.84  -0.81   0.95   2.51  2567    1
    phi_tilde[4,1]     0.43  0.84  -1.26   0.45   2.02  3449    1
    phi_tilde[4,2]    -0.04  0.83  -1.72  -0.02   1.61  4000    1
    phi_tilde[5,1]     0.24  0.87  -1.50   0.27   1.93  4000    1
    phi_tilde[5,2]     0.41  0.88  -1.28   0.42   2.14  2004    1
    theta_tilde[1,1]  -0.21  0.49  -1.30  -0.15   0.63  1035    1
    theta_tilde[1,2]   0.27  0.78  -1.41   0.31   1.76  3206    1
    theta_tilde[2,1]   0.30  0.41  -0.54   0.30   1.10  1305    1
    theta_tilde[2,2]  -0.16  0.77  -1.78  -0.14   1.31  2847    1
    theta_tilde[3,1]   1.60  0.57   0.62   1.55   2.85  2052    1
    theta_tilde[3,2]   0.79  0.74  -0.79   0.79   2.26  2782    1
    theta_tilde[4,1]   0.45  0.40  -0.33   0.45   1.23  1403    1
    theta_tilde[4,2]  -0.06  0.82  -1.74  -0.04   1.54  4000    1
    theta_tilde[5,1]  -0.34  0.61  -1.65  -0.31   0.80  1185    1
    theta_tilde[5,2]  -0.44  0.86  -2.14  -0.43   1.28  2574    1
    phi_m[1]           0.96  0.01   0.93   0.97   0.98  1586    1
    phi_m[2]           0.96  0.02   0.91   0.96   0.98  1713    1
    phi_s[1]           0.45  0.37   0.01   0.37   1.38  1821    1
    phi_s[2]           0.63  0.43   0.04   0.56   1.68  1451    1
    theta_m[1]         3.00  0.73   1.72   2.95   4.58  1163    1
    theta_m[2]         2.93  0.35   2.19   2.96   3.58  1627    1
    theta_s[1]         1.04  0.50   0.42   0.92   2.31  1260    1
    theta_s[2]         0.31  0.27   0.01   0.23   0.99  1206    1
    phi[1,1,1]         0.97  0.01   0.94   0.97   0.99  4000    1
    phi[1,1,2]         0.03  0.01   0.01   0.03   0.06  4000    1
    phi[1,2,1]         0.04  0.02   0.02   0.04   0.08  4000    1
    phi[1,2,2]         0.96  0.02   0.92   0.96   0.98  4000    1
    phi[2,1,1]         0.96  0.02   0.92   0.96   0.98  4000    1
    phi[2,1,2]         0.04  0.02   0.02   0.04   0.08  4000    1
    phi[2,2,1]         0.06  0.02   0.03   0.05   0.11  4000    1
    phi[2,2,2]         0.94  0.02   0.89   0.95   0.97  4000    1
    phi[3,1,1]         0.97  0.02   0.93   0.97   0.99  4000    1
    phi[3,1,2]         0.03  0.02   0.01   0.03   0.07  4000    1
    phi[3,2,1]         0.02  0.01   0.01   0.02   0.05  2439    1
    phi[3,2,2]         0.98  0.01   0.95   0.98   0.99  2439    1
    phi[4,1,1]         0.97  0.01   0.95   0.97   0.99  4000    1
    phi[4,1,2]         0.03  0.01   0.01   0.03   0.05  4000    1
    phi[4,2,1]         0.04  0.02   0.02   0.04   0.09  4000    1
    phi[4,2,2]         0.96  0.02   0.91   0.96   0.98  4000    1
    phi[5,1,1]         0.97  0.01   0.94   0.97   0.99  4000    1
    phi[5,1,2]         0.03  0.01   0.01   0.03   0.06  4000    1
    phi[5,2,1]         0.03  0.02   0.01   0.03   0.07  1840    1
    phi[5,2,2]         0.97  0.02   0.93   0.97   0.99  1840    1
    theta[1,1]         2.70  0.29   2.21   2.67   3.34  4000    1
    theta[1,2]         3.12  0.34   2.47   3.11   3.82  4000    1
    theta[2,1]         3.64  0.48   2.91   3.58   4.70  4000    1
    theta[2,2]         2.88  0.35   2.16   2.91   3.51  4000    1
    theta[3,1]         9.47  2.18   5.93   9.25  14.37  4000    1
    theta[3,2]         3.40  0.30   2.85   3.38   4.06  3116    1
    theta[4,1]         3.99  0.44   3.24   3.95   4.95  4000    1
    theta[4,2]         2.94  0.42   2.13   2.95   3.82  4000    1
    theta[5,1]         2.61  0.84   1.81   2.30   4.87   860    1
    theta[5,2]         2.72  0.48   1.81   2.80   3.52   958    1
    lp__             473.27  4.84 462.83 473.53 481.90  1158    1
     
    Inference for Stan model: lgp_multilevel.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
     
                     mean    sd   2.5%    50%  97.5% n_eff Rhat
    rho_tilde[1]    -0.48  0.71  -1.91  -0.48   0.91  2628 1.00
    rho_tilde[2]     1.07  0.78  -0.54   1.06   2.58  2620 1.00
    rho_tilde[3]    -0.05  0.80  -1.54  -0.08   1.69  2763 1.00
    rho_tilde[4]    -0.25  0.72  -1.64  -0.25   1.21  3139 1.00
    rho_tilde[5]    -0.37  0.75  -1.84  -0.37   1.20  2648 1.00
    alpha_tilde[1]   0.05  0.87  -1.64   0.04   1.78  4000 1.00
    alpha_tilde[2]  -0.11  0.91  -1.90  -0.13   1.74  4000 1.00
    alpha_tilde[3]  -0.04  0.88  -1.77  -0.02   1.69  4000 1.00
    alpha_tilde[4]  -0.08  0.90  -1.81  -0.09   1.72  4000 1.00
    alpha_tilde[5]   0.05  0.89  -1.75   0.07   1.78  4000 1.00
    sigma_tilde[1]  -0.30  0.84  -1.95  -0.29   1.36  4000 1.00
    sigma_tilde[2]  -0.17  0.81  -1.78  -0.16   1.46  3402 1.00
    sigma_tilde[3]  -0.34  0.83  -1.97  -0.34   1.31  4000 1.00
    sigma_tilde[4]   0.62  0.82  -1.09   0.62   2.19  3040 1.00
    sigma_tilde[5]   0.27  0.79  -1.35   0.28   1.85  4000 1.00
    rho_m            0.23  0.04   0.17   0.23   0.31  1835 1.00
    rho_s            0.26  0.17   0.02   0.23   0.67  1278 1.00
    alpha_m          1.63  0.13   1.40   1.63   1.92  2181 1.00
    alpha_s          0.08  0.09   0.00   0.06   0.32  1900 1.00
    sigma_m          1.62  0.06   1.49   1.62   1.74  1998 1.00
    sigma_s          0.05  0.05   0.00   0.04   0.18  1370 1.00
    rho[1]           0.20  0.03   0.15   0.20   0.26  4000 1.00
    rho[2]           0.31  0.06   0.20   0.30   0.44  2028 1.00
    rho[3]           0.23  0.06   0.17   0.22   0.41  2703 1.00
    rho[4]           0.22  0.03   0.16   0.21   0.29  4000 1.00
    rho[5]           0.21  0.03   0.15   0.21   0.28  4000 1.00
    alpha[1]         1.64  0.14   1.38   1.63   1.94  4000 1.00
    alpha[2]         1.62  0.14   1.35   1.61   1.92  4000 1.00
    alpha[3]         1.63  0.14   1.37   1.62   1.93  4000 1.00
    alpha[4]         1.62  0.14   1.35   1.62   1.92  4000 1.00
    alpha[5]         1.65  0.14   1.39   1.64   1.96  4000 1.00
    sigma[1]         1.59  0.06   1.46   1.60   1.72  4000 1.00
    sigma[2]         1.61  0.06   1.48   1.61   1.72  4000 1.00
    sigma[3]         1.59  0.07   1.45   1.59   1.71  4000 1.00
    sigma[4]         1.68  0.08   1.56   1.67   1.85  2781 1.00
    sigma[5]         1.65  0.07   1.53   1.64   1.80  4000 1.00
    lp__           298.62  4.21 289.77 298.92 305.96   879 1.01
     
    Inference for Stan model: hmm_multilevel.
    4 chains, each with iter=2000; warmup=1000; thin=1; 
    post-warmup draws per chain=1000, total post-warmup draws=4000.
     
                       mean    sd   2.5%    50%  97.5% n_eff Rhat
    phi_tilde[1,1]     0.00  0.76  -1.51   0.02   1.49  2642 1.00
    phi_tilde[1,2]    -0.31  0.70  -1.77  -0.30   1.06  2165 1.00
    phi_tilde[2,1]     0.07  0.82  -1.58   0.07   1.68  2802 1.00
    phi_tilde[2,2]     1.06  0.71  -0.35   1.05   2.45  2531 1.00
    phi_tilde[3,1]     0.95  0.80  -0.72   0.96   2.50  2421 1.00
    phi_tilde[3,2]     0.19  0.72  -1.21   0.20   1.62  2194 1.00
    phi_tilde[4,1]    -0.43  0.82  -2.10  -0.41   1.15  2819 1.00
    phi_tilde[4,2]     0.39  0.71  -0.99   0.38   1.86  1321 1.00
    phi_tilde[5,1]    -0.18  0.80  -1.74  -0.18   1.39  1909 1.00
    phi_tilde[5,2]    -0.73  0.72  -2.19  -0.71   0.67  2245 1.00
    theta_tilde[1,1]  -0.33  0.77  -1.84  -0.33   1.26  2576 1.00
    theta_tilde[1,2]   0.81  0.77  -0.80   0.83   2.26  2809 1.00
    theta_tilde[2,1]   0.69  0.82  -1.10   0.72   2.27  3189 1.00
    theta_tilde[2,2]  -0.44  0.74  -1.97  -0.40   0.92  2189 1.00
    theta_tilde[3,1]   0.36  0.75  -1.21   0.38   1.82  2975 1.00
    theta_tilde[3,2]   0.39  0.76  -1.18   0.41   1.85  3065 1.00
    theta_tilde[4,1]  -0.27  0.80  -1.85  -0.26   1.33  2897 1.00
    theta_tilde[4,2]  -0.29  0.72  -1.73  -0.29   1.11  2177 1.00
    theta_tilde[5,1]   0.12  0.78  -1.52   0.15   1.61  3321 1.00
    theta_tilde[5,2]   0.25  0.74  -1.29   0.30   1.61  2827 1.00
    phi_m[1]           0.91  0.03   0.84   0.92   0.95  1222 1.00
    phi_m[2]           0.92  0.03   0.84   0.93   0.96  1500 1.00
    phi_s[1]           0.43  0.32   0.02   0.37   1.25  1252 1.01
    phi_s[2]           0.65  0.37   0.10   0.58   1.56  1268 1.00
    theta_m[1]         3.78  0.31   3.10   3.80   4.36  1415 1.00
    theta_m[2]         3.55  0.37   2.63   3.59   4.22   503 1.00
    theta_s[1]         0.19  0.18   0.01   0.14   0.64  1038 1.00
    theta_s[2]         0.25  0.26   0.01   0.18   0.99   413 1.00
    phi[1,1,1]         0.91  0.02   0.87   0.92   0.95  4000 1.00
    phi[1,1,2]         0.09  0.02   0.05   0.08   0.13  4000 1.00
    phi[1,2,1]         0.09  0.03   0.05   0.09   0.15  4000 1.00
    phi[1,2,2]         0.91  0.03   0.85   0.91   0.95  4000 1.00
    phi[2,1,1]         0.91  0.02   0.86   0.92   0.96  4000 1.00
    phi[2,1,2]         0.09  0.02   0.04   0.08   0.14  4000 1.00
    phi[2,2,1]         0.04  0.02   0.02   0.04   0.08  1973 1.00
    phi[2,2,2]         0.96  0.02   0.92   0.96   0.98  1973 1.00
    phi[3,1,1]         0.94  0.02   0.90   0.94   0.98  1958 1.00
    phi[3,1,2]         0.06  0.02   0.02   0.06   0.10  1958 1.00
    phi[3,2,1]         0.07  0.02   0.03   0.07   0.12  4000 1.00
    phi[3,2,2]         0.93  0.02   0.88   0.93   0.97  4000 1.00
    phi[4,1,1]         0.90  0.03   0.82   0.90   0.94  4000 1.00
    phi[4,1,2]         0.10  0.03   0.06   0.10   0.18  4000 1.00
    phi[4,2,1]         0.06  0.02   0.03   0.06   0.10  4000 1.00
    phi[4,2,2]         0.94  0.02   0.90   0.94   0.97  4000 1.00
    phi[5,1,1]         0.91  0.02   0.86   0.91   0.94  4000 1.00
    phi[5,1,2]         0.09  0.02   0.06   0.09   0.14  4000 1.00
    phi[5,2,1]         0.11  0.03   0.06   0.11   0.19  2154 1.00
    phi[5,2,2]         0.89  0.03   0.81   0.89   0.94  2154 1.00
    theta[1,1]         3.65  0.29   3.06   3.66   4.18  4000 1.00
    theta[1,2]         4.12  0.48   3.37   4.06   5.21  2221 1.00
    theta[2,1]         4.22  0.46   3.54   4.14   5.33  2375 1.00
    theta[2,2]         3.38  0.25   2.89   3.39   3.88  4000 1.00
    theta[3,1]         3.99  0.29   3.48   3.97   4.62  4000 1.00
    theta[3,2]         3.83  0.36   3.22   3.80   4.61  4000 1.00
    theta[4,1]         3.67  0.35   2.95   3.69   4.31  4000 1.00
    theta[4,2]         3.45  0.26   2.92   3.45   3.94  4000 1.00
    theta[5,1]         3.88  0.31   3.29   3.87   4.55  4000 1.00
    theta[5,2]         3.77  0.36   3.13   3.73   4.60  4000 1.00
    lp__             423.87  5.30 412.49 424.22 433.19   759 1.01
  </code>
  </pre>
  </div>
</div>
    
<br />

Let's take a look at the HMC diagnostics.

``` r
# Check MCMC diagnostics for GP fit to GP data
check_hmc_diagnostics(fit_gp_to_gp)
```

    Divergences:
    0 of 4000 iterations ended with a divergence.
     
    Tree depth:
    0 of 4000 iterations saturated the maximum tree depth of 10.
     
    Energy:
    E-BFMI indicated no pathological behavior.

``` r
# Check MCMC diagnostics for HMM fit to GP data
check_hmc_diagnostics(fit_hmm_to_gp)
```

    Divergences:
    0 of 4000 iterations ended with a divergence.
     
    Tree depth:
    0 of 4000 iterations saturated the maximum tree depth of 10.
     
    Energy:
    E-BFMI indicated no pathological behavior.

``` r
# Check MCMC diagnostics for GP fit to HMM data
check_hmc_diagnostics(fit_gp_to_hmm)
```

    Divergences:
    1 of 4000 iterations ended with a divergence (0.025%).
     
    Tree depth:
    0 of 4000 iterations saturated the maximum tree depth of 10.
     
    Energy:
    E-BFMI indicated no pathological behavior.

``` r
# Check MCMC diagnostics for HMM fit to HMM data
check_hmc_diagnostics(fit_hmm_to_hmm)
```

    Divergences:
    10 of 4000 iterations ended with a divergence (0.25%).
     
    Tree depth:
    0 of 4000 iterations saturated the maximum tree depth of 10.
     
    Energy:
    E-BFMI indicated no pathological behavior.

The diagnostics look mostly good, however there were a few divergent
transitions.  However, an extremely small proportion of all the MCMC
transitions were divergent (less than half a percent of samples).  Let's take a
look at the fit with the most divergent transitions (the fit of the hidden
Markov model to the data generated by itself). Below we've plotted the value of
the standard deviation of the \\( \phi_{1,1} \\) population distribution (on the
y-axis) and the value of the first subject's \\( \phi_{1,1} \\) parameter (on the
x-axis). If divergent transitions were being caused by a misspecification of
our hierarchical model, we would expect to see the divergent transitions
concentrated at the "tip" of the hierarchical "funnel". Each dot is a MCMC
sample, and bright red dots indicate samples which ended in a divergence.  


``` r
# Show pairs plot for phi param
mcmc_scatter(as.array(fit_hmm_to_hmm), 
             pars=c("phi[1,1,1]", "phi_s[1]"), 
             transform=list("phi_s[1]"="log"), 
             np=nuts_params(fit_hmm_to_hmm))
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-30-1.svg)

The divergent transitions don't seem to be concentrating in any
particular region of parameter space, and so increasing our
`adapt_delta` parameter will probably help decrease the number of
divergent transitions, and we don't need to re-specify our model. If we
had fit the original version of the model (the centered
parameterization), divergent transitions would have been concentrated at
the tip of the funnel, and nearly half of the transitions would have
ended in a divergence!

Were the models able to recover the parameters used to generate the
data? The Gaussian process fit was able to sucessfully recover the
parameters of the Gaussian process used to generate the data.

``` r
# Plot true vs posterior for GP fit to GP
posterior = extract(fit_gp_to_gp)
par(mfrow=c(1, 3))
interval_density(posterior$rho_m,
                 xlab="Population median rho",
                 true=median(rho_true))
interval_density(posterior$alpha_m,
                 xlab="Population median alpha",
                 true=median(alpha_true))
interval_density(posterior$sigma_m,
                 xlab="Population median sigma",
                 true=median(sigma_true))
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-35-1.svg)

As was the hidden Markov model!

``` r
# Plot true vs posterior
posterior = extract(fit_hmm_to_hmm)
par(mfrow=c(2, 2))
interval_density(posterior$phi_m[,1], 
                 xlab="Population median phi[1,1]",
                 true=median(phi_true[,1]))
interval_density(posterior$phi_m[,2], 
                 xlab="Population median phi[2,2]",
                 true=median(phi_true[,2]))
interval_density(posterior$theta_m[,1], 
                 xlab="Population median theta[1]",
                 true=median(theta_true[,1]))
interval_density(posterior$theta_m[,2], 
                 xlab="Population median theta[2]",
                 true=median(theta_true[,2]))
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-36-1.svg)

We also want to ensure that the fit was able to recover the correct
values for each individual's parameters. The Gaussian process fit to
data generated by itself was (for the most part) able to recover the
correct parameter values for each individual:

``` r
# Plot true vs posterior for GP fit to GP
posterior = extract(fit_gp_to_gp)
par(mfrow=c(Ns, 3), mar=c(1,1,1,1))

for (sub in 1:Ns){
  interval_density(posterior$rho[,sub], xlim=xlim_rho,
                   xlab=sprintf("rho_%d", sub),
                   true=rho_true[sub])
  interval_density(posterior$alpha[,sub],  xlim=xlim_alpha,
                   xlab=sprintf("alpha_%d", sub),
                   true=alpha_true[sub])
  interval_density(posterior$sigma[,sub],  xlim=xlim_sigma,
                   xlab=sprintf("sigma_%d", sub),
                   true=sigma_true[sub])
}
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-37-1.svg)

As was the hidden Markov model (again, imperfectly for some parameters).

``` r
# Plot true vs posterior
posterior = extract(fit_hmm_to_hmm)
par(mfrow=c(Ns, 4), mar=c(1,1,1,1))

for (sub in 1:Ns){
  interval_density(posterior$phi[,sub,1,1], xlim=xlim_phi1,
                   xlab=sprintf("phi_1,1,%d", sub),
                   true=phi_true[sub,1])
  interval_density(posterior$phi[,sub,2,2], xlim=xlim_phi2,
                   xlab=sprintf("phi_2,2,%d", sub),
                   true=phi_true[sub,2])
  interval_density(posterior$theta[,sub,1], xlim=xlim_theta1,
                   xlab=sprintf("theta_1,%d", sub),
                   true=theta_true[sub,1])
  interval_density(posterior$theta[,sub,2], xlim=xlim_theta2,
                   xlab=sprintf("theta_2,%d", sub),
                   true=theta_true[sub,2])
}
```

![](/assets/img/hmm-vs-gp-part2/unnamed-chunk-38-1.svg)

Again, the Bayes factor should favor a fit of the model which generated
the data over a fit of the other model. We'll again use bridge sampling
to estimate the marginal probabilities of each of the four fits so that
we can estimate the Bayes factors.

``` r
# Perform bridge sampling for each model
bridge_gp_gp = bridge_sampler(fit_gp_to_gp)
bridge_hmm_gp = bridge_sampler(fit_hmm_to_gp)
bridge_gp_hmm = bridge_sampler(fit_gp_to_hmm)
bridge_hmm_hmm = bridge_sampler(fit_hmm_to_hmm)
```

The bridge-sampling-estimated Bayes factor favored the Gaussian process
fit to data generated by the Gaussian process.

``` r
lbf1 = bf(bridge_gp_gp, bridge_hmm_gp, log=TRUE)
cat(sprintf("LBF of GP over HMM on GP-generated data: %0.3g\n", 
            lbf1$bf))
```

    LBF of GP over HMM on GP-generated data: 1.57e+03

Conversely, the bridge-sampling-estimated Bayes factor favored the
hidden Markov model fit to data generated by the hidden Markov model.

``` r
lbf2 = bf(bridge_hmm_hmm, bridge_gp_hmm, log=TRUE)
cat(sprintf("LBF of HMM over GP on HMM-generated data: %0.3g\n", 
            lbf2$bf))
```

    LBF of HMM over GP on HMM-generated data: 131


## Original Computing Environment

``` r
writeLines(readLines(file.path(Sys.getenv("HOME"), ".R/Makevars")))
```

    CXXFLAGS=-O3 -Wno-unused-variable -Wno-unused-function

``` r
devtools::session_info()
```

    Session info --------------------------------------
     
     setting  value                       
     version  R version 3.5.1 (2018-07-02)
     system   x86_64, mingw32             
     ui       RTerm                       
     language (EN)                        
     collate  English_United States.1252  
     tz       America/Chicago             
     date     2018-11-21
     
    Packages ------------------------------------------
     
     package        * version date       source        
     assertthat       0.2.0   2017-04-11 CRAN (R 3.5.1)
     backports        1.1.2   2017-12-13 CRAN (R 3.5.0)
     base           * 3.5.1   2018-07-02 local         
     bayesplot      * 1.6.0   2018-08-02 CRAN (R 3.5.1)
     bindr            0.1.1   2018-03-13 CRAN (R 3.5.1)
     bindrcpp       * 0.2.2   2018-03-29 CRAN (R 3.5.1)
     bridgesampling * 0.5-2   2018-08-19 CRAN (R 3.5.1)
     Brobdingnag      1.2-6   2018-08-13 CRAN (R 3.5.1)
     coda             0.19-1  2016-12-08 CRAN (R 3.5.1)
     codetools        0.2-15  2016-10-05 CRAN (R 3.5.1)
     colorspace       1.3-2   2016-12-14 CRAN (R 3.5.1)
     compiler         3.5.1   2018-07-02 local         
     crayon           1.3.4   2017-09-16 CRAN (R 3.5.1)
     datasets       * 3.5.1   2018-07-02 local         
     devtools         1.13.6  2018-06-27 CRAN (R 3.5.1)
     digest           0.6.16  2018-08-22 CRAN (R 3.5.1)
     dplyr            0.7.6   2018-06-29 CRAN (R 3.5.1)
     evaluate         0.11    2018-07-17 CRAN (R 3.5.1)
     ggplot2        * 3.0.0   2018-07-03 CRAN (R 3.5.1)
     ggridges         0.5.0   2018-04-05 CRAN (R 3.5.1)
     glue             1.3.0   2018-07-17 CRAN (R 3.5.1)
     graphics       * 3.5.1   2018-07-02 local         
     grDevices      * 3.5.1   2018-07-02 local         
     grid             3.5.1   2018-07-02 local         
     gridExtra        2.3     2017-09-09 CRAN (R 3.5.1)
     gtable           0.2.0   2016-02-26 CRAN (R 3.5.1)
     htmltools        0.3.6   2017-04-28 CRAN (R 3.5.1)
     inline           0.3.15  2018-05-18 CRAN (R 3.5.1)
     invgamma       * 1.1     2017-05-07 CRAN (R 3.5.0)
     knitr            1.20    2018-02-20 CRAN (R 3.5.1)
     labeling         0.3     2014-08-23 CRAN (R 3.5.0)
     lattice          0.20-35 2017-03-25 CRAN (R 3.5.1)
     lazyeval         0.2.1   2017-10-29 CRAN (R 3.5.1)
     magrittr         1.5     2014-11-22 CRAN (R 3.5.1)
     Matrix           1.2-14  2018-04-13 CRAN (R 3.5.1)
     memoise          1.1.0   2017-04-21 CRAN (R 3.5.1)
     methods        * 3.5.1   2018-07-02 local         
     munsell          0.5.0   2018-06-12 CRAN (R 3.5.1)
     mvtnorm          1.0-8   2018-05-31 CRAN (R 3.5.0)
     parallel         3.5.1   2018-07-02 local         
     pillar           1.3.0   2018-07-14 CRAN (R 3.5.1)
     pkgconfig        2.0.2   2018-08-16 CRAN (R 3.5.1)
     plyr             1.8.4   2016-06-08 CRAN (R 3.5.1)
     purrr            0.2.5   2018-05-29 CRAN (R 3.5.1)
     R6               2.2.2   2017-06-17 CRAN (R 3.5.1)
     RColorBrewer   * 1.1-2   2014-12-07 CRAN (R 3.5.0)
     Rcpp             0.12.18 2018-07-23 CRAN (R 3.5.1)
     reshape2         1.4.3   2017-12-11 CRAN (R 3.5.1)
     rlang            0.2.2   2018-08-16 CRAN (R 3.5.1)
     rmarkdown        1.10    2018-06-11 CRAN (R 3.5.1)
     rprojroot        1.3-2   2018-01-03 CRAN (R 3.5.1)
     rstan          * 2.17.3  2018-01-20 CRAN (R 3.5.1)
     scales           1.0.0   2018-08-09 CRAN (R 3.5.1)
     StanHeaders    * 2.17.2  2018-01-20 CRAN (R 3.5.1)
     stats          * 3.5.1   2018-07-02 local         
     stats4           3.5.1   2018-07-02 local         
     stringi          1.1.7   2018-03-12 CRAN (R 3.5.0)
     stringr          1.3.1   2018-05-10 CRAN (R 3.5.1)
     tibble           1.4.2   2018-01-22 CRAN (R 3.5.1)
     tidyselect       0.2.4   2018-02-26 CRAN (R 3.5.1)
     tools            3.5.1   2018-07-02 local         
     utils          * 3.5.1   2018-07-02 local         
     withr            2.1.2   2018-03-15 CRAN (R 3.5.1)
     yaml             2.2.0   2018-07-25 CRAN (R 3.5.1)