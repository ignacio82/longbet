# Bayesian Ensemble Trees for Causal Inference on Longitudinal Data (LongBet)

> **This is a fork of [google/longbet](https://github.com/google/longbet).**
> It adds time-varying covariates, unit-level random intercepts,
> unbalanced-panel support, binary (probit) outcomes and a non-zero-mean
> Gaussian process; it wires up the propensity score argument that upstream
> accepts and ignores; it fixes the
> covariance factorisation used when sampling and extrapolating the shared time
> factor, makes regularization independent of the units of the outcome, and
> makes the projection reproducible. [NEWS.md](NEWS.md) has the rationale and
> the measurements for each, including the two changes that were implemented,
> measured, and then left switched off because they made things worse. Worked
> example: [Business Data Science, LongBet chapter](https://book.martinez.fyi/longbet.html).

## About

This package implements the Bayesian Ensemble Trees for Causal Inference on 
Longitudinal Data for estimating time-varying conditional average treatment effect 
estimation; the manuscript will be available soon. 
This approach builds on the methodology behind Bayesian Causal Forests outlined 
in [Hahn et al.](https://projecteuclid.org/euclid.ba/1580461461) (2020) and 
incorporates several improvements to Bayesian Additive Regression Trees 
implemented by [He et al.](http://proceedings.mlr.press/v89/he19a.html) (2019).

This package is based on the source code of the [XBCF](https://github.com/socket778/XBCF) 
package.

## Installation
It can be installed from GitHub directly using the devtools package in R.

```R
library(devtools)
install_github("ignacio82/longbet")
```

## Usage
```R
longbet(y, x, x_trt, z, t, pcat,
num_sweeps = 60, num_burnin = 20,
num_trees_pr = 20, num_trees_trt = 20,
mtry = 0L, n_min = 10,
sig_knl = 1, lambda_knl = 1,
split_time_ps = TRUE, split_time_trt = TRUE,
random_intercept = TRUE,
gamma_prior_a = 1, gamma_prior_b = 0.1,
tau_pr = NULL, tau_trt = NULL,
max_depth = 50, num_cutpoints = 20,
a_scaling = TRUE, b_scaling = FALSE,
random_seed = 0, parallel = TRUE, verbose = FALSE,
outcome = c("continuous", "binary"),
gp_constant_mean = TRUE,
ar1_errors = FALSE, rho_max = 0.95, sigma_u_init = 0.2,
ps = NULL, x_tv = NULL, x_tv_trt = NULL)
```

### Arguments
y: An n by t matrix of outcome variables.
x: n by p input matrix of covariates for the prognostic term. **Continuous columns must come first and categorical columns last**; `pcat` says how many of the trailing columns are categorical.
x_trt: n by p_trt input matrix of covariates for the treatment term. May be the same matrix as x.
z: An n by t matrix of treatment assignments.
t: time variable (post-treatment time for the treatment term is inferred from t and z).
pcat: The number of categorical inputs in matrix x.
num_sweeps: The total number of sweeps over data (default is 60).
num_burnin: The number of burn-in sweeps (default is 20).
num_trees_pr: The number of trees in the prognostic forest (default is 20).
num_trees_trt: The number of trees in the treatment forest (default is 20).
mtry: number of variables to be sampled as split candidate per tree.
n_min: The minimum node size. (default is 10)
sig_knl: variance parameter for squared exponential kernel (default is 1).
lambda_knl: lengthscale parameter for squared exponential kernel (default is 1).
split_time_trt: whether the treatment forest may split on time since adoption. TRUE lets different units have different effect *shapes*; FALSE gives every unit the same shape up to a scale factor, which extrapolates more cleanly.
random_intercept: whether to draw unit-level random intercepts (default TRUE, added in this fork). See NEWS.md.
gamma_prior_a, gamma_prior_b: inverse-gamma shape and rate for the random-intercept variance, on the standardized outcome scale.

random_seed: seed for the sampler. `longbet()` ignores R's RNG state, so this is the only way to get independent runs; vary it to run several chains and measure Monte Carlo error.
tau_pr, tau_trt: leaf variance priors on the **standardized** outcome scale (defaults 0.6/num_trees_pr and 0.1/num_trees_trt).

outcome: "continuous" (Gaussian) or "binary" (probit by Albert-Chib augmentation; effects come back on the probit scale).
gp_constant_mean: give the time-factor Gaussian process a constant mean estimated from the data rather than a mean of zero. Only matters when extrapolating, where a zero mean makes long projections revert to "no effect" because the prior says so.
ar1_errors: add a transitory AR(1) error on top of the unit intercept. Off by default, and see NEWS.md before turning it on -- it recovers rho accurately and makes conditional effects worse.

x_tv, x_tv_trt: time-varying covariates for the prognostic and treatment forests. A list of n by t matrices laid out like y, one such matrix, or an n by t by k array. Covariates in x stay time-invariant; these are the ones that are not. Must be observed in every cell, including cells where y is NA, and must be passed to predict.longbet() as well.
ps: estimated propensity scores, one per unit, appended to the *prognostic* covariates only. This is the BCF adjustment for targeted selection, and it was silently ignored through 0.3.1. Worth about 17% on CATT RMSE with a good score; worse than nothing with a badly specified one.

`y` may contain `NA` for periods in which a unit was not observed; those cells
are drawn from their full conditional each sweep rather than dropped.

The package is quiet by default. `verbose = TRUE` reports input coercions and
defaulted arguments through `message()`; `verbose_sampler = TRUE` turns on the
C++ per-tree trace.

`predict.longbet(..., summary_only = TRUE)` returns posterior means and
intervals instead of the `[n x t x draws]` arrays, which are the dominant
memory cost on a large panel.

The Gaussian process projection in `predict.longbet()` is a draw, not a
calculation; pass its own `random_seed` there to fix it. `predict.longbet()`
returns `muhats0` (the fitted outcome with the unit held untreated) alongside
`tauhats`, so the two potential outcomes are `muhats0` and
`muhats0 + tauhats` -- or `pnorm()` of each for a binary fit.

Do not read a trace of `beta_values` as a convergence diagnostic: the treatment
term is `b * beta_S * nu(X, S, t)` and only the product is identified. Use
`get_att()$att_full`, which is the average effect by time-since-adoption for
every post-burn-in sweep.

### See Also

### Example
```R
require(longbet)

set.seed(1)
n <- 100
t1 <- 4
t0 <- 3

# generate dcovariates
x1 <- rnorm(n)
x2 <- sample(1:3,n,replace=TRUE, prob = c(0.4,0.3,0.3))
x <- cbind(x1, x2)

# untreated outcome
mu <- outer(x1 * x2 , rnorm(t1, 5), '*')
# treatment effect
te <- outer(x1 + x2, rnorm(t1, 1), '*')

# generate treatment
z <- rbinom(n,1,0.6)
z_mat <- cbind(matrix(0, n, (t0 - 1)),  matrix(rep(z, t1 - t0 + 1), n, t1 - t0 + 1))

# generate observations
y0 <- mu + 0.2 * sd(mu) * matrix(rnorm(n * t1), n, t1)
y1 <- y0 + te
y <- z_mat * y1 + (1 - z_mat) * y0

t_longbet <- proc.time()
longbet.fit <- longbet(y = y, x = x, x_trt = x, z = z_mat, t = 1:t1, pcat = 1,
num_trees_pr =  50, num_trees_trt = 50, random_intercept = TRUE)

longbet.pred <- predict.longbet(longbet.fit, x, x, z_mat, t = 1:t1, random_seed = 1)
longbet.att <- get_att(longbet.pred, alpha = 0.05)
longbet.catt <- get_catt(longbet.pred, alpha = 0.05)
t_longbet <- proc.time() - t_longbet

ate <- colMeans(te)
print(paste0("longbet CATT RMSE: ", sqrt(mean((as.vector(longbet.catt$catt[z_mat==1]) - as.vector(te[z_mat==1]))^2))))
print(paste0("longbet ATT RMSE: ", sqrt(mean((as.vector(longbet.att$att) - as.vector(ate[t0:t1]))^2))))
print(paste0("longbet runtime: ", round(as.list(t_longbet)$elapsed,2)," seconds"))

# Unit random intercepts, on the scale of y, averaged over post-burnin sweeps
post <- (longbet.fit$model_params$burnin + 1):ncol(longbet.fit$gamma_draws)
gamma_hat <- rowMeans(longbet.fit$gamma_draws[, post])
print(paste0("sd of estimated unit effects: ", round(sd(gamma_hat), 3)))
```

## Tests

```R
Rscript tests/test_random_intercept.R
```
