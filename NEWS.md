# longbet 0.2.0

This release is a fork of [google/longbet](https://github.com/google/longbet)
at commit `5f85c32`. It adds unit-level random intercepts, fixes two defects in
how the shared time factor is sampled, and makes the forecast reproducible.

## New: unit-level random intercepts

`longbet(..., random_intercept = TRUE)` (now the default) adds

```
y_it = a * mu(X_i, t) + b_z * beta_S * nu(X_i, S_it, t) + gamma_i + eps_it
gamma_i ~ N(0, sigma_gamma^2),   sigma_gamma^2 ~ InvGamma(a, b)
```

**Why.** LongBet has no unit fixed or random effects, so every unit's
persistent level has to be reconstructed by the prognostic forest from the
time-invariant covariates `x`. Whatever the covariates cannot reach stays in
the residual, where it does three kinds of damage: it inflates `sigma`, it
leaves within-unit correlation in errors the likelihood assumes are
independent, and it spends tree capacity memorising individual units instead
of learning population structure.

**How.** Conditional on the forests, the residual `y - a*mu - b*beta*tau`
equals `gamma_i + eps_it`, so each `gamma_i` has an exact independent Gaussian
full conditional and `sigma_gamma^2` an exact inverse-gamma one. Both are drawn
by conjugate Gibbs at the end of each sweep, at `O(n * T)` cost. Internally the
sampler keeps `y_work = y - gamma_i` and hands every other component a pointer
to it, so the forests, the sigma draws, the `a`/`b` scalings and the Gaussian
process update all condition on the right quantity without any change to their
own code.

Unlike within-unit demeaning, `gamma_i` is drawn *conditional on the current
treatment fit*, so a unit's post-treatment periods do not drag its baseline
upward — the staggered-adoption contamination that makes upfront demeaning
unusable here. Units treated in every observed period carry no information
separating `gamma_i` from `tau`; `longbet()` now warns when it sees any.

**Measured effect** on the 3,000-seller staggered-rollout simulation from
[Business Data Science](https://book.martinez.fyi/longbet.html), where the true
residual SD is 0.28:

| covariates in `x`      | `random_intercept` | residual SD | ATT RMSE | 95% interval contains truth |
|------------------------|--------------------|-------------|----------|------------------------------|
| level and trend summaries plus structural | `FALSE` | 0.302 | 0.0081 | 14 of 14 |
| level and trend summaries plus structural | `TRUE`  | 0.281 | 0.0067 | 14 of 14 |
| structural only        | `FALSE`            | 0.701       | 0.0374   | 2 of 14                      |
| structural only        | `TRUE`             | 0.282       | 0.0084   | 14 of 14                     |

The last two rows are the point: without the random intercept, dropping the
hand-built lookback summaries multiplies the ATT RMSE by 4.6 and destroys
interval coverage. With it, the model recovers the true residual SD and the
correct answer from structural covariates alone. `sigma_gamma` shrinks toward
zero on its own when the covariates already explain the level, so turning it on
costs little when it is not needed.

New arguments: `random_intercept`, `gamma_prior_a`, `gamma_prior_b`.
New outputs: `gamma_draws` (`n` by `num_sweeps`, on the scale of `y`),
`sigma_gamma_draws`, `random_intercept`.

## Fixed: covariance factorisation of the time-factor draws

`update_time_coef()` and `predict_beta()` both drew from a multivariate normal
using `L = U * diag(s)` from the SVD of the covariance matrix. A factor has to
satisfy `L L' = var`, which requires `L = U * diag(sqrt(s))`. The old code
sampled with covariance `U diag(s^2) U'`.

Consequences, both now fixed:

* **During training** the conditional variance of `beta_S` is on the order of
  `sigma^2 / n`, well below one, so squaring it collapsed the draw onto its
  conditional mean. `beta` was effectively not being sampled, and the ATT
  credible intervals inherited the missing variance. On the simulation above,
  the fix widens the mean event-time interval from 0.019 to 0.034 and lifts
  the number of event times whose interval contains the truth from 11 of 14 to
  14 of 14.
* **When extrapolating** the effect was inverted for eigenvalues above one and
  extreme for a small `sigma`, which made the projection's coverage swing
  wildly with the prior scale. With the fix, the projection contains the truth
  at all six withheld horizons for every lengthscale tried at `sigma = 1`.

## Fixed: the amount of regularization depended on the units of `y`

`longbet()` computed the leaf-variance priors for both forests from the raw
outcome,

```r
tau_con = 0.6 * var(as.vector(y)) / num_trees_pr
tau_mod = 0.1 * var(as.vector(y)) / num_trees_trt
```

and then standardized `y` to unit variance before handing it to the sampler.
The priors were therefore off by a factor of `var(y)`: measure the same outcome
in dollars rather than log dollars and the forests get a leaf prior larger by
six orders of magnitude. The `0.6` and `0.1` are meant to be the *share of
outcome variance* each forest may use, which is only what they are on the
standardized scale. They are now computed after standardization, and can be
overridden through the new `tau_pr` and `tau_trt` arguments.

This is a correctness fix rather than an accuracy fix, and the distinction
matters. On a well-conditioned problem the point estimate barely moves. Where
it bites is where the prior binds — a modest heterogeneous effect against a
noisy baseline. Rescaling the *same* outcome, with everything else held fixed:

| scale applied to `y` | CATT RMSE, 0.1.2 | CATT RMSE, 0.2.0 |
|----------------------|------------------|------------------|
| 0.01                 | 0.315            | 0.263            |
| 1                    | 0.230            | 0.257            |
| 100                  | 0.189            | 0.266            |

(the true conditional effects have standard deviation 0.3). The old column
varies by two thirds across a change of units that carries no information;
whether a given analysis was over- or under-regularized was a property of the
unit it happened to be measured in. Note that the old behaviour is *better* at
`scale = 100` here, because this effect is genuinely heterogeneous and the
accidental under-shrinkage helped — which is the point: it was arbitrary, not
conservative. The new column is invariant up to floating point. Set `tau_trt`
explicitly if you want to move away from the default deliberately.

## New: `random_seed`, and the previously hidden hyperparameters

`longbet()` hardcoded `random_seed = 0`, so every fit on the same data returned
the identical answer and there was no way to run independent chains or measure
Monte Carlo error. `random_seed` is now an argument. So are `max_depth`,
`num_cutpoints`, `a_scaling`, `b_scaling`, `parallel` and `verbose`, all of
which were fixed inside the function body under a comment reading "deprecated
hyperparameters for user experience". Defaults are unchanged, so this is
backwards compatible.

`num_cutpoints` is the one most worth revisiting: it defaults to 20 candidate
splits per node, which is coarse for continuous covariates with a lot of
structure.

## Fixed: reproducibility of the projection

`predict_beta()` seeded a Mersenne twister from `std::random_device`, so the
extrapolated region of a prediction changed on every call. `predict.longbet()`
gains `random_seed` (default `1`); pass `NULL` for the old nondeterministic
behaviour. In-window predictions were and remain deterministic.

## Fixed: `predict.longbet(t = NULL)`

That branch read `longbet.fit$time` from the global environment rather than
`model$time`, so it failed with `object 'longbet.fit' not found` unless the
caller happened to have named the fitted object `longbet.fit` — as the
package's own examples do. It now reads the model it was handed.

## Considered and not adopted: AR(1) errors

A companion proposal suggested adding `u_it = rho * u_i,t-1 + eps_it` on top of
the random intercept, whitened by a Prais-Winsten quasi-difference of the tree
targets. The random intercept is implemented here; the AR(1) part is not, for
two reasons.

Quasi-differencing the forest targets breaks the property that makes XBART
fast. After the transform, the model's prediction at observation `(i, t)` is
`mu(X_i, t) - rho * mu(X_i, t-1)`, a difference of two leaf values that may sit
in *different* leaves once the tree splits on time. The leaf sufficient
statistics are then no longer sums over independent observations assigned to
that leaf, and the conjugate normal leaf update and split scoring are no longer
exact — contrary to the proposal's claim. Doing it correctly means sampling the
latent AR(1) path per unit and conditioning the forests on `y - gamma - u`,
which is exact but introduces a free latent value at every unit-period; with
`rho` near one that path is a random walk that can absorb the treatment effect
itself.

The permanent component is also the expensive one. A unit-level offset leaves
*every* pair of periods equally correlated, so its design effect grows roughly
like `1 + (T-1) * rho`, while an AR(1) contribution decays with the lag. On the
simulation above the random intercept alone takes the estimated residual SD to
0.281 against a true 0.28, which leaves nothing for an AR(1) term to remove.
That simulation has no serial correlation in it by construction, so it cannot
settle whether the AR(1) extension would pay on real data; the reason for
leaving it out is the first one, not the second.
