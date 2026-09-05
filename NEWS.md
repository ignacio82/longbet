# longbet 0.4.0

## Fixed: the propensity score was accepted and ignored

`longbet()` has always taken a `ps` argument. The code that used it was
commented out, so passing one changed nothing -- verifiably: fits with and
without `ps` returned bit-identical draws. That matters more than a dead
argument usually would, because conditioning the prognostic forest on an
estimate of `pi(x)` is precisely what separates Bayesian Causal Forest from
BART on a panel. Without it, strong confounding plus tree shrinkage produces
regularization-induced confounding, and the observational setting that LongBet
is *advertised for* is exactly where that bites.

`ps` is now appended to the prognostic covariates, after the continuous columns
and before the categorical block so `pcat` keeps its meaning. It is never given
to the treatment forest, which is the BCF design. `predict.longbet()` rebuilds
the column for you when you predict on the training rows and asks for `ps`
when you predict on new ones.

**Measured** on a panel version of the Hahn/Murray/Carvalho targeted-selection
design, where adoption probability is driven by the prognostic surface itself
(`cor(pi, mu) = 0.62`) and that surface is non-linear:

| prognostic covariates | CATT RMSE |
|---|---|
| no propensity score | 0.201 |
| true `pi(x)` | **0.166** |
| estimated, correctly specified | 0.196 |
| estimated, misspecified (linear in x) | 0.226 |

The true score is worth about 17%. An estimate is worth having only if it is a
good one: a logistic fit that misses the non-linearity in `mu` made things
*worse* than omitting it. That is the standard caveat about propensity
adjustment and it survives intact here.

## Fixed: getTaus() and getMus() were exported and broken

Both raised an error on every fit. They read `fit$tauhats.adjusted`, which the
C++ side returns as `NULL`. They could not have worked from `fit$tauhats`
either: that field holds the raw treatment-forest output `nu(X, S)` on the
standardized scale, not the effect. The effect is the contrast
`b1 * beta_S * nu(X, S) - b0 * beta_0 * nu(X, 0)`, which needs the forest
evaluated at `S = 0` as well and is what `predict.longbet()` assembles.

Both now take a *prediction* rather than a fit, and return what their names
say. Handing one a fit gives an error that says to call `predict.longbet()`
first, rather than a shape error from `rowMeans`.

## Fixed: the package printed to stdout and could not be silenced

`longbet()` and `predict.longbet()` reported input coercions, the defaulted
`pcat_trt`, and every Gaussian process extrapolation through `cat()` and
`print()`. Neither is suppressible with `suppressMessages()`, so any loop, any
knitted document and any test suite was flooded. They are now `message()` calls
behind a new `verbose` argument, default `FALSE`.

`verbose` deliberately does *not* switch on the C++ sampler's per-tree trace,
which is a genuine firehose -- one block per tree per sweep. That is
`verbose_sampler`, also default `FALSE`.

## New: `summary_only` for predictions

`predict.longbet()` returns three `[n x t x draws]` arrays. On the 3,000-seller
panel in the LongBet chapter that is 14 MB; at a hundred thousand units it is
the reason the session dies. `summary_only = TRUE` returns the posterior mean
and an `alpha`-level interval instead and drops the draws -- 13.9 MB to 0.5 MB
on that panel. `get_catt()` understands the summarised object; `get_att()`
needs the draws and says so.

The summary fields are `tau_summary` and `mu0_summary`, deliberately not
`tauhats_summary`: R's `$` does partial matching on lists, so the latter would
make `pred$tauhats` silently resolve to a list after the draws were dropped.

## Cleanup: the namespace

`NAMESPACE` used `exportPattern("^[[:alpha:]]+")`, which made `check_scalar`,
`sample_int_crank`, `longbet_cpp`, `predict_beta` and eight other internals
part of the public API. It is now an explicit list of eight functions, with
`predict.longbet` registered as an S3 method as well as exported, since
existing code -- including the package's own examples -- calls it directly.

# longbet 0.3.1

## Fixed: a unit treated in the first observed period crashed the sampler

`longbet()` segfaulted, on this fork and on `google/longbet`, whenever any unit
was already treated in the first period of the panel. Two buffers were sized by
the number of periods when they had to be sized by the number of distinct
times-since-adoption, and those differ by one in exactly that case:
`get_trt_time()` dates such a unit's adoption one step before the panel starts,
so `beta_size` becomes `p_y + 1`.

* The per-sweep `beta` draw buffer was `ini_matrix(beta_xinfo, p_y, ...)`, one
  element short of the `std::copy` that filled it.
* The Gaussian process kernel was built from the unique values of `t_mod`, but
  `update_time_coef()` indexes it by `s_values.size()`. `t_mod` loses its zero
  entry in this case, so the kernel came out one row short of what was read
  from it. The kernel is now built on the same grid it is indexed by, which is
  bit-for-bit identical whenever the two grids agree -- that is, on every panel
  with at least one period in which nobody is treated yet.

A panel in which *every* unit is treated in *every* period identifies nothing
and is now an error rather than a crash.

# longbet 0.3.0

Five further changes, each one recommended, implemented, and then *measured*.
Two of them did not survive the measurement, and are reported here as such
rather than quietly shipped as improvements.

## New: a constant mean for the beta_S Gaussian process

`gp_constant_mean = TRUE` (now the default) gives the process over
time-since-adoption a constant mean estimated alongside everything else,
instead of a mean of zero.

**Why.** It only matters when extrapolating, and there it matters a lot. A
zero-mean process reverts to *no treatment effect* once it is far enough past
the observed window. That is a statement about the prior, not about the data,
and it is rarely what anyone believes. With a constant mean the projection
reverts to the level the observed trajectory settled at.

**Measured**, on the staggered rollout from the LongBet chapter, projecting six
weeks past a fourteen-week window whose true effect is flat near 10.7%:

| lengthscale | RMSE, zero mean | RMSE, constant mean |
|---|---|---|
| 2  | 0.067 | **0.026** |
| 5  | 0.031 | **0.016** |
| 8  | **0.017** | 0.027 |
| 12 | **0.010** | 0.023 |
| 20 | **0.048** | 0.105 |

The pathology is gone: at short lengthscales the zero-mean process forgets the
data within a couple of periods and falls to zero, and the constant mean stops
it. At long lengthscales the projection is data-dominated anyway and the extra
pull is a cost, not a benefit. Average accuracy across the grid is a wash. It
is the default because the failure it removes is systematic and severe while
the one it introduces is mild, but `gp_constant_mean = FALSE` restores the old
behaviour and is the better choice if you extrapolate with a long lengthscale.
In-window results are unchanged either way.

## New: unbalanced panels

`y` may now contain `NA`. Cells with no observed outcome are drawn from their
own full conditional at the top of every sweep, so the tree-growing code, the
sufficient statistics, the variance draws and the Gaussian process update all
continue to see a rectangular panel and none of them needed to change.
Alternating `y_mis | theta` with `theta | y_obs, y_mis` is an ordinary Gibbs
sampler on the joint posterior, so the extra uncertainty is carried rather than
hidden. It is correct under missing at random. Units with no observed period at
all are an error, not a warning.

**Measured**: deleting 15% of the panel at random moves the ATT RMSE from
0.0079 to 0.0086 and the residual SD from 0.2814 to 0.2813 against a truth of
0.28. Dropping an eighth of the data costs almost nothing, which is the point:
the alternative -- casewise deletion of whole units -- costs a great deal.

## New: binary outcomes

`outcome = "binary"` fits a probit by Albert-Chib augmentation. A latent normal
is drawn each sweep, truncated to the half-line the observation implies, and
sigma is held at 1 because the link is what fixes the scale. The same
`draw_latent_outcome` hook carries this and the missing-data imputation; they
are the same operation seen twice.

The latent is centred at `qnorm(mean(y))` and the forests fit deviations from
it, exactly as they do for a centred continuous outcome. Getting this wrong is
not subtle: an early version left the forests seeded with an intercept equal to
the observed *proportion*, and the fitted probabilities came out 0.62 against
an observed 0.477 and stayed there no matter how long the chain ran.

**Measured**, on a probit DGP with a heterogeneous latent-scale effect of 0.8
for `x1 > 0` and 0.2 otherwise: the estimated conditional effects correlate
0.94 with the truth, the fitted probability on untreated cells is 0.477 against
an observed 0.477, and the counterfactual probability for treated units is
0.512 against a true 0.507.

Effects are returned on the **probit scale**. `predict.longbet()` now also
returns `muhats0`, the fitted outcome with the unit held untreated, so the two
potential outcomes are `pnorm(muhats0)` and `pnorm(muhats0 + tauhats)`.
(`muhats` evaluates the treatment forest at the unit's realised time since
adoption, so for a treated cell it is neither potential outcome. It is retained
unchanged for backwards compatibility.)

## New, and not recommended: AR(1) errors

`ar1_errors = TRUE` adds a transitory autoregressive component,
`y = fitted + gamma_i + u_it + eps_it` with `u_it = rho u_i,t-1 + e_it`. The
latent path is sampled by forward-filter backward-sample, so conditional on it
the forests still see an iid error and the conjugate leaf updates stay exact.
This is why the path is sampled rather than the tree targets quasi-differenced:
after a Prais-Winsten transform the fitted value at `(i,t)` is
`mu(X_i,t) - rho*mu(X_i,t-1)`, a difference of two leaf values that sit in
different leaves as soon as a tree splits on time, and the leaf sufficient
statistics stop being sums over independent observations.

It is off by default, and the measurements are the reason. On a panel with a
true `rho` of 0.7:

| | CATT RMSE | 95% interval containment | recovered rho |
|---|---|---|---|
| `ar1_errors = FALSE` | **0.120** | **0.938** | -- |
| `ar1_errors = TRUE`  | 0.152 | 0.538 | 0.76 |

and on a panel with no serial correlation at all, 0.058 / 0.877 against
0.067 / 0.687. It recovers `rho` accurately in both cases and makes the
conditional treatment effects worse in both cases. More sweeps do not fix it:
400 gives the same picture. The reason is the one that motivated bounding
`rho` in the first place -- a smooth unit-specific path can absorb a
unit-specific treatment effect, and here it does. The average effect survives
(the ATT is within 0.03 of the truth throughout, because it is pooled across
units through `beta_S`); the conditional effects do not.

Use it to *measure* whether your residuals are genuinely autoregressive, and
then turn it off to estimate effects. If the dependence you find is a permanent
per-unit offset rather than a decaying one -- which is the common case, and the
one that costs the most -- `random_intercept` removes it exactly and improves
everything.

## Resolved: the "non-converging sigma and b" note

`state.h` carried a note saying that the residual should use `beta_fit` rather
than `beta_t` but that doing so "leads to non-converging sigma and b values".
The cause was the covariance factorisation fixed in 0.2.0, which pinned `beta`
to its conditional mean. With that repaired, `b_scaling = TRUE` gives a
stationary sigma (sd 0.0006 with no drift across 200 sweeps), `b0` and `b1`
that settle at 0.35 and 0.60, and an ATT that is if anything slightly better
(RMSE 0.0077 covering 14 of 14 event times, against 0.0079 and 13 of 14). The
default is still `FALSE` -- that is one data-generating process, not a
validation -- but the option now works and the note is gone.

## Not done: constraining the beta_S scale

An earlier draft of these notes proposed removing the scale ridge between
`beta_S` and the treatment forest, so that `beta` could be interpreted and its
trace read as a convergence diagnostic. On reflection that is the wrong thing
to do. The treatment term is `b * beta_S * nu(X, S, t)`, and the redundant
scalings are not an oversight -- they are the parameter expansion XBCF adopts
precisely because it improves mixing of the quantities that *are* identified.
Constraining `beta` would fight that, and every reported quantity is already
invariant to the ridge.

What the ridge does require is that nobody diagnose convergence on `beta`. That
is now documented on `longbet()`'s return value, with a pointer to the
identified alternative: `get_att()` returns `att_full`, the average effect by
time-since-adoption for every post-burn-in sweep, and `get_catt()` the same per
unit. On the chapter's fit the coefficient of variation across sweeps is 0.23
for `beta` and 0.08 for the ATT it multiplies out to.

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
