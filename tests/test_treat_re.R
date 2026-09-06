# Unit-level random effects on the treatment effect (treat_effect_re).
#
# The case for them: when a unit's effect departs from what the covariates
# predict, a model with nowhere to put that departure calls it noise, and its
# intervals then miss badly. These tests check that the departure is
# recovered, that coverage is restored, and that the feature is honest about
# what it costs when there is no such departure to find.
suppressMessages(library(longbet))
np <- 0L; nf <- 0L
ok <- function(lab, cond) { if (isTRUE(cond)) { np <<- np+1L; cat("PASS ", lab, "\n") }
                            else { nf <<- nf+1L; cat("FAIL ", lab, "\n") } }

mk <- function(seed, n = 700, Tn = 12, sd_u = 0.25, cor_u = 0.9) {
  set.seed(seed)
  launch <- sample(c(5, 7, 9, Inf), n, TRUE, c(.2, .2, .2, .4))
  z <- outer(seq_len(n), 1:Tn, function(i, j) as.integer(j >= launch[i]))
  S <- t(apply(z, 1, cumsum)); x <- cbind(rnorm(n), rnorm(n), rbinom(n, 1, .5))
  f1 <- 0.30 * plogis(x[, 1]) + 0.05
  f2 <- 0.25 * plogis(x[, 2]) + 0.05
  Lu <- chol(matrix(c(1, cor_u, cor_u, 1), 2, 2))
  u <- (matrix(rnorm(2 * n), ncol = 2) %*% Lu) * sd_u
  tau1 <- ifelse(S > 0, (f1 + u[, 1]) * (1 - exp(-S / 4)), 0)
  tau2 <- ifelse(S > 0, (f2 + u[, 2]), 0)
  y1 <- 2 + .8 * x[, 1] + rnorm(n, 0, .5)[row(S)] + tau1 + matrix(rnorm(n * Tn, 0, .4), n, Tn)
  y2 <- 1 + .6 * x[, 2] + rnorm(n, 0, .4)[row(S)] + tau2 + matrix(rnorm(n * Tn, 0, .4), n, Tn)
  list(y = list(y1, y2), x = x, z = z, t = 1:Tn, tau1 = tau1, tau2 = tau2, u = u)
}
base <- function(d, ...) c(list(x = d$x, x_trt = d$x, z = d$z, t = d$t, pcat = 1,
  num_sweeps = 70, num_burnin = 23, num_trees_pr = 20, num_trees_trt = 20,
  random_intercept = TRUE, verbose = FALSE), list(...))
fit2 <- function(d, re, seed = 7)
  do.call(longbet, base(d, y = d$y[[2]], treat_effect_re = re, random_seed = seed))
sc <- function(f, d, seed = 7) {
  p <- predict.longbet(f, d$x, d$x, d$z, t = d$t, random_seed = seed)
  tr <- d$z == 1
  list(rmse = sqrt(mean((apply(p$tauhats, c(1,2), mean)[tr] - d$tau2[tr])^2)),
       cover = mean(apply(p$tauhats, c(1,2), quantile, .025)[tr] <= d$tau2[tr] &
                    apply(p$tauhats, c(1,2), quantile, .975)[tr] >= d$tau2[tr]),
       pred = p)
}

cat("== with real unit-level effect heterogeneity ==\n")
d  <- mk(1)
f0 <- fit2(d, FALSE); s0 <- sc(f0, d)
f1 <- fit2(d, TRUE);  s1 <- sc(f1, d)
ok("the flag is recorded on the fit", isTRUE(f1$treat_effect_re) && !isTRUE(f0$treat_effect_re))
ok("delta_draws is one row per unit",
   is.matrix(f1$delta_draws) && nrow(f1$delta_draws) == nrow(d$x))
ok(sprintf("without it, coverage collapses (%.2f)", s0$cover), s0$cover < 0.60)
ok(sprintf("with it, coverage is restored (%.2f)", s1$cover), s1$cover > 0.80)
ok(sprintf("and RMSE improves (%.4f from %.4f)", s1$rmse, s0$rmse), s1$rmse < s0$rmse)
post <- (f1$model_params$burnin + 1):ncol(f1$delta_draws)
dh <- rowMeans(f1$delta_draws[, post])
treated <- rowSums(d$z) > 0
ok(sprintf("the unit effects correlate with the truth (%.2f)",
           cor(dh[treated], d$u[treated, 2])), cor(dh[treated], d$u[treated, 2]) > 0.5)
ok(sprintf("sigma_delta is near the true 0.25 (%.2f)", f1$sigma_delta),
   abs(f1$sigma_delta - 0.25) < 0.12)

cat("\n== the estimand, with and without the unit part ==\n")
p_no <- predict.longbet(f1, d$x, d$x, d$z, t = d$t, random_seed = 7,
                        add_unit_effects = FALSE)
ok("add_unit_effects = FALSE returns the covariate-only effect",
   !isTRUE(p_no$unit_effects_added) && isTRUE(s1$pred$unit_effects_added))
ok("and that effect is less variable across units",
   sd(apply(p_no$tauhats, c(1,2), mean)[d$z == 1]) <
   sd(apply(s1$pred$tauhats, c(1,2), mean)[d$z == 1]))

cat("\n== when there is nothing to find ==\n")
dn <- mk(2, sd_u = 0)
n0 <- sc(fit2(dn, FALSE), dn); n1 <- sc(fit2(dn, TRUE), dn)
ok(sprintf("sigma_delta stays small (%.3f)", fit2(dn, TRUE)$sigma_delta),
   fit2(dn, TRUE)$sigma_delta < 0.15)
ok(sprintf("coverage is not broken (%.2f)", n1$cover), n1$cover > 0.90)
# The documented cost: turning it on when it is not needed is not free.
ok(sprintf("the cost is bounded (%.4f vs %.4f)", n1$rmse, n0$rmse),
   n1$rmse < n0$rmse * 1.8)

cat("\n== multi-outcome ==\n")
fm <- do.call(longbet_multi, base(d, y = d$y, outcome = "continuous",
                                  treat_effect_re = TRUE, random_seed = 5))
ok("a joint fit carries unit effects for every outcome",
   all(vapply(fm$fits, function(f) is.matrix(f$delta_draws), logical(1))))
cc <- effect_correlation(fm)
ok("effect_correlation returns an M x M correlation matrix",
   identical(dim(cc), c(2L, 2L)) && abs(cc[1,1] - 1) < 1e-8)
ok(sprintf("it is positive when the truth is positive (%.2f, attenuated)", cc[1,2]),
   cc[1,2] > 0.05)
ok("it is refused on a fit without unit effects",
   inherits(tryCatch(effect_correlation(
     do.call(longbet_multi, base(d, y = d$y, outcome = "continuous",
                                 random_seed = 5))),
     error = function(e) e), "error"))

cat(sprintf("\n%d passed, %d failed\n", np, nf))
if (nf > 0) quit(status = 1)
