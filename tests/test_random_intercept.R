# Regression tests for the fork's additions:
#   1. random_intercept recovers unit levels the covariates cannot explain
#   2. random_intercept = FALSE leaves the sampler untouched
#   3. the Gaussian process projection is reproducible under a fixed seed
#   4. units treated in every period raise a warning
#
# Run with:  Rscript tests/test_random_intercept.R

require(longbet)
set.seed(20260905)

n  <- 800
Tn <- 12
t0 <- 7                                    # first treated period

x1 <- rnorm(n)                             # observed, drives the level
x2 <- rbinom(n, 1, 0.5)                    # observed, categorical
level_obs   <- 1.5 * x1 + 0.5 * x2
level_unobs <- rnorm(n, 0, 1.2)            # NOT in x: only gamma can catch it
x <- cbind(x1, x2)                         # continuous first, categorical last

z <- cbind(matrix(0L, n, t0 - 1),
           matrix(rep(rbinom(n, 1, 0.5), Tn - t0 + 1), n, Tn - t0 + 1))
S <- t(apply(z, 1, cumsum))
tau_true <- 1.5 * (S > 0)                  # constant effect once treated

y <- matrix(level_obs + level_unobs, n, Tn) +
     matrix(rep(0.05 * (1:Tn), each = n), n, Tn) +
     tau_true + matrix(rnorm(n * Tn, 0, 0.3), n, Tn)

fit_args <- list(y = y, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
                 num_sweeps = 60, num_burnin = 20,
                 num_trees_pr = 20, num_trees_trt = 20)

resid_sd <- function(f) {
  post <- (f$model_params$burnin + 1):ncol(f$sigma0_draws)
  mean(f$sigma0_draws[, post]) * f$sdy
}
att_of <- function(f) {
  p <- predict.longbet(f, x, x, z, t = 1:Tn, random_seed = 1)
  mean(get_att(p)$att)
}

ok <- function(label, cond, extra = "") {
  cat(if (cond) "PASS  " else "FAIL  ", label, " ", extra, "\n", sep = "")
  if (!cond) stop("test failed: ", label, call. = FALSE)
}

# ---- 1. with the random intercept -------------------------------------
f_ri <- do.call(longbet, c(fit_args, list(random_intercept = TRUE)))
post <- (f_ri$model_params$burnin + 1):ncol(f_ri$gamma_draws)
g    <- rowMeans(f_ri$gamma_draws[, post])

ok("gamma recovers the unobserved unit level",
   cor(g, level_unobs) > 0.8,
   sprintf("(cor = %.3f)", cor(g, level_unobs)))
ok("residual sd is near the truth of 0.3",
   abs(resid_sd(f_ri) - 0.3) < 0.1, sprintf("(%.3f)", resid_sd(f_ri)))
ok("ATT is near the true effect of 1.5",
   abs(att_of(f_ri) - 1.5) < 0.15, sprintf("(%.3f)", att_of(f_ri)))
ok("sigma_gamma is positive and near sd(level_unobs) = 1.2",
   abs(mean(f_ri$sigma_gamma_draws[post]) - 1.2) < 0.4,
   sprintf("(%.3f)", mean(f_ri$sigma_gamma_draws[post])))

# ---- 2. without it, gamma is inert and sigma absorbs the level --------
f_no <- do.call(longbet, c(fit_args, list(random_intercept = FALSE)))
ok("gamma_draws are exactly zero when the flag is off",
   all(f_no$gamma_draws == 0))
ok("residual sd inflates without the random intercept",
   resid_sd(f_no) > resid_sd(f_ri) + 0.3,
   sprintf("(%.3f vs %.3f)", resid_sd(f_no), resid_sd(f_ri)))

# ---- 3. reproducible projection ---------------------------------------
z_ext <- cbind(z, matrix(rep(z[, Tn], 4), n, 4))          # 4 periods further
pargs <- list(f_ri, x, x, z_ext, t = 1:(Tn + 4))
a <- do.call(predict.longbet, c(pargs, list(random_seed = 7)))
b <- do.call(predict.longbet, c(pargs, list(random_seed = 7)))
d <- do.call(predict.longbet, c(pargs, list(random_seed = 8)))
ok("same seed gives the same projection", identical(get_att(a)$att, get_att(b)$att))
ok("different seed gives a different projection",
   !identical(get_att(a)$att, get_att(d)$att))

# ---- 4. always-treated units are flagged -------------------------------
z_bad <- z; z_bad[1:3, ] <- 1L
bad_args <- modifyList(fit_args, list(z = z_bad, random_intercept = TRUE,
                                      num_sweeps = 10, num_burnin = 3,
                                      num_trees_pr = 5, num_trees_trt = 5))
w <- tryCatch({ do.call(longbet, bad_args); NULL },
              warning = function(w) conditionMessage(w))
ok("always-treated units raise a warning",
   !is.null(w) && grepl("treated in every period", w))



# ---- 5. the estimator does not depend on the units of y ----------------
cat("\n-- scale invariance --\n")
att_scaled <- function(sc) {
  f <- do.call(longbet, modifyList(fit_args,
        list(y = y * sc, random_intercept = FALSE)))
  mean(get_att(predict.longbet(f, x, x, z, t = 1:Tn, random_seed = 1))$att) / sc
}
a_small <- att_scaled(0.01); a_one <- att_scaled(1); a_big <- att_scaled(1000)
ok("ATT is invariant to rescaling y",
   max(abs(c(a_small, a_big) - a_one)) < 0.01,
   sprintf("(%.4f / %.4f / %.4f)", a_small, a_one, a_big))

# ---- 6. random_seed produces independent fits --------------------------
s1 <- do.call(longbet, modifyList(fit_args, list(random_seed = 1)))
s1b <- do.call(longbet, modifyList(fit_args, list(random_seed = 1)))
s2 <- do.call(longbet, modifyList(fit_args, list(random_seed = 2)))
ok("same seed reproduces the fit", identical(s1$beta_values, s1b$beta_values))
ok("different seed gives an independent fit",
   !identical(s1$beta_values, s2$beta_values))



# ---- 7. unbalanced panel -----------------------------------------------
cat("\n-- unbalanced panel --\n")
y_na <- y
set.seed(31); y_na[sample(length(y_na), round(0.15 * length(y_na)))] <- NA
f_na <- suppressMessages(do.call(longbet,
          modifyList(fit_args, list(y = y_na, random_intercept = TRUE))))
ok("missing cells are counted", f_na$n_missing == sum(is.na(y_na)))
ok("residual sd survives 15% missingness",
   abs(resid_sd(f_na) - 0.3) < 0.12, sprintf("(%.3f)", resid_sd(f_na)))
ok("ATT survives 15% missingness",
   abs(att_of(f_na) - 1.5) < 0.2, sprintf("(%.3f)", att_of(f_na)))
ok("a unit with no observed period is an error",
   inherits(try({ yy <- y; yy[1, ] <- NA
     suppressMessages(do.call(longbet, modifyList(fit_args, list(y = yy)))) },
     silent = TRUE), "try-error"))

# ---- 8. binary outcome (probit) ----------------------------------------
cat("\n-- binary outcome --\n")
set.seed(77)
eta_b <- -0.3 + 0.9 * x1 + 1.2 * (S > 0)
yb <- matrix(rbinom(n * Tn, 1, pnorm(eta_b)), n, Tn)
fb <- longbet(y = yb, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
              outcome = "binary", num_sweeps = 80, num_burnin = 30,
              num_trees_pr = 20, num_trees_trt = 20, random_intercept = FALSE)
pb <- predict.longbet(fb, x, x, z, t = 1:Tn, random_seed = 1)
ok("sigma is held at 1 on the probit scale", all(fb$sigma0_draws == 1))
m0 <- apply(pb$muhats0, c(1, 2), mean)
fitted_rate <- mean(pnorm(m0[z == 0]))
ok("fitted probability matches the observed rate on untreated cells",
   abs(fitted_rate - mean(yb[z == 0])) < 0.03,
   sprintf("(%.3f vs %.3f)", fitted_rate, mean(yb[z == 0])))
ok("latent-scale ATT recovers the true 1.2",
   abs(mean(get_att(pb)$att) - 1.2) < 0.3,
   sprintf("(%.3f)", mean(get_att(pb)$att)))
ok("binary y must be 0/1",
   inherits(try(longbet(y = y, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
                        outcome = "binary", num_sweeps = 5, num_burnin = 2),
                silent = TRUE), "try-error"))

# ---- 9. AR(1) errors recover rho ---------------------------------------
cat("\n-- AR(1) errors --\n")
set.seed(53); rho_true <- 0.7
uu <- matrix(0, n, Tn); uu[, 1] <- rnorm(n, 0, 0.5 / sqrt(1 - rho_true^2))
for (tt in 2:Tn) uu[, tt] <- rho_true * uu[, tt - 1] + rnorm(n, 0, 0.5)
y_ar <- y + uu
f_ar <- longbet(y = y_ar, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
                num_sweeps = 120, num_burnin = 40, num_trees_pr = 20,
                num_trees_trt = 20, random_intercept = TRUE, ar1_errors = TRUE)
ok("rho is recovered", abs(f_ar$rho - rho_true) < 0.15,
   sprintf("(%.2f vs %.2f)", f_ar$rho, rho_true))
ok("rho respects its bound", abs(f_ar$rho) < 0.95)
f_no <- longbet(y = y_ar, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
                num_sweeps = 120, num_burnin = 40, num_trees_pr = 20,
                num_trees_trt = 20, random_intercept = TRUE, ar1_errors = FALSE)
ok("ar1_errors = FALSE leaves rho at zero", f_no$rho == 0)



# ---- 10. a unit treated from the first observed period ------------------
# This used to segfault, on this fork and upstream: beta_size becomes p_y + 1
# when get_trt_time() dates an adoption one step before the panel starts, and
# both the beta draw buffer and the Gaussian process kernel were sized p_y.
cat("\n-- unit treated from period 1 --\n")
z_first <- z; z_first[1:10, ] <- 1L
f_first <- longbet(y = y, x = x, x_trt = x, z = z_first, t = 1:Tn, pcat = 1,
                   num_sweeps = 30, num_burnin = 10, num_trees_pr = 8,
                   num_trees_trt = 8, random_intercept = FALSE)
ok("a unit treated in every period no longer crashes the sampler",
   is.finite(sum(f_first$beta_values)))
ok("beta buffers are sized by beta_size, not by the number of periods",
   nrow(f_first$beta_values) == max(rowSums(z_first)) + 1 &&
   nrow(f_first$beta_draws) == nrow(f_first$beta_values),
   sprintf("(%d rows)", nrow(f_first$beta_values)))
ok("a panel with no untreated observation at all is an error",
   inherits(try(longbet(y = y, x = x, x_trt = x,
                        z = matrix(1L, n, Tn), t = 1:Tn, pcat = 1,
                        num_sweeps = 5, num_burnin = 2), silent = TRUE),
            "try-error"))



# ---- 11. the propensity score is actually used ------------------------
cat("\n-- propensity score --\n")
ps_v <- pnorm(0.7 * x1 - 0.2)
f_ps <- do.call(longbet, modifyList(fit_args, list(ps = ps_v)))
f_no_ps <- do.call(longbet, fit_args)
ok("ps changes the fit (it was silently ignored through 0.3.1)",
   !isTRUE(all.equal(sum(f_ps$beta_values), sum(f_no_ps$beta_values))))
ok("ps enters the prognostic covariates only",
   f_ps$input_var_count$x_con == ncol(x) + 1 &&
   f_ps$input_var_count$x_mod == ncol(x))
ok("predicting on the training rows reuses the stored ps",
   is.finite(sum(get_catt(predict.longbet(f_ps, x, x, z, t = 1:Tn,
                                          random_seed = 1))$catt)))
ok("a wrong-length ps is an error",
   inherits(try(do.call(longbet, modifyList(fit_args, list(ps = ps_v[-1]))),
                silent = TRUE), "try-error"))

# ---- 12. the getters work and point at the right object ----------------
cat("\n-- getters --\n")
p_g <- predict.longbet(f_no_ps, x, x, z, t = 1:Tn, random_seed = 1)
ok("getTaus returns the effect, not the raw forest",
   abs(mean(getTaus(p_g)[z == 1]) - 1.5) < 0.2,
   sprintf("(%.3f vs a true 1.5)", mean(getTaus(p_g)[z == 1])))
ok("getMus returns an n by t matrix", all(dim(getMus(p_g)) == c(n, Tn)))
ok("handing getTaus a fit errors with a pointer to predict()",
   inherits(try(getTaus(f_no_ps), silent = TRUE), "try-error"))

# ---- 13. quiet by default ----------------------------------------------
cat("\n-- console output --\n")
noise <- capture.output({
  q <- do.call(longbet, fit_args)
  invisible(predict.longbet(q, x, x, z, t = 1:Tn, random_seed = 1))
}, type = "output")
ok("nothing is written to stdout", length(noise) == 0,
   sprintf("(%d lines)", length(noise)))
ok("verbose = TRUE still reports",
   length(capture.output(do.call(longbet, modifyList(fit_args,
     list(verbose = TRUE))), type = "message")) > 0)

# ---- 14. summary_only ---------------------------------------------------
cat("\n-- summary_only --\n")
p_full <- predict.longbet(f_no_ps, x, x, z, t = 1:Tn, random_seed = 1)
p_sum  <- predict.longbet(f_no_ps, x, x, z, t = 1:Tn, random_seed = 1,
                          summary_only = TRUE)
ok("draw arrays are dropped",
   is.null(p_sum[["tauhats"]]) && is.null(p_sum[["muhats0"]]))
ok("$tauhats does not partial-match the summary", is.null(p_sum$tauhats))
ok("the point estimate is preserved exactly",
   max(abs(p_sum$tau_summary$mean - apply(p_full$tauhats, c(1, 2), mean))) < 1e-12)
ok("get_catt works on a summary", all(dim(get_catt(p_sum)$catt) == c(n, Tn)))
ok("get_att refuses a summary",
   inherits(try(get_att(p_sum), silent = TRUE), "try-error"))
ok("memory drops by more than 5x",
   as.numeric(object.size(p_sum)) < 0.2 * as.numeric(object.size(p_full)),
   sprintf("(%.1f MB -> %.1f MB)", object.size(p_full) / 1e6,
           object.size(p_sum) / 1e6))



# ---- 15. time-varying covariates ---------------------------------------
# W is drawn per unit-week and independent across cells, so no unit-level
# summary of it says anything about which week is which. Only a tree that can
# split one unit's week 3 from its week 9 can use it.
cat("\n-- time-varying covariates --\n")
set.seed(404)
W <- matrix(rbinom(n * Tn, 1, 0.5), n, Tn)
tau_w <- (1.0 + 1.5 * W) * (S > 0)
y_w <- matrix(2 * x1 + 0.5 * x2, n, Tn) + 3.0 * W + tau_w +
       matrix(rnorm(n * Tn, 0, 0.4), n, Tn)

fit_tv <- function(tv_pr, tv_trt) {
  f <- longbet(y = y_w, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
               x_tv = tv_pr, x_tv_trt = tv_trt, num_sweeps = 80,
               num_burnin = 30, num_trees_pr = 20, num_trees_trt = 20,
               random_intercept = FALSE)
  p <- predict.longbet(f, x, x, z, t = 1:Tn, x_tv = tv_pr,
                       x_tv_trt = tv_trt, random_seed = 1)
  ct <- get_catt(p)$catt
  list(sd = resid_sd(f), w0 = mean(ct[z == 1 & W == 0]),
       w1 = mean(ct[z == 1 & W == 1]))
}
r_none <- fit_tv(NULL, NULL)
r_pr   <- fit_tv(list(W), NULL)
r_both <- fit_tv(list(W), list(W))

ok("without it, the residual carries the covariate", r_none$sd > 1.0,
   sprintf("(%.3f against a true 0.4)", r_none$sd))
ok("in the prognostic forest it recovers the baseline",
   abs(r_pr$sd - 0.4) < 0.15, sprintf("(%.3f)", r_pr$sd))
ok("without it in the treatment forest the heterogeneity is invisible",
   abs(r_pr$w0 - r_pr$w1) < 0.2,
   sprintf("(%.2f vs %.2f, truth 1.0 vs 2.5)", r_pr$w0, r_pr$w1))
ok("in both forests the effect heterogeneity is recovered",
   abs(r_both$w0 - 1.0) < 0.2 && abs(r_both$w1 - 2.5) < 0.2,
   sprintf("(%.2f and %.2f)", r_both$w0, r_both$w1))
ok("a wrong-shaped x_tv is an error",
   inherits(try(longbet(y = y_w, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
                        x_tv = list(W[, -1]), num_sweeps = 5, num_burnin = 2,
                        num_trees_pr = 3, num_trees_trt = 3),
                silent = TRUE), "try-error"))
f_tv <- longbet(y = y_w, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
                x_tv = list(W), num_sweeps = 20, num_burnin = 8,
                num_trees_pr = 5, num_trees_trt = 5, random_intercept = FALSE)
ok("the fit records how many were used", f_tv$n_tv_pr == 1 && f_tv$n_tv_trt == 0)
ok("predicting without them is an error",
   inherits(try(predict.longbet(f_tv, x, x, z, t = 1:Tn, random_seed = 1),
                silent = TRUE), "try-error"))
ok("an n by t by k array is accepted",
   !inherits(try(longbet(y = y_w, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
                         x_tv = array(W, c(n, Tn, 1)), num_sweeps = 10,
                         num_burnin = 4, num_trees_pr = 3, num_trees_trt = 3),
                 silent = TRUE), "try-error"))



# ---- 16. a *continuous* time-varying covariate --------------------------
# The candidate cutpoints on a cell-level axis have to be capped the way the x
# path caps them. Offering one per distinct value makes the likelihood loop
# quadratic in the panel and lets a continuous covariate outvote every x column
# purely on candidate count.
cat("\n-- continuous time-varying covariate --\n")
set.seed(88)
Wc <- matrix(runif(n * Tn), n, Tn)
tau_c <- (0.5 + 2.0 * (Wc > 0.6)) * (S > 0)
y_c <- matrix(2 * x1, n, Tn) + 3.0 * Wc + tau_c +
       matrix(rnorm(n * Tn, 0, 0.4), n, Tn)

el <- system.time({
  f_c <- longbet(y = y_c, x = x, x_trt = x, z = z, t = 1:Tn, pcat = 1,
                 x_tv = list(Wc), x_tv_trt = list(Wc), num_sweeps = 60,
                 num_burnin = 20, num_trees_pr = 15, num_trees_trt = 15,
                 random_intercept = FALSE)
  p_c <- predict.longbet(f_c, x, x, z, t = 1:Tn, x_tv = list(Wc),
                         x_tv_trt = list(Wc), random_seed = 1)
})[3]
ct_c <- get_catt(p_c)$catt
ok("a covariate with one distinct value per cell stays affordable",
   el < 120, sprintf("(%.0f s for %d distinct values)", el,
                     length(unique(as.vector(Wc)))))
ok("the residual SD comes back to the truth", abs(resid_sd(f_c) - 0.4) < 0.15,
   sprintf("(%.3f)", resid_sd(f_c)))
ok("the threshold in a continuous cell-level covariate is found",
   abs(mean(ct_c[z == 1 & Wc <= 0.6]) - 0.5) < 0.25 &&
   abs(mean(ct_c[z == 1 & Wc >  0.6]) - 2.5) < 0.25,
   sprintf("(%.2f below, %.2f above; truth 0.5 and 2.5)",
           mean(ct_c[z == 1 & Wc <= 0.6]), mean(ct_c[z == 1 & Wc > 0.6])))

cat("\nAll tests passed.\n")
