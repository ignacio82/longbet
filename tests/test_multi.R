# Multi-outcome (SUR) fits: contract, correctness, and the two properties the
# feature exists for -- a coherent joint posterior, and variance reduction
# from orthogonalizing correlated continuous outcomes.
suppressMessages(library(longbet))
np <- 0L; nf <- 0L
ok <- function(lab, cond) { if (isTRUE(cond)) { np <<- np+1L; cat("PASS ", lab, "\n") }
                            else { nf <<- nf+1L; cat("FAIL ", lab, "\n") } }
near <- function(a, b, tol) isTRUE(abs(a - b) < tol)

mk <- function(seed = 1, n = 900, Tn = 12, rho = 0.8, same_driver = TRUE) {
  set.seed(seed)
  launch <- sample(c(5, 7, 9, Inf), n, TRUE, c(.2, .2, .2, .4))
  z <- outer(seq_len(n), 1:Tn, function(i, j) as.integer(j >= launch[i]))
  S <- t(apply(z, 1, cumsum))
  x <- cbind(rnorm(n), rnorm(n), rbinom(n, 1, .5))
  w1 <- plogis(x[, 1]); w2 <- if (same_driver) w1 else plogis(x[, 2])
  tau1 <- ifelse(S > 0, (0.30 * w1 + 0.05) * (1 - exp(-S / 4)), 0)
  tau2 <- ifelse(S > 0, (0.25 * w2 + 0.03), 0)
  tau3 <- ifelse(S > 0, (0.50 * w1 - 0.10), 0)
  L <- chol(matrix(c(1, rho, rho, 1), 2, 2))
  e <- matrix(rnorm(2 * n * Tn), ncol = 2) %*% L
  y1 <- 2 + .8 * x[, 1] + tau1 + matrix(e[, 1], n, Tn) * 0.5
  y2 <- 1 + .6 * x[, 1] + tau2 + matrix(e[, 2], n, Tn) * 0.5
  lin3 <- 0.3 + .7 * x[, 2]
  y3 <- ((lin3 + tau3 + matrix(rnorm(n * Tn), n, Tn)) > 0) * 1
  list(y1 = y1, y2 = y2, y3 = y3, x = x, z = z, t = 1:Tn,
       tau1 = tau1, tau2 = tau2,
       tau3p = { p <- pnorm(lin3 + tau3) - pnorm(lin3); p[S == 0] <- 0; p })
}
args <- function(d, y, outcome, ...) c(list(y = y, x = d$x, x_trt = d$x, z = d$z,
  t = d$t, pcat = 1, outcome = outcome, num_sweeps = 60, num_burnin = 20,
  num_trees_pr = 15, num_trees_trt = 15, random_intercept = TRUE,
  random_seed = 3, verbose = FALSE), list(...))

cat("== contract ==\n")
d <- mk()
ok("one outcome is refused, with a pointer to longbet()",
   inherits(tryCatch(do.call(longbet_multi, args(d, list(d$y1), "continuous")),
                     error = function(e) e), "error"))
ok("mismatched outcome length is refused",
   inherits(tryCatch(do.call(longbet_multi,
     args(d, list(d$y1, d$y2), c("continuous", "binary", "continuous"))),
     error = function(e) e), "error"))
ok("a non-0/1 column declared binary is refused",
   inherits(tryCatch(do.call(longbet_multi,
     args(d, list(d$y1, d$y2), c("continuous", "binary"))),
     error = function(e) e), "error"))

f2 <- do.call(longbet_multi, args(d, list(d$y1, d$y2), "continuous"))
ok("returns a longbet_multi with one fit per outcome",
   inherits(f2, "longbet_multi") && length(f2$fits) == 2)
ok("each fit is an ordinary longbet object", all(vapply(f2$fits,
   function(f) inherits(f, "longbet"), logical(1))))
ok("Gamma_draws is M^2 by sweeps",
   identical(dim(f2$Gamma_draws), c(4L, 60L)))

p2 <- predict(f2, d$x, d$x, d$z, t = d$t)
ok("predict returns one prediction per outcome",
   inherits(p2, "longbet_multi.pred") && length(p2$preds) == 2)
ok("effect draws are n x t x post-burn-in",
   identical(dim(effect_draws(p2$preds[[1]])), c(900L, 12L, 40L)))

cat("\n== the joint posterior ==\n")
tt <- ncol(d$z); who <- d$z[, tt] == 1
e1 <- p2$preds[[1]]$tauhats; e2 <- p2$preds[[2]]$tauhats
pc <- median(vapply(which(who), function(i) cor(e1[i, tt, ], e2[i, tt, ]), numeric(1)))
ok(sprintf("effect draws are correlated when the effects share a driver (%.2f)", pc),
   pc > 0.15)
jp <- joint_prob(p2, list(function(a) a > 0, function(b) b > 0))
ok("joint_prob returns an n x t matrix of probabilities",
   identical(dim(jp), c(900L, 12L)) && all(jp >= 0 & jp <= 1))
ok("joint probability never exceeds either marginal",
   all(jp <= apply(e1 > 0, c(1, 2), mean) + 1e-9))
ok("joint_prob checks the number of conditions",
   inherits(tryCatch(joint_prob(p2, list(function(a) a > 0)),
                     error = function(e) e), "error"))
cr <- outcome_correlation(f2)
ok(sprintf("recovers the error correlation (%.2f against 0.80)", cr[1, 2]),
   near(cr[1, 2], 0.8, 0.15))
ok("correlation matrix is symmetric with unit diagonal",
   near(cr[1, 2], cr[2, 1], 1e-8) && near(cr[1, 1], 1, 1e-8))

cat("\n== what the coupling buys, and what it must not cost ==\n")
sd_of <- function(f, m) { post <- (f$burnin + 1):f$num_sweeps
  mean(f$fits[[m]]$sigma0_draws[, post]) * f$fits[[m]]$sdy }
f2i <- do.call(longbet_multi, args(d, list(d$y1, d$y2), "continuous", sur = FALSE))
ok(sprintf("orthogonalization shrinks the second innovation (%.3f vs %.3f)",
           sd_of(f2, 2), sd_of(f2i, 2)), sd_of(f2, 2) < 0.85 * sd_of(f2i, 2))
ok("the first outcome is untouched -- the loadings are triangular",
   near(sd_of(f2, 1), sd_of(f2i, 1), 0.02))

cat("\n== mixed continuous and binary ==\n")
f3 <- do.call(longbet_multi, args(d, list(d$y1, d$y2, d$y3),
                                  c("continuous", "continuous", "binary")))
ok("a mixed panel fits", inherits(f3, "longbet_multi") && f3$M == 3)
p3 <- predict(f3, d$x, d$x, d$z, t = d$t)
b <- effect_draws(p3$preds[[3]], "binary")
ok("a binary effect comes back on the probability scale",
   all(b >= -1 & b <= 1) && max(abs(b)) < 0.9)
tr <- d$z == 1
rm3 <- sqrt(mean((apply(b, c(1, 2), mean)[tr] - d$tau3p[tr])^2))
f3b <- do.call(longbet_multi, args(d, list(d$y1, d$y2, d$y3),
                                   c("continuous", "continuous", "binary"), sur = FALSE))
b_i <- effect_draws(predict(f3b, d$x, d$x, d$z, t = d$t)$preds[[3]], "binary")
rm3i <- sqrt(mean((apply(b_i, c(1, 2), mean)[tr] - d$tau3p[tr])^2))
ok(sprintf("the probit is not degraded by the coupling (%.4f vs %.4f)", rm3, rm3i),
   rm3 <= rm3i * 1.05)

cat("\n== ordering is an internal detail ==\n")
f3r <- do.call(longbet_multi, args(d, list(d$y3, d$y1, d$y2),
                                   c("binary", "continuous", "continuous")))
ok("outcomes come back in the order they were given",
   identical(f3r$outcome, c("binary", "continuous", "continuous")))
p3r <- predict(f3r, d$x, d$x, d$z, t = d$t)
br <- effect_draws(p3r$preds[[1]], "binary")
ok("and a reordered fit gives the same binary effect",
   near(sqrt(mean((apply(br, c(1,2), mean)[tr] - d$tau3p[tr])^2)), rm3, 0.01))

cat(sprintf("\n%d passed, %d failed\n", np, nf))
if (nf > 0) quit(status = 1)
