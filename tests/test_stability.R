# att_stability(): does it flag a sampler too short to report an interval from?
suppressMessages(library(longbet))
np <- 0L; nf <- 0L
ok <- function(lab, cond) { if (isTRUE(cond)) { np <<- np+1L; cat("PASS ", lab, "\n") }
                            else { nf <<- nf+1L; cat("FAIL ", lab, "\n") } }
quiet <- function(e) withCallingHandlers(e, warning = function(w) invokeRestart("muffleWarning"))

set.seed(42)
n <- 500; Tn <- 14; tt <- 1:Tn
launch <- sample(c(5, 7, 9, Inf), n, TRUE, c(.2, .2, .2, .4))
z <- outer(seq_len(n), tt, function(i, j) as.integer(j >= launch[i]))
x <- cbind(rnorm(n), runif(n), rbinom(n, 1, .5))
S <- t(apply(z, 1, cumsum))
y <- 2 + .7*x[,1] + outer(rep(1,n), .03*tt) + .4*(1-exp(-S/4)) +
     matrix(rnorm(n*Tn, 0, .3), n, Tn)

fit_at <- function(sw) {
  f <- longbet(y = y, x = x, x_trt = x, z = z, t = tt, pcat = 1,
               num_sweeps = sw, num_burnin = sw %/% 3,
               num_trees_pr = 15, num_trees_trt = 15,
               random_seed = 3, verbose = FALSE)
  predict.longbet(f, x, x, z, t = tt, random_seed = 3)
}

cat("== short sampler ==\n")
p_short <- fit_at(40)
s_short <- quiet(att_stability(p_short))
ok("returns summary and per-event-time detail",
   all(c("summary","by_event_time") %in% names(s_short)))
ok("reports an effective sweep count below the raw draw count",
   s_short$summary$ess_median < s_short$summary$draws)
ok("a short sampler is called unreliable", !s_short$summary$reliable)
w <- tryCatch({ att_stability(p_short); NULL }, warning = conditionMessage)
ok("and warns, naming effective draws", !is.null(w) && grepl("effectively independent", w))

cat("\n== longer sampler ==\n")
p_long <- fit_at(300)
s_long <- quiet(att_stability(p_long))
ok("more sweeps raise the effective count",
   s_long$summary$ess_median > s_short$summary$ess_median)
ok("a long sampler is called reliable", s_long$summary$reliable)
ok("no warning once it is long enough",
   is.null(tryCatch({ att_stability(p_long); NULL }, warning = conditionMessage)))

cat("\n== contract ==\n")
ok("accepts a get_att() result as well as a prediction",
   isTRUE(all.equal(quiet(att_stability(get_att(p_long)))$summary$ess_median,
                    s_long$summary$ess_median)))
ok("one row per event time",
   nrow(s_long$by_event_time) == nrow(get_att(p_long)$att_full))
ok("min <= median effective sweeps", s_long$summary$ess_min <= s_long$summary$ess_median)
ok("mcse is positive and small relative to the effect",
   s_long$summary$mcse > 0 && s_long$summary$mcse < 0.1)
ok("min_ess is honoured", !quiet(att_stability(p_long, min_ess = 1e6))$summary$reliable)
ok("rejects a summary-only prediction",
   inherits(tryCatch(att_stability(structure(list(summary_only = TRUE),
            class = "longbet.pred")), error = function(e) e), "error"))

cat(sprintf("\n%d passed, %d failed\n", np, nf))
if (nf > 0) quit(status = 1)
