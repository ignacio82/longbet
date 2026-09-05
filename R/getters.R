#' Posterior mean treatment effect for every unit and period
#'
#' Through 0.3.1 this read `fit$tauhats.adjusted`, which the C++ side returns
#' as NULL, so it raised an error on every fit. It also could not have worked
#' from `fit$tauhats` alone: that field holds the raw treatment-forest output
#' nu(X, S) on the standardized scale, not the effect. The effect is the
#' contrast b1 * beta_S * nu(X, S) - b0 * beta_0 * nu(X, 0), which needs the
#' forest evaluated at S = 0 as well and is what `predict.longbet()` assembles.
#'
#' So this now takes a prediction rather than a fit.
#'
#' @param object Output of `predict.longbet()`.
#' @return An n by t matrix of posterior mean treatment effects.
#' @export
getTaus <- function(object) {
    require_pred(object, "getTaus")
    get_catt(object)$catt
}

#' Posterior mean of the untreated outcome for every unit and period
#'
#' @param object Output of `predict.longbet()`.
#' @return An n by t matrix of posterior mean untreated outcomes.
#' @export
getMus <- function(object) {
    require_pred(object, "getMus")
    if (isTRUE(object$summary_only)) return(object$mu0_summary$mean)
    apply(object$muhats0, c(1, 2), mean)
}

require_pred <- function(object, what) {
    if (inherits(object, "longbet")) {
        stop(what, "() takes the output of predict.longbet(), not the fit ",
             "itself. The treatment effect is a contrast the fit object does ",
             "not hold: run predict.longbet(fit, x, x_trt, z, t = ...) first.",
             call. = FALSE)
    }
    if (!inherits(object, "longbet.pred")) {
        stop(what, "() needs the output of predict.longbet().", call. = FALSE)
    }
}
