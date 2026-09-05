#' Is the sampler long enough for the interval you are about to report?
#'
#' XBART does not run one Markov chain through tree space; it grows a fresh
#' forest every sweep, so what `longbet()` returns is an *approximation* to the
#' posterior whose quality depends on how many sweeps you average over. Too few
#' and you are looking at a slice of the posterior rather than the posterior,
#' and the credible interval that comes out is too narrow -- confidently, and
#' with nothing in the output to say so.
#'
#' That failure is silent and it is measurable. What measures it is the
#' **effective number of independent sweeps** behind the reported quantity:
#' consecutive forests are correlated, so 80 sweeps is not 80 draws, and a
#' 2.5% quantile estimated from a couple of dozen effective draws is not a
#' reliable interval endpoint.
#'
#' A split-half comparison is reported alongside it, and it is worth saying
#' what that one is and is not for. It detects *drift* -- a sampler still
#' moving, or burn-in set too short. It does **not** detect the narrowness
#' problem, because XBART's sweeps are roughly exchangeable after burn-in, so
#' both halves are narrow together and the ratio stays near 1 whether or not
#' you have run long enough. Read `ess_median` for the interval, `width_ratio`
#' for drift.
#'
#' The point estimate settles well before the interval does, so a sweep count
#' that looks fine on `att` can still be short for `intervals`. Check the
#' quantity you intend to report, which is why this works on the ATT rather
#' than on `beta` or `sigma`.
#'
#' @param object Output of [predict.longbet()], or of [get_att()].
#' @param alpha Credible level, matching the one you will report.
#' @param min_ess Effective sweeps below which the interval is called
#'   unreliable. The default of 40 is a smoke alarm calibrated on measured
#'   coverage, not a theorem. Across 40 replications of a staggered rollout,
#'   ATT intervals covered 92.0% at a median ESS near 23 and 95.2% -- nominal
#'   -- at a median ESS near 42, so the useful line sits between them. ESS
#'   depends on the panel, so on data unlike that, calibrate it yourself
#'   before trusting the verdict; the number the function is really for is
#'   `ess_median`, not the boolean.
#' @param drift_tol Relative disagreement between sweep halves that counts as
#'   drift rather than noise.
#' @param warn Emit a warning when the interval looks unreliable.
#' @return A list with `summary` (one row) and `by_event_time`. `summary` holds
#'   the post-burn-in sweep count, the median and minimum effective sweeps, the
#'   Monte Carlo standard error of the ATT, the split-half width ratio, and
#'   `reliable`.
#' @examples
#' \dontrun{
#'   p <- predict.longbet(fit, x, x, z, t = weeks)
#'   att_stability(p)
#' }
#' @export
att_stability <- function(object, alpha = 0.05, min_ess = 40,
                          drift_tol = 0.15, warn = TRUE) {
    full <- if (!is.null(object$att_full)) object$att_full
            else get_att(object, alpha = alpha)$att_full
    if (is.null(full)) {
        stop("att_stability() needs posterior draws: pass the output of ",
             "predict.longbet() (with summary_only = FALSE) or of get_att().",
             call. = FALSE)
    }

    D <- ncol(full)
    if (D < 8) {
        stop("att_stability() needs at least 8 post-burn-in sweeps; this fit ",
             "has ", D, ".", call. = FALSE)
    }
    mid <- D %/% 2
    h1  <- seq_len(mid)
    h2  <- (mid + 1):D

    width <- function(idx) {
        q <- apply(full[, idx, drop = FALSE], 1, stats::quantile,
                   probs = c(alpha / 2, 1 - alpha / 2))
        q[2, ] - q[1, ]
    }
    w1 <- width(h1)
    w2 <- width(h2)
    m1 <- rowMeans(full[, h1, drop = FALSE])
    m2 <- rowMeans(full[, h2, drop = FALSE])

    # Effective number of independent sweeps, from the autocorrelation of the
    # per-sweep ATT series. Sweeps are not a Markov chain, but consecutive
    # forests are correlated, and this is the honest denominator for how much
    # of the posterior has actually been seen.
    ess_one <- function(v) {
        v <- v[is.finite(v)]
        if (length(v) < 8 || stats::sd(v) == 0) return(NA_real_)
        a <- stats::acf(v, plot = FALSE, lag.max = min(length(v) - 2L, 50L))$acf[-1]
        k <- which(a < 0)[1]
        if (!is.na(k) && k > 1) a <- a[seq_len(k - 1L)] else if (!is.na(k)) a <- numeric(0)
        length(v) / (1 + 2 * sum(a))
    }
    ess <- apply(full, 1, ess_one)

    ratio    <- mean(w2, na.rm = TRUE) / mean(w1, na.rm = TRUE)
    ess_med  <- stats::median(ess, na.rm = TRUE)
    ess_min  <- min(ess, na.rm = TRUE)
    mcse     <- stats::median(apply(full, 1, stats::sd) / sqrt(ess), na.rm = TRUE)
    drifting <- is.finite(ratio) && abs(ratio - 1) > drift_tol
    reliable <- is.finite(ess_med) && ess_med >= min_ess && !drifting

    if (warn && is.finite(ess_med) && ess_med < min_ess) {
        warning("att_stability(): ", D, " post-burn-in sweeps give a median of ",
                "only ", sprintf("%.0f", ess_med), " effectively independent ",
                "draws. Interval endpoints estimated from that few draws are ",
                "unreliable and typically too narrow -- the point estimate is ",
                "fine. Increase num_sweeps if you intend to report the ",
                "interval.", call. = FALSE)
    }
    if (warn && drifting) {
        warning("att_stability(): the two halves of the sampler disagree by ",
                sprintf("%.0f%%", 100 * (ratio - 1)), " in interval width, ",
                "which suggests drift rather than noise. Increase num_burnin.",
                call. = FALSE)
    }

    list(
        summary = data.frame(
            draws            = D,
            width_first      = mean(w1, na.rm = TRUE),
            width_second     = mean(w2, na.rm = TRUE),
            ess_median       = ess_med,
            ess_min          = ess_min,
            mcse             = mcse,
            width_ratio      = ratio,
            max_shift        = max(abs(m2 - m1), na.rm = TRUE),
            reliable         = reliable
        ),
        by_event_time = data.frame(
            s            = seq_len(nrow(full)),
            width_first  = w1,
            width_second = w2,
            width_ratio  = w2 / w1,
            ess          = ess
        )
    )
}
