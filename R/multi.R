#' Fit several outcomes jointly on one panel
#'
#' Runs one LongBet per outcome, in a single sampler, with the outcomes'
#' contemporaneous errors tied together. Two things follow that separate fits
#' cannot give you.
#'
#' The first is **joint** inference. Sweep \eqn{d} of every outcome is drawn in
#' the same pass of the sampler, so draw \eqn{d} of outcome 1 and draw \eqn{d}
#' of outcome 2 are a sample from their *joint* posterior. Questions of the
#' form "what is the probability this seller gains revenue **and** does not
#' churn" are then a matter of counting draws. Fit the outcomes separately and
#' the draws are independent by construction, so the same count silently
#' assumes the two effects are uncorrelated -- which is the one thing you were
#' trying to find out.
#'
#' The second is precision, and it is the smaller of the two. Outcome \eqn{m}
#' is fitted to the *orthogonalized* target
#' \deqn{\tilde y^{(m)} = y^{(m)} - \sum_{l<m} \Gamma_{ml}\,\varepsilon^{(l)},}
#' where \eqn{\varepsilon^{(l)}} are the current residuals of the outcomes
#' before it. Shocks common to several outcomes are absorbed, so
#' \eqn{\sigma^{(m)}} falls and the intervals tighten. This is the triangular
#' factorization \eqn{\Sigma = A_0^{-1} H A_0^{-\top}} of Huber and Rossini
#' (2022), which is what makes an \eqn{M}-outcome model cost \eqn{M} univariate
#' fits rather than anything worse.
#'
#' @section What this does not do:
#' The orthogonalization conditions outcome \eqn{m} on other outcomes'
#' *residuals*, and a residual is a post-treatment quantity. That is
#' legitimate here for a specific reason: \eqn{\varepsilon^{(l)}} is what is
#' left of \eqn{y^{(l)}} once its own treatment effect has been removed, so it
#' carries the shock and not the effect. It is **not** licence to put another
#' outcome's *level* in `x` -- that is conditioning on a mediator and it will
#' bias \eqn{\tau}. The ordering matters for interpretation too: \eqn{\Gamma}
#' is triangular, so outcome 1 is never adjusted, and the estimand for every
#' outcome is still its own marginal effect regardless of order.
#'
#' @param y A list of n by t matrices, or an n by t by M array. One per outcome.
#' @param outcome Character vector, `"continuous"` or `"binary"`, length 1 or M.
#' @param sur If FALSE, the loadings are held at zero: M independent fits that
#'   still share a sampler, which is the honest comparison for what the
#'   coupling buys.
#' @param sur_prior_var Ridge prior variance on the loadings.
#' @param ... Passed to the sampler; same meaning as in [longbet()].
#' @return An object of class `longbet_multi`: `$fits`, a list of ordinary
#'   `longbet` objects usable with [predict.longbet()], and `$Gamma_draws`.
#' @export
longbet_multi <- function(y, x, x_trt, z, t, pcat, pcat_trt = NULL,
                          outcome = "continuous", sur = TRUE,
                          sur_prior_var = 1.0,
                          num_sweeps = 60, num_burnin = 20,
                          num_trees_pr = 20, num_trees_trt = 20,
                          mtry = 0L, n_min = 10,
                          sig_knl = 1, lambda_knl = 1,
                          split_time_ps = TRUE, split_time_trt = TRUE,
                          random_intercept = TRUE,
                          gamma_prior_a = 1, gamma_prior_b = 0.1,
                          gp_constant_mean = TRUE,
                          ar1_errors = FALSE, rho_max = 0.95, sigma_u_init = 0.2,
                          x_tv = NULL, x_tv_trt = NULL,
                          max_depth = 50, num_cutpoints = 20,
                          a_scaling = TRUE, b_scaling = FALSE,
                          random_seed = 0, parallel = TRUE, verbose = FALSE,
                          verbose_sampler = FALSE, ps = NULL) {

    if (is.array(y) && length(dim(y)) == 3) {
        y <- lapply(seq_len(dim(y)[3]), function(k) y[, , k])
    }
    if (!is.list(y)) stop("longbet_multi(): y must be a list of matrices or an n x t x M array. \n")
    M <- length(y)
    if (M < 2) stop("longbet_multi(): needs at least two outcomes; use longbet() for one. \n")
    y <- lapply(y, as.matrix)

    if (length(outcome) == 1) outcome <- rep(outcome, M)
    if (length(outcome) != M) stop("longbet_multi(): outcome must have length 1 or ", M, ". \n")
    # unname: vapply over a character vector carries the input across as
    # names, and a named outcome vector then fails any identical() check a
    # caller makes against a plain one.
    outcome <- unname(vapply(outcome,
        function(o) match.arg(o, c("continuous", "binary")), character(1)))
    binary <- outcome == "binary"

    n <- nrow(y[[1]]); tt <- ncol(y[[1]])
    for (m in seq_len(M)) {
        if (nrow(y[[m]]) != n || ncol(y[[m]]) != tt) {
            stop("longbet_multi(): outcome ", m, " is ", nrow(y[[m]]), " by ",
                 ncol(y[[m]]), "; expected ", n, " by ", tt, ". \n")
        }
    }
    # Continuous outcomes first. The loadings are triangular, so an outcome is
    # only ever adjusted by those before it; putting the continuous ones first
    # means they can benefit from each other, and the probits -- which are not
    # orthogonalized at all, their scale being pinned -- sit at the end where
    # they adjust nothing. Order does not change any outcome's estimand.
    if (any(binary) && !all(diff(as.integer(binary)) >= 0)) {
        ord <- order(binary)
        y <- y[ord]; outcome <- outcome[ord]; binary <- binary[ord]
        if (verbose) message("reordering outcomes continuous-first: ",
                             paste(ord, collapse = ", "))
        attr_ord <- ord
    } else {
        attr_ord <- seq_len(M)
    }

    if (length(t) != tt) stop("longbet_multi(): length(t) must match the columns of y. \n")
    if (nrow(x) != n) stop("longbet_multi(): x and y must have the same number of rows. \n")
    if (any(is.na(z))) stop("longbet_multi(): z cannot contain NA. \n")

    # --- shared design quantities, identical to longbet() -------------------
    if (is.null(pcat)) stop("number of categorical variables pcat is not specified. \n")
    if (is.null(pcat_trt)) pcat_trt <- pcat
    use_ps <- !is.null(ps)
    if (use_ps) {
        ps <- as.numeric(ps)
        if (length(ps) != nrow(x)) stop("ps must have one entry per row of x. \n")
        x <- insert_ps(x, ps, pcat)
    }
    if (all(z == 1)) {
        stop("every unit is treated in every period: no treatment effect is identified. \n")
    }
    if (any(apply(z, 1, function(v) any(diff(v) < 0)))) {
        stop("longbet_multi() requires absorbing treatment: z cannot return to 0 ",
             "once a unit is treated. \n", call. = FALSE)
    }
    get_trt_time <- function(z_vec, t) {
        tp <- which(z_vec == 1)
        if (!length(tp)) return(rep(0, length(z_vec)))
        t0 <- if (tp[1] == 1) t[1] - 1 else t[tp[1] - 1]
        sapply(t, function(x) max(0, x - t0))
    }
    post_trt_time <- t(apply(z, 1, get_trt_time, t = t))
    beta_size <- max(post_trt_time) + 1
    S <- t(apply(z, 1, cumsum))
    T <- matrix(rep(t, n), nrow = n, byrow = TRUE)
    t_con <- as.matrix(t)
    post_t <- max(rowSums(z))
    t0 <- tt - post_t + 1
    t_mod <- as.matrix(if (tt > 1) c(rep(0, t0 - 1), 1:post_t) else 1)

    x_tv     <- check_tv(x_tv,     n, tt, "x_tv")
    x_tv_trt <- check_tv(x_tv_trt, n, tt, "x_tv_trt")

    # --- per-outcome preprocessing, mirroring longbet() ---------------------
    meany <- numeric(M); sdy <- numeric(M)
    miss_list <- vector("list", M)
    for (m in seq_len(M)) {
        ym <- y[[m]]
        y_miss <- is.na(ym)
        n_miss <- sum(y_miss)
        if (n_miss > 0) {
            if (any(rowSums(y_miss) == ncol(ym))) {
                stop("outcome ", m, ": some units have no observed value in any period. \n")
            }
            ym[y_miss] <- mean(ym[!y_miss])
        }
        # `[[<- NULL` deletes a list element rather than setting it, which
        # silently shortens miss_list and then indexes past its end in C++.
        miss_list[m] <- list(if (n_miss > 0) y_miss * 1.0 else NULL)

        if (binary[m]) {
            obs <- ym[!y_miss]
            if (!all(obs %in% c(0, 1))) {
                stop("outcome ", m, ' is "binary" but contains values other than 0, 1 and NA. \n')
            }
            rate <- mean(obs)
            meany[m] <- qnorm(min(max(rate, 1e-4), 1 - 1e-4))
            sdy[m] <- 1
        } else {
            meany[m] <- mean(ym)
            ym <- ym - meany[m]
            sdy[m] <- sd(ym)
            if (sdy[m] == 0) stop("outcome ", m, " is constant. \n")
            ym <- ym / sdy[m]
        }
        y[[m]] <- ym
    }

    # var(y) is 1 on the working scale for both outcome types, so the leaf
    # priors are the same for every equation.
    tau_pr  <- 0.6 / num_trees_pr
    tau_trt <- 0.1 / num_trees_trt

    if (num_burnin >= num_sweeps) {
        stop("num_burnin cannot exceed or match num_sweeps. \n")
    }

    obj <- longbet_multi_cpp(
        y_list = y, X = x, X_tau = x_trt, z = z,
        t_con = t_con, t_mod = t_mod, post_t = post_trt_time, T = T, S = S,
        beta_size = beta_size, num_sweeps = num_sweeps, burnin = num_burnin,
        max_depth = max_depth, n_min = n_min, num_cutpoints = num_cutpoints,
        no_split_penality = log(num_cutpoints),
        mtry_pr = mtry, mtry_trt = mtry,
        p_categorical_pr = pcat, p_categorical_trt = pcat_trt,
        num_trees_pr = num_trees_pr,
        alpha_pr = 0.95, beta_pr = 1.25, tau_pr = tau_pr, kap_pr = 16, s_pr = 4,
        num_trees_trt = num_trees_trt,
        alpha_trt = 0.95, beta_trt = 1.25, tau_trt = tau_trt, kap_trt = 16, s_trt = 4,
        verbose = verbose_sampler, parallel = parallel,
        set_random_seed = TRUE, random_seed = random_seed,
        sample_weights_flag = TRUE,
        a_scaling = a_scaling, b_scaling = b_scaling,
        split_time_ps = split_time_ps, split_time_trt = split_time_trt,
        sig_knl = sig_knl, lambda_knl = lambda_knl,
        random_intercept = random_intercept,
        gamma_prior_a = gamma_prior_a, gamma_prior_b = gamma_prior_b,
        gp_constant_mean = gp_constant_mean,
        y_missing_list = miss_list,
        binary_vec = binary,
        binary_offset_vec = ifelse(binary, meany, 0),
        ar1_errors = ar1_errors, rho_max = rho_max, sigma_u_init = sigma_u_init,
        x_tv = if (length(x_tv)) x_tv else NULL,
        x_tv_trt = if (length(x_tv_trt)) x_tv_trt else NULL,
        # 0 switches the loading update off entirely, which is how
        # sur = FALSE gives genuinely independent equations in one sampler.
        sur_prior_var = if (sur) sur_prior_var else 0)

    # Each element of $fits is an ordinary longbet object once the scaling
    # metadata predict.longbet() needs is attached.
    fits <- vector("list", M)
    for (m in seq_len(M)) {
        f <- obj$fits[[m]]
        class(f) <- "longbet"
        f$n_tv_pr <- length(x_tv); f$n_tv_trt <- length(x_tv_trt)
        f$use_ps <- use_ps; f$ps <- if (use_ps) ps else NULL
        f$pcat <- pcat; f$time <- t_con; f$t0 <- t0
        f$sdy <- sdy[m]; f$meany <- meany[m]
        f$outcome <- outcome[m]
        f$gamma_draws <- f$gamma_draws * sdy[m]
        f$sigma_gamma_draws <- f$sigma_gamma_draws * sdy[m]
        f$exposure <- list(levels = seq_len(nrow(f$beta_values)) - 1,
                           carryover = FALSE,
                           lambda_knl = lambda_knl)
        fits[[m]] <- f
    }

    # Hand the outcomes back in the order they were supplied. The
    # continuous-first permutation is an internal detail of how the equations
    # are chained, and silently returning a permuted list would be a trap:
    # fits[[k]] must always be outcome k as the caller numbered it.
    inv <- order(attr_ord)
    fits <- fits[inv]
    outcome_user <- outcome[inv]

    out <- list(fits = fits, M = M, outcome = outcome_user, sur = sur,
                order = attr_ord,          # internal (fitting) order
                Gamma_draws = obj$Gamma_draws,   # in INTERNAL order
                num_sweeps = num_sweeps, burnin = num_burnin)
    class(out) <- "longbet_multi"
    out
}

#' Predict from a multi-outcome fit
#'
#' Calls [predict.longbet()] on each outcome. Because the outcomes were drawn
#' in one sampler, draw \eqn{d} of each is from the joint posterior, so the
#' returned predictions are aligned draw for draw and can be combined
#' directly -- which is what [joint_prob()] does.
#'
#' @param object A `longbet_multi` fit.
#' @param ... Passed to [predict.longbet()].
#' @export
predict.longbet_multi <- function(object, x, x_trt, z, t = NULL, ...) {
    preds <- lapply(object$fits, function(f)
        predict.longbet(f, x, x_trt, z, t = t, ...))
    out <- list(preds = preds, M = object$M, outcome = object$outcome)
    class(out) <- "longbet_multi.pred"
    out
}

#' Probability that several treatment effects hold at once
#'
#' The reason to fit outcomes jointly. `conditions` is a list of one-argument
#' functions, the m-th applied to the m-th outcome's effect draws; the result
#' is the posterior probability that all of them hold simultaneously, computed
#' by counting draws in which they do.
#'
#' For a binary outcome the effect is returned on the probability scale --
#' \eqn{\Phi(\mu_0 + \tau) - \Phi(\mu_0)}, a change in percentage points --
#' rather than on the latent probit scale, because that is the quantity a
#' decision is actually made on.
#'
#' @param object Output of `predict()` on a `longbet_multi` fit.
#' @param conditions List of M functions, each taking an n by t by draws array
#'   and returning a logical array of the same shape.
#' @param cells Optional logical n by t matrix restricting which cells count;
#'   a unit-level question usually wants a single period.
#' @return An n by t matrix of joint posterior probabilities, or a vector over
#'   units when `cells` selects one period per unit.
#' @export
joint_prob <- function(object, conditions, cells = NULL) {
    if (!inherits(object, "longbet_multi.pred")) {
        stop("joint_prob() needs the output of predict() on a longbet_multi fit.",
             call. = FALSE)
    }
    M <- object$M
    if (length(conditions) != M) {
        stop("joint_prob(): supply one condition per outcome (", M, "). \n")
    }
    eff <- lapply(seq_len(M), function(m) effect_draws(object$preds[[m]],
                                                       object$outcome[m]))
    ok <- conditions[[1]](eff[[1]])
    for (m in seq_len(M)[-1]) ok <- ok & conditions[[m]](eff[[m]])
    p <- apply(ok, c(1, 2), mean)
    if (!is.null(cells)) {
        stopifnot(identical(dim(cells), dim(p)))
        return(vapply(seq_len(nrow(p)), function(i) {
            k <- which(cells[i, ]); if (!length(k)) NA_real_ else mean(p[i, k])
        }, numeric(1)))
    }
    p
}

#' Effect draws on the scale a decision is made on
#'
#' Continuous outcomes come back as they are. Binary outcomes are converted
#' from the latent probit scale to a difference in probabilities.
#' @keywords internal
#' @export
effect_draws <- function(pred, outcome = "continuous") {
    if (isTRUE(pred$summary_only)) {
        stop("effect_draws() needs the posterior draws; call predict() with ",
             "summary_only = FALSE.", call. = FALSE)
    }
    if (identical(outcome, "binary")) {
        pnorm(pred$muhats0 + pred$tauhats) - pnorm(pred$muhats0)
    } else {
        pred$tauhats
    }
}

#' Posterior correlation between outcomes' innovations
#'
#' Reconstructs \eqn{\Sigma = A_0^{-1} H A_0^{-\top}} from the sampled
#' loadings and returns the implied correlation matrix, averaged over
#' post-burn-in sweeps. A value near zero says the outcomes shared nothing and
#' the joint fit bought only convenience.
#' @param object A `longbet_multi` fit.
#' @export
outcome_correlation <- function(object) {
    if (!inherits(object, "longbet_multi")) {
        stop("outcome_correlation() needs a longbet_multi fit.", call. = FALSE)
    }
    M <- object$M
    post <- (object$burnin + 1):object$num_sweeps
    # Gamma is stored in the internal (continuous-first) order, so assemble
    # Sigma there and permute back to the caller's order at the end.
    ord <- object$order
    sig <- vapply(object$fits[ord], function(f)
        mean(f$sigma0_draws[, post]), numeric(1))
    # The sampler defines eps_m = raw_m - sum_{l<m} Gamma_ml raw_l, that is
    # eps = (I - L) raw with L strictly lower triangular. So the composite
    # errors have Var(raw) = (I - L)^{-1} H (I - L)^{-T}, with H the diagonal
    # of orthogonal innovation variances.
    acc <- matrix(0, M, M)
    for (d in post) {
        L <- matrix(object$Gamma_draws[, d], M, M, byrow = TRUE)
        diag(L) <- 0
        Bi <- solve(diag(M) - L)
        acc <- acc + Bi %*% diag(sig^2, M) %*% t(Bi)
    }
    Sig <- acc / length(post)
    C <- Sig / sqrt(outer(diag(Sig), diag(Sig)))
    inv <- order(ord)
    C[inv, inv, drop = FALSE]
}
