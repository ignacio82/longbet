#' longbet model
#'
#' @param y An n by t matrix of outcome variables. NA marks a period in which
#'   a unit was not observed; such cells are drawn from their full conditional
#'   at each sweep rather than dropped, so an unbalanced panel needs no
#'   pre-processing. Units with no observed period at all are an error.
#' @param x n by p input matrix of covariates. (If the covariates matrix is different for the prognostic and treatment term, please use longbet_full).
#' @param x_trt n by p_trt input matrix of covariates for treatment trees
#' @param z An n by t matrix of treatment assignments.
#' @param t time variable (post-treatment time for treatment term will be infered based on input t and z).
#' @param pcat The number of categorical inputs in matrix x.
#' @param num_sweeps The total number of sweeps over data (default is 60).
#' @param num_burnin The number of burn-in sweeps (default is 20).
#' @param num_trees_pr The number of trees in the prognostic forest (default is 50).
#' @param num_trees_trt The number of trees in the treatment forest (default is 20).
#' @param mtry number of variables to be sampled as split candidate per tree.
#' @param n_min The minimum node size. (default is 1)
#' @param sig_knl variance parameter for squared exponential kernel (default is 1).
#' @param lambda_knl lengthscale parameter for squared exponential kernel (default is 1).
#' @param x_tv,x_tv_trt optional time-varying covariates for the prognostic and
#'   treatment forests. Each is a list of n by t matrices laid out like `y`, a
#'   single such matrix, or an n by t by k array. The sampler treats each as one
#'   more cell-level axis to split on, which is the same mechanism that already
#'   lets a tree split on calendar time or on time since adoption -- so a tree
#'   can send one unit's week 3 left and its week 9 right. Covariates in `x`
#'   remain time-invariant; these are the ones that are not. They must be
#'   observed in every cell, including cells where `y` is NA.
#' @param ps optional vector of estimated propensity scores, one per unit. It
#'   is appended to the *prognostic* covariates only, which is what Bayesian
#'   Causal Forest does and why it is robust to targeted selection; the
#'   treatment forest never sees it. Supplying it does nothing in an experiment
#'   where assignment is known and unconfounded, and matters a great deal in an
#'   observational panel. Note that this argument was silently ignored through
#'   version 0.3.1.
#' @param split_time_ps whether to split on time variable in prognostic trees (default is TRUE)
#' @param split_time_trt whether to split on time variable in treatment trees (default is FALSE)
#'
#' @return A fit object holding the tree ensembles and the per-sweep parameter
#'   draws. Note that `beta_values` (the shared time factor) is *not*
#'   separately identified: the treatment term enters as the product
#'   b * beta_S * nu(X, S, t), and nothing in the likelihood pins down how a
#'   common scale is divided between the three factors. That is deliberate --
#'   the redundant scalings are the parameter expansion XBCF uses to improve
#'   mixing -- but it means a trace of `beta_values` will wander even in a
#'   perfectly converged run. Diagnose and report the identified quantity
#'   instead: `get_att()` returns `att_full`, the average effect by
#'   time-since-adoption for every post-burn-in sweep, and `get_catt()` does
#'   the same per unit.
#' @export
longbet <- function(y, x, x_trt, z, t, pcat, pcat_trt = NULL,
                    num_sweeps = 60, num_burnin = 20,
                    num_trees_pr = 20, num_trees_trt = 20,
                    mtry = 0L, n_min = 10,
                    sig_knl = 1, lambda_knl = 1,
                    split_time_ps = TRUE, split_time_trt = TRUE,
                    random_intercept = TRUE,
                    gamma_prior_a = 1, gamma_prior_b = 0.1,
                    gp_constant_mean = TRUE,
                    outcome = c("continuous", "binary"),
                    ar1_errors = FALSE, rho_max = 0.95, sigma_u_init = 0.2,
                    treat_effect_re = FALSE, delta_scale = 0.5,
                    x_tv = NULL, x_tv_trt = NULL,
                    tau_pr = NULL, tau_trt = NULL,
                    max_depth = 50, num_cutpoints = 20,
                    a_scaling = TRUE, b_scaling = FALSE,
                    random_seed = 0, parallel = TRUE, verbose = FALSE,
                    verbose_sampler = FALSE,
                    ps = NULL) {

    outcome <- match.arg(outcome)
    binary_outcome <- identical(outcome, "binary")

    # Time-varying covariates. Each is an n by t matrix laid out like y, and
    # the sampler treats it as one more cell-level axis to split on -- the same
    # mechanism that already lets a tree split on calendar time or on time
    # since adoption. x_tv goes to the prognostic forest, x_tv_trt to the
    # treatment forest; supply either, both, or neither.
    x_tv     <- check_tv(x_tv, nrow(as.matrix(y)), ncol(as.matrix(y)), "x_tv")
    x_tv_trt <- check_tv(x_tv_trt, nrow(as.matrix(y)), ncol(as.matrix(y)), "x_tv_trt")

    if(!("matrix" %in% class(x))){
        if (verbose) message("input x is not a matrix; converting.")
        x = as.matrix(x)
    }
    if(!("matrix" %in% class(x_trt))){
        if (verbose) message("input x is not a matrix; converting.")
        x_trt = as.matrix(x_trt)
    }
    if(!("matrix" %in% class(z))){
        if (verbose) message("input z is not a matrix; converting.")
        z = as.matrix(z)
    }
    if(!("matrix" %in% class(y))){
        if (verbose) message("input y is not a matrix; converting.")
        y = as.matrix(y)
    }

    if(any(dim(z) != dim(y))) {
        stop("Dimensions of response y and treatment z do not match. \n")
    }

    # Unbalanced panels. Cells with no observed outcome are marked and then
    # completed inside the sampler, one draw per sweep, so the trees always see
    # a rectangular panel. Treatment status z must still be known everywhere:
    # whether a unit was treated in a week is a design fact, not an outcome.
    y_miss <- is.na(y)
    n_miss <- sum(y_miss)
    if (n_miss > 0) {
        if (any(is.na(z))) {
            stop("z cannot contain NA: treatment status must be known even ",
                 "in periods where the outcome is not observed. \n")
        }
        all_missing <- rowSums(y_miss) == ncol(y)
        if (any(all_missing)) {
            stop(sum(all_missing), " unit(s) have no observed outcome in any ",
                 "period. Drop them before fitting. \n")
        }
        message("Unbalanced panel: ", n_miss, " of ", length(y), " cells (",
                sprintf("%.1f%%", 100 * n_miss / length(y)),
                ") have no observed outcome; they will be imputed each sweep.")
        # Start them at the observed mean; the sampler replaces them from
        # sweep 0 onward. Only the standardization below reads these values.
        y[y_miss] <- mean(y[!y_miss])
    }

    if (length(t) != ncol(y)){
        stop("Lenght of input t should match the columns of y. \n")
    }

    # Propensity score. Through 0.3.1 this argument was accepted and then
    # ignored -- the code below was commented out -- which mattered because
    # conditioning the prognostic forest on an estimate of pi(x) is the whole
    # point of BCF over BART: without it, strong confounding plus tree
    # shrinkage produces regularization-induced confounding (Hahn, Murray and
    # Carvalho 2020). It belongs in the prognostic term only, never in the
    # treatment term, which is why only x is augmented and x_trt is not.
    #
    # It is appended after the continuous columns and before the categorical
    # block, so the "continuous first, categorical last" contract that pcat
    # relies on still holds and pcat itself does not change.
    use_ps <- !is.null(ps)
    if (use_ps) {
        ps <- as.numeric(ps)
        if (length(ps) != nrow(x)) {
            stop("ps must have one entry per row of x (", nrow(x), "); got ",
                 length(ps), ". \n")
        }
        if (any(!is.finite(ps))) stop("ps contains missing or infinite values. \n")
        if (min(ps) <= 0 || max(ps) >= 1) {
            warning("ps has values outside (0, 1); positivity is doubtful.",
                    call. = FALSE)
        }
        x <- insert_ps(x, ps, pcat)
    }
    
    # check if treatment all start at the same time
    # number of treated periods per unit should only be 0 or t1 - t0 + 1
    # unique_z_sum <- unique(rowSums(z))
    # if (length(unique_z_sum) != 2) {
    #     stop("Current version can only handle treamtments occured at the same time. \n")
    # }

    # get post-treatment time variable
    if (is.null(t)){
        t_con = 1:ncol(y)
    } else {
        t_con = t
    }

    # if (split_time_trt){
    #     stop("Can not handle split time on treatment tree with staggered adoption yet. \n")
    # }

    # A unit that is treated in every observed period carries no information
    # separating its own level from the treatment effect, so gamma_i and tau
    # are only held apart by the prior. Warn rather than fail: the fit is still
    # usable, but those units' effects are prior-driven.
    if (all(z == 1)) {
        stop("every unit is treated in every period: there is no untreated ",
             "observation anywhere, so no treatment effect is identified. \n")
    }
    if (random_intercept) {
        always_treated <- sum(rowSums(z) == ncol(z))
        if (always_treated > 0) {
            warning(always_treated, " unit(s) are treated in every period. ",
                    "With random_intercept = TRUE their unit effect and their ",
                    "treatment effect are separated only by the prior.",
                    call. = FALSE)
        }
    }

    # Treatment has to be absorbing. beta is indexed by "periods since this
    # unit's first treated period", counted forward from that first onset and
    # never consulting z again, so a unit that switches back off goes on being
    # credited with an effect at a state index that keeps climbing. A second
    # internal grid is built from cumsum(z) instead, which counts only the
    # periods actually switched on; the two agree exactly while z is
    # non-decreasing and diverge the moment it is not, leaving the Gaussian
    # process kernel and the beta vector different sizes. What the user saw
    # was an Armadillo error about incompatible matrix dimensions from deep
    # inside the sampler. Say so here instead.
    if (any(apply(z, 1, function(v) any(diff(v) < 0)))) {
        bad <- which(apply(z, 1, function(v) any(diff(v) < 0)))
        stop("longbet() requires absorbing treatment: z cannot return to 0 ",
             "once a unit is treated (", length(bad), " of ", nrow(z),
             " rows switch back off, first at row ", bad[1], "). ",
             "Switchback and on/off designs are not supported -- beta is a ",
             "single trajectory in time-since-adoption, which has no meaning ",
             "once treatment is withdrawn.", call. = FALSE)
    }

    # get post-treatment time matrix
    get_trt_time <- function(z_vec, t){
        treated_period <- which(z_vec == 1)
        if (length(treated_period) == 0){
            # no treated period
            return(rep(0, length(z_vec)))
        } else {
            if (treated_period[1] == 1){
                # case: no untreated period for this unit
                # assuming last untreated time point is lag 1
                t0 <- t[1] - 1
            } else {
                t0 <- t[treated_period[1] - 1]
            }
            trt_time <- sapply(t, function(x, t0) max(0, x - t0), t0 = t0)
            return(trt_time)
        }
    }
    post_trt_time <- t(apply(z, 1, get_trt_time, t = t))
    beta_size <- max(post_trt_time) + 1

    # get cumulative treated time
    S <- t(apply(z, 1, cumsum))

    # get a matrix of time for the panel data
    T <- matrix(rep(t, nrow(y)), nrow = nrow(y), byrow = TRUE)

    trt_time <- matrix(apply(z, 1, function(x) sum(x == 0)), nrow(z), 1)

    if (ncol(y) > 1) {
        post_t <- max(rowSums(z))
        t0 <- ncol(y) - post_t + 1
        t_mod <- c(rep(0, t0 - 1), 1:post_t)
    } else {
        t_mod <- c(1)
        t0 <- NULL
    }

    if(!("matrix" %in% class(t_con))){
        t_con = as.matrix(t_con)
    }
    if(!("matrix" %in% class(t_mod))){
        t_mod = as.matrix(t_mod)
    }


    if (nrow(x) != nrow(y)) {
        stop(paste0('row number mismatch between X (', nrow(x), ') and y (', nrow(y), ')'))
    }

    # check if p_categorical was not provided
    if(is.null(pcat)) {
        stop('number of categorical variables pcat_con is not specified')
    }
    
    # check if p_categorical exceeds the number of columns
    if(pcat > ncol(x)) {
        stop('number of categorical variables (pcat_con) cannot exceed number of columns')
    }

    # check if p_categorical is negative
    if(pcat < 0) {
        stop('number of categorical values can not be negative: check pcat_con and pcat_mod')
    }

    if(is.null(pcat_trt)){
        pcat_trt = pcat
        if (verbose) message("assuming pcat_trt = ", pcat, " for the treatment trees.")
    }

    # check if p_categorical exceeds the number of columns
    if(pcat_trt > ncol(x_trt)) {
        stop('number of categorical variables (pcat_trt) cannot exceed number of columns')
    }

    # check if mtry exceeds the number of columns
    if(mtry > ncol(x)) {
        if (verbose) message("mtry exceeds the number of columns; using the default.")
        mtry <- 0
    }
    # check if mtry is negative
    if(mtry < 0) {
        if (verbose) message("mtry exceeds the number of columns; using the default.")
        mtry <- 0
    }

    if (binary_outcome) {
        obs <- y[!is.na(y)]
        if (!all(obs %in% c(0, 1))) {
            stop('with outcome = "binary", y must contain only 0, 1 and NA. \n')
        }
        # No standardization: the latent scale is fixed by the probit link,
        # which is exactly what identifies it, and sigma is held at 1 in the
        # sampler for the same reason. What we do centre is the location:
        # qnorm of the observed rate is the latent grand mean, and handing it
        # to the sampler as an offset means the forests only ever fit
        # deviations from it, exactly as they do for a centred continuous
        # outcome.
        rate = mean(obs)
        meany = qnorm(min(max(rate, 1e-4), 1 - 1e-4))
        sdy = 1
    } else {
        meany = mean(y) # disable meany temporarily
        y = y - meany
        sdy = sd(y)

        if(sdy == 0) {
            stop('y is a constant variable; sdy = 0')
        } else {
            y = y / sdy
        }
    }

    # Leaf-variance priors, computed AFTER standardization, on the scale the
    # C++ sampler actually works on, where var(y) is 1 by construction.
    # Computing them from var(raw y) -- as this package did through 0.1.2 --
    # makes the amount of regularization depend on the units of the outcome:
    # the same data in dollars rather than log dollars gets a leaf prior larger
    # by a factor of var(y), leaving the forests effectively unregularized on
    # large-variance outcomes and over-shrunk on small ones. The 0.6 / 0.1
    # split is the share of outcome variance allotted to the two forests.
    # On the probit scale the latent has variance 1 by construction, which is
    # the analogue of standardizing a continuous outcome.
    y_var <- if (binary_outcome) 1 else var(as.vector(y))
    if (is.null(tau_pr))  tau_pr  <- 0.6 * y_var / num_trees_pr
    if (is.null(tau_trt)) tau_trt <- 0.1 * y_var / num_trees_trt
    tau_con = tau_pr
    tau_mod = tau_trt

    if(num_burnin >= num_sweeps){
        stop(paste0('num_burnin (',num_burnin,') cannot exceed or match the total number of sweeps (',num_sweeps,')'))
    }

    # still fixed: tree-prior shape and the sigma prior
    no_split_penality = log(num_cutpoints)
    alpha_con = 0.95; beta_con = 1.25
    kap_con = 16; s_con = 4
    pr_scale = FALSE
    alpha_mod = 0.95; beta_mod = 1.25
    kap_mod = 16; s_mod = 4
    trt_scale = FALSE
    sample_weights_flag = TRUE
    set_random_seed = TRUE

    obj = longbet_cpp(y = y,
                    X = x, 
                    X_tau = x_trt, 
                    z = z, 
                    t_con = t_con, 
                    t_mod = t_mod,
                    post_t = post_trt_time,
                    T = T, 
                    S = S,
                    beta_size = beta_size,
                    num_sweeps = num_sweeps, 
                    burnin = num_burnin,
                    max_depth = max_depth, 
                    n_min = n_min,
                    num_cutpoints = num_cutpoints,
                    no_split_penality = no_split_penality,
                    mtry_pr = mtry, 
                    mtry_trt = mtry,
                    p_categorical_pr = pcat, 
                    p_categorical_trt = pcat_trt,
                    num_trees_pr = num_trees_pr,
                    alpha_pr = alpha_con, 
                    beta_pr = beta_con, 
                    tau_pr = tau_con,
                    kap_pr = kap_con, 
                    s_pr = s_con,
                    pr_scale = pr_scale,
                    num_trees_trt = num_trees_trt,
                    alpha_trt = alpha_mod, 
                    beta_trt = beta_mod,
                    tau_trt = tau_mod,
                    kap_trt = kap_mod, 
                    s_trt = s_mod,
                    trt_scale = trt_scale,
                    verbose = verbose_sampler, 
                    parallel = parallel, 
                    set_random_seed = set_random_seed,
                    random_seed = random_seed, 
                    sample_weights_flag = sample_weights_flag,
                    a_scaling = a_scaling, 
                    b_scaling = b_scaling,
                    split_time_ps = split_time_ps, 
                    split_time_trt = split_time_trt,
                    sig_knl = sig_knl, 
                    lambda_knl = lambda_knl,
                    random_intercept = random_intercept,
                    gamma_prior_a = gamma_prior_a,
                    gamma_prior_b = gamma_prior_b,
                    gp_constant_mean = gp_constant_mean,
                    x_tv = if (length(x_tv)) x_tv else NULL,
                    x_tv_trt = if (length(x_tv_trt)) x_tv_trt else NULL,
                    y_missing = if (n_miss > 0) y_miss * 1.0 else NULL,
                    binary_outcome = binary_outcome,
                    binary_offset = if (binary_outcome) meany else 0,
                    ar1_errors = ar1_errors, rho_max = rho_max,
                    sigma_u_init = sigma_u_init,
                    treat_effect_re = treat_effect_re,
                    delta_scale = delta_scale)
    class(obj) = "longbet"

    obj$n_tv_pr = length(x_tv)
    obj$n_tv_trt = length(x_tv_trt)
    obj$use_ps = use_ps
    obj$ps = if (use_ps) ps else NULL
    obj$pcat = pcat
    obj$time = t_con
    obj$t0 = t0
    obj$sdy = sdy
    obj$meany = meany

    # The sampler works on the standardized outcome; return the unit effects on
    # the scale the user handed in. gamma_draws is [n x num_sweeps] in the row
    # order of x, so post-burnin unit effects are
    #   rowMeans(fit$gamma_draws[, (num_burnin + 1):num_sweeps]).
    obj$gamma_draws = obj$gamma_draws * sdy
    obj$sigma_gamma_draws = obj$sigma_gamma_draws * sdy
    # Unit treatment random effects, on the outcome scale. predict.longbet()
    # adds these to tauhats for in-sample predictions.
    obj$delta_draws = obj$delta_draws * sdy
    obj$sigma_delta = obj$sigma_delta * sdy

    # obj$beta_draws = obj$beta_draws[, (num_burnin+1):num_sweeps]
    return(obj)
}


`%||%` <- function(a, b) if (is.null(a)) b else a

# Time-varying covariates arrive as a list of n by t matrices, a single n by t
# matrix, or an n by t by k array. Normalise to a list of numeric matrices and
# check the shape, because a silent recycle here would be invisible in the
# output and wrong everywhere.
check_tv <- function(x_tv, n, tt, what) {
    if (is.null(x_tv)) return(list())
    if (is.array(x_tv) && length(dim(x_tv)) == 3) {
        x_tv <- lapply(seq_len(dim(x_tv)[3]), function(k) x_tv[, , k])
    }
    if (!is.list(x_tv)) x_tv <- list(x_tv)
    out <- vector("list", length(x_tv))
    for (k in seq_along(x_tv)) {
        m <- as.matrix(x_tv[[k]])
        if (nrow(m) != n || ncol(m) != tt) {
            stop(what, "[[", k, "]] is ", nrow(m), " by ", ncol(m),
                 "; it must match the panel, ", n, " by ", tt, ". \n")
        }
        if (any(!is.finite(m))) {
            stop(what, "[[", k, "]] contains missing or infinite values. ",
                 "Covariates must be observed even where the outcome is not. \n")
        }
        storage.mode(m) <- "double"
        out[[k]] <- m
    }
    out
}

# Put the propensity score after the continuous columns and before the
# categorical ones, so pcat keeps meaning what it meant.
insert_ps <- function(x, ps, pcat) {
    ncont <- ncol(x) - pcat
    cbind(x[, seq_len(ncont), drop = FALSE], ps = ps,
          x[, ncont + seq_len(pcat), drop = FALSE])
}

get_post_trt_time <- function(z_vec, t){
    treated_period <- which(z_vec == 1)
    if (length(treated_period) == 0){
        # no treated period
        return(rep(0, length(z_vec)))
    } else {
        if (treated_period[1] == 1){
            # case: no untreated period for this unit
            # assuming last untreated time point is lag 1
            t0 <- t[1] - 1
        } else {
            t0 <- t[treated_period[1] - 1]
        }
        trt_time <- sapply(t, function(x, t0) max(0, x - t0), t0 = t0)
        return(trt_time)
    }
}