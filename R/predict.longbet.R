#' Get post-burnin draws from longbet model
#'
#' @param model A trained longbet model.
#' @param x An input matrix for size n by p1. Column order matters: continuos features should all bgo before of categorical.
#' @param z n by p_y treatment matrix indicating whether each unit get treated at each step, should match the training period
#' @param gp bool, predict time coefficient beta using gaussian process
#' @param x_tv,x_tv_trt time-varying covariates, supplied exactly as at fit
#'   time. See `longbet()`.
#' @param ps propensity scores for the rows of `x`, required when the model was
#'   fit with `ps` and `x` is supplied without that column. Defaults to the
#'   training scores when predicting on the training rows.
#' @param summary_only return posterior means and `alpha`-level intervals
#'   instead of the full [n x t x draws] arrays. On a large panel the arrays are
#'   the dominant memory cost; this drops them.
#' @param alpha interval width used when `summary_only` is TRUE.
#' @param verbose print progress messages (default FALSE).
#' @param random_seed integer seed for the Gaussian process draws used when the
#'   prediction panel reaches further past treatment than the training panel
#'   did. The extrapolated beta is sampled, not computed, so without a seed the
#'   projected region of the output changes between otherwise identical calls.
#'   Pass NULL for a nondeterministic draw.
#'
#' @return A list with `tauhats` (the treatment effect, [n x t x draws]),
#'   `muhats0` (the fitted outcome with the unit held untreated), `muhats`
#'   (retained for backwards compatibility; see the note in the code) and the
#'   Gaussian process pieces. For a binary fit these are on the probit scale,
#'   so probabilities are `pnorm(muhats0)` and `pnorm(muhats0 + tauhats)`. 
#' @export
predict.longbet <- function(model, x, x_trt, z, t = NULL, sigma = NULL,
                            lambda = NULL, random_seed = 1, ps = NULL,
                            summary_only = FALSE, alpha = 0.05,
                            x_tv = NULL, x_tv_trt = NULL,
                            verbose = FALSE, ...,
                            add_unit_effects = TRUE) {

    # Time-varying covariates have to be supplied again, in the same order as
    # at fit time: the trees record which cell-level axis they split on.
    x_tv     <- check_tv(x_tv, nrow(as.matrix(x)), ncol(as.matrix(z)), "x_tv")
    x_tv_trt <- check_tv(x_tv_trt, nrow(as.matrix(x)), ncol(as.matrix(z)), "x_tv_trt")
    if (length(x_tv) != (model$n_tv_pr %||% 0)) {
        stop("this model was fit with ", model$n_tv_pr %||% 0,
             " time-varying prognostic covariate(s); predict() was given ",
             length(x_tv), ". \n")
    }
    if (length(x_tv_trt) != (model$n_tv_trt %||% 0)) {
        stop("this model was fit with ", model$n_tv_trt %||% 0,
             " time-varying treatment covariate(s); predict() was given ",
             length(x_tv_trt), ". \n")
    }
    if (length(x_tv) == 0) x_tv <- NULL
    if (length(x_tv_trt) == 0) x_tv_trt <- NULL

    # If the fit used a propensity score, the prognostic covariates it was
    # trained on carry an extra column. Rebuild it here rather than making the
    # caller remember, but insist on being told the scores for new units.
    if (isTRUE(model$use_ps)) {
        if (!("matrix" %in% class(x))) x <- as.matrix(x)
        if (ncol(x) == model$input_var_count$x_con - 1) {
            if (is.null(ps)) {
                if (nrow(x) == length(model$ps)) {
                    ps <- model$ps          # predicting on the training rows
                } else {
                    stop("this model was fit with a propensity score, so ",
                         "predict() needs `ps` for these rows. \n")
                }
            }
            if (length(ps) != nrow(x)) {
                stop("ps must have one entry per row of x. \n")
            }
            x <- insert_ps(x, as.numeric(ps), model$pcat)
        }
    }

    if(!("matrix" %in% class(x))) {
        if (verbose) message("input x is not a matrix; converting.")
        x = as.matrix(x)
    }

    if(ncol(x) != model$input_var_count$x_con) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x with ', model$input_var_count$x_con,
        ' columns; trying to predict on x with ', ncol(x),' columns.'))
    }

    if(!("matrix" %in% class(x_trt))) {
        if (verbose) message("input x is not a matrix; converting.")
        x_trt = as.matrix(x_trt)
    }

    if(ncol(x_trt) != model$input_var_count$x_mod) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x with ', model$input_var_count$x_mod,
        ' columns; trying to predict on x with ', ncol(x_trt),' columns.'))
    }

    if(!("matrix" %in% class(z))) {
        if (verbose) message("input z is not a matrix; converting.")
        z = as.matrix(z)
    }

    if (nrow(z) != nrow(x)){
        stop("X and Z should have the same number of rows. \n")
    }

    
    if (is.null(t)){
        # Was longbet.fit$time, which only resolved when the caller happened to
        # have named the fitted object longbet.fit in the global environment.
        if (verbose) message(paste(c("Predicting from time", model$time), collapse = " "))
        t_con <-  matrix(rep(model$time, nrow(x)), nrow = nrow(x), byrow = T)
    } else {
        if (length(t) != ncol(z)){
            stop("Msg: lenght of t should match the size of z. \n")
        }
        if (verbose) message(paste(c("Predicting from time", t), collapse = " "))
        t_con <-  matrix(rep(t, nrow(x)), nrow = nrow(x), byrow = T)
    }

    t_mod <- t( apply(z, 1, cumsum) )
    
    obj_mu = .Call(`_longbet_predict_longbet`, x, t_con,
        model$model_list$tree_pnt_pr, x_tv)

    obj_tau = .Call(`_longbet_predict_longbet`, x_trt, t_mod,
        model$model_list$tree_pnt_trt, x_tv_trt)
    
    # nu(X, S = 0). Without time-varying covariates in the treatment forest
    # this is one value per unit, so a single column suffices. With them it
    # varies by period too, and the whole panel has to be evaluated with the
    # time-since-adoption axis pinned to zero.
    if (is.null(x_tv_trt)) {
        obj_tau0 = .Call(`_longbet_predict_longbet`, x_trt,
            matrix(rep(0, nrow(x)), ncol = 1),
            model$model_list$tree_pnt_trt, NULL)
        tau0_wide <- FALSE
    } else {
        obj_tau0 = .Call(`_longbet_predict_longbet`, x_trt,
            matrix(0, nrow(x), ncol(z)),
            model$model_list$tree_pnt_trt, x_tv_trt)
        tau0_wide <- TRUE
    }

    # Match post treatment periods
    n <- nrow(z)
    p <- ncol(z)

    num_sweeps <- ncol(model$tauhats)
    num_burnin <- model$model_params$burnin

    if(num_burnin >= num_sweeps) {
        stop(paste0('burnin (',num_burnin,') cannot exceed or match the total number of sweeps (',num_sweeps,')'))
    }


    post_trt <- t(apply(z, 1, cumsum))
    beta_preds <- array(NA, dim = c(n, p, num_sweeps))

    max_post_trt <- max(post_trt)
    S <- nrow(model$beta_values) - 1 # max S observed
    if (max_post_trt > S){
        # predict beta
        # stop("TODO: update extrapolation code for staggered adoption, \n")
        if (is.null(sigma)) {  sigma = 1 }
        if (is.null(lambda)) { lambda = nrow(model$beta_values) / 2}
        if (verbose) message("extrapolating beta with the GP, sigma = ", sigma,
                             ", lambda = ", lambda)

        # beta to be predicted?
        beta_test <- as.matrix((S + 1) : max_post_trt)
        obj_beta = .Call(`_longbet_predict_beta`, beta_test,
            as.matrix(model$gp_info$t_values), model$gp_info$resid, model$gp_info$A_diag, model$gp_info$Sig_diag,
            sigma, lambda, !is.null(random_seed),
            if (is.null(random_seed)) 0 else as.integer(random_seed),
            model$gp_info$beta_mean)
        model$beta_values <- rbind(model$beta_values, obj_beta$beta)
    }
    for (i in 1:num_sweeps){
        beta_preds[,,i] <- t(apply(post_trt, 1, function(x, beta) beta[x + 1], beta = model$beta_values[,i]))
    }

    obj_mu$preds <- obj_mu$preds * model$sdy
    obj_tau$preds <- obj_tau$preds * model$sdy
    obj_tau0$preds <- obj_tau0$preds * model$sdy

    obj <- list()
    class(obj) = "longbet.pred"
    
    obj$muhats <- array(NA, dim = c(n, p, num_sweeps - num_burnin))
    obj$muhats0 <- array(NA, dim = c(n, p, num_sweeps - num_burnin))
    obj$tauhats <- array(NA, dim = c(n, p, num_sweeps - num_burnin))
    seq <- (num_burnin+1):num_sweeps
    for (i in seq) {
        obj$muhats[,, i - num_burnin] = matrix(obj_mu$preds[,i], n, p) * (model$a_draws[i]) + model$meany +  matrix(obj_tau$preds[,i], n, p) *  model$b_draws[i,1] * model$beta_draws[1, i]

        # Fitted value with the unit held untreated. muhats above evaluates the
        # treatment forest at the unit's *realised* time since adoption, so for
        # a treated cell it is neither potential outcome; muhats0 evaluates it
        # at S = 0 and is the untreated one. The treated potential outcome is
        # muhats0 + tauhats. Neither includes the unit random intercept, which
        # predict() cannot attach to rows of a new x -- add
        # rowMeans(fit$gamma_draws[, post]) yourself for in-sample fits.
        tau0_i <- if (tau0_wide) matrix(obj_tau0$preds[,i], n, p) else
                  matrix(rep(obj_tau0$preds[,i], p), ncol = p)
        obj$muhats0[,, i - num_burnin] = matrix(obj_mu$preds[,i], n, p) * (model$a_draws[i]) + model$meany +  tau0_i * model$b_draws[i,1] * model$beta_values[1, i]
        # obj$tauhats[,, i - num_burnin] = matrix(obj_tau$preds[,i], n, p) * (model$b_draws[i,2] * beta_preds[,,i] - model$b_draws[i,1] * model$beta_draws[1, i]) # * beta_preds[,,i]
        obj$tauhats[,, i - num_burnin] = model$b_draws[i,2] * beta_preds[,,i] * matrix(obj_tau$preds[,i], n, p)  - model$b_draws[i,1] * model$beta_values[1, i] * tau0_i
        # TODO: change tauhat to b1 * beta_s * tau_s - b0 * beta_0 * tau_0 when tau can split on post-treatment time
    }
    # Unit-level treatment random effects. Unlike gamma_i, delta_i is PART
    # of the treatment effect, so it belongs in tauhats -- but predict() can
    # only attach it when the rows of x are the training units in training
    # order, which it checks by count. Pass add_unit_effects = FALSE to get
    # the covariate-only effect nu(x) instead.
    if (isTRUE(model$treat_effect_re) && isTRUE(add_unit_effects) &&
        !is.null(model$delta_draws) && nrow(model$delta_draws) == n) {
        for (i in seq) {
            obj$tauhats[,, i - num_burnin] <- obj$tauhats[,, i - num_burnin] +
                model$delta_draws[, i] * z
        }
        obj$unit_effects_added <- TRUE
    }
    obj$beta_values <- model$beta_values
    obj$beta_preds <- beta_preds
    obj$z <- z

    # Full posterior arrays are [n x t x draws]; on a large panel the three of
    # them together are the biggest object in the session. summary_only keeps
    # the posterior mean and the interval and throws the draws away.
    if (summary_only) {
        summarise <- function(a) list(
            mean  = apply(a, c(1, 2), mean),
            lower = apply(a, c(1, 2), quantile, probs = alpha / 2),
            upper = apply(a, c(1, 2), quantile, probs = 1 - alpha / 2))
        # Deliberately not called tauhats_summary: R's `$` does partial
        # matching on lists, so obj$tauhats would silently resolve to it after
        # the draws are dropped, and callers would get a list where they
        # expected an array.
        obj$tau_summary <- summarise(obj$tauhats)
        obj$mu0_summary <- summarise(obj$muhats0)
        obj$tauhats <- NULL
        obj$muhats  <- NULL
        obj$muhats0 <- NULL
        obj$beta_preds <- NULL
        obj$summary_only <- TRUE
    }
    return(obj)
}

get_att <- function(object, alpha = 0.05, ...){
    if (!inherits(object, "longbet.pred")) {
        stop("get_att() needs the output of predict.longbet().", call. = FALSE)
    }
    if (isTRUE(object$summary_only)) {
        stop("get_att() needs the posterior draws; call predict.longbet() ",
             "with summary_only = FALSE.", call. = FALSE)
    }
    # att_full <- apply(object$tauhats[z,,], c(2, 3), mean)

    n <- dim(object$tauhats)[1]
    treatment_period <- nrow(object$beta_values) - 1
    num_sweeps <- dim(object$tauhats)[3]

    # Align treatment effect 
    align_catt <- array(NA, dim = c(n, treatment_period, num_sweeps))

    for (i in 1:n){
        if (sum(object$z[i,]) == 0) {next}
        post_t <- 1:sum(object$z[i,])
        align_catt[i,post_t,] = object$tauhats[i, object$z[i,]== 1,]
    }

    att_hat <- apply(align_catt, c(2, 3), mean, na.rm = T)

    obj <- list()
    obj$att <- rowMeans(att_hat)
    obj$intervals <- apply(att_hat, 1, quantile, probs = c(alpha / 2, 1- alpha / 2))
    obj$att_full <- att_hat
    return(obj)
}

get_catt <- function(object, alpha = 0.05, ...){
    if (!inherits(object, "longbet.pred")) {
        stop("get_catt() needs the output of predict.longbet().", call. = FALSE)
    }
    if (isTRUE(object$summary_only)) {
        # Already summarised at predict time; hand back the same shape.
        return(list(catt = object$tau_summary$mean,
                    intervals = array(c(object$tau_summary$lower,
                                        object$tau_summary$upper),
                                      dim = c(2, dim(object$tau_summary$mean)))))
    }
    obj <- list()
    obj$catt <- apply(object$tauhats, c(1, 2), mean)
    obj$intervals <- apply(object$tauhats, c(1, 2), quantile, probs = c(alpha / 2, 1 - alpha / 2))
    return(obj)
}
