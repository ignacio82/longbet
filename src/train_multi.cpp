#include <ctime>
#include "tree.h"
#include <chrono>
#include "X_struct.h"
#include "mcmc_loop.h"
#include "common.h"
#include "rcpp_utility.h"
#include <RcppArmadillo.h>

using namespace std;
using namespace chrono;

// Several outcomes fitted jointly on the same panel (SUR-LongBet).
//
// Structurally this is longbet_cpp with the outcome-specific parts -- the
// State, the two forests, the Gaussian process, the unit effects, the output
// arrays -- built once per outcome, and the sweep loop replaced by
// mcmc_loop_multi, which visits the outcomes in order inside every sweep and
// fits outcome m to y - sum_{l<m} Gamma_ml * eps_l. Everything that depends
// only on X, z and t is built once and shared by pointer.
//
// Deliberately a copy rather than a refactor of train_all.cpp: the
// single-outcome path is the one every existing result rests on, and leaving
// it byte-for-byte untouched is worth some duplication here.
//
// Returns one complete longbet fit per outcome -- the same fields
// longbet_cpp returns, so predict.longbet() works on each unchanged -- plus
// the per-sweep loadings Gamma. Because draw d of every outcome came from the
// same sweep, indexing the M fits by draw IS the joint posterior.
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List longbet_multi_cpp(Rcpp::List y_list, arma::mat X, arma::mat X_tau, arma::mat z,
                    arma::mat t_con, arma::mat t_mod, arma::mat post_t, arma::mat T, arma::mat S,
                    size_t beta_size,
                    size_t num_sweeps, size_t burnin,
                    size_t max_depth, size_t n_min, size_t num_cutpoints,
                    double no_split_penality,
                    size_t mtry_pr, size_t mtry_trt,
                    size_t p_categorical_pr, size_t p_categorical_trt,
                    size_t num_trees_pr,
                    double alpha_pr, double beta_pr, double tau_pr, double kap_pr, double s_pr,
                    size_t num_trees_trt,
                    double alpha_trt, double beta_trt, double tau_trt, double kap_trt, double s_trt,
                    bool verbose, bool parallel,
                    bool set_random_seed, size_t random_seed,
                    bool sample_weights_flag,
                    bool a_scaling, bool b_scaling,
                    bool split_time_ps, bool split_time_trt,
                    double sig_knl, double lambda_knl,
                    bool random_intercept,
                    double gamma_prior_a, double gamma_prior_b,
                    bool gp_constant_mean,
                    Rcpp::List y_missing_list,
                    Rcpp::LogicalVector binary_vec,
                    Rcpp::NumericVector binary_offset_vec,
                    bool ar1_errors, double rho_max, double sigma_u_init,
                    Rcpp::Nullable<Rcpp::List> x_tv,
                    Rcpp::Nullable<Rcpp::List> x_tv_trt,
                    double sur_prior_var)
{
    const size_t M = y_list.size();
    if (M < 1) Rcpp::stop("longbet_multi_cpp: need at least one outcome");

    size_t N = X.n_rows;
    size_t p_pr = X.n_cols;
    size_t p_trt = X_tau.n_cols;
    arma::mat y0 = Rcpp::as<arma::mat>(y_list[0]);
    size_t p_y = y0.n_cols;

    if (N * p_pr * p_trt * p_y == 0) Rcpp::stop("longbet_multi_cpp: wrong dimension");
    for (size_t m = 0; m < M; m++)
    {
        arma::mat ym = Rcpp::as<arma::mat>(y_list[m]);
        if (ym.n_rows != N || ym.n_cols != p_y)
            Rcpp::stop("longbet_multi_cpp: outcome %d is %d by %d; expected %d by %d",
                       (int) m + 1, (int) ym.n_rows, (int) ym.n_cols, (int) N, (int) p_y);
    }

    size_t p_continuous_pr = p_pr - p_categorical_pr;
    size_t p_continuous_trt = p_trt - p_categorical_trt;
    if (mtry_pr == 0)  mtry_pr = p_pr;
    if (mtry_trt == 0) mtry_trt = p_trt;

    // ---------------- shared: everything that depends only on X, z, t ----------------
    matrix<size_t> Xorder_std;      ini_matrix(Xorder_std, N, p_pr);
    matrix<size_t> torder_mu_std;   ini_matrix(torder_mu_std, t_con.n_rows, t_con.n_cols);
    matrix<size_t> torder_tau_std;  ini_matrix(torder_tau_std, t_mod.n_rows, t_mod.n_cols);
    matrix<size_t> Xorder_tau_std;  ini_matrix(Xorder_tau_std, N, p_trt);

    Rcpp::NumericMatrix z_std(N, p_y);
    Rcpp::NumericMatrix X_std(N, p_pr);
    Rcpp::NumericMatrix X_tau_std(N, p_trt);
    Rcpp::NumericMatrix tcon_std(t_con.n_rows, t_con.n_cols);
    Rcpp::NumericMatrix tmod_std(t_mod.n_rows, t_mod.n_cols);
    Rcpp::NumericMatrix post_t_std(N, p_y);

    arma_to_rcpp(X, X_std);
    arma_to_rcpp(z, z_std);
    arma_to_rcpp(t_con, tcon_std);
    arma_to_rcpp(t_mod, tmod_std);
    arma_to_rcpp(post_t, post_t_std);
    arma_to_std_ordered(X, Xorder_std);
    arma_to_std_ordered(t_con, torder_mu_std);
    arma_to_std_ordered(t_mod, torder_tau_std);
    arma_to_rcpp(X_tau, X_tau_std);
    arma_to_std_ordered(X_tau, Xorder_tau_std);

    double bscale0 = b_scaling ? -0.5 : 1.0;
    double bscale1 = b_scaling ?  0.5 : 1.0;
    std::vector<double> b_vec_init(2); b_vec_init[0] = bscale0; b_vec_init[1] = bscale1;

    std::vector<size_t> num_trees(2); num_trees[0] = num_trees_pr; num_trees[1] = num_trees_trt;

    std::vector<size_t> n_trt(p_y, 0);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < p_y; j++)
            if (z_std(i, j) == 1) n_trt[j]++;

    double *Xpointer = &X_std[0];
    double *Xpointer_tau = &X_tau_std[0];
    double *zpointer = &z_std[0];
    double *tpointer_mu = &tcon_std[0];
    double *tpointer_tau = &tmod_std[0];
    double *post_t_pointer = &post_t_std[0];

    matrix<size_t> Torder_std;  ini_matrix(Torder_std, p_y, N);
    matrix<size_t> Sorder_std;  ini_matrix(Sorder_std, p_y, N);
    for (size_t i = 0; i < N; i++)
    {
        std::iota(Torder_std[i].begin(), Torder_std[i].end(), 0);
        std::iota(Sorder_std[i].begin(), Sorder_std[i].end(), 0);
    }
    double *Spointer = S.memptr();
    double *Tpointer = T.memptr();
    std::vector<double> t_values = arma::conv_to<std::vector<double>>::from(arma::sort(arma::unique(arma::vectorise(T))));
    std::vector<double> s_values = arma::conv_to<std::vector<double>>::from(arma::sort(arma::unique(arma::vectorise(S))));

    // Time-varying covariates: shared across outcomes (same X, same panel).
    std::vector<Rcpp::NumericMatrix> tv_keep, tv_trt_keep;
    std::vector<const double *> tv_ptrs, tv_trt_ptrs;
    std::vector<std::vector<double>> tv_vals, tv_trt_vals;
    auto load_tv = [&](Rcpp::Nullable<Rcpp::List> src,
                       std::vector<Rcpp::NumericMatrix> &keep,
                       std::vector<const double *> &ptrs,
                       std::vector<std::vector<double>> &vals) {
        if (src.isNull()) return;
        Rcpp::List l(src);
        for (int k = 0; k < l.size(); k++)
            keep.push_back(Rcpp::as<Rcpp::NumericMatrix>(l[k]));
        for (size_t k = 0; k < keep.size(); k++)
        {
            ptrs.push_back(&keep[k][0]);
            std::vector<double> v(keep[k].begin(), keep[k].end());
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            vals.push_back(v);
        }
    };
    load_tv(x_tv, tv_keep, tv_ptrs, tv_vals);
    load_tv(x_tv_trt, tv_trt_keep, tv_trt_ptrs, tv_trt_vals);

    // ---------------- per outcome ----------------
    std::vector<Rcpp::NumericMatrix> y_std_v(M);
    std::vector<double> y_mean_v(M, 0.0);
    std::vector<std::unique_ptr<State>> states(M);
    std::vector<longbetModel *> model_pr_v(M, nullptr), model_trt_v(M, nullptr);
    std::vector<vector<vector<tree>> *> trees_pr_v(M, nullptr), trees_trt_v(M, nullptr);
    std::vector<std::unique_ptr<X_struct>> xs_pr(M), xs_trt(M), xs_gp(M);
    std::vector<std::unique_ptr<split_info>> sp_pr(M), sp_trt(M), sp_gp(M);
    std::vector<std::vector<double>> init_theta_pr(M), init_theta_trt(M);

    std::vector<std::vector<matrix<double>>> tauhats_xinfo(M), muhats_xinfo(M);
    std::vector<matrix<double>> sigma0_x(M), sigma1_x(M), a_x(M), b_x(M), beta_x(M),
                                resid_i(M), A_diag_i(M), Sig_diag_i(M), beta_i(M), gamma_x(M);
    std::vector<std::vector<double>> sigma_gamma_d(M), beta_mean_d(M);
    std::vector<SweepCtx> ctx(M);

    for (size_t m = 0; m < M; m++)
    {
        arma::mat ym = Rcpp::as<arma::mat>(y_list[m]);
        y_std_v[m] = Rcpp::NumericMatrix(N, p_y);
        arma_to_rcpp(ym, y_std_v[m]);
        const bool binary_m = binary_vec[m];
        y_mean_v[m] = binary_m ? 0.0 : compute_mat_mean(y_std_v[m]);
        double *ypointer = &y_std_v[m][0];

        std::vector<double> sigma_vec(2, 1.0);

        trees_pr_v[m] = new vector<vector<tree>>(num_sweeps);
        trees_trt_v[m] = new vector<vector<tree>>(num_sweeps);
        for (size_t i = 0; i < num_sweeps; i++)
        {
            (*trees_pr_v[m])[i] = vector<tree>(num_trees_pr);
            (*trees_trt_v[m])[i] = vector<tree>(num_trees_trt);
        }
        model_pr_v[m] = new longbetModel(kap_pr, s_pr, tau_pr, alpha_pr, beta_pr);
        model_pr_v[m]->setNoSplitPenality(no_split_penality);
        model_trt_v[m] = new longbetModel(kap_trt, s_trt, tau_trt, alpha_trt, beta_trt);
        model_trt_v[m]->setNoSplitPenality(no_split_penality);

        // Distinct seed per outcome so the streams do not coincide.
        states[m].reset(new longbetState(Xpointer, Xorder_std, N, n_trt, p_pr, p_trt, p_y,
            num_trees, p_categorical_pr, p_categorical_trt, p_continuous_pr, p_continuous_trt,
            set_random_seed, random_seed + m, n_min, num_cutpoints, parallel, mtry_pr, mtry_trt,
            Xpointer, num_sweeps, sample_weights_flag, ypointer, zpointer, post_t_pointer,
            Tpointer, Spointer, beta_size, sigma_vec, b_vec_init, max_depth, y_mean_v[m],
            burnin, model_trt_v[m]->dim_suffstat));
        State &st = *states[m];
        st.random_intercept = random_intercept;
        st.gp_constant_mean = gp_constant_mean;
        st.ar1_errors = ar1_errors;
        st.rho_max = rho_max;
        if (ar1_errors) { st.sigma_u = sigma_u_init; st.rho = 0.3; }
        st.binary_outcome = binary_m;
        if (binary_m)
        {
            for (size_t j = 0; j < p_y; j++)
                for (size_t i = 0; i < N; i++)
                    st.y_binary[j * N + i] = (y_std_v[m](i, j) > 0.5) ? 1 : 0;
            st.sigma_vec[0] = 1.0; st.sigma_vec[1] = 1.0;
            st.binary_offset = binary_offset_vec[m];
        }
        if (!Rf_isNull(y_missing_list[m]))
        {
            Rcpp::NumericMatrix miss = Rcpp::as<Rcpp::NumericMatrix>(y_missing_list[m]);
            size_t n_miss = 0;
            for (size_t j = 0; j < p_y; j++)
                for (size_t i = 0; i < N; i++)
                    if (miss(i, j) != 0) { st.y_missing[j * N + i] = 1; n_miss++; }
            st.has_missing = (n_miss > 0);
        }
        st.gamma_prior_a = gamma_prior_a;
        st.gamma_prior_b = gamma_prior_b;

        init_theta_pr[m] = std::vector<double>(1, y_mean_v[m] / (double) num_trees_pr);
        init_theta_trt[m] = std::vector<double>(1, 0.0);
        xs_pr[m].reset(new X_struct(Xpointer, ypointer, tpointer_mu, Tpointer, t_values, N, p_y,
            Xorder_std, torder_mu_std, Sorder_std, p_categorical_pr, p_continuous_pr,
            &init_theta_pr[m], num_trees_pr, sig_knl, lambda_knl));
        xs_trt[m].reset(new X_struct(Xpointer_tau, ypointer, tpointer_tau, Spointer, s_values, N, p_y,
            Xorder_tau_std, torder_tau_std, Sorder_std, p_categorical_trt, p_continuous_trt,
            &init_theta_trt[m], num_trees_trt, sig_knl, lambda_knl));
        xs_gp[m].reset(new X_struct(Xpointer_tau, ypointer, tpointer_tau, Spointer, s_values, N, p_y,
            Xorder_tau_std, torder_tau_std, Sorder_std, p_categorical_trt, p_continuous_trt,
            &init_theta_trt[m], num_trees_trt, sig_knl, lambda_knl));
        xs_gp[m]->ini_cov_kernel_s(sig_knl, lambda_knl);
        xs_pr[m]->set_time_varying(tv_ptrs, tv_vals);
        xs_trt[m]->set_time_varying(tv_trt_ptrs, tv_trt_vals);
        xs_gp[m]->set_time_varying(tv_trt_ptrs, tv_trt_vals);

        tauhats_xinfo[m].resize(num_sweeps); muhats_xinfo[m].resize(num_sweeps);
        for (size_t i = 0; i < num_sweeps; i++)
        {
            ini_matrix(tauhats_xinfo[m][i], p_y, N);
            ini_matrix(muhats_xinfo[m][i], p_y, N);
        }
        ini_matrix(sigma0_x[m], num_trees_trt + num_trees_pr, num_sweeps);
        ini_matrix(sigma1_x[m], num_trees_trt + num_trees_pr, num_sweeps);
        ini_matrix(a_x[m], num_sweeps, 1);
        ini_matrix(b_x[m], num_sweeps, 2);
        ini_matrix(beta_x[m], beta_size, num_sweeps);
        size_t t_size = st.beta_size;
        ini_matrix(resid_i[m], t_size, num_sweeps);
        ini_matrix(A_diag_i[m], t_size, num_sweeps);
        ini_matrix(Sig_diag_i[m], t_size, num_sweeps);
        ini_matrix(beta_i[m], t_size, num_sweeps);
        ini_matrix(gamma_x[m], N, num_sweeps);
        sigma_gamma_d[m].assign(num_sweeps, 0.0);
        beta_mean_d[m].assign(num_sweeps, 0.0);

        sp_pr[m].reset(new split_info(xs_pr[m], Xorder_std, Torder_std, t_values));
        sp_trt[m].reset(new split_info(xs_trt[m], Xorder_tau_std, Sorder_std, s_values));
        sp_gp[m].reset(new split_info(xs_gp[m], Xorder_tau_std, Sorder_std, s_values));

        SweepCtx &c = ctx[m];
        c.split_ps = &sp_pr[m]; c.split_trt = &sp_trt[m]; c.split_gp = &sp_gp[m];
        c.verbose = verbose;
        c.sigma0_draw_xinfo = &sigma0_x[m]; c.sigma1_draw_xinfo = &sigma1_x[m];
        c.b_xinfo = &b_x[m]; c.a_xinfo = &a_x[m]; c.beta_info = &beta_i[m]; c.beta_xinfo = &beta_x[m];
        c.trees_ps = trees_pr_v[m]; c.trees_trt = trees_trt_v[m];
        c.no_split_penality = no_split_penality;
        c.state = &states[m]; c.model_ps = model_pr_v[m]; c.model_trt = model_trt_v[m];
        c.x_struct_ps = &xs_pr[m]; c.x_struct_trt = &xs_trt[m]; c.x_struct_gp = &xs_gp[m];
        c.a_scaling = a_scaling; c.b_scaling = b_scaling;
        c.split_time_ps = split_time_ps; c.split_time_trt = split_time_trt;
        c.resid_info = &resid_i[m]; c.A_diag_info = &A_diag_i[m]; c.Sig_diag_info = &Sig_diag_i[m];
        c.gamma_xinfo = &gamma_x[m]; c.sigma_gamma_draws = &sigma_gamma_d[m]; c.beta_mean_draws = &beta_mean_d[m];
    }

    // ---------------- the joint sampler ----------------
    matrix<double> gamma_draws;
    ini_matrix(gamma_draws, M * M, num_sweeps);
    mcmc_loop_multi(ctx, num_sweeps, sur_prior_var, gamma_draws);

    // ---------------- package one fit per outcome ----------------
    Rcpp::List fits(M);
    for (size_t m = 0; m < M; m++)
    {
        State &st = *states[m];
        model_pr_v[m]->predict_std(Xpointer, tpointer_mu, N, p_y, num_sweeps, muhats_xinfo[m], *trees_pr_v[m], &tv_ptrs);
        model_trt_v[m]->predict_std(Xpointer_tau, tpointer_tau, N, p_y, num_sweeps, tauhats_xinfo[m], *trees_trt_v[m], &tv_trt_ptrs);

        size_t t_size = st.beta_size;
        Rcpp::NumericMatrix tauhats(N * p_y, num_sweeps), muhats(N * p_y, num_sweeps);
        Rcpp::NumericMatrix sigma0_draws(num_trees_trt + num_trees_pr, num_sweeps);
        Rcpp::NumericMatrix sigma1_draws(num_trees_trt + num_trees_pr, num_sweeps);
        Rcpp::NumericMatrix b_draws(num_sweeps, 2), a_draws(num_sweeps, 1);
        Rcpp::NumericMatrix beta_values(t_size, num_sweeps), beta_draws(beta_size, num_sweeps);
        Rcpp::NumericMatrix gamma_draws_m(N, num_sweeps);
        std_to_rcpp(gamma_x[m], gamma_draws_m);
        Rcpp::NumericVector sigma_gamma_out(num_sweeps), beta_mean_out(num_sweeps);
        for (size_t i = 0; i < num_sweeps; i++) { sigma_gamma_out[i] = sigma_gamma_d[m][i]; beta_mean_out[i] = beta_mean_d[m][i]; }
        Rcpp::NumericMatrix resid(t_size, num_sweeps), A_diag(t_size, num_sweeps), Sig_diag(t_size, num_sweeps), t_vector(t_size, 1);
        Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt_pr(trees_pr_v[m], true);
        Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt_trt(trees_trt_v[m], true);

        for (size_t sw = 0; sw < num_sweeps; sw++)
            for (size_t col = 0; col < p_y; col++)
                for (size_t row = 0; row < N; row++)
                {
                    tauhats(col * N + row, sw) = tauhats_xinfo[m][sw][row][col];
                    muhats(col * N + row, sw)  = muhats_xinfo[m][sw][row][col];
                }
        std_to_rcpp(sigma0_x[m], sigma0_draws);
        std_to_rcpp(sigma1_x[m], sigma1_draws);
        std_to_rcpp(b_x[m], b_draws);
        std_to_rcpp(a_x[m], a_draws);
        std_to_rcpp(beta_x[m], beta_draws);
        std_to_rcpp(beta_i[m], beta_values);
        std_to_rcpp(resid_i[m], resid);
        std_to_rcpp(A_diag_i[m], A_diag);
        std_to_rcpp(Sig_diag_i[m], Sig_diag);
        for (size_t i = 0; i < t_size; i++) t_vector(i, 0) = xs_trt[m]->s_values[i];

        std::stringstream treess_pr, treess_trt;
        Rcpp::StringVector output_tree_pr(num_sweeps), output_tree_trt(num_sweeps);
        for (size_t i = 0; i < num_sweeps; i++)
        {
            treess_pr.precision(10); treess_trt.precision(10);
            treess_pr.str(std::string()); treess_pr << num_trees_pr << " " << p_pr << endl;
            treess_trt.str(std::string()); treess_trt << num_trees_trt << " " << p_trt << endl;
            for (size_t t = 0; t < num_trees_pr; t++)  treess_pr  << (*trees_pr_v[m])[i][t];
            for (size_t t = 0; t < num_trees_trt; t++) treess_trt << (*trees_trt_v[m])[i][t];
            output_tree_pr(i) = treess_pr.str();
            output_tree_trt(i) = treess_trt.str();
        }

        const int n_missing_cells = (int) std::count(st.y_missing.begin(), st.y_missing.end(), 1);
        const bool binary_m = binary_vec[m];

        fits[m] = Rcpp::List::create(
            Rcpp::Named("tauhats") = tauhats,
            Rcpp::Named("muhats") = muhats,
            Rcpp::Named("gamma_draws") = gamma_draws_m,
            Rcpp::Named("sigma_gamma_draws") = sigma_gamma_out,
            Rcpp::Named("random_intercept") = random_intercept,
            Rcpp::Named("gp_constant_mean") = gp_constant_mean,
            Rcpp::Named("binary_outcome") = binary_m,
            Rcpp::Named("ar1_errors") = ar1_errors,
            Rcpp::Named("rho") = st.rho,
            Rcpp::Named("sigma_u") = st.sigma_u,
            Rcpp::Named("n_missing") = n_missing_cells,
            Rcpp::Named("n_tv_pr") = (int) tv_ptrs.size(),
            Rcpp::Named("n_tv_trt") = (int) tv_trt_ptrs.size(),
            Rcpp::Named("sigma0_draws") = sigma0_draws,
            Rcpp::Named("sigma1_draws") = sigma1_draws,
            Rcpp::Named("b_draws") = b_draws,
            Rcpp::Named("a_draws") = a_draws,
            Rcpp::Named("beta_draws") = beta_draws,
            Rcpp::Named("beta_values") = beta_values,
            Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt_pr") = tree_pnt_pr,
                                                           Rcpp::Named("tree_pnt_trt") = tree_pnt_trt,
                                                           Rcpp::Named("y_mean") = y_mean_v[m]),
            Rcpp::Named("treedraws_pr") = output_tree_pr,
            Rcpp::Named("treedraws_trt") = output_tree_trt,
            Rcpp::Named("model_params") = Rcpp::List::create(Rcpp::Named("num_sweeps") = num_sweeps,
                                                             Rcpp::Named("burnin") = burnin,
                                                             Rcpp::Named("num_obs") = N,
                                                             Rcpp::Named("p_y") = p_y,
                                                             Rcpp::Named("max_depth") = max_depth,
                                                             Rcpp::Named("Nmin") = n_min,
                                                             Rcpp::Named("num_cutpoints") = num_cutpoints,
                                                             Rcpp::Named("alpha_pr") = alpha_pr,
                                                             Rcpp::Named("beta_pr") = beta_pr,
                                                             Rcpp::Named("tau_pr") = tau_pr,
                                                             Rcpp::Named("p_categorical_pr") = p_categorical_pr,
                                                             Rcpp::Named("num_trees_pr") = num_trees_pr,
                                                             Rcpp::Named("alpha_trt") = alpha_trt,
                                                             Rcpp::Named("beta_trt") = beta_trt,
                                                             Rcpp::Named("tau_trt") = tau_trt,
                                                             Rcpp::Named("p_categorical_trt") = p_categorical_trt,
                                                             Rcpp::Named("num_trees_trt") = num_trees_trt,
                                                             Rcpp::Named("sig_knl") = sig_knl,
                                                             Rcpp::Named("lambda_knl") = lambda_knl),
            Rcpp::Named("input_var_count") = Rcpp::List::create(Rcpp::Named("x_con") = p_pr,
                                                                Rcpp::Named("x_mod") = p_trt),
            Rcpp::Named("gp_info") = Rcpp::List::create(
                Rcpp::Named("t_values") = t_vector,
                Rcpp::Named("resid") = resid,
                Rcpp::Named("A_diag") = A_diag,
                Rcpp::Named("Sig_diag") = Sig_diag,
                Rcpp::Named("beta_mean") = beta_mean_out));
    }

    Rcpp::NumericMatrix gamma_out(M * M, num_sweeps);
    std_to_rcpp(gamma_draws, gamma_out);

    for (size_t m = 0; m < M; m++)
    {
        delete model_pr_v[m];
        delete model_trt_v[m];
        states[m].reset();
        xs_pr[m].reset(); xs_trt[m].reset(); xs_gp[m].reset();
    }

    return Rcpp::List::create(
        Rcpp::Named("fits") = fits,
        Rcpp::Named("Gamma_draws") = gamma_out,
        Rcpp::Named("M") = (int) M);
}
