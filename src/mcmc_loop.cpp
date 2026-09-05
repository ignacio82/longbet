#include "mcmc_loop.h"
#include <memory>
#include <ostream>
#include "split_info.h"
// BCF main loop
// input includes information about two sets of trees 
// (one for prognostic term, the other for treatment term)

void mcmc_loop_longbet( 
  std::unique_ptr<split_info> &split_pr,  
  std::unique_ptr<split_info> &split_trt,
  std::unique_ptr<split_info> &split_gp,
  bool verbose,
  matrix<double> &sigma0_draw_xinfo,
  matrix<double> &sigma1_draw_xinfo,
  matrix<double> &b_xinfo,
  matrix<double> &a_xinfo,
  matrix<double> &beta_info,
  matrix<double> &beta_xinfo,
  vector<vector<tree>> &trees_ps,
  vector<vector<tree>> &trees_trt,
  double no_split_penality,
  std::unique_ptr<State> &state,
  longbetModel *model_ps,
  longbetModel *model_trt,
  std::unique_ptr<X_struct> &x_struct_ps,
  std::unique_ptr<X_struct> &x_struct_trt,
  std::unique_ptr<X_struct> &x_struct_gp,
  bool a_scaling,
  bool b_scaling,
  bool split_time_ps,
  bool split_time_trt,
  matrix<double> &resid_info,
  matrix<double> &A_diag_info,
  matrix<double> &Sig_diag_info,
  matrix<double> &gamma_xinfo,
  std::vector<double> &sigma_gamma_draws,
  std::vector<double> &beta_mean_draws
  )
{

  if (state->parallel)
    thread_pool.start();

  // verbose = true;

  SweepCtx c;
  c.split_ps = &split_pr; c.split_trt = &split_trt; c.split_gp = &split_gp;
  c.verbose = verbose;
  c.sigma0_draw_xinfo = &sigma0_draw_xinfo; c.sigma1_draw_xinfo = &sigma1_draw_xinfo;
  c.b_xinfo = &b_xinfo; c.a_xinfo = &a_xinfo; c.beta_info = &beta_info; c.beta_xinfo = &beta_xinfo;
  c.trees_ps = &trees_ps; c.trees_trt = &trees_trt;
  c.no_split_penality = no_split_penality;
  c.state = &state; c.model_ps = model_ps; c.model_trt = model_trt;
  c.x_struct_ps = &x_struct_ps; c.x_struct_trt = &x_struct_trt; c.x_struct_gp = &x_struct_gp;
  c.a_scaling = a_scaling; c.b_scaling = b_scaling;
  c.split_time_ps = split_time_ps; c.split_time_trt = split_time_trt;
  c.resid_info = &resid_info; c.A_diag_info = &A_diag_info; c.Sig_diag_info = &Sig_diag_info;
  c.gamma_xinfo = &gamma_xinfo; c.sigma_gamma_draws = &sigma_gamma_draws; c.beta_mean_draws = &beta_mean_draws;

  for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
  {
    mcmc_one_sweep(sweeps, c);
  }

  thread_pool.stop();
}

void mcmc_one_sweep(size_t sweeps, SweepCtx &c)
{
    if (c.verbose == true)
    {
      COUT << "--------------------------------" << endl;
      COUT << "number of sweeps " << sweeps << endl;
      COUT << "--------------------------------" << endl;
    }

    // Complete the panel before anything reads it this sweep. On sweep 0 the
    // forests are empty, so the draw is around the (standardized) mean.
    c.model_ps->draw_latent_outcome((*c.state));

    c.model_ps->set_state_status((*c.state), 0, (*c.x_struct_ps)->X_std, (*c.split_ps)->Xorder_std, (*c.x_struct_ps)->t_std);

    ////////////// Prognostic term loop
    for (size_t tree_ind = 0; tree_ind < (*c.state)->num_trees_vec[0]; tree_ind++)
    {
      if (c.verbose == true)
      {
        COUT << "--------------------------------" << endl;
        COUT << "number of prognostic trees " << tree_ind << endl;
        COUT << "--------------------------------" << endl;
      }
      (*c.state)->update_residuals();  // update residuals
      c.model_ps->draw_sigma((*c.state), 0);  // draw sigmas
      // store sigma draws
      (*c.sigma0_draw_xinfo)[sweeps][tree_ind] = (*c.state)->sigma_vec[0];
      (*c.sigma1_draw_xinfo)[sweeps][tree_ind] = (*c.state)->sigma_vec[1];
      // cout << "sigma = " << (*c.state)->sigma_vec << endl;

      if ((*c.state)->use_all && (sweeps > (*c.state)->burnin) &&
      ((*c.state)->mtry_pr != (*c.state)->p_pr))
      {
        (*c.state)->use_all = false;
      }

      // clear counts of splits for one tre
      std::fill((*c.state)->split_count_current_tree.begin(),
      (*c.state)->split_count_current_tree.end(), 0.0);

      if ((*c.state)->sample_weights_flag)
      {
        // subtract old tree for sampling case
        (*c.state)->mtry_weight_current_tree = (*c.state)->mtry_weight_current_tree -
        (*c.state)->split_count_all_tree_pr[tree_ind];
      }

      // get partial mu_fit -- thus take out the old fitted values
      c.model_ps->subtract_old_tree_fit(tree_ind, (*c.state)->mu_fit, (*c.x_struct_ps));
      // initialize suff stat using partial fit
      c.model_ps->initialize_root_suffstat((*c.state),
      (*c.trees_ps)[sweeps][tree_ind].suff_stat);
      // cout << "root suffstat = " << (*c.trees_ps)[sweeps][tree_ind].suff_stat << endl;
      // GFR
      (*c.trees_ps)[sweeps][tree_ind].grow_from_root((*c.state), (*c.split_ps), c.model_ps,
      (*c.x_struct_ps), sweeps, tree_ind, c.split_time_ps);
      c.model_ps->state_sweep(tree_ind, (*c.state)->mu_fit, (*c.x_struct_ps));  // update total mu_fit by adding just fitted values
      // cout << "finish prognostic tree" << endl;

      (*c.state)->update_split_counts(tree_ind, 0);  // update split counts for mu 
    }

    c.model_ps->set_state_status((*c.state), 1, (*c.x_struct_trt)->X_std, (*c.split_trt)->Xorder_std, (*c.x_struct_trt)->t_std);
    
    ////////////// Treatment term loop
    for (size_t tree_ind = 0; tree_ind < (*c.state)->num_trees_vec[1]; tree_ind++)
    {
       if (c.verbose == true)
      {
        COUT << "--------------------------------" << endl;
        COUT << "number of treatment trees " << tree_ind << endl;
        COUT << "--------------------------------" << endl;
      }
      // cout << "beta_t = " << (*c.state)->beta_t << endl;
      // cout << "b_vec = " << (*c.state)->b_vec << endl;
      (*c.state)->update_residuals(); // update residuals
      c.model_trt->draw_sigma((*c.state), 1); // draw sigmas (and update them in the (*c.state) obj)

      // store sigma draws
      (*c.sigma0_draw_xinfo)[sweeps][(*c.state)->num_trees_vec[0]+tree_ind] = (*c.state)->sigma_vec[0]; // storing sigmas
      (*c.sigma1_draw_xinfo)[sweeps][(*c.state)->num_trees_vec[0]+tree_ind] = (*c.state)->sigma_vec[1]; // storing sigmas

      if ((*c.state)->use_all && (sweeps > (*c.state)->burnin) && ((*c.state)->mtry_trt != (*c.state)->p_trt))
      {
        (*c.state)->use_all = false;
      }

      std::fill((*c.state)->split_count_current_tree.begin(), (*c.state)->split_count_current_tree.end(), 0.0); // clear counts of splits for one tree

      if ((*c.state)->sample_weights_flag)
      {
        (*c.state)->mtry_weight_current_tree = (*c.state)->mtry_weight_current_tree - (*c.state)->split_count_all_tree_trt[tree_ind]; // subtract old tree for sampling case
      }

      c.model_trt->subtract_old_tree_fit(tree_ind, (*c.state)->tau_fit, (*c.x_struct_trt)); // for GFR we will need partial tau_fit -- thus take out the old fitted values

      c.model_trt->initialize_root_suffstat((*c.state), (*c.trees_trt)[sweeps][tree_ind].suff_stat); // initialize suff stat using partial fit
      // cout << "root suffstat = " << (*c.trees_trt)[sweeps][tree_ind].suff_stat << endl;
      // GFR
      (*c.trees_trt)[sweeps][tree_ind].grow_from_root((*c.state), (*c.split_trt), c.model_trt, (*c.x_struct_trt), sweeps, tree_ind, c.split_time_trt);
      // cout << "finish treatment " << tree_ind << endl;

      c.model_trt->state_sweep(tree_ind, (*c.state)->tau_fit, (*c.x_struct_trt)); // update total tau_fit by adding just fitted values

      (*c.state)->update_split_counts(tree_ind, 1); // update split counts for tau
      
    }

    if (sweeps != 0)
    {
      if (c.a_scaling) // in case c.b_scaling on, we update b0 and b1
      {
        c.model_ps->update_a_value((*c.state));
      }
      if (c.b_scaling) // in case c.b_scaling on, we update b0 and b1
      {
        c.model_trt->update_b_values((*c.state));
      }
    }

    // TODO: replace torder with sorder
    c.model_ps->update_time_coef((*c.state), (*c.x_struct_gp), (*c.split_gp)->sorder_std,
      (*c.resid_info)[sweeps], (*c.A_diag_info)[sweeps], (*c.Sig_diag_info)[sweeps], (*c.beta_info)[sweeps]); 
    // cout << "beta " << (*c.state)->beta_t << endl;

    // Unit random intercepts. Drawn last so that they condition on this
    // sweep's forests, scalings and time coefficients; skipped on the first
    // sweep, when the forests are still empty and gamma would swallow the
    // whole mean structure. Order matters within the block: gamma first,
    // then its prior scale, otherwise sigma_gamma is drawn from all-zero
    // gammas and collapses to zero.
    if (sweeps != 0)
    {
      c.model_ps->update_random_intercept((*c.state));
      c.model_ps->update_sigma_gamma((*c.state));
      c.model_ps->update_ar1((*c.state));
    }
    std::copy((*c.state)->gamma.begin(), (*c.state)->gamma.end(), (*c.gamma_xinfo)[sweeps].begin());
    (*c.sigma_gamma_draws)[sweeps] = (*c.state)->sigma_gamma;
    (*c.beta_mean_draws)[sweeps]   = (*c.state)->beta_mean;

    std::copy((*c.state)->beta_t.begin(), (*c.state)->beta_t.end(),
    (*c.beta_xinfo)[sweeps].begin());
    // store draws for b0, b1 and a, although they are updated per tree, we save results per forest (sweep)
    (*c.b_xinfo)[0][sweeps] = (*c.state)->b_vec[0];
    (*c.b_xinfo)[1][sweeps] = (*c.state)->b_vec[1];
    (*c.a_xinfo)[0][sweeps] = (*c.state)->a;
    // cout << "finish " << endl;
}


void mcmc_loop_multi(std::vector<SweepCtx> &ctx, size_t num_sweeps,
                     double gamma_prior_var, matrix<double> &gamma_draws)
{
  const size_t M = ctx.size();
  bool any_parallel = false;
  for (size_t m = 0; m < M; m++) any_parallel |= (*ctx[m].state)->parallel;
  if (any_parallel) thread_pool.start();

  State &s0 = **ctx[0].state;
  const size_t N = s0.n_y, P = s0.p_y, NP = N * P;

  std::vector<std::vector<double>> raw(M, std::vector<double>(NP, 0.0));
  std::vector<std::vector<double>> Gamma(M);
  for (size_t m = 0; m < M; m++) Gamma[m].assign(m, 0.0);

  for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
  {
    for (size_t m = 0; m < M; m++)
    {
      State &st = **ctx[m].state;

      // Orthogonalized target: subtract what earlier outcomes' shocks explain.
      //
      // Except for a probit. Its latent scale is not identified, so sigma is
      // pinned at 1; orthogonalizing genuinely lowers the innovation variance
      // but the model cannot represent that, and the mismatch shows up as a
      // worse effect estimate -- measured at about 10% on a three-outcome
      // panel. A binary equation is therefore fitted exactly as it would be
      // alone. It still shares the sampler, so its draws remain paired with
      // the others' for joint inference, which is the point of fitting
      // together in the first place.
      const bool skip_offset = st.binary_outcome;
      for (size_t k = 0; k < NP; k++)
      {
        double off = 0.0;
        if (!skip_offset)
          for (size_t l = 0; l < m; l++) off += Gamma[m][l] * raw[l][k];
        st.sur_offset[k] = off;
      }
      st.refresh_y_work();

      mcmc_one_sweep(sweeps, ctx[m]);

      // Raw residual of this equation: y minus everything the model fitted
      // to it, but NOT minus the offset -- that is what the next equations
      // condition on.
      for (size_t j = 0; j < P; j++)
      {
        for (size_t i = 0; i < N; i++)
        {
          const size_t k = j * N + i;
          const bool treated = (*(st.z + k) == 1);
          const double b = treated ? st.b_vec[1] : st.b_vec[0];
          raw[m][k] = st.y_orig[k]
                    - st.a * st.mu_fit[i][j]
                    - b * st.beta_fit[i][j] * st.tau_fit[i][j]
                    - st.gamma[i]
                    - (st.ar1_errors ? st.u[k] : 0.0);
        }
      }

      // Gamma_m | rest: conjugate ridge regression of raw[m] on raw[0..m-1],
      // weighted by this equation's own precision per cell.
      if (m > 0 && sweeps > 0 && gamma_prior_var > 0 && !st.binary_outcome)
      {
        arma::mat  XtWX(m, m, arma::fill::zeros);
        arma::vec  XtWy(m, arma::fill::zeros);
        for (size_t k = 0; k < NP; k++)
        {
          const bool treated = (*(st.z + k) == 1);
          const double s2 = pow(treated ? st.sigma_vec[1] : st.sigma_vec[0], 2);
          const double w  = 1.0 / s2;
          for (size_t a = 0; a < m; a++)
          {
            XtWy(a) += w * raw[a][k] * raw[m][k];
            for (size_t bb = 0; bb <= a; bb++)
              XtWX(a, bb) += w * raw[a][k] * raw[bb][k];
          }
        }
        XtWX = arma::symmatl(XtWX);
        arma::mat prec = XtWX + arma::eye(m, m) / gamma_prior_var;
        arma::mat V    = arma::inv_sympd(prec);
        arma::vec mu   = V * XtWy;
        arma::mat L    = arma::chol(V, "lower");
        arma::vec zdraw(m);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (size_t a = 0; a < m; a++) zdraw(a) = nd(st.gen);
        arma::vec g = mu + L * zdraw;
        for (size_t a = 0; a < m; a++) Gamma[m][a] = g(a);
      }
    }

    // Store the loadings for this sweep, row-major, unit diagonal.
    for (size_t r = 0; r < M; r++)
      for (size_t cidx = 0; cidx < M; cidx++)
        gamma_draws[sweeps][r * M + cidx] =
          (r == cidx) ? 1.0 : ((cidx < r) ? Gamma[r][cidx] : 0.0);
  }

  if (any_parallel) thread_pool.stop();
}
