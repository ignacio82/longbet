#include "tree.h"
#include "model.h"
// #include <armadillo>
#include <cstddef>
#include <memory>
#include <numeric>

using namespace arma;

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  XBCF Model
//
//
//////////////////////////////////////////////////////////////////////////////////////

// adds residual to suff stats
// called from calcSuffStat_categorical, calcSuffStat_continuous in tree.cpp
// void longbetModel::incSuffStat(std::unique_ptr<State> &state,
// size_t index_next_obs, matrix<double> &suffstats)
// {
//   // TODO: reconstruct this for n*t y matrix
//   double gp;
//   double resid;
//   for (size_t j = 0; j < state->p_y; j++){
//     gp = state->z[index_next_obs][j];  // treatment group
//     resid = *(state->y_std + state->n_y * j + index_next_obs) -
//     state->a * state->mu_fit[index_next_obs][j] - state->b_vec[gp] *
//     state->beta_t[j] * state->tau_fit[index_next_obs][j];

//     if (state->fl == 0)  // suff stat for prognostic trees
//     {
//       if (state->z[index_next_obs][j] == 1)
//       {
//         suffstats[j][1] += resid / state->a;
//         suffstats[j][3] += 1;
//       } else {
//         suffstats[j][0] += resid / state->a;
//         suffstats[j][2] += 1;
//       }
//     } else {  // suff stat for treatment trees
//       if (state->z[index_next_obs][j] == 1)
//       {
//         // beta_t^2 * r / b / beta_t = beta_t * r / b
//         suffstats[j][1] += state->beta_t[j] * resid / state->b_vec[1];
//         suffstats[j][3] += pow(state->beta_t[j], 2);
//       } else {
//         suffstats[j][0] += state->beta_t[j] * resid / state->b_vec[0];
//         suffstats[j][2] += pow(state->beta_t[j], 2);
//       }
//     }
//   }
// }

void longbetModel::incSuffStat(std::unique_ptr<State> &state,
size_t index_next_obs, size_t index_next_t, std::vector<double> &suffstats)
{
  if (index_next_t >= state->p_y)  {
    cout << "index_next_t = " << index_next_t << endl;
    abort();
  }
  double gp = *(state->z + index_next_t * state->n_y + index_next_obs);
  double resid = *(state->y_std + state->n_y * index_next_t + index_next_obs) -
  state->a * state->mu_fit[index_next_obs][index_next_t] - state->b_vec[gp] *
  state->beta_fit[index_next_obs][index_next_t] * state->tau_fit[index_next_obs][index_next_t];

  if (state->fl == 0)  // suff stat for prognostic trees
  {
    // r = resid / a
    // s0 = sum(r)
    // s1 = sum(r)
    // s2 = n0  
    // s3 = n1
    if (gp == 1)
    {
      suffstats[1] += resid / state->a;
      suffstats[3] += 1;
    } else {
      suffstats[0] += resid / state->a;
      suffstats[2] += 1;
    }
  } else {
    // r = resid / b / beta
    // s0 = sum(beta^2 * r)
    // s1 = sum(beta^2 * r)
    // s2 = sum(beta^2)
    // s3 = sum(beta^2)
    if (gp == 1)
    {
      // beta_t^2 * r / b / beta_t = beta_t * r / b
      suffstats[1] += state->beta_fit[index_next_obs][index_next_t] * resid / state->b_vec[1];
      suffstats[3] += pow(state->beta_fit[index_next_obs][index_next_t], 2);
    } else {
      suffstats[0] += state->beta_fit[index_next_obs][index_next_t] * resid / state->b_vec[0];
      suffstats[2] += pow(state->beta_fit[index_next_obs][index_next_t], 2);
    }
  }

}

// samples leaf parameter
// called from GFR in tree.cpp
void longbetModel::samplePars(std::unique_ptr<State> &state,
std::vector<double> &suff_stat, std::vector<double> &theta_vector,
double &prob_leaf)
{
  std::normal_distribution<double> normal_samp(0.0, 1.0);
  double s0 = 0;
  double s1 = 0;

  if (state->fl == 0)  // no sum of beta_t sufficient for prognostic trees
  {
    s0 = pow(state->a / state->sigma_vec[0], 2);
    s1 = pow(state->a / state->sigma_vec[1], 2);
  } else {
    s0 = pow(state->b_vec[0] / state->sigma_vec[0], 2);
    s1 = pow(state->b_vec[1] / state->sigma_vec[1], 2);
  }

  double denominator = suff_stat[2] * s0 + suff_stat[3] * s1  + 1 / tau;
  double m1 = (suff_stat[0] * s0 + suff_stat[1] * s1) / denominator;
  double v1 = 1 / denominator;
  // cout << "s0 = " << s0 << "; s1 = " << s1 << "; m1 = " << m1 << "; v1 = " << v1 << endl;
  // cout << "suff0 = " << suff_stat[0] << "; suff1 = " << suff_stat[1] << "; suff2 = " << suff_stat[2] << "; suff3 = " << suff_stat[3] << "; denom = " << denominator << endl;

  // sample leaf parameter
  theta_vector[0] = m1 + sqrt(v1) * normal_samp(state->gen);
  // cout << "suff_stat = " << suff_stat << ", s0 = " << s0 << ", s1 = " << s1 << endl;
  // cout << "m1 = " << m1 << ", v1 = " << v1 << endl;
  // cout << "theta = " << theta_vector[0] << endl;

  // also update probability of leaf parameters
  prob_leaf = 1.0;
  if (isnan(theta_vector[0])) {
    cout << "theta is nan, m1 = " << m1 << ", v1 = " << v1 << endl;
    cout << "suff_stat = " << suff_stat << ", s0 = " << s0 << ", s1 = " << s1 << endl;
    abort();
  }
}

// updates sigmas (new)
void longbetModel::draw_sigma(std::unique_ptr<State> &state, size_t ind)
{
  // A probit has no free residual scale: the latent variance is what fixes
  // the link, so sigma stays at 1 rather than being sampled.
  if (state->binary_outcome)
  {
    state->update_sigma(1.0, ind);
    return;
  }

  double m1 = 0;
  double v1 = 0;
  double sigma;
  double squared_resid = 0;
  double N_trt = std::accumulate(state->n_trt.begin(), state->n_trt.end(), 0.0);

  if (ind == 0){  // update sigma0
    for (size_t i = 0; i < state->p_y; i++){
      squared_resid += sum_squared(state->full_residual_ctrl[i]);
    }
    m1 = (state->n_y * state->p_y - N_trt + kap) / 2.0;
    v1 =  2.0 / (squared_resid + s);
  } else {
    for (size_t i = 0; i < state->p_y; i++){
      squared_resid += sum_squared(state->full_residual_trt[i]);
    }
    m1 = (N_trt + kap) / 2.0;
    v1 = 2.0 / (squared_resid + s);
  }

  // computing both sigmas here due to structural complexity of splitting them
  std::gamma_distribution<double> gamma_samp(m1, v1);
  sigma = 1.0 / sqrt(gamma_samp(state->gen));

  // update the corresponding value in the state object
  state->update_sigma(sigma, ind);
}

// initializes root suffstats
// called from mcmc_loop_longbet in mcmc_loop.cpp
void longbetModel::initialize_root_suffstat(std::unique_ptr<State> &state,
std::vector<double> &suff_stat)
{
  // ini_matrix(suff_stat, 4, state->p_y);
  // for (size_t i = 0; i < state->p_y; i++)
  //   std::fill(suff_stat[i].begin(), suff_stat[i].end(), 0.0);
  suff_stat.resize(4);
  std::fill(suff_stat.begin(), suff_stat.end(), 0.0);

  for (size_t i = 0; i < state->n_y; i++)
  {
    for (size_t j = 0; j < state->p_y; j++)
    {
      incSuffStat(state, i, j, suff_stat);
    }
  }
}

// updates node suffstats for the split
// called from split_xorder_std_continuous, split_xorder_std_categorical in tree.cpp
// it is executed after suffstats for the node has been initialized by suff_stats_ini [defined in tree.h]
void longbetModel::updateNodeSuffStat(std::vector<double> &suff_stat, std::unique_ptr<State> &state, matrix<size_t> &Xorder_std, matrix<size_t> &torder_std, size_t &split_var, size_t row_ind)
{
  for (auto i: torder_std[Xorder_std[split_var][row_ind]]){
    incSuffStat(state, Xorder_std[split_var][row_ind], i, suff_stat);
  }
}

// updates the other side node's side suffstats for the split
// called from split_xorder_std_continuous, split_xorder_std_categorical in tree.cpp
void longbetModel::calculateOtherSideSuffStat(std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, std::vector<double> &rchild_suff_stat, bool &compute_left_side)
{

  // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
  // this function calculate the other side based on parent and the other child

  if (compute_left_side)
  {
    rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
  }
  else
  {
    lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
  }
}

// updates partial residual for the next tree to fit
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void longbetModel::state_sweep(size_t tree_ind, matrix<double> &fit, std::unique_ptr<X_struct> &x_struct) const
{
  matrix<double> mu_ft;
  ini_matrix(mu_ft, fit[0].size(), fit.size());
  for (size_t i = 0; i < fit.size(); i++)
  {
    for (size_t j = 0; j < fit[i].size(); j++){
      // fit[i][j] += (*(x_struct->data_pointers[tree_ind][i * fit[i].size() + j]))[0];
      mu_ft[i][j] = (*(x_struct->data_pointers[tree_ind][i * fit[i].size() + j]))[0];
      fit[i][j] += mu_ft[i][j];
    }
  }

  for (size_t i = 0; i < fit.size(); i++){
    // cout << "mu " << i << " = " << mu_ft[i] << endl;
  }
}

// computes likelihood of a split
// called from GFR in tree.cpp
double longbetModel::likelihood(std::vector<double> &temp_suff_stat,
std::vector<double> &suff_stat_all, bool left_side,
bool no_split, std::unique_ptr<State> &state) const
{
  // helper variables
  double s0 = 0;
  double s1 = 0;
  double denominator = 1;   // (1 + tau * precision_squared)
  double s_psi_squared = 0;  // (residual * precision_squared)^2

  if (state->fl == 0)  // no sum of beta_t sufficient for prognostic trees
  {
    s0 = pow(state->a, 2) / pow(state->sigma_vec[0], 2);
    s1 = pow(state->a, 2) / pow(state->sigma_vec[1], 2);
  } else {
    s0 = pow(state->b_vec[0], 2) / pow(state->sigma_vec[0], 2);
    s1 = pow(state->b_vec[1], 2) / pow(state->sigma_vec[1], 2);
  }

  if (no_split)
  {
    denominator = 1 + tau  * (suff_stat_all[2] * s0 + suff_stat_all[3] * s1);
    s_psi_squared = suff_stat_all[0] * s0 + suff_stat_all[1] * s1;
  } else {
    if (left_side)
    {
      denominator = 1 + tau * (temp_suff_stat[2] * s0 + temp_suff_stat[3]*s1);
      s_psi_squared = temp_suff_stat[0] * s0 + temp_suff_stat[1] * s1;
    } else {
      denominator = 1 + tau * ((suff_stat_all[2] - temp_suff_stat[2])
      * s0 + (suff_stat_all[3] - temp_suff_stat[3]) * s1);
      s_psi_squared = (suff_stat_all[0] - temp_suff_stat[0]) * s0 +
      (suff_stat_all[1] - temp_suff_stat[1]) * s1;
    }
  }

  return 0.5 * log(1 / denominator) +
  0.5 * pow(s_psi_squared, 2) * tau / denominator;
}

// makes a prediction for treatment effect on the given Xtestpointer data
void longbetModel::predict_std(const double *Xtestpointer, const double *tpointer, size_t N_test, size_t p, size_t num_sweeps, std::vector<matrix<double>> &yhats_test_xinfo, vector<vector<tree>> &trees, const std::vector<const double *> *tv)
{
  std::vector<double> output(this->dim_theta, 0.0);
  for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
  {
    for (size_t data_ind = 0; data_ind < N_test; data_ind++)
    {
      for (size_t time_ind = 0; time_ind < p; time_ind++)
      {
        for (size_t tree_ind  = 0; tree_ind < trees[0].size(); tree_ind++){
          // cout << "data = " << data_ind << ", time = " << time_ind << ", tree = " << tree_ind << endl;
          getThetaForObs_Outsample(output, trees[sweeps][tree_ind],
          data_ind, time_ind, Xtestpointer, tpointer, N_test, p, tv);

          yhats_test_xinfo[sweeps][data_ind][time_ind] += output[0];
        }
      }
    }
  }
}

// updates parameter a
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
// Conjugate draw of the unit-level random intercepts.
//
// The model is  y_it = a*mu_it + b_{z}*beta_it*tau_it + gamma_i + eps_it,
// with gamma_i ~ N(0, sigma_gamma^2) and eps_it ~ N(0, sigma_{z_it}^2).
// Conditional on everything else the residual r_it = y_it - a*mu_it -
// b*beta_it*tau_it equals gamma_i + eps_it, so each gamma_i has an exact
// independent Gaussian full conditional. Cost is O(n * T).
//
// Note that gamma_i is drawn *conditional on the current treatment fit*, so a
// unit's post-treatment periods do not drag its baseline upward the way naive
// within-unit demeaning would. Units observed only under treatment have no
// information separating gamma_i from tau, and the prior is what holds them
// apart; the R wrapper warns when that happens.
void longbetModel::update_random_intercept(std::unique_ptr<State> &state)
{
  if (!state->random_intercept) return;

  const double sig02 = pow(state->sigma_vec[0], 2);
  const double sig12 = pow(state->sigma_vec[1], 2);
  const double prior_prec = (state->sigma_gamma > 0) ?
      1.0 / pow(state->sigma_gamma, 2) : 0.0;

  std::normal_distribution<double> normal_samp(0.0, 1.0);

  for (size_t i = 0; i < state->n_y; i++)
  {
    double prec = prior_prec;
    double wsum = 0.0;

    for (size_t j = 0; j < state->p_y; j++)
    {
      const bool treated = (*(state->z + j * state->n_y + i) == 1);
      const double s2 = treated ? sig12 : sig02;
      const double b  = treated ? state->b_vec[1] : state->b_vec[0];

      // Residual against the *original* outcome, i.e. gamma_i is still in it.
      // The SUR offset is removed too: gamma_i belongs to this equation's
      // own error, not to the part already explained by earlier outcomes.
      double r = state->y_orig[j * state->n_y + i]
               - state->sur_offset[j * state->n_y + i]
               - state->a * state->mu_fit[i][j]
               - b * state->beta_fit[i][j] * state->tau_fit[i][j]
               - (state->treat_effect_re && treated ? state->delta[i] : 0.0);

      prec += 1.0 / s2;
      wsum += r / s2;
    }

    const double post_var  = 1.0 / prec;
    const double post_mean = wsum * post_var;
    state->gamma[i] = post_mean + sqrt(post_var) * normal_samp(state->gen);
  }

  // Everything downstream reads y through state->y_std.
  state->refresh_y_work();
}

// Draw a standard normal truncated to [a, infinity), by Robert's (1995)
// algorithm: plain rejection when the bound is not binding, exponential
// rejection when it is. No inverse CDF and no external dependency.
static double rtruncnorm_lower(double a, std::mt19937 &gen)
{
  std::normal_distribution<double> norm(0.0, 1.0);
  if (a <= 0.0)
  {
    // Acceptance probability is at least 1/2 here.
    for (int it = 0; it < 1000; it++)
    {
      double x = norm(gen);
      if (x >= a) return x;
    }
    return a;
  }
  const double alpha = 0.5 * (a + sqrt(a * a + 4.0));
  std::exponential_distribution<double> expo(alpha);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  for (int it = 0; it < 1000; it++)
  {
    double x = a + expo(gen);
    double rho = exp(-0.5 * (x - alpha) * (x - alpha));
    if (unif(gen) <= rho) return x;
  }
  return a;
}

void longbetModel::draw_latent_outcome(std::unique_ptr<State> &state)
{
  if (!state->has_missing && !state->binary_outcome) return;

  std::normal_distribution<double> normal_samp(0.0, 1.0);

  for (size_t j = 0; j < state->p_y; j++)
  {
    for (size_t i = 0; i < state->n_y; i++)
    {
      const size_t k = j * state->n_y + i;
      const bool missing = (state->y_missing[k] == 1);
      if (!missing && !state->binary_outcome) continue;

      const bool treated = (*(state->z + k) == 1);
      const double sd = treated ? state->sigma_vec[1] : state->sigma_vec[0];
      const double b  = treated ? state->b_vec[1] : state->b_vec[0];

      // y_orig is the full latent, so its mean carries everything that is
      // later subtracted to form y_work -- including the SUR offset. Zero for
      // a single outcome.
      const double fitted = state->a * state->mu_fit[i][j]
                          + b * state->beta_fit[i][j] * state->tau_fit[i][j]
                          + state->gamma[i]
                          + state->sur_offset[k]
                          + (state->treat_effect_re && treated ? state->delta[i] : 0.0);

      if (state->binary_outcome && !missing)
      {
        // The sampler holds the latent minus binary_offset, so a success means
        // centred > -offset rather than centred > 0. sd is 1 on this scale.
        const double bound = -state->binary_offset - fitted;
        if (state->y_binary[k] == 1)
        {
          state->y_orig[k] = fitted + rtruncnorm_lower(bound, state->gen);
        }
        else
        {
          state->y_orig[k] = fitted - rtruncnorm_lower(-bound, state->gen);
        }
      }
      else
      {
        state->y_orig[k] = fitted + sd * normal_samp(state->gen);
      }
    }
  }
  state->refresh_y_work();
}

// sigma_gamma^2 | gamma ~ InvGamma(a + n/2, b + sum(gamma^2)/2).
// Conjugate Gibbs draw of the unit-level treatment random effects delta_i.
// For unit i the treated cells satisfy r_it = delta_i + e_it with
// e_it ~ N(0, sigma_1^2), against the prior N(m_i, v) held on the state. Units
// with no treated cell get a draw from the prior: it does not enter their
// (nonexistent) effect, and it keeps the Psi update over the whole vector
// coherent when several outcomes are fitted together.
void longbetModel::update_delta(std::unique_ptr<State> &state)
{
  if (!state->treat_effect_re) return;

  const double sig12 = pow(state->sigma_vec[1], 2);
  const double v     = state->delta_prior_var;
  const double prior_prec = (v > 0) ? 1.0 / v : 0.0;
  std::normal_distribution<double> normal_samp(0.0, 1.0);

  for (size_t i = 0; i < state->n_y; i++)
  {
    double prec = prior_prec;
    double wsum = prior_prec * state->delta_prior_mean[i];

    for (size_t j = 0; j < state->p_y; j++)
    {
      const size_t k = j * state->n_y + i;
      if (*(state->z + k) != 1) continue;
      const double b = state->b_vec[1];
      // Residual with everything except delta removed.
      const double r = state->y_orig[k]
                     - state->sur_offset[k]
                     - state->a * state->mu_fit[i][j]
                     - b * state->beta_fit[i][j] * state->tau_fit[i][j]
                     - state->gamma[i];
      prec += 1.0 / sig12;
      wsum += r / sig12;
    }

    if (prec <= 0) { state->delta[i] = state->delta_prior_mean[i]; continue; }
    const double post_var  = 1.0 / prec;
    const double post_mean = wsum * post_var;
    state->delta[i] = post_mean + sqrt(post_var) * normal_samp(state->gen);
  }
  state->refresh_y_work();
}

// Single-outcome prior scale for delta: sigma_delta^2 ~ IG(a, b), updated
// from the deltas of units that were actually treated. Never-treated units
// carry only a prior draw and would pull the estimate toward the prior.
void longbetModel::update_sigma_delta(std::unique_ptr<State> &state)
{
  if (!state->treat_effect_re) return;
  double ss = 0.0; size_t n_tr = 0;
  for (size_t i = 0; i < state->n_y; i++)
  {
    bool any = false;
    for (size_t j = 0; j < state->p_y && !any; j++)
      any = (*(state->z + j * state->n_y + i) == 1);
    if (!any) continue;
    ss += pow(state->delta[i], 2); n_tr++;
  }
  // Half-Cauchy(0, A) on sigma_delta, via Makalic & Schmidt (2016):
  //   sigma^2 | xi ~ IG(1/2, 1/xi),   xi ~ IG(1/2, 1/A^2).
  // Compared with the IG(1, 0.1) this replaces, it puts far more mass at
  // zero, which is what stops a set of noisy deltas from sustaining their own
  // scale when the true unit heterogeneity is nil. A is on the standardized
  // outcome scale; 0.5 is "an effect the size of half a residual sd", which
  // is generous for the tail and still tight at the origin.
  const double A2 = pow(state->delta_prior_b, 2);   // delta_prior_b now = A
  const double s2 = pow(state->sigma_delta, 2);
  std::gamma_distribution<double> gxi(1.0, 1.0 / (1.0 / A2 + 1.0 / s2));
  const double xi = 1.0 / gxi(state->gen);                       // IG(1, ...)
  std::gamma_distribution<double> gs(0.5 * (1.0 + (double) n_tr),
                                     1.0 / (1.0 / xi + 0.5 * ss));
  const double prec = gs(state->gen);                            // 1/sigma^2
  state->sigma_delta = (prec > 0 && std::isfinite(prec)) ? 1.0 / sqrt(prec) : state->sigma_delta;
  state->delta_prior_var = pow(state->sigma_delta, 2);
}

void longbetModel::update_sigma_gamma(std::unique_ptr<State> &state)
{
  if (!state->random_intercept) return;

  double ss = 0.0;
  for (size_t i = 0; i < state->n_y; i++)
  {
    ss += state->gamma[i] * state->gamma[i];
  }

  const double shape = state->gamma_prior_a + 0.5 * (double)state->n_y;
  const double rate  = state->gamma_prior_b + 0.5 * ss;

  // std::gamma_distribution is parameterised by (shape, scale).
  std::gamma_distribution<double> gamma_samp(shape, 1.0 / rate);
  const double prec = gamma_samp(state->gen);
  state->sigma_gamma = 1.0 / sqrt(prec);
}

void longbetModel::update_a_value(std::unique_ptr<State> &state)
{
  std::normal_distribution<double> normal_samp(0.0, 1.0);

  double mu2sum_ctrl = 0;
  double mu2sum_trt = 0;
  double muressum_ctrl = 0;
  double muressum_trt = 0;
  double s0 = pow(state->sigma_vec[0], 2);
  double s1 = pow(state->sigma_vec[1], 2);
  size_t s = 0;

  // compute the residual y-b*beta_t*tau(x)
  for (size_t i = 0; i < state->n_y; i++)
  {
    for (size_t j = 0; j < state->p_y; j++){
      if ((*(state->z + j * state->n_y + i)) == 1)
      {
        state->residual[i][j] = *(state->y_std + state->n_y * j + i) -
        state->b_vec[1] * state->beta_fit[i][j] * state->tau_fit[i][j];

        mu2sum_trt += state->mu_fit[i][j] * state->mu_fit[i][j];
        muressum_trt += state->mu_fit[i][j] * state->residual[i][j];
      } else {
        state->residual[i][j] = *(state->y_std + state->n_y * j + i) -
        state->b_vec[0] * state->beta_fit[i][j] * state->tau_fit[i][j];

        mu2sum_ctrl += state->mu_fit[i][j] * state->mu_fit[i][j];
        muressum_ctrl += state->mu_fit[i][j] * state->residual[i][j];
      }
    }
  }

  // update parameters
  double denominator = mu2sum_ctrl / s0 + mu2sum_trt / s1 + 1;
  double m1 = (muressum_ctrl / s0 + muressum_trt / s1) / denominator;
  double v1 = 1 / denominator;

  // sample a
  state->a = m1 + sqrt(v1) * normal_samp(state->gen);
}

// updates parameters b0, b1
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void longbetModel::update_b_values(std::unique_ptr<State> &state)
{
  std::normal_distribution<double> normal_samp(0.0, 1.0);

  double tau2sum_ctrl = 0;
  double tau2sum_trt = 0;
  double tauressum_ctrl = 0;
  double tauressum_trt = 0;
  double s0 = pow(state->sigma_vec[0], 2);
  double s1 = pow(state->sigma_vec[1], 2);
  size_t s = 0;

  for (size_t i = 0; i < state->n_y; i++)
  {
    for (size_t j = 0; j < state->p_y; j++){
      state->residual[i][j] = *(state->y_std + state->n_y * j + i) -
      state->a * state->mu_fit[i][j];

      if (*(state->z + j * state->n_y + i) == 1)
      {
        tau2sum_trt += pow(state->tau_fit[i][j] * state->beta_fit[i][j], 2);
        tauressum_trt += state->beta_fit[i][j] * state->tau_fit[i][j] *
        state->residual[i][j];
      } else {
        tau2sum_ctrl += pow(state->tau_fit[i][j] * state->beta_fit[i][j], 2);
        tauressum_ctrl += state->beta_fit[i][j] * state->tau_fit[i][j] *
        state->residual[i][j];
      }
    }
  }

  // update parameters
  double v0 = 1 / (2 + tau2sum_ctrl / s0);
  double v1 = 1 / (2 + tau2sum_trt / s1);

  double m0 = v0 * (tauressum_ctrl) / s0;
  double m1 = v1 * (tauressum_trt) / s1;

  // sample b0, b1
  double b0 = m0 + sqrt(v0) * normal_samp(state->gen);
  double b1 = m1 + sqrt(v1) * normal_samp(state->gen);

  state->b_vec[1] = b1;
  state->b_vec[0] = b0;
}

void longbetModel::update_time_coef(std::unique_ptr<State> &state, std::unique_ptr<X_struct> &x_struct, matrix<size_t> &sorder_std, std::vector<double> &resid, std::vector<double> &diag, std::vector<double> &sig, std::vector<double> &beta)
{  
  // get total number of time
  double t_size = x_struct->s_values.size();
  double n = state->n_y;  // n obs per period. TODO: need update

  std::vector<double> res_ctrl(t_size, 0);  // residuals
  std::vector<double> res_trt(t_size, 0);

  // diagonal element of matrix A: sigma_{z_i}^{-1} * b_{z_i} * tau_i
  std::vector<double> diag_ctrl(t_size, 0);
  std::vector<double> diag_trt(t_size, 0);
  // std::vector<double> diag(t_size, 0);

  double sig02 = pow(state->sigma_vec[0], 2);
  double sig12 = pow(state->sigma_vec[1], 2);
  // vec sig(t_size, fill::zeros);


  std::vector<size_t> idx(state->p_y);  // keep track of t-values
  size_t t_idx;
  size_t counts = 0;
  size_t s;
  const double *z_pointer;
  const double *y_pointer;

  std::vector<size_t> t_counts(t_size, 0);

  for (size_t i = 0; i < state->n_y; i++){
    for (size_t j = 0; j < state->p_y; j++){
      s = *(state->post_trt_time + j * state->n_y + i);
      t_counts[s] += 1;
      if (*(state->z + state->n_y * j + i) == 0){
        res_ctrl[s] += *(state->y_std + state->n_y * j + i) - state->a * state->mu_fit[i][j];
        diag_ctrl[s] += state->tau_fit[i][j];
        sig[s] += sig02;
      } else {
        res_trt[s] += *(state->y_std + state->n_y * j + i) - state->a * state->mu_fit[i][j];
        diag_trt[s] += state->tau_fit[i][j];
        sig[s] += sig12;
      }
    }
  }

  for (size_t i = 0; i < t_size; i++){
    resid[i] = (res_trt[i] + res_ctrl[i]) / t_counts[i];
    diag[i] = (state->b_vec[1] * diag_trt[i] + state->b_vec[0] * diag_ctrl[i])/ t_counts[i];
    sig[i] = sig[i] / pow(t_counts[i], 2) ;
  }
  // cout << "res " << resid  << endl;
  // cout << "diag " << diag << endl;
  // cout << "sig " << sig << endl;
  // solve by var = (Sigma0^-1 + Sigma^-1)^-1
  // Sigma0 = A*cov_kernel*A'
  // Sigma = diag(sig)
  // mu = var * (Sigma0^-1 * mu0 + res)
  arma::mat Sigma0(t_size, t_size);
  // arma::mat Sigma_inv(t_size, t_size);
  arma::mat Sigma_inv = diagmat(conv_to<mat>::from(sig));
  for (size_t i = 0; i < t_size; i++){
    for (size_t j = 0; j < t_size; j++){
      Sigma0(i, j) = diag[i] * x_struct->cov_kernel[i][j] * diag[j];
    }
    Sigma_inv(i, i) = 1 / sig[i];
  }

  arma::mat Sigma0_inv = pinv(Sigma0);
  arma::mat var_inv = Sigma0_inv + Sigma_inv;
  arma::mat var = pinv(var_inv);

  // Prior mean of beta_tilde = A * beta. With a zero-mean GP this is zero and
  // everything below reduces to the original code.
  arma::mat mu0(t_size, 1, arma::fill::zeros);
  if (state->gp_constant_mean)
  {
    for (size_t i = 0; i < t_size; i++) mu0(i, 0) = diag[i] * state->beta_mean;
  }

  arma::mat U, V;
  arma::vec scale;
  svd(U, scale, V, var);

  // A factor L of a covariance matrix has to satisfy L * L' = var. With
  // var = U * diag(scale) * U', that is L = U * diag(sqrt(scale)); using
  // diag(scale) makes the sampled covariance U * diag(scale^2) * U', which
  // here is orders of magnitude too small and collapses the beta draw onto
  // its conditional mean. clamp() guards against tiny negative eigenvalues
  // returned by the SVD of a numerically semi-definite matrix.
  arma::mat L = U * diagmat(sqrt(arma::clamp(scale, 0.0, arma::datum::inf)));
  // mean
  arma::mat res_vec(t_size, 1);
  for (size_t i = 0; i < t_size; i++){
    res_vec(i, 0) = resid[i];
  }
  arma::mat mu = var * (Sigma_inv * res_vec + Sigma0_inv * mu0);

  std::normal_distribution<double> normal_samp(0.0, 1.0);
  arma::mat draws(t_size, 1);
  for (size_t i = 0; i < t_size; i++){ draws(i, 0) = normal_samp(state->gen); }

  arma::mat beta_tilde = mu + L * draws;

  // beta = diag^-1 * beta_tilde
  // arma::mat beta(t_size, 1);
  for (size_t i = 0; i < t_size; i++){
    // beta[i] = 1; // disable beta for debug
    beta[i] = beta_tilde(i, 0) / diag[i];
    state->beta_t[i] = beta[i];
  }

  // Draw the GP's constant mean from its own conditional. beta ~ N(m*1, K)
  // with m ~ N(0, sig_knl^2) gives m | beta ~ N(S/P, 1/P) for
  // P = 1/sig_knl^2 + 1'K^-1 1 and S = 1'K^-1 beta. The kernel's diagonal is
  // sig_knl^2, which is where the prior variance comes from.
  if (state->gp_constant_mean)
  {
    arma::mat K(t_size, t_size);
    for (size_t i = 0; i < t_size; i++)
      for (size_t j = 0; j < t_size; j++) K(i, j) = x_struct->cov_kernel[i][j];

    arma::mat K_inv = pinv(K);
    arma::mat one(t_size, 1, arma::fill::ones);
    arma::mat beta_col(t_size, 1);
    for (size_t i = 0; i < t_size; i++) beta_col(i, 0) = beta[i];

    const double kernel_var = x_struct->cov_kernel[0][0];
    const double prior_prec = (kernel_var > 0) ? 1.0 / kernel_var : 0.0;
    const double prec = prior_prec + arma::as_scalar(one.t() * K_inv * one);
    const double mean = arma::as_scalar(one.t() * K_inv * beta_col) / prec;

    std::normal_distribution<double> mean_samp(0.0, 1.0);
    state->beta_mean = mean + sqrt(1.0 / prec) * mean_samp(state->gen);
  }

  // // match beta_t to beta_fit
  for (size_t i = 0; i < state->n_y; i++){
    for (size_t j = 0; j < state->p_y; j++){
      state->beta_fit[i][j] = state->beta_t[*(state->post_trt_time + j * state->n_y + i)];
    }
  }

}


// subtracts old tree contribution from the fit
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void longbetModel::subtract_old_tree_fit(size_t tree_ind, matrix<double> &fit, std::unique_ptr<X_struct> &x_struct)
{
  for (size_t i = 0; i < fit.size(); i++)  // N
  {
    for (size_t j = 0; j < fit[i].size(); j++){  // p_y
      fit[i][j] -= (*(x_struct->data_pointers[tree_ind][i * fit[i].size() + j]))[0];
    }
  }
}

// sets unique term parameters in the state object depending on the term being updated
// called from mcmc_loop_xbcf in xbcf_mcmc_loop.cpp
void longbetModel::set_state_status(std::unique_ptr<State> &state, size_t value, const double *X, matrix<size_t> &Xorder, const double *t_std)
{
  state->fl = value; // value can only be 0 or 1 (to alternate between arms)
  state->iniSplitStorage(state->fl);
  state->adjustMtry(state->fl);
  state->X_std = X;
  state->Xorder_std = Xorder;
  state->t_std = t_std;
  if(value == 0)
  {
    state->p = state->p_pr;
    state->p_categorical = state->p_categorical_pr;
    state->p_continuous = state->p_continuous_pr;
  } else {
    state->p = state->p_trt;
    state->p_categorical = state->p_categorical_trt;
    state->p_continuous = state->p_continuous_trt;
  }

}

void longbetModel::predict_beta(std::vector<double> &beta,
  std::vector<double> &res_vec, std::vector<double> &a_vec, std::vector<double> &sig_vec, 
  matrix<double> &Sigma_tr_std, matrix<double> &Sigma_te_std, matrix<double> &Sigma_tt_std,
  std::mt19937 &gen, double beta_mean)
{
  vec a_diag = conv_to<vec>::from(a_vec);
  vec sig_diag = conv_to<vec>::from(sig_vec);
  vec res = conv_to<vec>::from(res_vec);

  size_t tr_size = Sigma_tr_std.size();
  size_t te_size = Sigma_te_std.size();
  mat Sigma_tr(tr_size, tr_size);
  mat Sigma_te(te_size, te_size);
  mat Sigma_tt(tr_size, te_size);

  std_to_arma(Sigma_tr_std, Sigma_tr);
  std_to_arma(Sigma_te_std, Sigma_te);
  std_to_arma(Sigma_tt_std, Sigma_tt);

  mat A = diagmat(a_diag);
  mat Sig = diagmat(sig_diag);
  mat Sig_inv = pinv(Sig + A * Sigma_tr * A.t());
  mat common_mat = Sigma_tt.t() * A.t() * Sig_inv;

  // beta = beta_mean + f with f ~ GP(0, K), so condition on the residual after
  // removing the mean's contribution and add it back to the prediction.
  // beta_mean = 0 recovers the zero-mean GP exactly.
  mat res_adj = res - beta_mean * a_diag;
  
  mat mu = beta_mean + common_mat * res_adj;
  mat var = Sigma_te - common_mat * A * Sigma_tt;

  arma::mat U, V;
  arma::vec s;
  svd(U, s, V, var);

  // See update_time_coef: the covariance factor is U * diag(sqrt(s)).
  arma::mat L = U * diagmat(sqrt(arma::clamp(s, 0.0, arma::datum::inf)));

  std::normal_distribution<double> normal_samp(0.0, 1.0);
  arma::mat draws(te_size, 1);
  for (size_t i = 0; i < te_size; i++){ draws(i, 0) = normal_samp(gen); }

  arma::mat beta_tilde = mu + L * draws;

  for (size_t i = 0; i < te_size; i++){
    beta[i] = beta_tilde(i, 0);
  }
  
}
