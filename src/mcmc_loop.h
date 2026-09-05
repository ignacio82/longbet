#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "model.h"
#include "state.h"
#include "cdf.h"
#include "X_struct.h"
//#include "MH.h"

// Everything one sweep of the sampler needs, bundled so the same body can be
// driven either by the single-outcome loop below or interleaved across
// several outcomes by mcmc_loop_multi. Pointers to the caller's objects; no
// ownership.
struct SweepCtx {
  std::unique_ptr<split_info> *split_ps, *split_trt, *split_gp;
  bool verbose;
  matrix<double> *sigma0_draw_xinfo, *sigma1_draw_xinfo, *b_xinfo, *a_xinfo,
                 *beta_info, *beta_xinfo;
  vector<vector<tree>> *trees_ps, *trees_trt;
  double no_split_penality;
  std::unique_ptr<State> *state;
  longbetModel *model_ps, *model_trt;
  std::unique_ptr<X_struct> *x_struct_ps, *x_struct_trt, *x_struct_gp;
  bool a_scaling, b_scaling, split_time_ps, split_time_trt;
  matrix<double> *resid_info, *A_diag_info, *Sig_diag_info, *gamma_xinfo;
  std::vector<double> *sigma_gamma_draws, *beta_mean_draws;
};

// One full sweep -- both forests, scalings, GP, unit effects -- for one
// context. Does not touch the thread pool.
void mcmc_one_sweep(size_t sweeps, SweepCtx &c);

// Several outcomes fitted jointly (SUR). Each sweep visits the outcomes in
// order; outcome m is fitted to y - sum_{l<m} Gamma_ml * eps_l, with eps_l the
// raw residuals of the outcomes before it, then Gamma_m is redrawn from a
// conjugate ridge regression of its raw residual on theirs. gamma_draws
// receives, per sweep, the lower-triangular loadings flattened row-major
// (M*M entries, zeros above the diagonal, ones on it).
void mcmc_loop_multi(std::vector<SweepCtx> &ctx, size_t num_sweeps,
                     double gamma_prior_var, matrix<double> &gamma_draws);

void mcmc_loop_longbet( 
  std::unique_ptr<split_info> &split_ps,  
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
  );