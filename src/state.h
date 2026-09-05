#ifndef GUARD_fit_info_h
#define GUARD_fit_info_h

#include <algorithm>
#include <cstddef>
#include <ctime>
#include "common.h"
#include "utility.h"
#include <chrono>
#include <vector>

class State
{
public:
    size_t dim_suffstat;


    // residual vectors
    // total residual vector
    matrix<double> residual;
    // residual for treated group, length n_trt
    matrix<double> full_residual_trt;
    matrix<double> full_residual_ctrl;

    std::vector<double> beta_t;
    std::vector<double> t;
    std::vector<double> unique_t;

    // ---- unit-level random intercept ---------------------------------
    // gamma_i soaks up the part of a unit's level that the prognostic
    // forest cannot reconstruct from the time-invariant covariates X_i.
    // y_work = y_orig - gamma_i is what every other part of the sampler
    // reads through y_std, so no other update needs to know gamma exists.
    // ---- constant mean for the beta_S Gaussian process -----------------
    // With a zero mean, a projection far past the observed window reverts to
    // "no effect", because that is what the prior says and not because any
    // data said so. A constant mean, estimated alongside everything else,
    // makes it revert to the level the observed trajectory settled at.
    bool gp_constant_mean;
    double beta_mean;

    // ---- unbalanced panels ---------------------------------------------
    // Cells with no observed outcome. They are filled at the top of every
    // sweep with a draw from their own full conditional, so the forests and
    // every sufficient statistic downstream see a complete panel. This is
    // ordinary Bayesian data augmentation: alternating
    //   y_mis | theta   and   theta | y_obs, y_mis
    // is a valid Gibbs sampler on the joint posterior, which is why nothing
    // else in the sampler has to learn about missingness.
    bool has_missing;
    std::vector<int> y_missing;       // n_y * p_y, 1 = unobserved

    // ---- binary outcomes -----------------------------------------------
    // Albert and Chib data augmentation. y_binary holds the observed 0/1
    // outcome; y_orig holds a latent normal drawn each sweep, truncated to
    // the half-line the observation implies. Everything downstream then runs
    // the ordinary Gaussian machinery on the latent scale with sigma fixed
    // at 1, which is what makes the model a probit.
    bool binary_outcome;
    std::vector<int> y_binary;        // n_y * p_y, 0/1 (ignored where missing)
    // Grand mean of the latent, qnorm(mean(y)). The sampler works on the
    // latent minus this, exactly as it works on a centred continuous outcome,
    // so the forests never have to carry the intercept.
    double binary_offset;

    // ---- AR(1) transitory errors ---------------------------------------
    // y_it = fitted_it + gamma_i + u_it + eps_it with u an AR(1) process.
    // Conditional on the latent path u, the forests see y - gamma - u, whose
    // error is iid again, so XBART's conjugate leaf updates stay exact. This
    // is the reason for sampling u rather than quasi-differencing the tree
    // targets: after a Prais-Winsten transform the fitted value at (i,t)
    // becomes mu(X_i,t) - rho*mu(X_i,t-1), a difference of two leaf values
    // that sit in different leaves once a tree splits on time, and the leaf
    // sufficient statistics stop being sums over independent observations.
    bool ar1_errors;
    std::vector<double> u;            // n_y * p_y latent AR(1) path
    double rho;                       // persistence
    double sigma_u;                   // innovation sd
    double rho_max;                   // hard bound, keeps u from random-walking

    bool random_intercept;
    std::vector<double> gamma;        // length n_y
    double sigma_gamma;               // sd of the gamma prior
    double gamma_prior_a;             // inverse-gamma shape for sigma_gamma^2
    double gamma_prior_b;             // inverse-gamma rate  for sigma_gamma^2
    std::vector<double> y_orig;       // standardized outcome as supplied
    std::vector<double> y_work;       // y_orig - gamma_i - sur_offset

    // Multi-outcome (SUR) support. When several outcomes are fitted jointly,
    // equation m is estimated on the ORTHOGONALIZED target
    //     y_orig - sum_{l<m} Gamma_{ml} * eps_l
    // where eps_l are the raw residuals of the equations before it. That
    // subtraction lives here, folded into y_work exactly as gamma_i is, so
    // every consumer downstream -- trees, sigma, the GP -- sees the adjusted
    // target with no change of its own. All zeros for a single outcome.
    std::vector<double> sur_offset;   // n_y * p_y, laid out like y

    // Random
    std::vector<double> prob;
    std::random_device rd;
    std::mt19937 gen;
    std::discrete_distribution<> d;

    // Splits
    matrix<double> split_count_all_tree;
    matrix<double> split_count_all_tree_pr;  // TODO: move to xbcfState
    matrix<double> split_count_all_tree_trt; // TODO: move to xbcfState
    std::vector<double> split_count_current_tree;
    std::vector<double> mtry_weight_current_tree;

    // mtry
    bool use_all = true;

    // fitinfo
    size_t n_min;
    size_t n_cutpoints;
    bool parallel;
    size_t p_categorical;
    size_t p_continuous;
    size_t p; // total number of variables = p_categorical + p_continuous
    size_t mtry;
    size_t n_y;  // number of total data points in root node
    size_t p_y;  // dimension of response variables
    size_t beta_size; // unique post-treatment period

    const double *X_std;  // pointer to original data
    const double *y_std;  // pointer to y data
    const double *z;            // the scaled treatment vector            TODO: move to xbcfState
    const double *post_trt_time;
    const double *Spt;
    const double *Tpt;
    
    std::vector<size_t> n_trt;                     // the number of treated individuals      TODO: check if it's used anywhere after restructuring
    matrix<double> mu_fit;       // total mu_fit                           TODO: move to xbcfState
    matrix<double> tau_fit;      // total tau_fit                          TODO: move to xbcfState
    matrix<double> beta_fit;     // track beta values 
    std::vector<double> b_vec;        // scaling parameters for tau (b0,b1)     TODO: move to xbcfState
    std::vector<double> sigma_vec;    // residual standard deviations           TODO: move to xbcfState
    double a;                         // scaling parameter for mu               TODO: move to xbcfState
    size_t p_categorical_pr;          // TODO: move to xbcfState
    size_t p_continuous_pr;           // TODO: move to xbcfState
    size_t p_categorical_trt;         // TODO: move to xbcfState
    size_t p_continuous_trt;          // TODO: move to xbcfState
    size_t p_pr;                      // total number of variables for mu          TODO: move to xbcfState
    size_t p_trt;                     // total number of variables for tau          TODO: move to xbcfState
    size_t mtry_pr;                   // TODO: move to xbcfState
    size_t mtry_trt;                  // TODO: move to xbcfState

    size_t max_depth;
    size_t num_trees;
    std::vector<size_t> num_trees_vec;
    size_t num_sweeps;
    size_t burnin;
    bool sample_weights_flag;
    double ini_var_yhat;
    size_t fl; // flag for likelihood function to alternate between mu loop and tau loop calculations  TODO: move to xbcfState

    matrix<size_t> Xorder_std;

    // residual standard deviation
    double sigma;
    double sigma2; // sigma squared

    // time information
    const double *t_std;
    size_t n_t;
    size_t p_t;
    //std::vector<double> precision_squared;

    void update_residuals()
    {
        // cout << "update_residuals in state.h need to be updated for staggered adoption" << endl;
        size_t index_trt;
        size_t index_ctrl;
        const double *temp_pointer;

        for (size_t j = 0; j < this->p_y; j++){
            index_trt = 0;
            index_ctrl = 0;
            temp_pointer = this->z + j * this->n_y;

            for (size_t i = 0; i < this->n_y; i++)
            {
                if (*(temp_pointer + i) == 1)
                {
                    // this->full_residual_trt[j][index_trt] = *(this->y_std + this->n_y * j + i) - this->a * this->mu_fit[i][j] - this->b_vec[1] * this->beta_t[j] * this->tau_fit[i][j];
                    this->full_residual_trt[j][index_trt] = *(this->y_std + this->n_y * j + i) - this->a * this->mu_fit[i][j] - this->b_vec[1] * this->beta_fit[i][j] * this->tau_fit[i][j];
                    index_trt++;

                    // This used to carry a note saying that beta_fit is the
                    // correct term here but "leads to non-converging sigma and
                    // b values". It no longer does. The cause was the
                    // covariance factorisation in update_time_coef, which
                    // sampled beta with variance U*diag(s^2)*U' instead of
                    // U*diag(s)*U' and so effectively pinned beta to its
                    // conditional mean. With that fixed, b_scaling = TRUE
                    // gives a stationary sigma (sd 0.0006, no drift over 200
                    // sweeps) and b0/b1 that settle away from 1.
                }
                else
                {
                    // this->full_residual_ctrl[j][index_ctrl] = *(this->y_std + this->n_y * j + i) - this->a * this->mu_fit[i][j] - this->b_vec[0] * this->beta_t[0] * this->tau_fit[i][j];
                    this->full_residual_ctrl[j][index_ctrl] = *(this->y_std + this->n_y * j + i) - this->a * this->mu_fit[i][j] - this->b_vec[0] * this->beta_fit[i][j] * this->tau_fit[i][j];
                    index_ctrl++;
                }
            }
        }
    }

    // Rebuild the outcome the forests see after gamma changes.
    void refresh_y_work()
    {
        for (size_t j = 0; j < this->p_y; j++)
        {
            for (size_t i = 0; i < this->n_y; i++)
            {
                const size_t k = j * this->n_y + i;
                this->y_work[k] = this->y_orig[k] - this->gamma[i]
                                - (this->ar1_errors ? this->u[k] : 0.0)
                                - this->sur_offset[k];
            }
        }
    }

    void update_sigma(double sigma)
    {
        this->sigma = sigma;
        this->sigma2 = pow(sigma, 2);
   }

    // sigma update for longbetModel       TODO: move to xbcfClass
    void update_sigma(double sigma, size_t ind)
    {
        this->sigma_vec[ind] = sigma;    }

    // sigma update for longbetModel       TODO: move to xbcfClass
    void update_bscales(double b0, double b1)
    {
        this->b_vec[0] = b0;  // sigma for the control group
        this->b_vec[1] = b1;
    }

    //  TODO: update the constructor / get rid of it
    State(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N,
    size_t p_pr, size_t p_trt, size_t p_y, std::vector<size_t> num_trees_vec,
    size_t p_categorical_pr, size_t p_categorical_trt, size_t p_continuous_pr,
    size_t p_continuous_trt, bool set_random_seed, size_t random_seed,
    size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry_pr,
    size_t mtry_trt, const double *X_std, size_t num_sweeps, bool
    sample_weights_flag, const double *y_std,
    const double *z, std::vector<double> sigma_vec,
    std::vector<double> b_vec, size_t max_depth, double ini_var_yhat,
    size_t burnin)
    {
        // Random
        this->prob = std::vector<double>(2, 0.5);
        this->gen = std::mt19937(rd());
        if (set_random_seed)
        {
            gen.seed(random_seed);
        }
        this->d = std::discrete_distribution<>(prob.begin(), prob.end());

        // Splits
        ini_xinfo(this->split_count_all_tree_pr, p_pr, num_trees_vec[0]);
        ini_xinfo(this->split_count_all_tree_trt, p_trt, num_trees_vec[1]);

        this->n_min = n_min;
        this->n_cutpoints = n_cutpoints;
        this->parallel = parallel;
        this->p_categorical_pr = p_categorical_pr;
        this->p_continuous_pr = p_continuous_pr;
        this->p_categorical_trt = p_categorical_trt;
        this->p_continuous_trt = p_continuous_trt;
        this->mtry_pr = mtry_pr;
        this->mtry_trt = mtry_trt;
        this->X_std = X_std;
        this->p_pr = p_categorical_pr + p_continuous_pr;
        this->p_trt = p_categorical_trt + p_continuous_trt;
        this->n_y = N;
        this->p_y = p_y;
        this->num_trees_vec = num_trees_vec;  // stays the same even for vector
        this->num_sweeps = num_sweeps;
        this->sample_weights_flag = sample_weights_flag;

        // Own a mutable copy of the outcome and hand every consumer a
        // pointer to it. With random_intercept off, y_work == y_orig and
        // the sampler is bit-for-bit the original one.
        this->y_orig.assign(y_std, y_std + N * p_y);
        this->y_work = this->y_orig;
        this->y_std  = this->y_work.data();
        this->sur_offset = std::vector<double>(N * p_y, 0.0);
        this->gamma  = std::vector<double>(N, 0.0);
        this->random_intercept = false;
        this->gp_constant_mean = false;
        this->beta_mean        = 0.0;
        this->has_missing      = false;
        this->y_missing        = std::vector<int>(N * p_y, 0);
        this->binary_outcome   = false;
        this->y_binary         = std::vector<int>(N * p_y, 0);
        this->binary_offset    = 0.0;
        this->ar1_errors       = false;
        this->u                = std::vector<double>(N * p_y, 0.0);
        this->rho              = 0.0;
        this->sigma_u          = 0.0;
        this->rho_max          = 0.95;
        this->sigma_gamma      = 1.0;   // diffuse start, adapts after sweep 1
        this->gamma_prior_a    = 1.0;
        this->gamma_prior_b    = 0.1;
        this->max_depth = max_depth;
        this->burnin = burnin;
        this->ini_var_yhat = ini_var_yhat;
        this->Xorder_std = Xorder_std;

   }

    void update_split_counts(size_t tree_ind)
    {
        mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;
        split_count_all_tree[tree_ind] = split_count_current_tree;
    }

    void update_split_counts(size_t tree_ind, size_t flag)
    {
        mtry_weight_current_tree = mtry_weight_current_tree + split_count_current_tree;
        if (flag == 0)
        {
            split_count_all_tree_pr[tree_ind] = split_count_current_tree;
        }
        else
        {
            split_count_all_tree_trt[tree_ind] = split_count_current_tree;
        }
    }

    void iniSplitStorage(size_t flag)
    {
        if (flag == 0)
        {
            this->split_count_current_tree = std::vector<double>(this->p_pr, 0);
            this->mtry_weight_current_tree = std::vector<double>(this->p_pr, 0);
        }
        else if (flag == 1)
        {
            this->split_count_current_tree = std::vector<double>(this->p_trt, 0);
            this->mtry_weight_current_tree = std::vector<double>(this->p_trt, 0);
        }
    }

    void adjustMtry(size_t flag)
    {
        if (flag == 0)
        {
            this->mtry = this->mtry_pr;
        }
        else if (flag == 1)
        {
            this->mtry = this->mtry_trt;
        }
    }

    void set_t_info(const double *t_std, size_t n_t)
    {
        this->t_std = t_std;
        this->n_t = n_t;
    }
};

class longbetState : public State
{
 public:
    longbetState(const double *Xpointer, matrix<size_t> &Xorder_std, size_t N,
    std::vector<size_t> n_trt, size_t p, size_t p_tau, size_t p_y,
    std::vector<size_t> num_trees_vec,
    size_t p_categorical_pr, size_t p_categorical_trt, size_t p_continuous_pr,
    size_t p_continuous_trt, bool set_random_seed, size_t random_seed,
    size_t n_min, size_t n_cutpoints, bool parallel, size_t mtry_pr,
    size_t mtry_trt, const double *X_std, size_t num_sweeps,
    bool sample_weights_flag, const double *y_std,
    const double *z, const double *post_trt_time, const double *Tpointer, const double *Spointer, size_t beta_size,
    std::vector<double> sigma_vec, std::vector<double> b_vec, size_t max_depth,
    double ini_var_yhat, size_t burnin, size_t dim_suffstat) :
    State(Xpointer, Xorder_std, N, p, p_tau, p_y, num_trees_vec,
    p_categorical_pr, p_categorical_trt, p_continuous_pr, p_continuous_trt,
    set_random_seed, random_seed, n_min, n_cutpoints, parallel, mtry_pr,
    mtry_trt, X_std, num_sweeps, sample_weights_flag, y_std, z, 
    sigma_vec, b_vec, max_depth, ini_var_yhat, burnin)
    {
        this->sigma_vec = sigma_vec;
        this->b_vec = b_vec;
        this->n_trt = n_trt;
        this->num_trees_vec = num_trees_vec;
        this->z = z;
        this->post_trt_time = post_trt_time;
        this->Tpt = Tpointer;
        this->Spt = Spointer;
        this->a = 1;  // initialize a at 1 for now

        this->dim_suffstat = dim_suffstat;

        ini_matrix(this->mu_fit, p_y, N);
        ini_matrix(this->tau_fit, p_y, N);
        ini_matrix(this->beta_fit, p_y, N);
        for (size_t i = 0; i < N; i++){ std::fill(this->beta_fit[i].begin(), this->beta_fit[i].end(), 1); }
        // TODO: Shrink beta_t size to max trt_time in state.h

        
        this->beta_size = beta_size;
        this->beta_t = std::vector<double>(beta_size, 1);

        // those are for XBCF, initialize at a length 1 vector
        // this->residual = std::vector<double>(N, 0);
        // this->full_residual_ctrl = std::vector<double>(N - n_trt, 0);
        // this->full_residual_trt = std::vector<double>(n_trt, 0);
        ini_matrix(this->residual, p_y, N);
        ini_residuals(N, p_y, n_trt);
    }

    void ini_residuals(size_t N, size_t p_y, std::vector<size_t> n_trt){
        this->full_residual_trt.resize(p_y);
        this->full_residual_ctrl.resize(p_y);
        for (size_t i = 0; i < p_y; i++){
            this->full_residual_trt[i].resize(n_trt[i]);
            this->full_residual_ctrl[i].resize(N - n_trt[i]);
        }
    }

};


#endif