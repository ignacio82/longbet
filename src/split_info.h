#ifndef THIRD_PARTY_R_PACKAGES_LONGBET_LONGBET_SRC_SPLIT_INFO_H_
#define THIRD_PARTY_R_PACKAGES_LONGBET_LONGBET_SRC_SPLIT_INFO_H_
#include <iostream>
#include <memory>
#include <vector>
#include "common.h"
#include "state.h"
#include "model.h"
#include "X_struct.h"

struct split_info
{
public:
    matrix<size_t> Xorder_std;
    matrix<size_t> sorder_std;
    std::vector<double> s_values;
    // Alive values for each time-varying covariate, same role as s_values for
    // the time axis. Empty unless the caller supplied any.
    std::vector<std::vector<double>> tv_alive;

    std::vector<size_t> X_counts;
    std::vector<size_t> X_num_unique;

    size_t N_Xorder;
    size_t p_Xorder;

    split_info(std::unique_ptr<X_struct> &x_struct, matrix<size_t> &Xorder_std, matrix<size_t> &sorder_std, std::vector<double> s_values){
      this->Xorder_std = Xorder_std;
      this->X_counts  = x_struct->X_counts;
      this->X_num_unique = x_struct->X_num_unique;

      this->sorder_std = sorder_std;
      this->s_values = s_values;
      this->tv_alive = x_struct->tv_values;

      this->N_Xorder = Xorder_std[0].size();
      this->p_Xorder = Xorder_std.size();
    }

    split_info(std::unique_ptr<split_info> &parent, size_t &split_var, size_t &split_point, bool split_t, bool left)
    {
      this->X_num_unique.resize(parent->X_num_unique.size());
      this->X_counts.resize(parent->X_counts.size());
      this->sorder_std.resize(parent->sorder_std.size());

      if (split_t)
      {
        ini_xinfo_sizet(this->Xorder_std, parent->N_Xorder, parent->p_Xorder);
        // copy Xorder
        for (size_t i = 0; i < Xorder_std.size(); i++){
          std::copy(parent->Xorder_std[i].begin(), parent->Xorder_std[i].end(), this->Xorder_std[i].begin());
        }

        // split_var identifies the cell-level axis: 0 the time axis, 1 + a the
        // a-th time-varying covariate. Only that axis's value set is cut; the
        // others carry over whole.
        this->s_values = parent->s_values;
        this->tv_alive = parent->tv_alive;

        std::vector<double> &parent_vals = (split_var == 0)
            ? parent->s_values : parent->tv_alive[split_var - 1];
        std::vector<double> cut;
        if (left)
        {
          cut.assign(parent_vals.begin(), parent_vals.begin() + split_point + 1);
        } else {
          cut.assign(parent_vals.begin() + split_point + 1, parent_vals.end());
        }
        if (split_var == 0) { this->s_values = cut; }
        else                { this->tv_alive[split_var - 1] = cut; }
      } else {
        // copy value sets
        this->s_values = parent->s_values;
        this->tv_alive = parent->tv_alive;
        if (left) 
        {
          ini_xinfo_sizet(this->Xorder_std, split_point + 1, parent->p_Xorder);
        } else {
          ini_xinfo_sizet(this->Xorder_std, parent->N_Xorder - split_point - 1, parent->p_Xorder);
        }
      }

      this->N_Xorder = Xorder_std[0].size();
      this->p_Xorder = Xorder_std.size();
    }

    void split_this(std::unique_ptr<split_info> &split_left,
      std::unique_ptr<split_info> &split_right,
      size_t split_var, size_t split_point, bool split_t,
      Model *model, std::unique_ptr<X_struct> &x_struct,
      std::unique_ptr<State> &state, std::vector<double> &suff_stat,
      std::vector<double> &left_suff_stat, std::vector<double> &right_suff_stat);

    void split_xorder_std_continuous(std::unique_ptr<split_info> &split_left,
      std::unique_ptr<split_info> &split_right, size_t split_var, size_t split_point,
      Model *model, std::unique_ptr<X_struct> &x_struct,
      std::unique_ptr<State> &state, std::vector<double> &suff_stat,
      std::vector<double> &left_suff_stat, std::vector<double> &right_suff_stat);

    void split_xorder_std_categorical(std::unique_ptr<split_info> &split_left,
      std::unique_ptr<split_info> &split_right,
      size_t split_var, size_t split_point,
      Model *model, std::unique_ptr<X_struct> &x_struct,
      std::unique_ptr<State> &state, std::vector<double> &suff_stat,
      std::vector<double> &left_suff_stat, std::vector<double> &right_suff_stat);

    void split_torder_std(std::unique_ptr<split_info> &split_left,
      std::unique_ptr<split_info> &split_right,
      size_t split_var, size_t split_point,
      Model *model, std::unique_ptr<X_struct> &x_struct,
      std::unique_ptr<State> &state, std::vector<double> &suff_stat,
      std::vector<double> &left_suff_stat, std::vector<double> &right_suff_stat);

};

#endif  // THIRD_PARTY_R_PACKAGES_LONGBET_LONGBET_SRC_SPLIT_INFO_H_
