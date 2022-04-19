#include "socp_derivatives.h"

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::seq;
using Eigen::seqN;
using Eigen::VectorXd;

MatrixXd CalcC(const Eigen::Ref<const Eigen::VectorXd> &v) {
  const auto n = v.size();
  MatrixXd V(n, n);
  V.setZero();
  V.row(0) = v;
  V(seq(1, Eigen::last), 0) = v.tail(n - 1);
  V(seq(1, Eigen::last), seq(1, Eigen::last)).diagonal().setConstant(v[0]);
  return V;
}

void SocpDerivatives::UpdateProblem(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &b,
    const std::vector<Eigen::MatrixXd> &G_list,
    const std::vector<Eigen::VectorXd> &e_list,
    const Eigen::Ref<const Eigen::VectorXd> &z_star,
    const std::vector<Eigen::VectorXd> &lambda_star_list,
    double lambda_threshold, bool calc_G_grad) {
  const auto n_z = z_star.size();
  const auto n_c = lambda_star_list.size();
  const auto m = e_list[0].size(); // For contact problems, m == 3.
  lambda_star_active_indices_.clear();

  // Find active constraints with large lagrange multipliers.
  for (int i = 0; i < n_c; i++) {
    const auto &lambda_star_i = lambda_star_list[i];
    if (lambda_star_i.norm() > lambda_threshold) {
      lambda_star_active_indices_.push_back(i);
    }
  }
  const auto n_c_active = lambda_star_active_indices_.size();

  // Form A_inv and find A using pseudo-inverse.
  // Length of all active Lagrange multipliers combined.
  const auto n_la = n_c_active * m;
  const auto n_A = n_z + n_la;
  MatrixXd A_inv(n_A, n_A);
  A_inv.setZero();
  A_inv.topLeftCorner(n_z, n_z) = Q;

  std::vector<MatrixXd> C_lambda_list;
  for (int i = 0; i < n_c_active; i++) {
    const auto idx = lambda_star_active_indices_[i];
    const MatrixXd &G_i = G_list[idx];
    const VectorXd &e_i = e_list[idx];
    const VectorXd &lambda_i = lambda_star_list[idx];

    VectorXd w_i = -G_i * z_star + e_i;
    MatrixXd C_lambda_i = CalcC(lambda_i);

    const auto k = n_z + m * i;
    A_inv.block(0, k, n_z, m) = G_i.transpose();
    A_inv.block(k, 0, m, n_z) = -C_lambda_i * G_i;
    A_inv.block(k, k, m, m) = CalcC(w_i);

    C_lambda_list.emplace_back(std::move(C_lambda_i));
  }

  const MatrixXd A = QpDerivatives::CalcInverseAndCheck(A_inv, tol_);

  const MatrixXd &A_11 = A.topLeftCorner(n_z, n_z);
  DzDb_ = -A_11;

  DzDe_ = MatrixXd::Zero(n_z, n_c * m);
  for (int i = 0; i < n_c_active; i++) {
    const auto idx = lambda_star_active_indices_[i];
    DzDe_(Eigen::all, seqN(idx * m, m)) =
        -A.block(0, i * m, n_z, m) * C_lambda_list[i];
  }
}
