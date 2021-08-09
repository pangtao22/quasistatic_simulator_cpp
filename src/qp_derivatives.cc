#include <iostream>
#include <sstream>

#include <spdlog/spdlog.h>

#include "qp_derivatives.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void QpDerivatives::UpdateProblem(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &b,
    const Eigen::Ref<const Eigen::MatrixXd> &G,
    const Eigen::Ref<const Eigen::MatrixXd> &e,
    const Eigen::Ref<const Eigen::VectorXd> &z_star,
    const Eigen::Ref<const Eigen::VectorXd> &lambda_star) {
  const size_t n_z = z_star.size();
  const size_t n_l = lambda_star.size();
  MatrixXd A_inv(n_z + n_l, n_z + n_l);
  A_inv.setZero();
  A_inv.topLeftCorner(n_z, n_z) = Q;
  A_inv.topRightCorner(n_z, n_l) = G.transpose();
  A_inv.bottomLeftCorner(n_l, n_z) = lambda_star.asDiagonal() * G;
  A_inv.bottomRightCorner(n_l, n_l).diagonal() = G * (z_star) - e;

  MatrixXd rhs(n_z + n_l, n_z + n_l);
  rhs.setZero();
  rhs.bottomLeftCorner(n_l, n_l).diagonal() = lambda_star;
  rhs.topRightCorner(n_z, n_z).diagonal().setConstant(-1);

  MatrixXd sol = A_inv.colPivHouseholderQr().solve(rhs);
  const double error = (A_inv * sol - rhs).norm();
  is_solution_valid_ = error < tol_;
  if (not is_solution_valid_) {
    std::stringstream ss;
    ss << "bad gradient. |Ax - b| norm is " << error
       << ". Tolerance is " << tol_ << ".";
    spdlog::warn(ss.str());
  }
  DzDe_ = sol.topLeftCorner(n_z, n_l);
  DzDb_ = sol.topRightCorner(n_z, n_z);
}
