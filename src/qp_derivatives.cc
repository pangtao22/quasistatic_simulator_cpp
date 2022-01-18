#include <iostream>
#include <cmath>

#include <spdlog/spdlog.h>

#include "qp_derivatives.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void QpDerivativesBase::check_solution_error(double error) {
  is_solution_valid_ = error < tol_;

  if (isnan(error)) {
    throw std::runtime_error("Gradient is nan.");
  }

  if (not is_solution_valid_) {
    std::stringstream ss;
    ss << "bad gradient. |Ax - b| norm is " << error << ". Tolerance is "
       << tol_ << ".";
    spdlog::warn(ss.str());

  }
}

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
  A_inv.bottomRightCorner(n_l, n_l).diagonal() = G * (z_star)-e;

  MatrixXd rhs(n_z + n_l, n_z + n_l);
  rhs.setZero();
  rhs.bottomLeftCorner(n_l, n_l).diagonal() = lambda_star;
  rhs.topRightCorner(n_z, n_z).diagonal().setConstant(-1);

  MatrixXd sol = A_inv.colPivHouseholderQr().solve(rhs);
  check_solution_error((A_inv * sol - rhs).norm());

  DzDe_ = sol.topLeftCorner(n_z, n_l);
  DzDb_ = sol.topRightCorner(n_z, n_z);
}

void QpDerivativesActive::UpdateProblem(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &b,
    const Eigen::Ref<const Eigen::MatrixXd> &G,
    const Eigen::Ref<const Eigen::MatrixXd> &e,
    const Eigen::Ref<const Eigen::VectorXd> &z_star,
    const Eigen::Ref<const Eigen::VectorXd> &lambda_star,
    double lambda_threshold) {
  const int n_z = z_star.size();
  const int n_l = lambda_star.size();

  std::vector<double> nu_star_v;
  std::vector<int> nu_star_indices;

  // Find active constraints with large lagrange multipliers.
  for (int i = 0; i < n_l; i++) {
    double lambda_star_i = lambda_star[i];
    if (lambda_star_i > lambda_threshold) {
      nu_star_v.push_back(lambda_star_i);
      nu_star_indices.push_back(i);
    }
  }

  const int n_nu = nu_star_v.size();
  auto nu_star = Eigen::Map<VectorXd>(nu_star_v.data(), n_nu);
  MatrixXd B(n_nu, n_z);
  for (int i = 0; i < n_nu; i++) {
    B.row(i) = G.row(nu_star_indices[i]);
  }

  // Form A_inv and find A using pseudo-inverse.
  MatrixXd A_inv(n_z + n_nu, n_z + n_nu);
  A_inv.setZero();
  A_inv.topLeftCorner(n_z, n_z) = Q;
  A_inv.topRightCorner(n_z, n_nu) = B.transpose();
  A_inv.bottomLeftCorner(n_nu, n_z) = B;

  const auto I = MatrixXd::Identity(n_z + n_nu, n_z + n_nu);
  const MatrixXd A =
      A_inv.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(I);

  check_solution_error((A_inv * A - I).norm());

  DzDb_ = -A.topLeftCorner(n_z, n_z);

  const MatrixXd DzDe_active = A.topRightCorner(n_z, n_nu);
  DzDe_ = MatrixXd::Zero(n_z, n_l);
  for (int i = 0; i < n_nu; i++) {
    DzDe_.col(nu_star_indices[i]) = DzDe_active.col(i);
  }
}
