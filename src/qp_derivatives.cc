#include <cmath>
#include <iostream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <unsupported/Eigen/KroneckerProduct>

#include "qp_derivatives.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void QpDerivativesBase::check_solution_error(double error, int n) {
  auto rel_err = error / n;
  is_relative_err_small_ = rel_err < tol_;

  if (std::isnan(error)) {
    throw std::runtime_error("Gradient is nan.");
  }
  if (!is_relative_err_small_) {
    std::stringstream ss;
    ss << "Relative error " << rel_err << " is greater than " << tol_ << ".";
    throw std::runtime_error(ss.str());
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
  check_solution_error((A_inv * sol - rhs).norm(), n_z + n_l);

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
    double lambda_threshold, bool calc_G_grad) {
  const int n_z = z_star.size();
  const int n_l = lambda_star.size();

  std::vector<double> lambda_star_active_vec;
  std::vector<int> lambda_star_active_indices;

  // Find active constraints with large lagrange multipliers.
  for (int i = 0; i < n_l; i++) {
    double lambda_star_i = lambda_star[i];
    if (lambda_star_i > lambda_threshold) {
      lambda_star_active_vec.push_back(lambda_star_i);
      lambda_star_active_indices.push_back(i);
    }
  }

  const int n_nu = lambda_star_active_vec.size();
  auto lambda_star_active =
      Eigen::Map<VectorXd>(lambda_star_active_vec.data(), n_nu);
  MatrixXd B(n_nu, n_z);
  for (int i = 0; i < n_nu; i++) {
    B.row(i) = G.row(lambda_star_active_indices[i]);
  }

  // Form A_inv and find A using pseudo-inverse.
  MatrixXd A_inv(n_z + n_nu, n_z + n_nu);
  A_inv.setZero();
  A_inv.topLeftCorner(n_z, n_z) = Q;
  A_inv.topRightCorner(n_z, n_nu) = B.transpose();
  A_inv.bottomLeftCorner(n_nu, n_z) = B;

  const auto I = MatrixXd::Identity(n_z + n_nu, n_z + n_nu);
  auto cod = Eigen::CompleteOrthogonalDecomposition<MatrixXd>(A_inv);
  const MatrixXd A = cod.solve(I);

  check_solution_error((A_inv * A - I).norm(), n_z + n_l);
  const MatrixXd &A_11 = A.topLeftCorner(n_z, n_z);
  DzDb_ = -A_11;

  const MatrixXd DzDe_active = A.topRightCorner(n_z, n_nu);
  DzDe_ = MatrixXd::Zero(n_z, n_l);
  for (int i = 0; i < n_nu; i++) {
    DzDe_.col(lambda_star_active_indices[i]) = DzDe_active.col(i);
  }

  if (not calc_G_grad) {
    return;
  }
  const MatrixXd& A_12 = A.topRightCorner(n_z, n_nu);
  DzDvecG_ =
      -Eigen::kroneckerProduct(A_11, lambda_star_active.transpose()).eval();
  DzDvecG_ -= Eigen::kroneckerProduct(z_star.transpose(), A_12).eval();
}
