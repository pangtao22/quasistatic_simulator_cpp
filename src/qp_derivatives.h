#pragma once

#include <Eigen/Dense>

/*
QP:
min. 1 / 2 * z.dot(Q).dot(z) + b.dot(z)
s.t. G.dot(z) <= e
 */

class QpDerivativesBase {
 public:
  explicit QpDerivativesBase(double tol) : tol_(tol) {};
  [[nodiscard]] const Eigen::MatrixXd& get_DzDe() const {return DzDe_;};
  [[nodiscard]] const Eigen::MatrixXd& get_DzDb() const {return DzDb_;};
  [[nodiscard]] const Eigen::MatrixXd& get_DzDvecG() const {return DzDvecG_;};
 protected:
  void check_solution_error(double error, int n);
  const double tol_;
  bool is_relative_err_small_{false};
  Eigen::MatrixXd DzDe_;
  Eigen::MatrixXd DzDb_;
  Eigen::MatrixXd DzDvecG_;
};

class QpDerivatives : public QpDerivativesBase {
 public:
  explicit QpDerivatives(double tol) : QpDerivativesBase(tol) {};
  void UpdateProblem(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                     const Eigen::Ref<const Eigen::VectorXd>& b,
                     const Eigen::Ref<const Eigen::MatrixXd>& G,
                     const Eigen::Ref<const Eigen::MatrixXd>& e,
                     const Eigen::Ref<const Eigen::VectorXd>& z_star,
                     const Eigen::Ref<const Eigen::VectorXd>& lambda_star);
};


class QpDerivativesActive : public QpDerivativesBase {
 public:
  explicit QpDerivativesActive(double tol) : QpDerivativesBase(tol) {};
  void UpdateProblem(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                     const Eigen::Ref<const Eigen::VectorXd> &b,
                     const Eigen::Ref<const Eigen::MatrixXd> &G,
                     const Eigen::Ref<const Eigen::MatrixXd> &e,
                     const Eigen::Ref<const Eigen::VectorXd> &z_star,
                     const Eigen::Ref<const Eigen::VectorXd> &lambda_star,
                     double lambda_threshold,
                     bool calc_G_grad);
};
