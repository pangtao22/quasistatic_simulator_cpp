#pragma once

#include <Eigen/Dense>

/*
QP:
min. 1 / 2 * z.dot(Q).dot(z) + b.dot(z)
s.t. G.dot(z) <= e
 */

class QpDerivatives {
 public:
  explicit QpDerivatives(double tol) : tol_(tol) {};
  void UpdateProblem(const Eigen::Ref<const Eigen::MatrixXd>& Q,
                     const Eigen::Ref<const Eigen::VectorXd>& b,
                     const Eigen::Ref<const Eigen::MatrixXd>& G,
                     const Eigen::Ref<const Eigen::MatrixXd>& e,
                     const Eigen::Ref<const Eigen::VectorXd>& z_star,
                     const Eigen::Ref<const Eigen::VectorXd>& lambda_star);
  [[nodiscard]] const Eigen::MatrixXd& get_DzDe() const {return DzDe_;};
  [[nodiscard]] const Eigen::MatrixXd& get_DzDb() const {return DzDb_;};
  [[nodiscard]] bool is_solution_valid() const {return is_solution_valid_;};
 private:
  const double tol_;
  bool is_solution_valid_{false};
  Eigen::MatrixXd DzDe_;
  Eigen::MatrixXd DzDb_;
};
