#pragma once
#include <Eigen/Dense>

#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program.h"

/*
 * Consider the QP
 * min. 0.5 * v.T * Q * v + b.T * v
 *  s.t. G * v - e <= 0,
 * which has a log-barrier formulation
 * min. kappa * (0.5 * v.T * Q * v + b * v) - sum_log(e - G * v),
 * where sum_log refers to taking the log of every row and then summing them;
 * kappa is the log barrier weight.
 *
 * The phase-1 program, which finds a feasible solution to the QP, is given by
 * min. s
 *  s.t. G * v - e <= s.
 * The QP is feasible if s < 0.
 */
class QpLogBarrierSolver {
public:
  QpLogBarrierSolver();
  void SolvePhaseOne(const Eigen::Ref<const Eigen::MatrixXd> &G,
                     const Eigen::Ref<const Eigen::VectorXd> &e,
                     drake::EigenPtr<Eigen::VectorXd> v0_ptr) const;

  Eigen::VectorXd Solve(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                        const Eigen::Ref<const Eigen::VectorXd> &b,
                        const Eigen::Ref<const Eigen::MatrixXd> &G,
                        const Eigen::Ref<const Eigen::VectorXd> &e,
                        double kappa) const;
  /*
   * F is the log-barrier objective which we'd like to minimize.
   */
  static double CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                      const Eigen::Ref<const Eigen::VectorXd> &b,
                      const Eigen::Ref<const Eigen::MatrixXd> &G,
                      const Eigen::Ref<const Eigen::VectorXd> &e,
                      const double kappa,
                      const Eigen::Ref<const Eigen::VectorXd> &v);

  static void CalcGradientAndHessian(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                                     const Eigen::Ref<const Eigen::VectorXd> &b,
                                     const Eigen::Ref<const Eigen::MatrixXd> &G,
                                     const Eigen::Ref<const Eigen::VectorXd> &e,
                                     const Eigen::Ref<const Eigen::VectorXd> &v,
                                     double kappa,
                                     drake::EigenPtr<Eigen::VectorXd> Df_ptr,
                                     drake::EigenPtr<Eigen::MatrixXd> H_ptr);

  static double BackStepLineSearch(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                            const Eigen::Ref<const Eigen::VectorXd> &b,
                            const Eigen::Ref<const Eigen::MatrixXd> &G,
                            const Eigen::Ref<const Eigen::VectorXd> &e,
                            const Eigen::Ref<const Eigen::VectorXd> &v,
                            const Eigen::Ref<const Eigen::VectorXd> &dv,
                            const Eigen::Ref<const Eigen::VectorXd> &Df,
                            const double kappa);

private:
  std::unique_ptr<drake::solvers::GurobiSolver> solver_;
  mutable drake::solvers::MathematicalProgramResult mp_result_;

  // Hyperparameters for line search.
  static constexpr double alpha_{0.4};
  static constexpr double beta_{0.5};
  static constexpr int line_search_iter_limit_{20};

  // Hyperparamteres for Newton's method.
  static constexpr int newton_steps_limit_{40};
  // Considered converge if Newton's decrement / 2 < tol_.
  static constexpr double tol_{1e-6};
};
