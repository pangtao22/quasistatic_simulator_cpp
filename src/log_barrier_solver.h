#pragma once
#include <Eigen/Dense>

#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program.h"

class LogBarrierSolver {
public:
  LogBarrierSolver();
  virtual void SolvePhaseOne(const Eigen::Ref<const Eigen::MatrixXd> &G,
                             const Eigen::Ref<const Eigen::VectorXd> &e,
                             drake::EigenPtr<Eigen::VectorXd> v0_ptr) const = 0;

  virtual double CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                       const Eigen::Ref<const Eigen::VectorXd> &b,
                       const Eigen::Ref<const Eigen::MatrixXd> &G,
                       const Eigen::Ref<const Eigen::VectorXd> &e,
                       const double kappa,
                       const Eigen::Ref<const Eigen::VectorXd> &v) const = 0;

  virtual void
  CalcGradientAndHessian(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                         const Eigen::Ref<const Eigen::VectorXd> &b,
                         const Eigen::Ref<const Eigen::MatrixXd> &G,
                         const Eigen::Ref<const Eigen::VectorXd> &e,
                         const Eigen::Ref<const Eigen::VectorXd> &v,
                         double kappa, drake::EigenPtr<Eigen::VectorXd> Df_ptr,
                         drake::EigenPtr<Eigen::MatrixXd> H_ptr) const = 0;

  void GetPhaseOneSolution(
      const drake::solvers::VectorXDecisionVariable &v,
      const drake::solvers::DecisionVariable &s,
      drake::EigenPtr<Eigen::VectorXd> v0_ptr) const;

  Eigen::VectorXd Solve(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                        const Eigen::Ref<const Eigen::VectorXd> &b,
                        const Eigen::Ref<const Eigen::MatrixXd> &G,
                        const Eigen::Ref<const Eigen::VectorXd> &e,
                        double kappa) const;

  double BackStepLineSearch(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                            const Eigen::Ref<const Eigen::VectorXd> &b,
                            const Eigen::Ref<const Eigen::MatrixXd> &G,
                            const Eigen::Ref<const Eigen::VectorXd> &e,
                            const Eigen::Ref<const Eigen::VectorXd> &v,
                            const Eigen::Ref<const Eigen::VectorXd> &dv,
                            const Eigen::Ref<const Eigen::VectorXd> &Df,
                            const double kappa) const;

 protected:
  std::unique_ptr<drake::solvers::GurobiSolver> solver_;
  mutable drake::solvers::MathematicalProgramResult mp_result_;

  // Hyperparameters for line search.
  static constexpr double alpha_{0.4};
  static constexpr double beta_{0.5};
  static constexpr int line_search_iter_limit_{20};

  // Hyperparameters for Newton's method.
  static constexpr int newton_steps_limit_{50};
  // Considered converge if Newton's decrement / 2 < tol_.
  static constexpr double tol_{1e-6};
};

/*
 * Consider the QP
 * min. 0.5 * v.T * Q * v + b.T * v
 *  s.t. G * v - e <= 0,
 * which has a log-barrier formulation
 * min. kappa * (0.5 * v.T * Q * v + b * v) - sum_log(e - G * v),
 * where sum_log refers to taking the log of every entry in a vector and then
 * summing them; kappa is the log barrier weight.
 *
 * The phase-1 program, which finds a feasible solution to the QP, is given by
 * min. s
 *  s.t. G * v - e <= s.
 * The QP is feasible if s < 0.
 */
class QpLogBarrierSolver : public LogBarrierSolver {
public:
  QpLogBarrierSolver() : LogBarrierSolver() {};
  void SolvePhaseOne(const Eigen::Ref<const Eigen::MatrixXd> &G,
                     const Eigen::Ref<const Eigen::VectorXd> &e,
                     drake::EigenPtr<Eigen::VectorXd> v0_ptr) const override;

  /*
   * F is the log-barrier objective which we'd like to minimize.
   */
  double CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
               const Eigen::Ref<const Eigen::VectorXd> &b,
               const Eigen::Ref<const Eigen::MatrixXd> &G,
               const Eigen::Ref<const Eigen::VectorXd> &e, const double kappa,
               const Eigen::Ref<const Eigen::VectorXd> &v) const override;

  void
  CalcGradientAndHessian(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                         const Eigen::Ref<const Eigen::VectorXd> &b,
                         const Eigen::Ref<const Eigen::MatrixXd> &G,
                         const Eigen::Ref<const Eigen::VectorXd> &e,
                         const Eigen::Ref<const Eigen::VectorXd> &v,
                         double kappa, drake::EigenPtr<Eigen::VectorXd> Df_ptr,
                         drake::EigenPtr<Eigen::MatrixXd> H_ptr) const override;

};

/*
 * Consider the SOCP
 * min. 0.5 * v.T * Q * v + b.T * v
 *  s.t. -G_i * v + [e_i, 0, 0] \in Q^3,
 *  where Q^3 is the 3-dimensional second-order cone; G_i is a (3, n_v)
 *  matrix; e_i is a scalar. We concatenate G_i and e_i vertically:
 *  G := [[G_1], ...[G_n]], with shape (3 * n, n_v), and
 *  e := [e_1, ... e_n], with shape (n,).
 *
 * For convenience, we define
 * w_i := -G_i * v + [e_i, 0, 0],
 * so that the cone constraints in the SOCP can be expressed as
 * w_i[0]**2 >= w_i[1]**2 + w_i[2]**2.
 *
 * The SOCP has a log-barrier formulation
 * min. kappa * (0.5 * v.T * Q * v + b * v)
 *      - sum_log(w_i[0]**2 - w_i[1]**2 - w_i[2]**),
 * where sum_log refers to taking the log of every entry in a vector and then
 * summing them; kappa is the log barrier weight.
 */
class SocpLogBarrierSolver : public LogBarrierSolver {
 public:
  SocpLogBarrierSolver() : LogBarrierSolver() {};
  void SolvePhaseOne(const Eigen::Ref<const Eigen::MatrixXd> &G,
                     const Eigen::Ref<const Eigen::VectorXd> &e,
                     drake::EigenPtr<Eigen::VectorXd> v0_ptr) const override;

  /*
   * F is the log-barrier objective which we'd like to minimize.
   */
  double CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
               const Eigen::Ref<const Eigen::VectorXd> &b,
               const Eigen::Ref<const Eigen::MatrixXd> &G,
               const Eigen::Ref<const Eigen::VectorXd> &e, const double kappa,
               const Eigen::Ref<const Eigen::VectorXd> &v) const override;

  void
  CalcGradientAndHessian(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                         const Eigen::Ref<const Eigen::VectorXd> &b,
                         const Eigen::Ref<const Eigen::MatrixXd> &G,
                         const Eigen::Ref<const Eigen::VectorXd> &e,
                         const Eigen::Ref<const Eigen::VectorXd> &v,
                         double kappa, drake::EigenPtr<Eigen::VectorXd> Df_ptr,
                         drake::EigenPtr<Eigen::MatrixXd> H_ptr) const override;
};
