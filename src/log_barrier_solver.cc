#include <iostream>

#include "log_barrier_solver.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

QpLogBarrierSolver::QpLogBarrierSolver() {
  solver_ = std::make_unique<drake::solvers::GurobiSolver>();
}

void QpLogBarrierSolver::SolvePhaseOne(
    const Eigen::Ref<const Eigen::MatrixXd> &G,
    const Eigen::Ref<const Eigen::VectorXd> &e,
    drake::EigenPtr<Eigen::VectorXd> v0_ptr) const {
  const auto n_f = G.rows();
  const auto n_v = G.cols();
  auto prog = drake::solvers::MathematicalProgram();
  // v_s is the concatenation of [v, s], where v is the vector of generalized
  // velocities of the system and s is the scalar slack variables.
  auto v_s = prog.NewContinuousVariables(n_v + 1, "v");
  const auto& v = v_s.head(n_v);
  const auto& s = v_s[n_v];

  // G * v - e <= s  <==> [G, -1] * v_s <= e.
  MatrixXd G_1(n_f, n_v + 1);
  G_1.leftCols(n_v) = G;
  G_1.rightCols(1) = -VectorXd::Ones(n_f);

  prog.AddLinearCost(s);
  prog.AddLinearConstraint(
      G_1, VectorXd::Constant(n_f, -std::numeric_limits<double>::infinity()), e,
      v_s);
  prog.AddBoundingBoxConstraint(-1, 1, v);

  solver_->Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error("Phase 1 program cannot be solved.");
  }

  const auto s_value = mp_result_.GetSolution(s);
  if (s_value > -1e-6) {
    v0_ptr = nullptr;
    std::stringstream ss;
    ss << "Phase 1 cannot find a feasible solution. s = " << s_value << endl;
    throw std::runtime_error(ss.str());
  }

  *v0_ptr = mp_result_.GetSolution(v);
}

double QpLogBarrierSolver::CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                                 const Eigen::Ref<const Eigen::VectorXd> &b,
                                 const Eigen::Ref<const Eigen::MatrixXd> &G,
                                 const Eigen::Ref<const Eigen::VectorXd> &e,
                                 const double kappa,
                                 const Eigen::Ref<const Eigen::VectorXd> &v) {
  double f = kappa * 0.5 * v.transpose() * Q * v;
  f += kappa * b.transpose() * v;
  for (int i = 0; i < G.rows(); i++) {
    double d = G.row(i) * v - e[i];
    if (d > 0) {
      // Out of domain of log(.), i.e. one of the inequality constraints is
      // infeasible.
      return std::numeric_limits<double>::infinity();
    }
    f -= log(-d);
  }
  return f;
}

void QpLogBarrierSolver::CalcGradientAndHessian(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &b,
    const Eigen::Ref<const Eigen::MatrixXd> &G,
    const Eigen::Ref<const Eigen::VectorXd> &e,
    const Eigen::Ref<const Eigen::VectorXd> &v, const double kappa,
    drake::EigenPtr<Eigen::VectorXd> Df_ptr,
    drake::EigenPtr<Eigen::MatrixXd> H_ptr) {
  *H_ptr = Q * kappa;
  *Df_ptr = (Q * v + b) * kappa;
  for (int i = 0; i < G.rows(); i++) {
    double d = G.row(i) * v - e[i];
    *H_ptr += G.row(i).transpose() * G.row(i) / d / d;
    *Df_ptr += -G.row(i) / d;
  }
}

Eigen::VectorXd
QpLogBarrierSolver::Solve(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                          const Eigen::Ref<const Eigen::VectorXd> &b,
                          const Eigen::Ref<const Eigen::MatrixXd> &G,
                          const Eigen::Ref<const Eigen::VectorXd> &e,
                          double kappa) const {
  const auto n_v = Q.rows();
  MatrixXd H(n_v, n_v);
  VectorXd Df(n_v);
  VectorXd v(n_v);
  SolvePhaseOne(G, e, &v);
  int n_iters = 0;
  bool converged = false;
  while (n_iters < iteration_limit_) {
    CalcGradientAndHessian(Q, b, G, e, v, kappa, &Df, &H);
    VectorXd dv = -H.llt().solve(Df);
    double lambda_squared = -Df.transpose() * dv;
    if (lambda_squared / 2 < 1e-6) {
      converged = true;
      break;
    }
//    cout << "------------------------------------------" << endl;
//    cout << "Iter " << n_iters << endl;
//    cout << "H\n" << H << endl;
//    cout << "dv: " << dv.transpose() << endl;
//    cout << "v: " << v.transpose() << endl;

    // back-tracking line search.
    double t = 1;
    double f0 = CalcF(Q, b, G, e, kappa, v);
    while (true) {
      double f = CalcF(Q, b, G, e, kappa, v + t * dv);
      double f1 = f0 + alpha_ * t * Df.transpose() * dv;
      if (f < f1) {
        break;
      }
      t *= beta_;
    }
//    cout << "t " << t << endl;
    v += t * dv;
    n_iters++;
  }

  if (not converged) {
    throw std::runtime_error("QpLogBarrier Newton's method did not converge.");
  }

  return v;
}
