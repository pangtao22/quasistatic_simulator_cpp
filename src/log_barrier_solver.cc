#include <iostream>

#include "log_barrier_solver.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;
using std::cout;
using std::endl;

LogBarrierSolver::LogBarrierSolver() {
  solver_ = std::make_unique<drake::solvers::GurobiSolver>();
}

double LogBarrierSolver::BackStepLineSearch(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &b,
    const Eigen::Ref<const Eigen::MatrixXd> &G,
    const Eigen::Ref<const Eigen::VectorXd> &e,
    const Eigen::Ref<const Eigen::VectorXd> &v,
    const Eigen::Ref<const Eigen::VectorXd> &dv,
    const Eigen::Ref<const Eigen::VectorXd> &Df, const double kappa) const {
  double t = 1;
  int line_search_iters = 0;
  bool line_search_success = false;
  double f0 = CalcF(Q, b, G, e, kappa, v);

  while (line_search_iters < line_search_iter_limit_) {
    double f = CalcF(Q, b, G, e, kappa, v + t * dv);
    double f1 = f0 + alpha_ * t * Df.transpose() * dv;
    if (f < f1) {
      line_search_success = true;
      break;
    }
    t *= beta_;
    line_search_iters++;
  }

  if (not line_search_success) {
    throw std::runtime_error(
        "Back stepping Line search exceeded iteration limit.");
  }

  // cout << "t " << t << endl;
  return t;
}

Eigen::VectorXd
LogBarrierSolver::Solve(const Eigen::Ref<const Eigen::MatrixXd> &Q,
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
  while (n_iters < newton_steps_limit_) {
    CalcGradientAndHessian(Q, b, G, e, v, kappa, &Df, &H);
    VectorXd dv = -H.llt().solve(Df);
    double lambda_squared = -Df.transpose() * dv;
    if (lambda_squared / 2 < tol_) {
      converged = true;
      break;
    }
    //    cout << "------------------------------------------" << endl;
    //    cout << "Iter " << n_iters << endl;
    //    cout << "H\n" << H << endl;
    //    cout << "dv: " << dv.transpose() << endl;
    //    cout << "v: " << v.transpose() << endl;
    double t = BackStepLineSearch(Q, b, G, e, v, dv, Df, kappa);
    v += t * dv;
    n_iters++;
  }

  if (not converged) {
    throw std::runtime_error("QpLogBarrier Newton's method did not converge.");
  }

  return v;
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
  const auto &v = v_s.head(n_v);
  const auto &s = v_s[n_v];

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

double
QpLogBarrierSolver::CalcF(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                          const Eigen::Ref<const Eigen::VectorXd> &b,
                          const Eigen::Ref<const Eigen::MatrixXd> &G,
                          const Eigen::Ref<const Eigen::VectorXd> &e,
                          const double kappa,
                          const Eigen::Ref<const Eigen::VectorXd> &v) const {
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
    drake::EigenPtr<Eigen::MatrixXd> H_ptr) const {
  *H_ptr = Q * kappa;
  *Df_ptr = (Q * v + b) * kappa;
  for (int i = 0; i < G.rows(); i++) {
    double d = G.row(i) * v - e[i];
    *H_ptr += G.row(i).transpose() * G.row(i) / d / d;
    *Df_ptr += -G.row(i) / d;
  }
}

void SocpLogBarrierSolver::SolvePhaseOne(
    const Eigen::Ref<const Eigen::MatrixXd> &G,
    const Eigen::Ref<const Eigen::VectorXd> &e,
    drake::EigenPtr<Eigen::VectorXd> v0_ptr) const {}


Eigen::Vector3d CalcWi(
    const Eigen::Ref<const Eigen::Matrix3Xd> &G_i,
    const double e_i, const Eigen::Ref<const Eigen::VectorXd> &v) {
  Vector3d w = -G_i * v;
  w[0] += e_i;
  return w;
}

double SocpLogBarrierSolver::CalcF(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &b,
    const Eigen::Ref<const Eigen::MatrixXd> &G,
    const Eigen::Ref<const Eigen::VectorXd> &e, const double kappa,
    const Eigen::Ref<const Eigen::VectorXd> &v) const {
  const int n_c = G.rows() / 3;
  const int n_v = G.cols();
  DRAKE_ASSERT(G.rows() % 3 == 0);
  DRAKE_ASSERT(e.size() == n_c);

  double output = 0.5 * v.transpose() * Q * v + (b.array() * v.array()).sum();
  output *= kappa;
  for (int i_c = 0; i_c < n_c; i_c++) {
    Vector3d w = CalcWi(G.block(i_c, 0, 3, n_v), e[i_c], v);
    const double d = -w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if (d > 0) {
      return std::numeric_limits<double>::infinity();
    }
    output += -log(-d);
  }
  return output;
}

void SocpLogBarrierSolver::CalcGradientAndHessian(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &b,
    const Eigen::Ref<const Eigen::MatrixXd> &G,
    const Eigen::Ref<const Eigen::VectorXd> &e,
    const Eigen::Ref<const Eigen::VectorXd> &v, double kappa,
    drake::EigenPtr<Eigen::VectorXd> Df_ptr,
    drake::EigenPtr<Eigen::MatrixXd> H_ptr) const {
  *H_ptr = Q * kappa;
  *Df_ptr = (Q * v + b) * kappa;
  const int n_c = G.rows() / 3;
  const int n_v = G.cols();

  Eigen::RowVectorXd Dd(n_v);
  Eigen::Matrix3d A;
  A.setIdentity();
  A *= 2;
  A(0, 0) = -2;
  for (int i = 0; i < n_c; i++) {
    const Eigen::Matrix3Xd& G_i = G.block(i, 0, 3, n_v);
    Vector3d w = CalcWi(G_i, e[i], v);
    const double d = -w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    Dd = 2 * Eigen::RowVector3d(-w[0], w[1], w[2]) * G_i;
    *Df_ptr += Dd.transpose() / d;
    *H_ptr += Dd.transpose() * Dd / d / d;
    *H_ptr += G_i.transpose() * A * G_i / -d;
  }
}
