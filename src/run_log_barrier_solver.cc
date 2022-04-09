#include "log_barrier_solver.h"

#include "drake/math/autodiff.h"
#include "drake/math/jacobian.h"

using drake::AutoDiffXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;

int main() {
  const int n_v = 3;
  MatrixXd Q(n_v, n_v);
  Q.setIdentity();

  VectorXd tau_h(n_v);
  tau_h << -0.41669826, -0.46413128, 0;

  MatrixXd J(2, n_v);
  J.row(0) << -1., 1., 1;
  J.row(1) << 1., 1., -1;

  const double kappa{100}, h{0.1};

  VectorXd phi_constraints(2);
  phi_constraints << 0, 0;

  auto solver = QpLogBarrierSolver();

  cout << "log QP solution" << endl;
  cout << solver.Solve(Q, -tau_h, -J, phi_constraints / h, kappa) << endl;
  // -------------------------------------------------------------------
  const double mu = 1;
  MatrixXd Jn(1, 3);
  Jn << 0, 1, 0;

  MatrixXd Jt(2, 3);
  Jt.row(0) << -1, 0, 1;
  Jt.row(1) << 0, 0, 0;

  MatrixXd J2(3, n_v);
  J2.row(0) = Jn / mu;
  J2.row(1) = Jt.row(0);
  J2.row(2) = Jt.row(1);

  drake::Vector1d phi(phi_constraints[0]);

  auto f = [&](const auto &x) {
    using Scalar = typename std::remove_reference_t<decltype(x)>::Scalar;
    drake::Vector1<Scalar> output;
    output = x.transpose() * Q.template cast<Scalar>() * x;
    output[0] *= 0.5;
    output -= tau_h.template cast<Scalar>().transpose() * x;
    output[0] *= kappa;

    drake::Vector3<Scalar> w = J2.template cast<Scalar>() * x;
    w[0] += phi[0] / h / mu;
    output[0] -= Eigen::log(w[0] * w[0] - w[1] * w[1] - w[2] * w[2]);

    return output;
  };

  VectorXd v(3);
  v << 0.5, 2, 1;

  auto H = drake::math::hessian(f, v);
  auto &cost = H(0, 0);
  cout << "phi " << cost.value() << endl;
  cout << "phi derivatives\n" << cost.derivatives() << endl;
  cout << "phi hessian\n";
  int n = cost.derivatives().size();
  for (int i = 0; i < n; i++) {
    cout << cost.derivatives()[i].derivatives().transpose() << endl;
  }

  cout << "-----My----" << endl;
  auto solver_log_socp = SocpLogBarrierSolver();
  cout << "cost "
       << solver_log_socp.CalcF(Q, -tau_h, -J2, phi / mu / h, kappa, v)
       << endl;

  Eigen::VectorXd Df(n_v);
  Eigen::MatrixXd H_my(n_v, n_v);
  solver_log_socp.CalcGradientAndHessian(Q, -tau_h, -J2, phi / mu / h, v,
                                         kappa, &Df, &H_my);
  cout << "Df\n" << Df << endl;
  cout << "H_my\n" << H_my << endl;
  cout << "log SOCP solution" << endl;
  cout << solver_log_socp.Solve(Q, -tau_h, -J2, phi / mu / h,
                                kappa) << endl;
  return 0;
}
