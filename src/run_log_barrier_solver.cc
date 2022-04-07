#include "log_barrier_solver.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

int main() {
  //--------------------------- tst 
  const int n_v = 3;
  MatrixXd Q(n_v, n_v);
  Q.setIdentity();

  VectorXd tau_h(n_v);
  tau_h << 0.31390925,  0.1954178 , 0;

  MatrixXd J(2, n_v);
  J.row(0) << -1.,  1.,  1;
  J.row(1) << 1.,  1., -1;

  const double kappa{100}, h{0.1};

  VectorXd phi_constraints(2);
  phi_constraints.setZero();

  auto solver = QpLogBarrierSolver();
  solver.Solve(Q, -tau_h, -J, phi_constraints / h, kappa);
  return 0;
}
