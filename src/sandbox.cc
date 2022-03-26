#include <iostream>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using Eigen::MatrixXd;

int main() {
  MatrixXd A(3, 4);
  A << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  cout << "A\n" << A << endl;

  const auto B = Eigen::Map<MatrixXd>(A.row(1).data(), 2, 2);
  cout << "B\n" << B << endl;

  return 0;
}