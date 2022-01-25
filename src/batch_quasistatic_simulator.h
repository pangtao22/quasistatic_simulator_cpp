#include "quasistatic_simulator.h"
#include <list>
#include <tuple>

class BatchQuasistaticSimulator {
public:
  BatchQuasistaticSimulator(
      const std::string &model_directive_path,
      const std::unordered_map<std::string, Eigen::VectorXd>
          &robot_stiffness_str,
      const std::unordered_map<std::string, std::string> &object_sdf_paths,
      QuasistaticSimParameters sim_params);

  static Eigen::VectorXd
  CalcDynamics(QuasistaticSimulator *q_sim,
               const Eigen::Ref<const Eigen::VectorXd> &q,
               const Eigen::Ref<const Eigen::VectorXd> &u, double h,
               const GradientMode gradient_mode);

  static Eigen::MatrixXd
  CalcBundledB(QuasistaticSimulator *q_sim,
               const Eigen::Ref<const Eigen::VectorXd> &q,
               const Eigen::Ref<const Eigen::VectorXd> &u, double h,
               const Eigen::Ref<const Eigen::MatrixXd> &du);

  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<bool>>
  CalcDynamicsSerial(
      const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
      const Eigen::Ref<const Eigen::MatrixXd> &u_batch, double h,
      const GradientMode gradient_mode);

  /*
   * Each row in x_batch and u_batch represent a pair of current states and
   * inputs. The function returns a tuple of
   *  (x_next_batch, B_batch, is_valid_batch), where
   *  - x_next_batch.row(i) = f(x_batch.row(i), u_batch.row(i))
   *  - B_batch[i] is a B matrix, as in x_next = A * x + B * u.
   *  - is_valid_batch[i] is false if the forward QP fails to solve or
   *    B_batch[i] is nan, which can happen if the least square solve during
   *    the application of implicit function theorem to the KKT condition of
   *    the QP fails.
   *
   *  Behaviors under different gradient_mode:
   *    kNone: B_batch has 0 length.
   *    kAB: an exception is thrown.
   *    kBOnly: B_batch[i] is a (n_q, n_a) matrix.
   */
  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<bool>>
  CalcDynamicsParallel(const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
                       const Eigen::Ref<const Eigen::MatrixXd> &u_batch,
                       double h, GradientMode gradient_mode) const;

  std::vector<Eigen::MatrixXd> CalcBundledBTrj(
      const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
      const Eigen::Ref<const Eigen::MatrixXd> &u_trj,
      double h, double std_u, int n_samples, std::optional<int> seed);


  /*
   * Implements multi-threaded computation of bundled gradient based on drake's
   * Monte-Carlo simulation:
   * https://github.com/RobotLocomotion/drake/blob/5316536420413b51871ceb4b9c1f77aedd559f71/systems/analysis/monte_carlo.cc#L42
   * But this implementation does not seem to be faster than CalcBundledBTrj,
   * which is again  slower than the original ZMQ-based PUSH-PULL scheme.
   *
   * It is a sad conclusion after almost two weeks of effort ¯\_(ツ)_/¯.
   * Well, at least I learned more about C++ and saw quite a bit of San
   * Francisco :)
   */
  std::vector<Eigen::MatrixXd> CalcBundledBTrjDirect(
      const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
      const Eigen::Ref<const Eigen::MatrixXd> &u_trj,
      double h, double std_u, int n_samples, std::optional<int> seed);

  size_t get_hardware_concurrency() const { return hardware_concurrency_; };

  QuasistaticSimulator& get_q_sim() { return *q_sims_.begin(); };

private:
  std::stack<int> InitializeAvailableSimulatorStack() const;

  const size_t hardware_concurrency_{0};
  mutable std::vector<QuasistaticSimulator> q_sims_;
  std::mt19937 gen_;
};
