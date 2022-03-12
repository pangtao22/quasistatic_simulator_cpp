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
      const QuasistaticSimParameters &sim_params);

  static Eigen::VectorXd
  CalcDynamics(QuasistaticSimulator *q_sim,
               const Eigen::Ref<const Eigen::VectorXd> &q,
               const Eigen::Ref<const Eigen::VectorXd> &u,
               const QuasistaticSimParameters &sim_params);

  static Eigen::MatrixXd
  CalcBundledB(QuasistaticSimulator *q_sim,
               const Eigen::Ref<const Eigen::VectorXd> &q,
               const Eigen::Ref<const Eigen::VectorXd> &u,
               const Eigen::Ref<const Eigen::MatrixXd> &du,
               const QuasistaticSimParameters &sim_params);

  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<bool>>
  CalcDynamicsSerial(const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
                     const Eigen::Ref<const Eigen::MatrixXd> &u_batch,
                     const QuasistaticSimParameters &sim_params) const;

  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<bool>>
  CalcDynamicsSerial(const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
                     const Eigen::Ref<const Eigen::MatrixXd> &u_batch) const {
    return CalcDynamicsSerial(x_batch, u_batch, get_q_sim().get_sim_params());
  }

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
                       const QuasistaticSimParameters &sim_params) const;

  std::vector<Eigen::MatrixXd>
  CalcBundledBTrjScalarStd(const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
                           const Eigen::Ref<const Eigen::MatrixXd> &u_trj,
                           double std_u,
                           const QuasistaticSimParameters &sim_params,
                           int n_samples, std::optional<int> seed) const;

  std::vector<Eigen::MatrixXd>
  CalcBundledBTrj(const Eigen::Ref<
      const Eigen::MatrixXd> &x_trj,
                  const Eigen::Ref<
                      const Eigen::MatrixXd> &u_trj,
                  const Eigen::Ref<
                      const Eigen::VectorXd> &std_u,
                  const QuasistaticSimParameters &sim_params,
                  int n_samples,
                  std::optional<
                      int> seed) const;

  Eigen::MatrixXd
  SampleGaussianMatrix(int n_rows, const Eigen::Ref<const Eigen::VectorXd> &mu,
                       const Eigen::Ref<const Eigen::VectorXd> &std) const;

  /*
   * Implements multi-threaded computation of bundled gradient based on drake's
   * Monte-Carlo simulation:
   * https://github.com/RobotLocomotion/drake/blob/5316536420413b51871ceb4b9c1f77aedd559f71/systems/analysis/monte_carlo.cc#L42
   * But this implementation does not seem to be faster than
   * CalcBundledBTrjScalarStd, which is again  slower than the original
   * ZMQ-based PUSH-PULL scheme.
   *
   * It is a sad conclusion after almost two weeks of effort ¯\_(ツ)_/¯.
   * Well, at least I learned more about C++ and saw quite a bit of San
   * Francisco :)
   */
  std::vector<Eigen::MatrixXd>
  CalcBundledBTrjDirect(
      const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
      const Eigen::Ref<const Eigen::MatrixXd> &u_trj,
      double std_u,
      const QuasistaticSimParameters &sim_params,
      int n_samples,
      std::optional<int> seed) const;

  size_t get_num_max_parallel_executions() const {
    return num_max_parallel_executions;
  };
  void set_num_max_parallel_executions(size_t n) {
    num_max_parallel_executions = n;
  };

  QuasistaticSimulator &get_q_sim() const { return *q_sims_.begin(); };

private:
  static std::vector<size_t> CalcBatchSizes(size_t n_tasks, size_t n_threads);

  std::stack<int> InitializeAvailableSimulatorStack() const;
  size_t num_max_parallel_executions{0};

  mutable std::vector<QuasistaticSimulator> q_sims_;
  mutable std::mt19937 gen_;
};
