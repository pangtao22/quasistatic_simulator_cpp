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

  std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<bool>>
  CalcDynamicsSingleThread(
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

  std::vector<Eigen::MatrixXd> CalcBundledB(
      const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
      const Eigen::Ref<const Eigen::MatrixXd> &u_trj,
      double h, double std_u, int n_samples);

  size_t get_hardware_concurrency() const { return hardware_concurrency_; };

  QuasistaticSimulator& get_q_sim() { return *(q_sims_.begin()); };

private:
  const size_t hardware_concurrency_{0};
  mutable std::list<QuasistaticSimulator> q_sims_;
  std::mt19937 gen_;
};
