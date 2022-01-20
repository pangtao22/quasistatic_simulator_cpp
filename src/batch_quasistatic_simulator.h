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

  Eigen::MatrixXd CalcDynamicsSingleThread(
      const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
      const Eigen::Ref<const Eigen::MatrixXd> &u_batch, double h);

  /*
   * Each row in x_batch and u_batch represent a pair of current states and
   * inputs. The function returns x_next_batch, where
   *  x_next_batch.row(i) = f(x_batch.row(i), u_batch.row(i))
   */
  Eigen::MatrixXd
  CalcDynamicsParallel(const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
                       const Eigen::Ref<const Eigen::MatrixXd> &u_batch,
                       double h) const;

  size_t get_hardware_concurrency() const { return hardware_concurrency_; };

  QuasistaticSimulator& get_q_sim() { return *(q_sims_.begin()); };

private:
  const size_t hardware_concurrency_{0};
  mutable std::list<QuasistaticSimulator> q_sims_;
};
