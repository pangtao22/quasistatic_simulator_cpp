#include <future>
#include <mutex>

#include "batch_quasistatic_simulator.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

ModelInstanceIndexToVecMap
GetQaCmdDictFromU(const QuasistaticSimulator &q_sim,
                  const Eigen::Ref<const Eigen::VectorXd> &u) {
  ModelInstanceIndexToVecMap q_a_cmd_dict;
  size_t i_start = 0;
  for (const auto &model : q_sim.get_actuated_models()) {
    auto n_v_i = q_sim.get_plant().num_velocities(model);
    q_a_cmd_dict[model] = u.segment(i_start, n_v_i);
    i_start += n_v_i;
  }

  return q_a_cmd_dict;
}

VectorXd CalcDynamics(QuasistaticSimulator *q_sim,
                      const Eigen::Ref<const VectorXd> &q,
                      const Eigen::Ref<const VectorXd> &u, double h,
                      const GradientMode gradient_mode) {
  q_sim->UpdateMbpPositions(q);
  auto tau_ext_dict = q_sim->CalcTauExt({});
  auto q_a_cmd_dict = GetQaCmdDictFromU(*q_sim, u);
  q_sim->Step(q_a_cmd_dict, tau_ext_dict, h,
              q_sim->get_sim_params().contact_detection_tolerance,
              gradient_mode, true);
  return q_sim->GetMbpPositionsVec();
}

BatchQuasistaticSimulator::BatchQuasistaticSimulator(
    const std::string &model_directive_path,
    const std::unordered_map<std::string, Eigen::VectorXd> &robot_stiffness_str,
    const std::unordered_map<std::string, std::string> &object_sdf_paths,
    QuasistaticSimParameters sim_params)
    : hardware_concurrency_(std::thread::hardware_concurrency()) {
  for (int i = 0; i < hardware_concurrency_; i++) {
    q_sims_.emplace_back(model_directive_path, robot_stiffness_str,
                         object_sdf_paths, sim_params);
  }
}

Eigen::MatrixXd BatchQuasistaticSimulator::CalcForwardDynamics(
    const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
    const Eigen::Ref<const Eigen::MatrixXd> &u_batch, double h) const {
  const size_t n_tasks = x_batch.rows();
  DRAKE_THROW_UNLESS(n_tasks == u_batch.rows());

  const auto n_threads = std::min(hardware_concurrency_, n_tasks);
  // TODO: remove this check.
  DRAKE_THROW_UNLESS(n_tasks % n_threads == 0);
  const size_t batch_size = n_tasks / n_threads;

  // Storage for forward dynamics results.
  const size_t n_q = x_batch.cols();
  const size_t n_u = u_batch.cols();

  std::list<MatrixXd> x_next_batch_v;
  for (int i = 0; i < n_threads; i++) {
    x_next_batch_v.emplace_back(batch_size, n_q);
  }

  std::list<std::future<size_t>> operations;

  // Launch threads.
  auto q_sim_iter = q_sims_.begin();
  auto x_next_batch_iter = x_next_batch_v.begin();
  for (size_t i = 0; i < n_threads; i++) {
    auto calc_dynamics_batch = [&q_sim = *q_sim_iter, &x_batch, &u_batch,
                                &x_next_thread = *x_next_batch_iter, batch_size,
                                h, i_thread = i] {
      const auto offset = i_thread * batch_size;
      for (size_t i = 0; i < batch_size; i++) {
        x_next_thread.row(i) =
            CalcDynamics(&q_sim, x_batch.row(i + offset),
                         u_batch.row(i + offset), h, GradientMode::kNone);
      }
      return i_thread;
    };

    operations.emplace_back(
        std::async(std::launch::async, std::move(calc_dynamics_batch)));
    q_sim_iter++;
    x_next_batch_iter++;
  }

  // Collect results from threads.
  int i = 0;
  MatrixXd x_next_batch(n_tasks, n_q);
  x_next_batch_iter = x_next_batch_v.begin();
  for (auto &op : operations) {
    op.get(); // catch exceptions.
    x_next_batch.block(i * batch_size, 0, batch_size, n_q) = *x_next_batch_iter;

    i++;
    x_next_batch_iter++;
  }

  return x_next_batch;
}