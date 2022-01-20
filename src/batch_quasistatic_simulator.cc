#include <future>
#include <mutex>

#include "batch_quasistatic_simulator.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

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

VectorXd BatchQuasistaticSimulator::CalcDynamics(
    QuasistaticSimulator *q_sim, const Eigen::Ref<const VectorXd> &q,
    const Eigen::Ref<const VectorXd> &u, double h,
    const GradientMode gradient_mode) {
  q_sim->UpdateMbpPositions(q);
  auto tau_ext_dict = q_sim->CalcTauExt({});
  auto q_a_cmd_dict = q_sim->GetQaCmdDictFromQaCmd(u);
  q_sim->Step(q_a_cmd_dict, tau_ext_dict, h,
              q_sim->get_sim_params().contact_detection_tolerance,
              gradient_mode, true);
  return q_sim->GetMbpPositionsAsVec();
}

Eigen::MatrixXd BatchQuasistaticSimulator::CalcDynamicsSingleThread(
    const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
    const Eigen::Ref<const Eigen::MatrixXd> &u_batch, double h) {
  const size_t n_tasks = x_batch.rows();
  DRAKE_THROW_UNLESS(n_tasks == u_batch.rows());
  MatrixXd x_next_batch(x_batch);

  auto &q_sim = *(q_sims_.begin());

  for (int i = 0; i < n_tasks; i++) {
    x_next_batch.row(i) = CalcDynamics(&q_sim, x_batch.row(i), u_batch.row(i),
                                       h, GradientMode::kNone);
  }
  return x_next_batch;
}

Eigen::MatrixXd BatchQuasistaticSimulator::CalcDynamicsParallel(
    const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
    const Eigen::Ref<const Eigen::MatrixXd> &u_batch, double h) const {
  const size_t n_tasks = x_batch.rows();
  DRAKE_THROW_UNLESS(n_tasks == u_batch.rows());

  const auto n_threads = std::min(hardware_concurrency_, n_tasks);
  const size_t batch_size = n_tasks / n_threads;
  std::vector<size_t> batch_sizes(n_threads);
  for (int i = 0; i < n_threads - 1; i++) {
    batch_sizes[i] = batch_size;
  }
  batch_sizes[n_threads - 1] = n_tasks - (n_threads - 1) * batch_size;

  // Storage for forward dynamics results.
  const size_t n_q = x_batch.cols();
  const size_t n_u = u_batch.cols();

  std::list<MatrixXd> x_next_batch_v;
  for (int i = 0; i < n_threads; i++) {
    x_next_batch_v.emplace_back(batch_sizes[i], n_q);
  }

  std::list<std::future<void>> operations;

  // Launch threads.
  auto q_sim_iter = q_sims_.begin();
  auto x_next_batch_iter = x_next_batch_v.begin();
  for (int i_thread = 0; i_thread < n_threads; i_thread++) {
    auto calc_dynamics_batch = [&q_sim = *q_sim_iter, &x_batch, &u_batch,
                                &x_next_thread = *x_next_batch_iter,
                                &batch_sizes, h, i_thread] {
      const auto offset = std::accumulate(batch_sizes.begin(),
                                          batch_sizes.begin() + i_thread, 0);
      for (int i = 0; i < batch_sizes[i_thread]; i++) {
        x_next_thread.row(i) =
            CalcDynamics(&q_sim, x_batch.row(i + offset),
                         u_batch.row(i + offset), h, GradientMode::kNone);
      }
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
    x_next_batch.block(i * batch_size, 0, batch_sizes[i], n_q) =
        *x_next_batch_iter;

    i++;
    x_next_batch_iter++;
  }

  return x_next_batch;
}