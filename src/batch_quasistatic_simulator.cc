#include <future>
#include <queue>
#include <spdlog/spdlog.h>
#include <stack>

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
  std::random_device rd;
  gen_.seed(rd());

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

Eigen::MatrixXd BatchQuasistaticSimulator::CalcBundledB(
    QuasistaticSimulator *q_sim, const Eigen::Ref<const Eigen::VectorXd> &q,
    const Eigen::Ref<const Eigen::VectorXd> &u, double h,
    const Eigen::Ref<const Eigen::MatrixXd> &du) {
  const auto n_q = q.size();
  const auto n_u = u.size();
  MatrixXd B_bundled(n_q, n_u);
  B_bundled.setZero();

  int n_valid = 0;
  for (int i = 0; i < du.rows(); i++) {
    VectorXd u_new = u + du.row(i).transpose();
    try {
      CalcDynamics(q_sim, q, u_new, h, GradientMode::kBOnly);
      B_bundled += q_sim->get_Dq_nextDqa_cmd();
      n_valid++;
    } catch (std::runtime_error &err) {
      spdlog::warn(err.what());
    }
  }
  B_bundled /= n_valid;

  return B_bundled;
}

std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<bool>>
BatchQuasistaticSimulator::CalcDynamicsSerial(
    const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
    const Eigen::Ref<const Eigen::MatrixXd> &u_batch, double h,
    const GradientMode gradient_mode) {
  DRAKE_THROW_UNLESS(gradient_mode != GradientMode::kAB);

  const size_t n_tasks = x_batch.rows();
  DRAKE_THROW_UNLESS(n_tasks == u_batch.rows());
  MatrixXd x_next_batch(x_batch);
  std::vector<MatrixXd> B_batch;
  std::vector<bool> is_valid_batch(n_tasks);
  const auto n_q = x_batch.cols();
  const auto n_u = u_batch.cols();

  auto &q_sim = get_q_sim();

  for (int i = 0; i < n_tasks; i++) {
    if (gradient_mode == GradientMode::kBOnly) {
      B_batch.emplace_back(MatrixXd::Zero(n_q, n_u));
    }
    try {
      x_next_batch.row(i) = CalcDynamics(&q_sim, x_batch.row(i), u_batch.row(i),
                                         h, gradient_mode);
      if (gradient_mode == GradientMode::kBOnly) {
        B_batch.back() = q_sim.get_Dq_nextDqa_cmd();
      }

      is_valid_batch[i] = true;
    } catch (std::runtime_error &err) {
      is_valid_batch[i] = false;
    }
  }

  return {x_next_batch, B_batch, is_valid_batch};
}

std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, std::vector<bool>>
BatchQuasistaticSimulator::CalcDynamicsParallel(
    const Eigen::Ref<const Eigen::MatrixXd> &x_batch,
    const Eigen::Ref<const Eigen::MatrixXd> &u_batch, const double h,
    const GradientMode gradient_mode) const {
  DRAKE_THROW_UNLESS(gradient_mode != GradientMode::kAB);

  // Compute number of threads and batch size for each thread.
  const size_t n_tasks = x_batch.rows();
  DRAKE_THROW_UNLESS(n_tasks == u_batch.rows());
  const auto n_threads = std::min(hardware_concurrency_, n_tasks);
  const size_t batch_size = n_tasks / n_threads;
  std::vector<size_t> batch_sizes(n_threads);
  for (int i = 0; i < n_threads - 1; i++) {
    batch_sizes[i] = batch_size;
  }
  batch_sizes[n_threads - 1] = n_tasks - (n_threads - 1) * batch_size;

  // Allocate storage for results.
  const auto n_q = x_batch.cols();
  const auto n_u = u_batch.cols();

  std::list<MatrixXd> x_next_list;
  std::list<std::vector<MatrixXd>> B_list;
  std::list<std::vector<bool>> is_valid_list;
  for (int i = 0; i < n_threads; i++) {
    x_next_list.emplace_back(batch_sizes[i], n_q);

    if (gradient_mode == GradientMode::kBOnly) {
      B_list.emplace_back(batch_sizes[i]);
      auto &B_batch = B_list.back();
      std::for_each(
          B_batch.begin(), B_batch.end(),
          [n_q, n_u](Eigen::MatrixXd &A) { A = MatrixXd::Zero(n_q, n_u); });
    }

    is_valid_list.emplace_back(batch_sizes[i]);
  }

  // Launch threads.
  std::list<std::future<void>> operations;
  auto q_sim_iter = q_sims_.begin();
  auto x_next_iter = x_next_list.begin();
  auto B_iter = B_list.begin();
  auto is_valid_iter = is_valid_list.begin();

  for (int i_thread = 0; i_thread < n_threads; i_thread++) {
    // subscript _t indicates a quantity for a thread.
    auto calc_dynamics_batch = [&q_sim = *q_sim_iter, &x_batch, &u_batch,
                                &x_next_t = *x_next_iter, &B_t = *B_iter,
                                &is_valid_t = *is_valid_iter, &batch_sizes, h,
                                i_thread, gradient_mode] {
      const auto offset = std::accumulate(batch_sizes.begin(),
                                          batch_sizes.begin() + i_thread, 0);
      for (int i = 0; i < batch_sizes[i_thread]; i++) {
        try {
          x_next_t.row(i) =
              CalcDynamics(&q_sim, x_batch.row(i + offset),
                           u_batch.row(i + offset), h, gradient_mode);

          if (gradient_mode == GradientMode::kBOnly) {
            B_t[i] = q_sim.get_Dq_nextDqa_cmd();
          }

          is_valid_t[i] = true;
        } catch (std::runtime_error &err) {
          is_valid_t[i] = false;
          spdlog::warn(err.what());
        }
      }
    };

    operations.emplace_back(
        std::async(std::launch::async, std::move(calc_dynamics_batch)));

    q_sim_iter++;
    x_next_iter++;
    B_iter++;
    is_valid_iter++;
  }

  // Collect results from threads.
  // x_next;
  int i = 0;
  MatrixXd x_next_batch(n_tasks, n_q);
  x_next_iter = x_next_list.begin();
  for (auto &op : operations) {
    op.get(); // catch exceptions.
    x_next_batch.block(i * batch_size, 0, batch_sizes[i], n_q) = *x_next_iter;

    i++;
    x_next_iter++;
  }

  // B.
  std::vector<MatrixXd> B_batch;
  for (B_iter = B_list.begin(); B_iter != B_list.end(); B_iter++) {
    B_batch.insert(B_batch.end(), std::make_move_iterator(B_iter->begin()),
                   std::make_move_iterator(B_iter->end()));
  }

  // is_valid_batch.
  std::vector<bool> is_valid_batch;
  for (is_valid_iter = is_valid_list.begin();
       is_valid_iter != is_valid_list.end(); is_valid_iter++) {
    is_valid_batch.insert(is_valid_batch.end(), is_valid_iter->begin(),
                          is_valid_iter->end());
  }

  return {x_next_batch, B_batch, is_valid_batch};
}

/*
 * x_trj: (T + 1, dim_x)
 * u_trj: (T, dim_u)
 */
std::vector<Eigen::MatrixXd> BatchQuasistaticSimulator::CalcBundledBTrj(
    const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
    const Eigen::Ref<const Eigen::MatrixXd> &u_trj, double h, double std_u,
    int n_samples, std::optional<int> seed) {
  if (seed.has_value()) {
    gen_.seed(seed.value());
  }

  const int T = u_trj.rows();
  DRAKE_THROW_UNLESS(x_trj.rows() == T + 1);

  const int n_x = x_trj.cols();
  const int n_u = u_trj.cols();

  MatrixXd x_batch(T * n_samples, n_x);
  MatrixXd u_batch(T * n_samples, n_u);
  std::normal_distribution<> d{0, std_u};
  for (int t = 0; t < T; t++) {
    int i_start = t * n_samples;
    MatrixXd du =
        MatrixXd::NullaryExpr(n_samples, n_u, [&]() { return d(gen_); });
    for (int i = 0; i < n_samples; i++) {
      x_batch.row(i_start + i) = x_trj.row(t);
      u_batch.row(i_start + i) = u_trj.row(t) + du.row(i);
    }
  }

  auto [x_next_batch, B_batch, is_valid_batch] =
      CalcDynamicsParallel(x_batch, u_batch, h, GradientMode::kBOnly);

  std::vector<MatrixXd> B_bundled;
  for (int t = 0; t < T; t++) {
    int i_start = t * n_samples;
    int n_valid_samples = 0;
    B_bundled.emplace_back(n_x, n_u);
    B_bundled.back().setZero();

    for (int i = 0; i < n_samples; i++) {
      if (is_valid_batch[i_start + i]) {
        n_valid_samples++;
        B_bundled.back() += B_batch[i_start + i];
      }
    }

    B_bundled.back() /= n_valid_samples;
  }

  return B_bundled;
}

template <typename T> bool IsFutureReady(const std::future<T> &future) {
  // future.wait_for() is the only method to check the status of a future
  // without waiting for it to complete.
  const std::future_status status =
      future.wait_for(std::chrono::milliseconds(1));
  return (status == std::future_status::ready);
}

std::stack<int>
BatchQuasistaticSimulator::InitializeAvailableSimulatorStack() const {
  std::stack<int> available_sims;
  for (int i = 0; i < hardware_concurrency_; i++) {
    available_sims.push(i);
  }
  return available_sims;
}

std::vector<MatrixXd> BatchQuasistaticSimulator::CalcBundledBTrjDirect(
    const Eigen::Ref<const Eigen::MatrixXd> &x_trj,
    const Eigen::Ref<const Eigen::MatrixXd> &u_trj, double h, double std_u,
    int n_samples, std::optional<int> seed) {
  if (seed.has_value()) {
    gen_.seed(seed.value());
  }

  // Determine the number of threads.
  const size_t T = u_trj.rows();
  DRAKE_THROW_UNLESS(x_trj.rows() == T + 1);
  const auto n_threads = std::min(hardware_concurrency_, T);

  // Allocate storage for results.
  const auto n_q = x_trj.cols();
  const auto n_u = u_trj.cols();
  std::vector<MatrixXd> B_batch(T, MatrixXd::Zero(n_q, n_u));

  // Generate samples.
  std::vector<MatrixXd> du_trj(T);
  std::normal_distribution<> d{0, std_u};
  for (int t = 0; t < T; t++) {
    du_trj[t] =
        MatrixXd::NullaryExpr(n_samples, n_u, [&]() { return d(gen_); });
  }

  // Storage for active parallel simulation operations.
  std::list<std::future<int>> active_operations;
  int n_bundled_B_dispatched = 0;
  auto available_sims = InitializeAvailableSimulatorStack();

  while (!active_operations.empty() || n_bundled_B_dispatched < T) {
    // Check for completed operations.
    for (auto op = active_operations.begin(); op != active_operations.end();) {
      if (IsFutureReady(*op)) {
        auto sim_idx = op->get();
        op = active_operations.erase(op);
        available_sims.push(sim_idx);
      } else {
        ++op;
      }
    }

    // Dispatch new operations.
    while (static_cast<int>(active_operations.size()) < n_threads &&
           n_bundled_B_dispatched < T) {
      DRAKE_THROW_UNLESS(!available_sims.empty());
      auto idx_sim = available_sims.top();
      available_sims.pop();

      auto calc_B_bundled =
          [&q_sim = q_sims_[idx_sim], &x_trj = std::as_const(x_trj),
           &u_trj = std::as_const(u_trj), &du_trj = std::as_const(du_trj),
           &B_batch, t = n_bundled_B_dispatched, h, idx_sim] {
            B_batch[t] =
                CalcBundledB(&q_sim, x_trj.row(t), u_trj.row(t), h, du_trj[t]);
            return idx_sim;
          };

      active_operations.emplace_back(
          std::async(std::launch::async, std::move(calc_B_bundled)));
      ++n_bundled_B_dispatched;
    }

    // Wait a bit before checking for completion.
    // For the planar hand and ball system, computing forward dynamics and
    // its gradient takes a bit more than 1ms on average.
    std::this_thread::sleep_for(std::chrono::milliseconds(n_samples));
  }

  return B_batch;
}
