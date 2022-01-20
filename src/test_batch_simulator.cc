#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include "batch_quasistatic_simulator.h"
#include "get_model_paths.h"

using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;

const string kObjectSdfPath =
    GetQsimModelsPath() / "sphere_yz_rotation_r_0.25m.sdf";

const string kModelDirectivePath = GetQsimModelsPath() / "planar_hand.yml";

MatrixXd CreateRandomMatrix(int n_rows, int n_cols, std::mt19937 &gen) {
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  return MatrixXd::NullaryExpr(n_rows, n_cols, [&]() { return dis(gen); });
}

int main() {
  QuasistaticSimParameters sim_params;
  sim_params.gravity = Vector3d(0, 0, -10);
  sim_params.nd_per_contact = 2;
  sim_params.contact_detection_tolerance = 1.0;
  sim_params.is_quasi_dynamic = true;
  sim_params.gradient_from_active_constraints = true;

  VectorXd Kp;
  Kp.resize(2);
  Kp << 50, 25;
  const string robot_l_name = "arm_left";
  const string robot_r_name = "arm_right";

  std::unordered_map<string, VectorXd> robot_stiffness_dict = {
      {robot_l_name, Kp}, {robot_r_name, Kp}};

  const string object_name("sphere");
  std::unordered_map<string, string> object_sdf_dict;
  object_sdf_dict[object_name] = kObjectSdfPath;

  auto q_sim_batch = BatchQuasistaticSimulator(
      kModelDirectivePath, robot_stiffness_dict, object_sdf_dict, sim_params);
  auto &q_sim = q_sim_batch.get_q_sim();
  const auto name_to_idx_map = q_sim.GetModelInstanceNameToIndexMap();
  const auto idx_l = name_to_idx_map.at(robot_l_name);
  const auto idx_r = name_to_idx_map.at(robot_r_name);
  const auto idx_o = name_to_idx_map.at(object_name);

  ModelInstanceIndexToVecMap q0_dict = {{idx_o, Vector3d(0, 0.316, 0)},
                                        {idx_l, Vector2d(-0.775, -0.785)},
                                        {idx_r, Vector2d(0.775, 0.785)}};

  VectorXd q0 = q_sim.GetQFromQdict(q0_dict);
  VectorXd u0 = q_sim.GetQaCmdFromQaCmdDict(q0_dict);

  auto n_tasks = 11;
  const int n_q = q_sim.get_plant().num_positions();
  const double h = 0.1;

  std::mt19937 gen(1);
  MatrixXd u_batch =
      0.1 * CreateRandomMatrix(n_tasks, q_sim.num_actuated_dofs(), gen);
  u_batch.rowwise() += u0.transpose();

  MatrixXd x_batch(n_tasks, n_q);
  x_batch.setZero();
  x_batch.rowwise() += q0.transpose();
  cout << "Start" << endl;
  MatrixXd x_next_batch_parallel =
      q_sim_batch.CalcDynamicsParallel(x_batch, u_batch, h);
  cout << "Parallel done" << endl;

  MatrixXd x_next_batch_serial =
      q_sim_batch.CalcDynamicsSingleThread(x_batch, u_batch, h);
  cout << "Serial done" << endl;

  cout << "Diff "
       << (x_next_batch_serial - x_next_batch_parallel)
                  .matrix()
                  .rowwise()
                  .norm()
                  .sum() /
              n_tasks
       << endl;

  return 0;
}