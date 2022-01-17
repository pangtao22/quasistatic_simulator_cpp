#include <chrono>

#include "get_model_paths.h"
#include "quasistatic_simulator.h"
#include "batch_quasistatic_simulator.h"

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::string;
using drake::multibody::ModelInstanceIndex;
using std::cout;
using std::endl;

const string kObjectSdfPath = GetQsimModelsPath() / "box_1m.sdf";

const string kModelDirectivePath =
    GetQsimModelsPath() / "three_link_arm_and_ground.yml";

int main() {
  QuasistaticSimParameters sim_params;
  sim_params.gravity = Vector3d(0, 0, -10);
  sim_params.nd_per_contact = 4;
  sim_params.contact_detection_tolerance = INFINITY;
  sim_params.is_quasi_dynamic = true;
  sim_params.gradient_from_active_constraints = true;

  VectorXd  Kp;
  Kp.resize(3);
  Kp << 1000, 1000, 1000;
  const string robot_name("arm");

  std::unordered_map<string, VectorXd> robot_stiffness_dict = {
      {robot_name, Kp}};

  const string object_name("box0");
  std::unordered_map<string, string> object_sdf_dict;
  object_sdf_dict[object_name] = kObjectSdfPath;

  auto q_sim = QuasistaticSimulator(
      kModelDirectivePath, robot_stiffness_dict, object_sdf_dict, sim_params);

  const auto name_to_idx_map = q_sim.GetModelInstanceNameToIndexMap();
  const auto idx_r = name_to_idx_map.at(robot_name);
  const auto idx_o = name_to_idx_map.at(object_name);

  VectorXd q_u0(7);
  q_u0 << 1, 0, 0, 0, 0.0, 1.7, 0.5;
  ModelInstanceIndexToVecMap q0_dict = {
      {idx_o, q_u0},
      {idx_r, Vector3d(M_PI / 2, -M_PI / 2, -M_PI / 2)},
  };

  const int n_tasks = 1200;

  cout << "==Time batch execution==" << endl;
  auto q_sim_batch = BatchQuasistaticSimulator(
      kModelDirectivePath, robot_stiffness_dict, object_sdf_dict, sim_params);

  MatrixXd x_batch(n_tasks, 10);
  MatrixXd u_batch(n_tasks, 3);
  for (int i = 0; i < n_tasks; i++) {
    x_batch.row(i).head(3) = q0_dict[idx_r];
    x_batch.row(i).tail(7) = q0_dict[idx_o];
    u_batch.row(i) = q0_dict[idx_r];
  }

  auto t_start = std::chrono::steady_clock::now();
  auto x_next = q_sim_batch.CalcForwardDynamics(x_batch, u_batch, 0.1);
  auto t_end = std::chrono::steady_clock::now();
  cout << "wall time ms parallel: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(t_end -
           t_start)
           .count()
       << endl;


  cout << "==Time single-thread execution==" << endl;
  t_start = std::chrono::steady_clock::now();
  for (int i = 0; i < n_tasks; i++) {
    q_sim.UpdateMbpPositions(q0_dict);
    ModelInstanceIndexToVecMap tau_ext_dict = q_sim.CalcTauExt({});
    q_sim.Step(q0_dict, tau_ext_dict, 0.1);
  }
  t_end = std::chrono::steady_clock::now();
  cout << "wall time ms serial: "
       << std::chrono::duration_cast<std::chrono::milliseconds>(t_end -
           t_start)
           .count()
       << endl;

//  cout << "x_next\n" << x_next << endl;

  return 0;
}