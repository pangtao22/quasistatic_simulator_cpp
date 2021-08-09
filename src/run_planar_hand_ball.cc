#include <chrono>

#include "quasistatic_simulator.h"

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::string;
using drake::multibody::ModelInstanceIndex;
using std::cout;
using std::endl;

const string kObjectSdfPath =
    "/Users/pangtao/PycharmProjects/quasistatic_simulator/models/sphere_yz_rotation_r_0.25m.sdf";

const string kModelDirectivePath =
    "/Users/pangtao/PycharmProjects/quasistatic_simulator/models/planar_hand.yml";

std::unordered_map<ModelInstanceIndex, VectorXd>
CreateMapKeyedByModelInstanceIndex(
    const drake::multibody::MultibodyPlant<double>& plant,
    const std::unordered_map<string, VectorXd>& map_str) {
  std::unordered_map<ModelInstanceIndex, VectorXd> map_model;
  for(const auto& [name, v] : map_str) {
    auto model = plant.GetModelInstanceByName(name);
    map_model[model] = v;
  }
  return map_model;
}


int main() {
  QuasistaticSimParameters sim_params;
  sim_params.gravity = Vector3d(0, 0, -10);
  sim_params.nd_per_contact = 2;
  sim_params.contact_detection_tolerance = 1.0;
  sim_params.is_quasi_dynamic = true;
  sim_params.requires_grad = true;

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

  auto q_sim = QuasistaticSimulator(
      kModelDirectivePath, robot_stiffness_dict, object_sdf_dict, sim_params);

  std::unordered_map<string, VectorXd> q0_dict_str = {
      {object_name, Vector3d(0, 0.35, 0)},
      {robot_l_name, Vector2d(-M_PI / 4, -M_PI / 4)},
      {robot_r_name, Vector2d(M_PI / 4, M_PI / 4)}
  };

  auto q0_dict = CreateMapKeyedByModelInstanceIndex(
      q_sim.get_plant(), q0_dict_str);

  auto t_start = std::chrono::high_resolution_clock::now();
  const int n = 1000;
  for (int i = 0; i < n; i++) {
    q_sim.UpdateMbpPositions(q0_dict);
    ModelInstanceToVecMap tau_ext_dict = q_sim.CalcTauExt({});
    q_sim.Step(q0_dict, tau_ext_dict, 0.1);
    auto q_next_dict = q_sim.GetMbpPositions();
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  cout << "wall time microseconds per dynamics: " <<
    std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start)
        .count() / n
    << endl;

  cout << "Dq_nextDq\n" << q_sim.get_Dq_nextDq() << endl;
  cout << "Dq_nextDqa_cmd\n" << q_sim.get_Dq_nextDqa_cmd() << endl;

//  for(const auto& [model, q_i] : q_next_dict) {
//    cout << model << " " << q_i.transpose() << endl;
//  }

  return 0;
}
