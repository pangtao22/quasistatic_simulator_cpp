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
    "/Users/pangtao/PycharmProjects/quasistatic_simulator/models/box_1m.sdf";

const string kModelDirectivePath =
    "/Users/pangtao/PycharmProjects/quasistatic_simulator/models/three_link_arm_and_ground.yml";


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


  VectorXd q_u0(7);
  q_u0 << 1, 0, 0, 0, 0.0, 1.7, 0.5;
  std::unordered_map<string, VectorXd> q0_dict_str = {
      {object_name, q_u0},
      {robot_name, Vector3d(M_PI / 2, -M_PI / 2, -M_PI / 2)},
  };

  auto q0_dict = CreateMapKeyedByModelInstanceIndex(
      q_sim.get_plant(), q0_dict_str);

  q_sim.UpdateMbpPositions(q0_dict);
  ModelInstanceToVecMap tau_ext_dict = q_sim.CalcTauExt({});
  q_sim.Step(q0_dict, tau_ext_dict, 0.1);

  return 0;
}