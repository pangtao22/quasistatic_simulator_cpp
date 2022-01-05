#include "get_model_paths.h"
#include "quasistatic_simulator.h"

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

  q_sim.UpdateMbpPositions(q0_dict);
  ModelInstanceIndexToVecMap tau_ext_dict = q_sim.CalcTauExt({});
  q_sim.Step(q0_dict, tau_ext_dict, 0.1);

  return 0;
}