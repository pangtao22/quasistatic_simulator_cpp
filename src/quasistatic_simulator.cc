#include <set>
#include <vector>

#include "drake/common/drake_path.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/parsing/process_model_directives.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/systems/framework/diagram_builder.h"

#include "get_model_paths.h"
#include "quasistatic_simulator.h"

using drake::AutoDiffXd;
using drake::Matrix3X;
using drake::MatrixX;
using drake::Vector3;
using drake::Vector4;
using drake::VectorX;
using drake::math::ExtractValue;
using drake::math::InitializeAutoDiff;
using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;
using std::vector;

void CreateMbp(
    drake::systems::DiagramBuilder<double> *builder,
    const string &model_directive_path,
    const std::unordered_map<string, VectorXd> &robot_stiffness_str,
    const std::unordered_map<string, string> &object_sdf_paths,
    const Eigen::Ref<const Vector3d> &gravity,
    drake::multibody::MultibodyPlant<double> **plant,
    drake::geometry::SceneGraph<double> **scene_graph,
    std::set<ModelInstanceIndex> *robot_models,
    std::set<ModelInstanceIndex> *object_models,
    std::unordered_map<ModelInstanceIndex, Eigen::VectorXd> *robot_stiffness) {
  std::tie(*plant, *scene_graph) =
      drake::multibody::AddMultibodyPlantSceneGraph(builder, 1e-3);
  auto parser = drake::multibody::Parser(*plant, *scene_graph);
  // TODO: add package paths from yaml file? Hard-coding paths is clearly not
  //  the solution...
  parser.package_map().Add("quasistatic_simulator", GetQsimModelsPath());
  parser.package_map().Add("drake_manipulation_models",
                           drake::MaybeGetDrakePath().value() +
                               "/manipulation/models");
  parser.package_map().Add("iiwa_controller", GetRoboticsUtilitiesModelsPath());

  drake::multibody::parsing::ProcessModelDirectives(
      drake::multibody::parsing::LoadModelDirectives(model_directive_path),
      *plant, nullptr, &parser);

  // Objects.
  // Use a set to sort object names.
  std::set<std::string> object_names;
  for (const auto &item : object_sdf_paths) {
    object_names.insert(item.first);
  }
  for (const auto &name : object_names) {
    object_models->insert(
        parser.AddModelFromFile(object_sdf_paths.at(name), name));
  }

  // Robots.
  for (const auto &[name, Kp] : robot_stiffness_str) {
    auto robot_model = (*plant)->GetModelInstanceByName(name);
    robot_models->insert(robot_model);
    (*robot_stiffness)[robot_model] = Kp;
  }

  // Gravity.
  (*plant)->mutable_gravity_field().set_gravity_vector(gravity);
  (*plant)->Finalize();
}

template <class T>
drake::Matrix3X<T> CalcTangentVectors(const Vector3<T> &normal,
                                      const size_t nd) {
  Vector3<T> n = normal.normalized();
  Vector4<T> n4(n.x(), n.y(), n.z(), 0);
  Matrix3X<T> tangents(3, nd);
  if (nd == 2) {
    // Makes sure that dC is in the yz plane.
    Vector4<T> n_x4(1, 0, 0, 0);
    tangents.col(0) = n_x4.cross3(n4).head(3);
    tangents.col(1) = -tangents.col(0);
  } else {
    const auto R = drake::math::RotationMatrix<T>::MakeFromOneUnitVector(n, 2);

    for (int i = 0; i < nd; i++) {
      const double theta = 2 * M_PI / nd * i;
      tangents.col(i) << cos(theta), sin(theta), 0;
    }
    tangents = R * tangents;
  }

  return tangents;
}

QuasistaticSimulator::QuasistaticSimulator(
    const std::string &model_directive_path,
    const std::unordered_map<std::string, Eigen::VectorXd> &robot_stiffness_str,
    const std::unordered_map<std::string, std::string> &object_sdf_paths,
    QuasistaticSimParameters sim_params)
    : sim_params_(std::move(sim_params)),
      solver_grb_(std::make_unique<drake::solvers::GurobiSolver>()),
      solver_msk_(std::make_unique<drake::solvers::MosekSolver>()) {
  auto builder = drake::systems::DiagramBuilder<double>();

  CreateMbp(&builder, model_directive_path, robot_stiffness_str,
            object_sdf_paths, sim_params_.gravity, &plant_, &sg_,
            &models_actuated_, &models_unactuated_, &robot_stiffness_);
  // All models instances.
  models_all_ = models_unactuated_;
  models_all_.insert(models_actuated_.begin(), models_actuated_.end());
  diagram_ = builder.Build();

  // Contexts.
  context_ = diagram_->CreateDefaultContext();
  context_plant_ =
      &(diagram_->GetMutableSubsystemContext(*plant_, context_.get()));
  context_sg_ = &(diagram_->GetMutableSubsystemContext(*sg_, context_.get()));

  // MBP introspection.
  n_q_ = plant_->num_positions();
  n_v_ = plant_->num_velocities();

  for (const auto &model : models_all_) {
    velocity_indices_[model] = GetIndicesForModel(model, ModelIndicesMode::kV);
    position_indices_[model] = GetIndicesForModel(model, ModelIndicesMode::kQ);
    const auto body_indices = plant_->GetBodyIndices(model);
    bodies_indices_[model].insert(body_indices.begin(), body_indices.end());
  }

  n_v_a_ = 0;
  for (const auto &model : models_actuated_) {
    n_v_a_ += plant_->num_velocities(model);
  }

  n_v_u_ = 0;
  for (const auto &model : models_unactuated_) {
    n_v_u_ += plant_->num_velocities(model);
  }

  // Find planar model instances.
  /* Features of a 3D un-actuated model instance:
   *
   * 1. The model instance has only 1 rigid body.
   * 2. The model instance has a floating base.
   * 3. The model instance has 6 velocities and 7 positions.
   */
  for (const auto &model : models_unactuated_) {
    const auto n_v = plant_->num_velocities(model);
    const auto n_q = plant_->num_positions(model);
    if (n_v == 6 and n_q == 7) {
      const auto body_indices = plant_->GetBodyIndices(model);
      DRAKE_THROW_UNLESS(body_indices.size() == 1);
      DRAKE_THROW_UNLESS(plant_->get_body(body_indices.at(0)).is_floating());
      is_3d_floating_[model] = true;
    } else {
      is_3d_floating_[model] = false;
    }
  }

  for (const auto &model : models_actuated_) {
    is_3d_floating_[model] = false;
  }

  // friction coefficients.
  const auto &inspector = sg_->model_inspector();
  const auto cc = inspector.GetCollisionCandidates();
  for (const auto &[g_idA, g_idB] : cc) {
    const double mu = GetFrictionCoefficientForSignedDistancePair(g_idA, g_idB);
    friction_coefficients_[g_idA][g_idB] = mu;
    friction_coefficients_[g_idB][g_idA] = mu;
  }

  // QP derivative.
  dqp_ = std::make_unique<QpDerivativesActive>(
      sim_params_.gradient_lstsq_tolerance);

  // Find smallest stiffness.
  VectorXd min_stiffness_vec(models_actuated_.size());
  int i = 0;
  for (const auto &model : models_actuated_) {
    min_stiffness_vec[i] = robot_stiffness_.at(model).minCoeff();
    i++;
  }
  min_K_a_ = min_stiffness_vec.minCoeff();

  // AutoDiff plants.
  diagram_ad_ =
      drake::systems::System<double>::ToAutoDiffXd<drake::systems::Diagram>(
          *diagram_);
  plant_ad_ =
      dynamic_cast<const drake::multibody::MultibodyPlant<AutoDiffXd> *>(
          &(diagram_ad_->GetSubsystemByName(plant_->get_name())));
  sg_ad_ = dynamic_cast<const drake::geometry::SceneGraph<drake::AutoDiffXd> *>(
      &(diagram_ad_->GetSubsystemByName(sg_->get_name())));

  // AutoDiff contexts.
  context_ad_ = diagram_ad_->CreateDefaultContext();
  context_plant_ad_ =
      &(diagram_ad_->GetMutableSubsystemContext(*plant_ad_, context_ad_.get()));
  context_sg_ad_ =
      &(diagram_ad_->GetMutableSubsystemContext(*sg_ad_, context_ad_.get()));
}

std::vector<int> QuasistaticSimulator::GetIndicesForModel(
    drake::multibody::ModelInstanceIndex idx, ModelIndicesMode mode) const {
  std::vector<double> selector;
  if (mode == ModelIndicesMode::kQ) {
    selector.resize(n_q_);
  } else {
    selector.resize(n_v_);
  }
  std::iota(selector.begin(), selector.end(), 0);
  Eigen::Map<VectorXd> selector_eigen(selector.data(), selector.size());

  VectorXd indices_d;
  if (mode == ModelIndicesMode::kQ) {
    indices_d = plant_->GetPositionsFromArray(idx, selector_eigen);
  } else {
    indices_d = plant_->GetVelocitiesFromArray(idx, selector_eigen);
  }
  std::vector<int> indices(indices_d.size());
  for (size_t i = 0; i < indices_d.size(); i++) {
    indices[i] = roundl(indices_d[i]);
  }
  return indices;
}

double QuasistaticSimulator::GetFrictionCoefficientForSignedDistancePair(
    drake::geometry::GeometryId id_A, drake::geometry::GeometryId id_B) const {
  const auto &inspector = sg_->model_inspector();
  const auto props_A = inspector.GetProximityProperties(id_A);
  const auto props_B = inspector.GetProximityProperties(id_B);
  const auto &geometryA_friction =
      props_A->GetProperty<drake::multibody::CoulombFriction<double>>(
          "material", "coulomb_friction");
  const auto &geometryB_friction =
      props_B->GetProperty<drake::multibody::CoulombFriction<double>>(
          "material", "coulomb_friction");
  auto cf = drake::multibody::CalcContactFrictionFromSurfaceProperties(
      geometryA_friction, geometryB_friction);
  return cf.static_friction();
}

drake::multibody::BodyIndex QuasistaticSimulator::GetMbpBodyFromGeometry(
    drake::geometry::GeometryId g_id) const {
  const auto &inspector = sg_->model_inspector();
  return plant_->GetBodyFromFrameId(inspector.GetFrameId(g_id))->index();
}

/*
 * Similar to the python implementation, this function updates context_plant_
 * and query_object_.
 */
void QuasistaticSimulator::UpdateMbpPositions(
    const ModelInstanceIndexToVecMap &q_dict) {
  for (const auto &model : models_all_) {
    plant_->SetPositions(context_plant_, model, q_dict.at(model));
  }

  query_object_ =
      &(sg_->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
          *context_sg_));
}

void QuasistaticSimulator::UpdateMbpPositions(
    const Eigen::Ref<const Eigen::VectorXd> &q) {
  plant_->SetPositions(context_plant_, q);
  query_object_ =
      &(sg_->get_query_output_port().Eval<drake::geometry::QueryObject<double>>(
          *context_sg_));
}

void QuasistaticSimulator::UpdateMbpAdPositions(
    const ModelInstanceIndexToVecAdMap &q_dict) const {
  for (const auto &model : models_all_) {
    plant_ad_->SetPositions(context_plant_ad_, model, q_dict.at(model));
  }

  query_object_ad_ =
      &(sg_ad_->get_query_output_port()
            .Eval<drake::geometry::QueryObject<AutoDiffXd>>(*context_sg_ad_));
}

void QuasistaticSimulator::UpdateMbpAdPositions(
    const Eigen::Ref<const drake::AutoDiffVecXd> &q) const {
  plant_ad_->SetPositions(context_plant_ad_, q);

  query_object_ad_ =
      &(sg_ad_->get_query_output_port()
            .Eval<drake::geometry::QueryObject<AutoDiffXd>>(*context_sg_ad_));
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetMbpPositions() const {
  ModelInstanceIndexToVecMap q_dict;
  for (const auto &model : models_all_) {
    q_dict[model] = plant_->GetPositions(*context_plant_, model);
  }
  return q_dict;
}

Eigen::VectorXd QuasistaticSimulator::GetPositions(
    drake::multibody::ModelInstanceIndex model) const {
  return plant_->GetPositions(*context_plant_, model);
}

/*
 * Returns nullptr if body_idx is not in any of the values of bodies_indices_;
 * Otherwise returns the model instance to which body_idx belongs.
 */
std::unique_ptr<ModelInstanceIndex> QuasistaticSimulator::FindModelForBody(
    drake::multibody::BodyIndex body_idx) const {
  for (const auto &[model, body_indices] : bodies_indices_) {
    auto search = body_indices.find(body_idx);
    if (search != body_indices.end()) {
      return std::make_unique<ModelInstanceIndex>(model);
    }
  }
  return nullptr;
}

template <class T>
void QuasistaticSimulator::UpdateJacobianRows(
    const drake::multibody::MultibodyPlant<T> *plant,
    const drake::systems::Context<T> *context_plant,
    const drake::multibody::BodyIndex &body_idx,
    const drake::VectorX<T> &pC_Body, const drake::VectorX<T> &n_W,
    const drake::MatrixX<T> &d_W, size_t i_c, size_t n_d, size_t i_f_start,
    drake::EigenPtr<drake::MatrixX<T>> Jn_ptr,
    drake::EigenPtr<drake::MatrixX<T>> Jf_ptr) {
  drake::Matrix3X<T> Ji(3, plant->num_velocities());
  const auto &frameB = plant->get_body(body_idx).body_frame();
  plant->CalcJacobianTranslationalVelocity(
      *context_plant, drake::multibody::JacobianWrtVariable::kV, frameB,
      pC_Body, plant->world_frame(), plant->world_frame(), &Ji);

  (*Jn_ptr).row(i_c) += n_W.transpose() * Ji;
  for (size_t i = 0; i < n_d; i++) {
    (*Jf_ptr).row(i + i_f_start) += d_W.col(i).transpose() * Ji;
  }
}

template <class T>
void QuasistaticSimulator::CalcJacobianAndPhi(
    const drake::multibody::MultibodyPlant<T> *plant,
    const drake::geometry::SceneGraph<T> *sg,
    const drake::systems::Context<T> *context_plant,
    const vector<drake::geometry::SignedDistancePair<T>> &sdps, const int n_d,
    drake::VectorX<T> *phi_ptr, drake::VectorX<T> *phi_constraints_ptr,
    drake::MatrixX<T> *Jn_ptr, drake::MatrixX<T> *J_ptr) const {
  VectorX<T> &phi = *phi_ptr;
  VectorX<T> &phi_constraints = *phi_constraints_ptr;
  MatrixX<T> &Jn = *Jn_ptr;
  MatrixX<T> &J = *J_ptr;

  // Contact Jacobians.
  const auto n_c = sdps.size();
  const int n_v = plant->num_velocities();
  const auto n_f = n_d * n_c;

  phi.resize(n_c);
  Jn.resize(n_c, n_v);
  Jn.setZero();

  VectorX<T> U(n_c);
  MatrixX<T> Jf(n_f, n_v);
  Jf.setZero();
  const auto &inspector = sg->model_inspector();

  int i_f_start = 0;
  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto &sdp = sdps[i_c];
    phi[i_c] = sdp.distance;
    U[i_c] = friction_coefficients_.at(sdp.id_A).at(sdp.id_B);
    const auto bodyA_idx = GetMbpBodyFromGeometry(sdp.id_A);
    const auto bodyB_idx = GetMbpBodyFromGeometry(sdp.id_B);
    const auto &X_AGa = inspector.GetPoseInFrame(sdp.id_A).template cast<T>();
    const auto &X_AGb = inspector.GetPoseInFrame(sdp.id_B).template cast<T>();
    const auto p_ACa_A = X_AGa * sdp.p_ACa;
    const auto p_BCb_B = X_AGb * sdp.p_BCb;

    const auto model_A_ptr = FindModelForBody(bodyA_idx);
    const auto model_B_ptr = FindModelForBody(bodyB_idx);

    if (model_A_ptr and model_B_ptr) {
      const auto n_A_W = sdp.nhat_BA_W;
      const auto d_A_W = CalcTangentVectors<T>(n_A_W, n_d);
      const auto n_B_W = -n_A_W;
      const auto d_B_W = -d_A_W;
      UpdateJacobianRows<T>(plant, context_plant, bodyA_idx, p_ACa_A, n_A_W,
                            d_A_W, i_c, n_d, i_f_start, &Jn, &Jf);
      UpdateJacobianRows<T>(plant, context_plant, bodyB_idx, p_BCb_B, n_B_W,
                            d_B_W, i_c, n_d, i_f_start, &Jn, &Jf);
    } else if (model_A_ptr) {
      const auto n_A_W = sdp.nhat_BA_W;
      const auto d_A_W = CalcTangentVectors<T>(n_A_W, n_d);
      UpdateJacobianRows<T>(plant, context_plant, bodyA_idx, p_ACa_A, n_A_W,
                            d_A_W, i_c, n_d, i_f_start, &Jn, &Jf);
    } else if (model_B_ptr) {
      const auto n_B_W = -sdp.nhat_BA_W;
      const auto d_B_W = CalcTangentVectors<T>(n_B_W, n_d);
      UpdateJacobianRows<T>(plant, context_plant, bodyB_idx, p_BCb_B, n_B_W,
                            d_B_W, i_c, n_d, i_f_start, &Jn, &Jf);
    } else {
      throw std::runtime_error(
          "One body in a contact pair is not in body_indices_");
    }
    i_f_start += n_d;
  }

  // Jacobian for constraints.
  phi_constraints.resize(n_f);
  J = Jf;

  int j_start = 0;
  for (int i_c = 0; i_c < n_c; i_c++) {
    for (int j = 0; j < n_d; j++) {
      int idx = j_start + j;
      J.row(idx) = Jn.row(i_c) + U[i_c] * Jf.row(idx);
      phi_constraints[idx] = phi[i_c];
    }
    j_start += n_d;
  }
}

void QuasistaticSimulator::CalcQAndTauH(
    const ModelInstanceIndexToVecMap &q_dict,
    const ModelInstanceIndexToVecMap &q_a_cmd_dict,
    const ModelInstanceIndexToVecMap &tau_ext_dict, const double h,
    MatrixXd *Q_ptr, VectorXd *tau_h_ptr,
    const double unactuated_mass_scale) const {
  MatrixXd &Q = *Q_ptr;
  Q = MatrixXd::Zero(n_v_, n_v_);
  VectorXd &tau_h = *tau_h_ptr;
  tau_h = VectorXd::Zero(n_v_);
  ModelInstanceIndexToMatrixMap M_u_dict;
  if (sim_params_.is_quasi_dynamic) {
    M_u_dict = CalcScaledMassMatrix(h, unactuated_mass_scale);
  }

  for (const auto &model : models_unactuated_) {
    const auto &idx_v = velocity_indices_.at(model);
    const auto n_v_i = idx_v.size();
    const VectorXd &tau_ext = tau_ext_dict.at(model);

    for (int i = 0; i < tau_ext.size(); i++) {
      tau_h(idx_v[i]) = tau_ext(i) * h;
    }

    if (sim_params_.is_quasi_dynamic) {
      for (int i = 0; i < n_v_i; i++) {
        for (int j = 0; j < n_v_i; j++) {
          Q(idx_v[i], idx_v[j]) = M_u_dict.at(model)(i, j);
        }
      }
    }
  }

  for (const auto &model : models_actuated_) {
    const auto &idx_v = velocity_indices_.at(model);
    VectorXd dq_a_cmd = q_a_cmd_dict.at(model) - q_dict.at(model);
    const auto &Kp = robot_stiffness_.at(model);
    VectorXd tau_a_h = Kp.array() * dq_a_cmd.array();
    tau_a_h += tau_ext_dict.at(model);
    tau_a_h *= h;

    for (int i = 0; i < tau_a_h.size(); i++) {
      tau_h(idx_v[i]) = tau_a_h(i);
    }

    for (int i = 0; i < idx_v.size(); i++) {
      int idx = idx_v[i];
      Q(idx, idx) = Kp(i) * h * h;
    }
  }
}

void QuasistaticSimulator::Step(const ModelInstanceIndexToVecMap &q_a_cmd_dict,
                                const ModelInstanceIndexToVecMap &tau_ext_dict,
                                const QuasistaticSimParameters &params) {
  // TODO: handle this better.
  const auto fm = params.forward_mode;
  const auto q_dict = GetMbpPositions();
  auto q_dict_next(q_dict);

  if (fm == ForwardDynamicsMode::kQpMp) {
    // Optimization coefficient matrices and vectors.
    MatrixXd Q, Jn, J;
    VectorXd tau_h, phi, phi_constraints;
    // Primal and dual solutions.
    VectorXd v_star, beta_star;

    CalcPyramidMatrices(q_dict, q_a_cmd_dict, tau_ext_dict, params, &Q, &tau_h,
                        &Jn, &J, &phi, &phi_constraints);
    StepForwardQp(Q, tau_h, J, phi_constraints, params, &q_dict_next, &v_star,
                  &beta_star);
    BackwardQp(Q, tau_h, Jn, J, phi_constraints, q_dict, q_dict_next, v_star,
               beta_star, params);
    return;
  }

  if (fm == ForwardDynamicsMode::kLogPyramidMp) {
    MatrixXd Q, Jn, J;
    VectorXd tau_h, phi, phi_constraints;
    // Primal and dual solutions.
    VectorXd v_star;
    CalcPyramidMatrices(q_dict, q_a_cmd_dict, tau_ext_dict, params, &Q, &tau_h,
                        &Jn, &J, &phi, &phi_constraints);
    StepForwardLogPyramid(Q, tau_h, J, phi_constraints, params, &q_dict_next,
                          &v_star);
    BackwardLogPyramid(Q, J, phi_constraints, q_dict_next, v_star, params);
    return;
  }

  std::stringstream ss;
  ss << "Forward dynamics mode " << static_cast<int>(fm)
     << " is not supported in C++.";
  throw std::logic_error(ss.str());
}

void QuasistaticSimulator::CalcPyramidMatrices(
    const ModelInstanceIndexToVecMap &q_dict,
    const ModelInstanceIndexToVecMap &q_a_cmd_dict,
    const ModelInstanceIndexToVecMap &tau_ext_dict,
    const QuasistaticSimParameters &params, Eigen::MatrixXd *Q,
    Eigen::VectorXd *tau_h, Eigen::MatrixXd *Jn, Eigen::MatrixXd *J,
    Eigen::VectorXd *phi, Eigen::VectorXd *phi_constraints) const {
  // Collision queries.
  const auto &sdps = query_object_->ComputeSignedDistancePairwiseClosestPoints(
      params.contact_detection_tolerance);
  collision_pairs_.clear();
  for (const auto &sdp : sdps) {
    collision_pairs_.emplace_back(sdp.id_A, sdp.id_B);
  }

  CalcJacobianAndPhi(plant_, sg_, context_plant_, sdps, params.nd_per_contact,
                     phi, phi_constraints, Jn, J);
  CalcQAndTauH(q_dict, q_a_cmd_dict, tau_ext_dict, params.h, Q, tau_h,
               params.unactuated_mass_scale);
}

void QuasistaticSimulator::StepForwardQp(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &tau_h,
    const Eigen::Ref<const Eigen::MatrixXd> &J,
    const Eigen::Ref<const Eigen::VectorXd> &phi_constraints,
    const QuasistaticSimParameters &params,
    ModelInstanceIndexToVecMap *q_dict_ptr, Eigen::VectorXd *v_star_ptr,
    Eigen::VectorXd *beta_star_ptr) {
  auto &q_dict = *q_dict_ptr;
  VectorXd &v_star = *v_star_ptr;
  VectorXd &beta_star = *beta_star_ptr;
  const auto n_f = phi_constraints.size();
  const auto h = params.h;

  // construct and solve MathematicalProgram.
  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");
  prog.AddQuadraticCost(Q, -tau_h, v);

  const VectorXd e = phi_constraints / h;
  auto constraints = prog.AddLinearConstraint(
      -J, VectorXd::Constant(n_f, -std::numeric_limits<double>::infinity()), e,
      v);

  solver_grb_->Solve(prog, {}, solver_options_, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error("Quasistatic dynamics QP cannot be solved.");
  }

  v_star = mp_result_.GetSolution(v);
  beta_star = -mp_result_.GetDualSolution(constraints);

  // Update q_dict.
  UpdateQdictFromV(v_star, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

void QuasistaticSimulator::BackwardQp(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &tau_h,
    const Eigen::Ref<const Eigen::MatrixXd> &Jn,
    const Eigen::Ref<const Eigen::MatrixXd> &J,
    const Eigen::Ref<const Eigen::VectorXd> &phi_constraints,
    const ModelInstanceIndexToVecMap &q_dict,
    const ModelInstanceIndexToVecMap &q_dict_next,
    const Eigen::Ref<const Eigen::VectorXd> &v_star,
    const Eigen::Ref<const Eigen::VectorXd> &beta_star,
    const QuasistaticSimParameters &params) {
  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }
  const auto h = params.h;
  const auto n_d = params.nd_per_contact;

  if (params.gradient_mode == GradientMode::kAB) {
    dqp_->UpdateProblem(Q, -tau_h, -J, phi_constraints / h, v_star, beta_star,
                        0.1 * params.h, true);
    const auto &Dv_nextDe = dqp_->get_DzDe();
    const auto &Dv_nextDb = dqp_->get_DzDb();

    Dq_nextDq_ = CalcDfDx(Dv_nextDb, Dv_nextDe, q_dict, h, Jn, n_d);
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, h, q_dict_next);
    return;
  }

  if (params.gradient_mode == GradientMode::kBOnly) {
    dqp_->UpdateProblem(Q, -tau_h, -J, phi_constraints / h, v_star, beta_star,
                        0.1 * params.h, false);
    const auto &Dv_nextDb = dqp_->get_DzDb();
    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, h, q_dict_next);
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    return;
  }

  throw std::runtime_error("Invalid gradient_mode.");
}

void QuasistaticSimulator::BackwardLogPyramid(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::MatrixXd> &J,
    const Eigen::Ref<const Eigen::VectorXd> &phi_constraints,
    const ModelInstanceIndexToVecMap &q_dict,
    const Eigen::Ref<const Eigen::VectorXd> &v_star,
    const QuasistaticSimParameters &params) {
  if (params.gradient_mode == GradientMode::kNone) {
    return;
  }

  if (params.gradient_mode == GradientMode::kBOnly) {
    Eigen::MatrixXd H(Q);
    for (int i = 0; i < J.rows(); i++) {
      double d = phi_constraints[i] / params.h + J.row(i) * v_star;
      H += J.row(i).transpose() * J.row(i) / d / d / params.log_barrier_weight;
    }
    MatrixXd Dv_nextDb(n_v_, n_v_);
    Dv_nextDb.setIdentity();
    Dv_nextDb *= -1;
    H.llt().solveInPlace(Dv_nextDb); // H is positive definite.

    Dq_nextDqa_cmd_ = CalcDfDu(Dv_nextDb, params.h, q_dict);
    Dq_nextDq_ = MatrixXd::Zero(n_q_, n_q_);
    return;
  }

  throw std::runtime_error("Invalid gradient_mode.");
}

void QuasistaticSimulator::StepForwardLogPyramid(
    const Eigen::Ref<const Eigen::MatrixXd> &Q,
    const Eigen::Ref<const Eigen::VectorXd> &tau_h,
    const Eigen::Ref<const Eigen::MatrixXd> &J,
    const Eigen::Ref<const Eigen::VectorXd> &phi_constraints,
    const QuasistaticSimParameters &params,
    ModelInstanceIndexToVecMap *q_dict_ptr, Eigen::VectorXd *v_star_ptr) {
  auto &q_dict = *q_dict_ptr;
  VectorXd &v_star = *v_star_ptr;
  const auto n_f = J.rows();
  const auto h = params.h;

  drake::solvers::MathematicalProgram prog;
  auto v = prog.NewContinuousVariables(n_v_, "v");
  auto s = prog.NewContinuousVariables(n_f, "s");

  prog.AddQuadraticCost(Q, -tau_h, v, true);
  prog.AddLinearCost(-VectorXd::Constant(n_f, 1 / params.log_barrier_weight), 0,
                     s);

  for (int i = 0; i < n_f; i++) {
    MatrixXd A = MatrixXd::Zero(3, n_v_ + 1);
    A.row(0).head(n_v_) = J.row(i);
    A(2, n_v_) = 1;

    Vector3d b(phi_constraints[i] / h, 1, 0);

    drake::solvers::VectorXDecisionVariable v_s_i(n_v_ + 1);
    v_s_i.head(n_v_) = v;
    v_s_i[n_v_] = s[i];
    prog.AddExponentialConeConstraint(A.sparseView(), b, v_s_i);
  }

  solver_msk_->Solve(prog, {}, {}, &mp_result_);
  if (!mp_result_.is_success()) {
    throw std::runtime_error(
        "Quasistatic dynamics Log Pyramid cannot be solved.");
  }

  v_star = mp_result_.GetSolution(v);

  // Update q_dict.
  UpdateQdictFromV(v_star, params, &q_dict);

  // Update context_plant_ using the new q_dict.
  UpdateMbpPositions(q_dict);
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetVdictFromVec(
    const Eigen::Ref<const Eigen::VectorXd> &v) const {
  DRAKE_THROW_UNLESS(v.size() == n_v_);
  std::unordered_map<ModelInstanceIndex, VectorXd> v_dict;

  for (const auto &model : models_all_) {
    const auto &idx_v = velocity_indices_.at(model);
    auto v_model = VectorXd(idx_v.size());

    for (int i = 0; i < idx_v.size(); i++) {
      v_model[i] = v[idx_v[i]];
    }
    v_dict[model] = v_model;
  }
  return v_dict;
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetQDictFromVec(
    const Eigen::Ref<const Eigen::VectorXd> &q) const {
  DRAKE_THROW_UNLESS(q.size() == n_q_);
  ModelInstanceIndexToVecMap q_dict;

  for (const auto &model : models_all_) {
    const auto &idx_q = position_indices_.at(model);
    auto q_model = VectorXd(idx_q.size());

    for (int i = 0; i < idx_q.size(); i++) {
      q_model[i] = q[idx_q[i]];
    }
    q_dict[model] = q_model;
  }
  return q_dict;
}

Eigen::VectorXd QuasistaticSimulator::GetQVecFromDict(
    const ModelInstanceIndexToVecMap &q_dict) const {
  VectorXd q(n_q_);
  for (const auto &model : models_all_) {
    const auto &idx_q = position_indices_.at(model);
    const auto &q_model = q_dict.at(model);
    for (int i = 0; i < idx_q.size(); i++) {
      q[idx_q[i]] = q_model[i];
    }
  }
  return q;
}

Eigen::VectorXd QuasistaticSimulator::GetQaCmdVecFromDict(
    const ModelInstanceIndexToVecMap &q_a_cmd_dict) const {
  int i_start = 0;
  VectorXd q_a_cmd(n_v_a_);
  for (const auto &model : models_actuated_) {
    auto n_v_i = plant_->num_velocities(model);
    q_a_cmd.segment(i_start, n_v_i) = q_a_cmd_dict.at(model);
    i_start += n_v_i;
  }

  return q_a_cmd;
}

ModelInstanceIndexToVecMap QuasistaticSimulator::GetQaCmdDictFromVec(
    const Eigen::Ref<const Eigen::VectorXd> &q_a_cmd) const {
  ModelInstanceIndexToVecMap q_a_cmd_dict;
  int i_start = 0;
  for (const auto &model : models_actuated_) {
    auto n_v_i = plant_->num_velocities(model);
    q_a_cmd_dict[model] = q_a_cmd.segment(i_start, n_v_i);
    i_start += n_v_i;
  }

  return q_a_cmd_dict;
}

void QuasistaticSimulator::Step(
    const ModelInstanceIndexToVecMap &q_a_cmd_dict,
    const ModelInstanceIndexToVecMap &tau_ext_dict) {
  Step(q_a_cmd_dict, tau_ext_dict, sim_params_);
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDu(
    const Eigen::Ref<const Eigen::MatrixXd> &Dv_nextDb, const double h,
    const ModelInstanceIndexToVecMap &q_dict) const {
  MatrixXd DbDqa_cmd = MatrixXd::Zero(n_v_, n_v_a_);
  int j_start = 0;
  for (const auto &model : models_actuated_) {
    const auto &idx_v = velocity_indices_.at(model);
    const int n_v_i = idx_v.size();
    const auto &Kq_i = robot_stiffness_.at(model);

    for (int k = 0; k < n_v_i; k++) {
      int i = idx_v[k];
      int j = j_start + k;
      DbDqa_cmd(i, j) = -h * Kq_i[k];
    }

    j_start += n_v_i;
  }

  const MatrixXd Dv_nextDqa_cmd = Dv_nextDb * DbDqa_cmd;

  // 2D systems.
  if (n_v_ == n_q_) {
    return h * Dv_nextDqa_cmd;
  }

  // 3D systems.
  MatrixXd Dq_dot_nextDqa_cmd(n_q_, n_v_a_);
  for (const auto &model : models_all_) {
    const auto &idx_v_model = velocity_indices_.at(model);
    const auto &idx_q_model = position_indices_.at(model);

    if (is_3d_floating_.at(model)) {
      // If q contains a quaternion.
      const Eigen::Vector4d &Q_WB = q_dict.at(model).head(4);
      const Eigen::Matrix<double, 4, 3> E = CalcE(Q_WB);

      MatrixXd Dv_nextDqa_cmd_model_rot(3, n_v_a_);
      for (int i = 0; i < 3; i++) {
        Dv_nextDqa_cmd_model_rot.row(i) = Dv_nextDqa_cmd.row(idx_v_model[i]);
      }

      MatrixXd E_X_Dv_nextDqa_cmd_model_rot = E * Dv_nextDqa_cmd_model_rot;
      for (int i = 0; i < 7; i++) {
        const int i_q = idx_q_model[i];
        if (i < 4) {
          // Rotation.
          Dq_dot_nextDqa_cmd.row(i_q) = E_X_Dv_nextDqa_cmd_model_rot.row(i);
        } else {
          // Translation.
          const int i_v = idx_v_model[i - 1];
          Dq_dot_nextDqa_cmd.row(i_q) = Dv_nextDqa_cmd.row(i_v);
        }
      }
    } else {
      for (int i = 0; i < idx_v_model.size(); i++) {
        const int i_q = idx_q_model[i];
        const int i_v = idx_v_model[i];
        Dq_dot_nextDqa_cmd.row(i_q) = Dv_nextDqa_cmd.row(i_v);
      }
    }
  }
  return h * Dq_dot_nextDqa_cmd;
}

/*
 * Arranges DGactiveDq[i] into a (n_lambda_active, n_v) matrix.
 */
std::vector<MatrixXd>
CalcDGactiveDq(const Eigen::Ref<const MatrixX<AutoDiffXd>> &J_ad,
               const std::vector<int> &lambda_star_active_indices) {
  const auto n_v = J_ad.cols();
  const auto n_q = J_ad(0, 0).derivatives().size();
  const auto n_lambda_active = lambda_star_active_indices.size();

  vector<MatrixXd> DGactiveDq;
  for (int i_q = 0; i_q < n_q; i_q++) {
    DGactiveDq.emplace_back(n_lambda_active, n_v);
    for (int i = 0; i < n_lambda_active; i++) {
      for (int j = 0; j < n_v; j++) {
        DGactiveDq[i_q](i, j) =
            J_ad(lambda_star_active_indices[i], j).derivatives()[i_q];
      }
    }
  }
  return DGactiveDq;
}

Eigen::MatrixXd QuasistaticSimulator::CalcDfDx(
    const Eigen::Ref<const Eigen::MatrixXd> &Dv_nextDb,
    const Eigen::Ref<const Eigen::MatrixXd> &Dv_nextDe,
    const ModelInstanceIndexToVecMap &q_dict, const double h,
    const Eigen::Ref<const Eigen::MatrixXd> &Jn, const int n_d) const {
  const auto n_c = Jn.rows();
  const auto n_f = n_c * n_d;

  /*----------------------------------------------------------------*/
  // TODO: consider only active constraints.
  MatrixXd Dphi_constraints_Dq(n_f, n_v_);
  int i_start = 0;
  for (int i_c = 0; i_c < n_c; i_c++) {
    for (int i = i_start; i < i_start + n_d; i++) {
      Dphi_constraints_Dq.row(i) = Jn.row(i_c);
    }
    i_start += n_d;
  }

  /*----------------------------------------------------------------*/
  MatrixXd DbDq = MatrixXd::Zero(n_v_, n_v_);
  int j_start = 0;
  for (const auto &model : models_actuated_) {
    const auto &idx_v = velocity_indices_.at(model);
    const int n_v_i = idx_v.size();
    const auto &Kq_i = robot_stiffness_.at(model);

    for (int k = 0; k < n_v_i; k++) {
      int i = idx_v[k];
      int j = j_start + k;
      DbDq(i, i) = h * Kq_i[k];
    }

    j_start += n_v_i;
  }

  /*----------------------------------------------------------------*/
  MatrixXd Dv_nextDq = Dv_nextDb * DbDq + Dv_nextDe * Dphi_constraints_Dq / h;

  /*----------------------------------------------------------------*/
  // Compute Dv_nextDvecG from the KKT conditions of the QP.
  const auto &[Dv_nextDvecG_active, lambda_star_active_indices] =
      dqp_->get_DzDvecG_active();

  if (not lambda_star_active_indices.empty()) {
    // This is skipped if there is no contact.
    // Compute DvecGDq using Autodiff through MBP.
    const auto q = GetQVecFromDict(q_dict);
    const auto q_ad = InitializeAutoDiff(q);
    UpdateMbpAdPositions(q_ad);
    const auto sdps = CalcSignedDistancePairsFromCollisionPairs();
    MatrixX<AutoDiffXd> Jn_ad, J_ad;
    VectorX<AutoDiffXd> phi_ad, phi_constraints_ad;
    CalcJacobianAndPhi<AutoDiffXd>(plant_ad_, sg_ad_, context_plant_ad_, sdps,
                                   n_d, &phi_ad, &phi_constraints_ad, &Jn_ad,
                                   &J_ad);

    const auto DGactiveDq = CalcDGactiveDq(J_ad, lambda_star_active_indices);

    const auto n_lambda_active = lambda_star_active_indices.size();
    for (int i_v = 0; i_v < n_v_; i_v++) {
      MatrixXd Dv_next_i_vDGactive(n_lambda_active, n_q_);
      for (int i = 0; i < n_lambda_active; i++) {
        for (int j = 0; j < n_q_; j++) {
          Dv_next_i_vDGactive(i, j) =
              Dv_nextDvecG_active(i_v, n_lambda_active * j + i);
        }
      }
      for (int i_q = 0; i_q < n_q_; i_q++) {
        Dv_nextDq(i_v, i_q) +=
            (Dv_next_i_vDGactive.array() * DGactiveDq[i_q].array()).sum();
      }
    }
  }

  return MatrixXd::Identity(n_v_, n_v_) + h * Dv_nextDq;
}

void QuasistaticSimulator::GetGeneralizedForceFromExternalSpatialForce(
    const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>
        &easf,
    ModelInstanceIndexToVecMap *tau_ext) const {
  // TODO: actually process externally applied spatial force.
  for (const auto &model : models_actuated_) {
    (*tau_ext)[model] = Eigen::VectorXd::Zero(plant_->num_velocities(model));
  }
}

void QuasistaticSimulator::CalcGravityForUnactuatedModels(
    ModelInstanceIndexToVecMap *tau_ext) const {
  const auto gravity_all =
      plant_->CalcGravityGeneralizedForces(*context_plant_);

  for (const auto &model : models_unactuated_) {
    const auto &indices = velocity_indices_.at(model);
    const int n_v_i = indices.size();
    (*tau_ext)[model] = VectorXd(n_v_i);
    for (int i = 0; i < n_v_i; i++) {
      (*tau_ext)[model][i] = gravity_all[indices[i]];
    }
  }
}

ModelInstanceIndexToVecMap QuasistaticSimulator::CalcTauExt(
    const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>
        &easf_list) const {
  ModelInstanceIndexToVecMap tau_ext;
  GetGeneralizedForceFromExternalSpatialForce(easf_list, &tau_ext);
  CalcGravityForUnactuatedModels(&tau_ext);
  return tau_ext;
}

ModelInstanceNameToIndexMap
QuasistaticSimulator::GetModelInstanceNameToIndexMap() const {
  ModelInstanceNameToIndexMap name_to_index_map;
  for (const auto &model : models_all_) {
    name_to_index_map[plant_->GetModelInstanceName(model)] = model;
  }
  return name_to_index_map;
}

Eigen::Matrix<double, 4, 3>
QuasistaticSimulator::CalcE(const Eigen::Ref<const Eigen::Vector4d> &Q) {
  Eigen::Matrix<double, 4, 3> E;
  E.row(0) << -Q[1], -Q[2], -Q[3];
  E.row(1) << Q[0], Q[3], -Q[2];
  E.row(2) << -Q[3], Q[0], Q[1];
  E.row(3) << Q[2], -Q[1], Q[0];
  E *= 0.5;
  return E;
}

std::vector<drake::geometry::SignedDistancePair<drake::AutoDiffXd>>
QuasistaticSimulator::CalcSignedDistancePairsFromCollisionPairs() const {
  std::vector<drake::geometry::SignedDistancePair<drake::AutoDiffXd>> sdps_ad;
  for (const auto &collision_pair : collision_pairs_) {
    sdps_ad.push_back(query_object_ad_->ComputeSignedDistancePairClosestPoints(
        collision_pair.first, collision_pair.second));
  }
  return sdps_ad;
}

ModelInstanceIndexToMatrixMap
QuasistaticSimulator::CalcScaledMassMatrix(double h,
                                           double unactuated_mass_scale) const {
  MatrixXd M(n_v_, n_v_);
  plant_->CalcMassMatrix(*context_plant_, &M);

  ModelInstanceIndexToMatrixMap M_u_dict;
  for (const auto &model : models_unactuated_) {
    const auto &idx_v_model = velocity_indices_.at(model);
    const auto n_v_i = idx_v_model.size();
    MatrixXd M_u(n_v_i, n_v_i);
    for (int i = 0; i < n_v_i; i++) {
      for (int j = 0; j < n_v_i; j++) {
        M_u(i, j) = M(idx_v_model[i], idx_v_model[j]);
      }
    }
    M_u_dict[model] = M_u;
  }

  if (unactuated_mass_scale == 0 or std::isnan(unactuated_mass_scale)) {
    return M_u_dict;
  }

  std::unordered_map<drake::multibody::ModelInstanceIndex, double>
      max_eigen_value_M_u;
  for (const auto &model : models_unactuated_) {
    max_eigen_value_M_u[model] = M_u_dict.at(model).diagonal().maxCoeff();
  }

  const double min_K_a_h2 = min_K_a_ * h * h;

  for (auto &[model, M_u] : M_u_dict) {
    auto epsilon =
        min_K_a_h2 / max_eigen_value_M_u[model] / unactuated_mass_scale;
    M_u *= epsilon;
  }

  return M_u_dict;
}

void QuasistaticSimulator::UpdateQdictFromV(
    const Eigen::Ref<const Eigen::VectorXd> &v_star,
    const QuasistaticSimParameters &params,
    ModelInstanceIndexToVecMap *q_dict_ptr) const {
  const auto v_dict = GetVdictFromVec(v_star);
  auto &q_dict = *q_dict_ptr;
  const auto h = params.h;

  std::unordered_map<ModelInstanceIndex, VectorXd> dq_dict;
  for (const auto &model : models_all_) {
    const auto &idx_v = velocity_indices_.at(model);
    const auto n_q_i = plant_->num_positions(model);

    if (is_3d_floating_.at(model)) {
      // Positions of the model contains a quaternion. Conversion from
      // angular velocities to quaternion dot is necessary.
      const auto &q_u = q_dict[model];
      const Eigen::Vector4d Q(q_u.head(4));

      VectorXd dq_u(7);
      const auto &v_u = v_dict.at(model);
      dq_u.head(4) = CalcE(Q) * v_u.head(3) * h;
      dq_u.tail(3) = v_u.tail(3) * h;

      dq_dict[model] = dq_u;
    } else {
      dq_dict[model] = v_dict.at(model) * h;
    }
  }

  // TODO: not updating unactuated object poses can lead to penetration at
  //  the next time step. A better solution is needed.
  if (params.unactuated_mass_scale > 0 or
      std::isnan(params.unactuated_mass_scale)) {
    for (const auto &model : models_all_) {
      auto &q_model = q_dict[model];
      q_model += dq_dict[model];

      if (is_3d_floating_.at(model)) {
        // Normalize quaternion.
        q_model.head(4).normalize();
      }
    }
  } else {
    // un-actuated objects remain fixed.
    for (const auto &model : models_actuated_) {
      auto &q_model = q_dict[model];
      q_model += dq_dict[model];
    }
  }
}
