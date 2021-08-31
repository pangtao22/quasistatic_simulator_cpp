#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <unordered_map>

#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program_result.h"

#include "qp_derivatives.h"

using ModelInstanceToVecMap =
    std::unordered_map<drake::multibody::ModelInstanceIndex, Eigen::VectorXd>;

struct QuasistaticSimParameters {
  Eigen::Vector3d gravity;
  size_t nd_per_contact;
  double contact_detection_tolerance;
  bool is_quasi_dynamic;
  bool requires_grad;
  double gradient_lstsq_tolerance{1e-8};
  bool gradient_from_active_constraints{false};
};

class QuasistaticSimulator {
public:
  QuasistaticSimulator(
      const std::string &model_directive_path,
      const std::unordered_map<std::string, Eigen::VectorXd>
          &robot_stiffness_str,
      const std::unordered_map<std::string, std::string> &object_sdf_paths,
      QuasistaticSimParameters sim_params);

  void UpdateMbpPositions(const ModelInstanceToVecMap &q_dict);

  [[nodiscard]] ModelInstanceToVecMap GetMbpPositions() const;

  Eigen::VectorXd
  GetPositions(drake::multibody::ModelInstanceIndex model) const;

  void Step(const ModelInstanceToVecMap &q_a_cmd_dict,
            const ModelInstanceToVecMap &tau_ext_dict,
            const double h,
            const double contact_detection_tolerance,
            const bool requires_grad,
            const bool grad_from_active_constraints);

  void Step(const ModelInstanceToVecMap &q_a_cmd_dict,
            const ModelInstanceToVecMap &tau_ext_dict, const double h);

  void GetGeneralizedForceFromExternalSpatialForce(
      const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>
          &easf,
      ModelInstanceToVecMap *tau_ext) const;

  void CalcGravityForUnactautedModels(ModelInstanceToVecMap *tau_ext) const;

  ModelInstanceToVecMap CalcTauExt(
      const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>
          &easf_list) const;

  [[nodiscard]] const std::set<drake::multibody::ModelInstanceIndex> &
  get_all_models() const {
    return models_all_;
  };

  [[nodiscard]] const std::set<drake::multibody::ModelInstanceIndex> &
  get_actuated_models() const {
    return models_actuated_;
  };

  [[nodiscard]] const std::set<drake::multibody::ModelInstanceIndex> &
  get_unactuated_models() const {
    return models_unactuated_;
  };

  const drake::geometry::QueryObject<double> &get_query_object() const {
    return *query_object_;
  };

  const drake::multibody::MultibodyPlant<double> &get_plant() const {
    return *plant_;
  }

  const drake::geometry::SceneGraph<double> &get_scene_graph() const {
    return *sg_;
  }

  const drake::multibody::ContactResults<double> &get_contact_results() const {
    // TODO: return non-empty contact results.
    return contact_results_;
  }

  int num_actuated_dofs() const { return n_v_a_;};
  int num_unactuated_dofs() const { return n_v_u_;};

  Eigen::MatrixXd get_Dq_nextDq() const {return Dq_nextDq_;};
  Eigen::MatrixXd get_Dq_nextDqa_cmd() const {return Dq_nextDqa_cmd_;};

  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
    get_velocity_indices() const { return velocity_indices_; };

private:
  [[nodiscard]] std::vector<int>
  GetVelocityIndicesForModel(drake::multibody::ModelInstanceIndex idx) const;

  [[nodiscard]] double GetFrictionCoefficientForSignedDistancePair(
      drake::geometry::GeometryId id_A, drake::geometry::GeometryId id_B) const;

  [[nodiscard]] drake::multibody::BodyIndex
  GetMbpBodyFromGeometry(drake::geometry::GeometryId g_id) const;

  [[nodiscard]] std::unique_ptr<drake::multibody::ModelInstanceIndex>
      FindModelForBody(drake::multibody::BodyIndex) const;

  void CalcJacobianAndPhi(const double contact_detection_tol, int *n_c_ptr,
                          int *n_f_ptr, Eigen::VectorXd *phi_ptr,
                          Eigen::VectorXd *phi_constraints_ptr,
                          Eigen::MatrixXd *Jn_ptr,
                          Eigen::MatrixXd *J_ptr) const;

  void UpdateJacobianRows(const drake::multibody::BodyIndex &body_idx,
                          const Eigen::Ref<const Eigen::Vector3d> &pC_Body,
                          const Eigen::Ref<const Eigen::Vector3d> &n_W,
                          const Eigen::Ref<const Eigen::Matrix3Xd> &d_W,
                          int i_c, int n_d, int i_f_start,
                          drake::EigenPtr<Eigen::MatrixXd> Jn_ptr,
                          drake::EigenPtr<Eigen::MatrixXd> Jf_ptr) const;

  void FormQAndTauH(const ModelInstanceToVecMap &q_dict,
                    const ModelInstanceToVecMap &q_a_cmd_dict,
                    const ModelInstanceToVecMap &tau_ext_dict, const double h,
                    Eigen::MatrixXd *Q_ptr, Eigen::VectorXd *tau_h_ptr) const;

  const QuasistaticSimParameters sim_params_;

  // QP solver.
  std::unique_ptr<drake::solvers::GurobiSolver> solver_;
  mutable drake::solvers::MathematicalProgramResult mp_result_;

  // QP derivatives. Refer to the python implementation of
  //  QuasistaticSimulator for more details.
  std::unique_ptr<QpDerivatives> dqp_;
  std::unique_ptr<QpDerivativesActive> dqp_active_;
  Eigen::MatrixXd Dq_nextDq_;
  Eigen::MatrixXd Dq_nextDqa_cmd_;

  // Systems.
  std::unique_ptr<drake::systems::Diagram<double>> diagram_;
  drake::multibody::MultibodyPlant<double> *plant_{nullptr};
  drake::geometry::SceneGraph<double> *sg_{nullptr};

  // Contexts.
  std::unique_ptr<drake::systems::Context<double>> context_; // Diagram.
  drake::systems::Context<double> *context_plant_{nullptr};
  drake::systems::Context<double> *context_sg_{nullptr};

  // Internal state (for interfacing with QuasistaticSystem).
  const drake::geometry::QueryObject<double> *query_object_{nullptr};
  drake::multibody::ContactResults<double> contact_results_;

  // MBP introspection.
  int n_v_a_{0};  // number of actuated DOFs.
  int n_v_u_{0};  // number of un-actuated DOFs.
  std::set<drake::multibody::ModelInstanceIndex> models_actuated_;
  std::set<drake::multibody::ModelInstanceIndex> models_unactuated_;
  std::set<drake::multibody::ModelInstanceIndex> models_all_;
  ModelInstanceToVecMap robot_stiffness_;
  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
      velocity_indices_;
  std::unordered_map<drake::multibody::ModelInstanceIndex,
                     std::unordered_set<drake::multibody::BodyIndex>>
      bodies_indices_;

  // friction_coefficients[g_idA][g_idB] and friction_coefficients[g_idB][g_idA]
  //  gives the coefficient of friction between contact geometries g_idA
  //  and g_idB.
  std::unordered_map<drake::geometry::GeometryId,
                     std::unordered_map<drake::geometry::GeometryId, double>>
      friction_coefficients_;
};
