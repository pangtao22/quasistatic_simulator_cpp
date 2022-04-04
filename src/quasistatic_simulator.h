#pragma once
#include <iostream>

#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/mosek_solver.h"

#include "qp_derivatives.h"
#include "contact_jacobian_calculator.h"


/*
 * Denotes whether the indices are those of a model's configuration vector
 * into the configuration vector of the system, or those of a model's velocity
 * vector into the velocity vector of the system.
 */
enum class ModelIndicesMode { kQ, kV };

class QuasistaticSimulator {
public:
  QuasistaticSimulator(
      const std::string &model_directive_path,
      const std::unordered_map<std::string, Eigen::VectorXd>
          &robot_stiffness_str,
      const std::unordered_map<std::string, std::string> &object_sdf_paths,
      QuasistaticSimParameters sim_params);

  void UpdateMbpPositions(const ModelInstanceIndexToVecMap &q_dict);
  void UpdateMbpPositions(const Eigen::Ref<const Eigen::VectorXd> &q);
  // These methods are naturally const because context will eventually be
  // moved outside this class.
  void UpdateMbpAdPositions(const ModelInstanceIndexToVecAdMap &q_dict) const;
  void
  UpdateMbpAdPositions(const Eigen::Ref<const drake::AutoDiffVecXd> &q) const;

  [[nodiscard]] ModelInstanceIndexToVecMap GetMbpPositions() const;
  [[nodiscard]] Eigen::VectorXd GetMbpPositionsAsVec() const {
    return plant_->GetPositions(*context_plant_);
  }

  Eigen::VectorXd
  GetPositions(drake::multibody::ModelInstanceIndex model) const;

  void Step(const ModelInstanceIndexToVecMap &q_a_cmd_dict,
            const ModelInstanceIndexToVecMap &tau_ext_dict,
            const QuasistaticSimParameters &params);

  void Step(const ModelInstanceIndexToVecMap &q_a_cmd_dict,
            const ModelInstanceIndexToVecMap &tau_ext_dict);

  void GetGeneralizedForceFromExternalSpatialForce(
      const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>
          &easf,
      ModelInstanceIndexToVecMap *tau_ext) const;

  void
  CalcGravityForUnactuatedModels(ModelInstanceIndexToVecMap *tau_ext) const;

  ModelInstanceIndexToVecMap CalcTauExt(
      const std::vector<drake::multibody::ExternallyAppliedSpatialForce<double>>
          &easf_list) const;

  ModelInstanceNameToIndexMap GetModelInstanceNameToIndexMap() const;

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

  [[nodiscard]] const QuasistaticSimParameters &get_sim_params() const {
    return sim_params_;
  }

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

  void update_sim_params(const QuasistaticSimParameters &new_params) {
    sim_params_ = new_params;
  }

  int num_actuated_dofs() const { return n_v_a_; };
  int num_unactuated_dofs() const { return n_v_u_; };

  Eigen::MatrixXd get_Dq_nextDq() const { return Dq_nextDq_; };
  Eigen::MatrixXd get_Dq_nextDqa_cmd() const { return Dq_nextDqa_cmd_; };

  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
  GetVelocityIndices() const {
    return velocity_indices_;
  };

  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
  GetPositionIndices() const {
    return position_indices_;
  };

  ModelInstanceIndexToVecMap
  GetVdictFromVec(const Eigen::Ref<const Eigen::VectorXd> &v) const;

  ModelInstanceIndexToVecMap
  GetQDictFromVec(const Eigen::Ref<const Eigen::VectorXd> &q) const;

  Eigen::VectorXd
  GetQVecFromDict(const ModelInstanceIndexToVecMap &q_dict) const;

  /*
   * QaCmd, sometimes denoted by u, is the concatenation of position vectors
   * for all models in this->models_actuated_, which is sorted in ascending
   * order.
   * They keys of q_a_cmd_dict does not need to be the same as
   * this->models_actuated. It only needs to be a superset of
   * this->models_actuated. This means that it is possible to pass in a
   * dictionary containing position vectors for all model instances in the
   * system, including potentially position vectors of un-actuated models,
   * and this method will extract the actuated position vectors and
   * concatenate them into a single vector.
   */
  Eigen::VectorXd
  GetQaCmdVecFromDict(const ModelInstanceIndexToVecMap &q_a_cmd_dict) const;

  ModelInstanceIndexToVecMap
  GetQaCmdDictFromVec(const Eigen::Ref<const Eigen::VectorXd> &q_a_cmd) const;

private:
  [[nodiscard]] std::vector<int>
  GetIndicesForModel(drake::multibody::ModelInstanceIndex idx,
                     ModelIndicesMode mode) const;

  void CalcQAndTauH(const ModelInstanceIndexToVecMap &q_dict,
                    const ModelInstanceIndexToVecMap &q_a_cmd_dict,
                    const ModelInstanceIndexToVecMap &tau_ext_dict,
                    const double h, Eigen::MatrixXd *Q_ptr,
                    Eigen::VectorXd *tau_h_ptr,
                    const double unactuated_mass_scale) const;

  Eigen::MatrixXd CalcDfDu(const Eigen::Ref<const Eigen::MatrixXd> &Dv_nextDb,
                           const double h,
                           const ModelInstanceIndexToVecMap &q_dict) const;
  Eigen::MatrixXd CalcDfDx(const Eigen::Ref<const Eigen::MatrixXd> &Dv_nextDb,
                           const Eigen::Ref<const Eigen::MatrixXd> &Dv_nextDe,
                           const ModelInstanceIndexToVecMap &q_dict,
                           const double h,
                           const Eigen::Ref<const Eigen::MatrixXd> &Jn,
                           const int n_d) const;
  static Eigen::Matrix<double, 4, 3>
  CalcE(const Eigen::Ref<const Eigen::Vector4d> &Q);
  
  const std::vector<drake::geometry::SignedDistancePair<double>>
  CalcCollisionPairs(double contact_detection_tolerance) const;

  std::vector<drake::geometry::SignedDistancePair<drake::AutoDiffXd>>
  CalcSignedDistancePairsFromCollisionPairs() const;

  ModelInstanceIndexToMatrixMap
  CalcScaledMassMatrix(double h, double unactuated_mass_scale) const;

  void UpdateQdictFromV(const Eigen::Ref<const Eigen::VectorXd> &v_star,
                        const QuasistaticSimParameters &params,
                        ModelInstanceIndexToVecMap *q_dict_ptr) const;

  void CalcPyramidMatrices(const ModelInstanceIndexToVecMap &q_dict,
                           const ModelInstanceIndexToVecMap &q_a_cmd_dict,
                           const ModelInstanceIndexToVecMap &tau_ext_dict,
                           const QuasistaticSimParameters &params,
                           Eigen::MatrixXd *Q, Eigen::VectorXd *tau_h,
                           Eigen::MatrixXd *Jn, Eigen::MatrixXd *J,
                           Eigen::VectorXd *phi,
                           Eigen::VectorXd *phi_constraints) const;

  void StepForwardQp(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                     const Eigen::Ref<const Eigen::VectorXd> &tau_h,
                     const Eigen::Ref<const Eigen::MatrixXd> &J,
                     const Eigen::Ref<const Eigen::VectorXd> &phi_constraints,
                     const QuasistaticSimParameters &params,
                     ModelInstanceIndexToVecMap *q_dict_ptr,
                     Eigen::VectorXd *v_star_ptr,
                     Eigen::VectorXd *beta_star_ptr);

  void StepForwardLogPyramid(
      const Eigen::Ref<const Eigen::MatrixXd> &Q,
      const Eigen::Ref<const Eigen::VectorXd> &tau_h,
      const Eigen::Ref<const Eigen::MatrixXd> &J,
      const Eigen::Ref<const Eigen::VectorXd> &phi_constraints,
      const QuasistaticSimParameters &params,
      ModelInstanceIndexToVecMap *q_dict_ptr, Eigen::VectorXd *v_star_ptr);

  void BackwardQp(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                  const Eigen::Ref<const Eigen::VectorXd> &tau_h,
                  const Eigen::Ref<const Eigen::MatrixXd> &Jn,
                  const Eigen::Ref<const Eigen::MatrixXd> &J,
                  const Eigen::Ref<const Eigen::VectorXd> &phi_constraints,
                  const ModelInstanceIndexToVecMap &q_dict,
                  const ModelInstanceIndexToVecMap &q_dict_next,
                  const Eigen::Ref<const Eigen::VectorXd> &v_star,
                  const Eigen::Ref<const Eigen::VectorXd> &beta_star,
                  const QuasistaticSimParameters &params);

  void
  BackwardLogPyramid(const Eigen::Ref<const Eigen::MatrixXd> &Q,
                     const Eigen::Ref<const Eigen::MatrixXd> &J,
                     const Eigen::Ref<const Eigen::VectorXd> &phi_constraints,
                     const ModelInstanceIndexToVecMap &q_dict,
                     const Eigen::Ref<const Eigen::VectorXd> &v_star,
                     const QuasistaticSimParameters &params);

  QuasistaticSimParameters sim_params_;

  // Solvers.
  std::unique_ptr<drake::solvers::GurobiSolver> solver_grb_;
  std::unique_ptr<drake::solvers::MosekSolver> solver_msk_;
  mutable drake::solvers::MathematicalProgramResult mp_result_;
  drake::solvers::SolverOptions solver_options_;

  // QP derivatives. Refer to the python implementation of
  //  QuasistaticSimulator for more details.
  std::unique_ptr<QpDerivativesActive> dqp_;
  Eigen::MatrixXd Dq_nextDq_;
  Eigen::MatrixXd Dq_nextDqa_cmd_;

  // Systems.
  std::unique_ptr<drake::systems::Diagram<double>> diagram_;
  drake::multibody::MultibodyPlant<double> *plant_{nullptr};
  drake::geometry::SceneGraph<double> *sg_{nullptr};

  // AutoDiff Systems.
  std::unique_ptr<drake::systems::Diagram<drake::AutoDiffXd>> diagram_ad_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd> *plant_ad_{nullptr};
  const drake::geometry::SceneGraph<drake::AutoDiffXd> *sg_ad_{nullptr};

  // Contexts.
  std::unique_ptr<drake::systems::Context<double>> context_; // Diagram.
  drake::systems::Context<double> *context_plant_{nullptr};
  drake::systems::Context<double> *context_sg_{nullptr};

  // AutoDiff contexts
  std::unique_ptr<drake::systems::Context<drake::AutoDiffXd>> context_ad_;
  mutable drake::systems::Context<drake::AutoDiffXd> *context_plant_ad_{
      nullptr};
  mutable drake::systems::Context<drake::AutoDiffXd> *context_sg_ad_{nullptr};

  // Internal state (for interfacing with QuasistaticSystem).
  const drake::geometry::QueryObject<double> *query_object_{nullptr};
  mutable std::vector<CollisionPair> collision_pairs_;
  mutable const drake::geometry::QueryObject<drake::AutoDiffXd>
      *query_object_ad_{nullptr};

  drake::multibody::ContactResults<double> contact_results_;

  // MBP introspection.
  int n_v_a_{0}; // number of actuated DOFs.
  int n_v_u_{0}; // number of un-actuated DOFs.
  int n_v_{0};   // total number of velocities.
  int n_q_{0};   // total number of positions.
  std::set<drake::multibody::ModelInstanceIndex> models_actuated_;
  std::set<drake::multibody::ModelInstanceIndex> models_unactuated_;
  std::set<drake::multibody::ModelInstanceIndex> models_all_;
  std::unordered_map<drake::multibody::ModelInstanceIndex, bool>
      is_3d_floating_;
  ModelInstanceIndexToVecMap robot_stiffness_;
  double min_K_a_{0}; // smallest stiffness of all joints.
  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
      velocity_indices_;
  std::unordered_map<drake::multibody::ModelInstanceIndex, std::vector<int>>
      position_indices_;

  std::unique_ptr<ContactJacobianCalculator<double>> cjc_;
  std::unique_ptr<ContactJacobianCalculator<drake::AutoDiffXd>> cjc_ad_;
};
