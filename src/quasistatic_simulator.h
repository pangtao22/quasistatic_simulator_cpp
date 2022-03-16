#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <unordered_map>

#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/mosek_solver.h"

#include "qp_derivatives.h"

using ModelInstanceIndexToVecMap =
    std::unordered_map<drake::multibody::ModelInstanceIndex, Eigen::VectorXd>;
using ModelInstanceIndexToMatrixMap =
    std::unordered_map<drake::multibody::ModelInstanceIndex, Eigen::MatrixXd>;
using ModelInstanceNameToIndexMap =
    std::unordered_map<std::string, drake::multibody::ModelInstanceIndex>;

/*
 * Gradient computation mode of QuasistaticSimulator.
 * Using an analogy from torch, GradientMode is the mode when "backward()" is
 *  called, after the forward dynamics is done.
 * - kNone: do not compute gradient, just roll out the dynamics.
 * - kBOnly: only computes dfdu, where x_next = f(x, u).
 * - kAB: computes both dfdx and dfdu.
 */
enum class GradientMode { kNone, kBOnly, kAB };

enum class ForwardDynamicsMode {
  kQpMp,
  kQpCvx,
  kSocpMp,
  kLogPyramidMp,
  kLogPyramidCvx,
  kLogIcecreamMp,
  kLogIcecreamCvx
};

/*
h: simulation time step in seconds.
gravity: 3-vector indicating the gravity feild in world frame.
 WARNING: it CANNOT be changed after the simulator object is constructed.
 TODO: differentiate gravity from other simulation parameters, which are not
  ignored when calling QuasistaticSimulator.Step(...).
nd_per_contact: int, number of extreme rays per contact point. Only
 useful in QP mode.

contact_detection_tolerance: Signed distance pairs whose distances are
 greater than this value are ignored in the simulator's non-penetration
 constraints. Unit is in meters.

is_quasi_dynamic: bool. If True, dynamics of unactauted objects is
 given by sum(F) = M @ (v_(l+1) - 0). If False, it becomes sum(F) = 0 instead.

 The mass matrix for unactuated objects is always added when the
 unconstrained (log-barrier) version of the problem is solved. Not having a mass
 matrix can sometimes make the unconstrained program unbounded.

mode:
Note that C++ does not support modes using CVX.
                 | Friction Cone | Force Field | Parser |
kQpMp            | Pyramid       | No          | MP     |
kQpCvx           | Pyramid       | No          | CVXPY  |
kSocpMp          | Icecream      | No          | MP     |
kLogPyramidMp    | Pyramid       | Yes         | MP     |
kLogPyramidCvx   | Pyramid       | Yes         | CVXPY  |
kLogIcecreamMp   | Icecream      | Yes         | MP     |
kLogIcecreamCvx  | Icecream      | Yes         | CVXPY  |

log_barrier_weight: float, used only in log-barrier modes.

unactuated_mass_scale:
scales the mass matrix of un-actuated objects by epsilon, so that
(max_M_u_eigen_value * epsilon) * unactuated_mass_scale = min_h_squared_K.
    If 0, the mass matrix is not scaled. Refer to the function that computes
    mass matrix for details.
*------------------------------C++ only-----------------------------------*
gradient_lstsq_tolerance: float
   When solving for A during dynamics gradient computation, i.e.
   A * A_inv = I_n, --------(*)
   the relative error is defined as
   (A_sol * A_inv - I_n) / n,
   where A_sol is the least squares solution to (*), or the pseudo-inverse
   of A_inv.
   A warning is printed when the relative error is greater than this number.
*/
// TODO: the inputs to QuasistaticSimulator's constructor should be
//  collected into a "QuasistaticPlantParameters" structure, which
//  cannot be changed after the constructor call. "gravity" belongs there.
struct QuasistaticSimParameters {
  double h{NAN};
  Eigen::Vector3d gravity;
  size_t nd_per_contact;
  double contact_detection_tolerance;
  bool is_quasi_dynamic;
  ForwardDynamicsMode forward_mode{ForwardDynamicsMode::kQpMp};
  GradientMode gradient_mode{GradientMode::kNone};
  double log_barrier_weight{NAN};
  double unactuated_mass_scale{NAN};
  // -------------------------- CPP only --------------------------
  double gradient_lstsq_tolerance{2e-2};
};

/*
 * Denotes whether the indices are those of a configuration vector of a model
 * into the configuration vector of the system, or those of a velocity vector
 * of a model into the velocity vector of the system.
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

  [[nodiscard]] double GetFrictionCoefficientForSignedDistancePair(
      drake::geometry::GeometryId id_A, drake::geometry::GeometryId id_B) const;

  [[nodiscard]] drake::multibody::BodyIndex
  GetMbpBodyFromGeometry(drake::geometry::GeometryId g_id) const;

  [[nodiscard]] std::unique_ptr<drake::multibody::ModelInstanceIndex>
      FindModelForBody(drake::multibody::BodyIndex) const;

  void CalcJacobianAndPhi(const double contact_detection_tol,
                          Eigen::VectorXd *phi_ptr,
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
                           const double h,
                           const Eigen::Ref<const Eigen::MatrixXd> &Jn,
                           const int n_d) const;
  static Eigen::Matrix<double, 4, 3>
  GetE(const Eigen::Ref<const Eigen::Vector4d> &Q);

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
                  const Eigen::Ref<const Eigen::VectorXd> &v_star,
                  const Eigen::Ref<const Eigen::VectorXd> &beta_star,
                  const QuasistaticSimParameters &params);

  void BackwardLogPyramid(const Eigen::Ref<const Eigen::MatrixXd> &Q,
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

  // Contexts.
  std::unique_ptr<drake::systems::Context<double>> context_; // Diagram.
  drake::systems::Context<double> *context_plant_{nullptr};
  drake::systems::Context<double> *context_sg_{nullptr};

  // Internal state (for interfacing with QuasistaticSystem).
  const drake::geometry::QueryObject<double> *query_object_{nullptr};
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
