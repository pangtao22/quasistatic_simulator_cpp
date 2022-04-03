#pragma once
#include <Eigen/Dense>

#include "drake/common/default_scalars.h"
#include "drake/multibody/plant/multibody_plant.h"

#include "quasistatic_sim_params.h"

template <class T> struct ContactPairInfo {
  // Contact normal pointing to body A from body B.
  drake::Vector3<T> nhat_BA_W;

  // The (3, n_v) contact Jacobian defined in the docs.
  drake::Matrix3X<T> Jc;
  double mu{0}; // coefficient of friction.
};

template <class T> class ContactJacobianCalculator {
public:
  ContactJacobianCalculator(
      const drake::systems::Diagram<T> *diagram,
      std::set<drake::multibody::ModelInstanceIndex> models_all);
  /*
   * Computes contact Jacobians for the list of SignedDistancePairs. This
   * should be the first function that gets called when computing Jacobians
   * for a given state, before computing normal / tangent Jacobians.
   */
  void UpdateContactPairInfo(
      const drake::systems::Context<T> *context_plant,
      const std::vector<drake::geometry::SignedDistancePair<T>> &sdps) const;

  /*
   *  Retrieve coefficient of friction from the cached list of contact info.
   */
  double get_friction_coefficient(int i_c) const {
    return contact_pairs_[i_c].mu;
  }

  void CalcJacobianAndPhi(
      const drake::systems::Context<T> *context_plant,
      const std::vector<drake::geometry::SignedDistancePair<T>> &sdps,
      const int n_d, drake::VectorX<T> *phi_ptr,
      drake::VectorX<T> *phi_constraints_ptr, drake::MatrixX<T> *Jn_ptr,
      drake::MatrixX<T> *J_ptr) const;

private:
  double GetFrictionCoefficientForSignedDistancePair(
      drake::geometry::GeometryId id_A, drake::geometry::GeometryId id_B) const;

  std::unique_ptr<drake::multibody::ModelInstanceIndex>
  FindModelForBody(drake::multibody::BodyIndex body_idx) const;

  drake::multibody::BodyIndex
  GetMbpBodyFromGeometry(drake::geometry::GeometryId g_id) const;

  /*
   * Each contact Jacobian is the subtraction of the Jacobians of two points
   * on the two bodies in the contact pair. This function computes the
   * contribution from one of the two bodies.
   *
   * Jc_ptr must have shape (3, n_v).
   */
  drake::Matrix3X<T> CalcContactJaocibanFromPoint(
      const drake::systems::Context<T> *context_plant,
      const drake::multibody::BodyIndex &body_idx,
      const drake::VectorX<T> &pC_Body) const;

  const drake::multibody::MultibodyPlant<T> *plant_{nullptr};
  const drake::geometry::SceneGraph<T> *sg_{nullptr};

  // MBP.
  const std::set<drake::multibody::ModelInstanceIndex> models_all_;

  // friction_coefficients[g_idA][g_idB] and friction_coefficients[g_idB][g_idA]
  //  gives the coefficient of friction between contact geometries g_idA
  //  and g_idB.
  std::unordered_map<drake::geometry::GeometryId,
                     std::unordered_map<drake::geometry::GeometryId, double>>
      friction_coefficients_;

  // Mutable storage for the current contact.
  mutable std::vector<ContactPairInfo<T>> contact_pairs_;
};

template class ContactJacobianCalculator<double>;
template class ContactJacobianCalculator<drake::AutoDiffXd>;
