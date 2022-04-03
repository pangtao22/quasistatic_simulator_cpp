#pragma once
#include <Eigen/Dense>

#include "drake/common/default_scalars.h"
#include "drake/multibody/plant/multibody_plant.h"

#include "quasistatic_sim_params.h"

template <class T> class ContactComputer {
public:
  ContactComputer(
      const drake::systems::Diagram<T> *diagram,
      const std::set<drake::multibody::ModelInstanceIndex> &models_all);
  void CalcContactJacobians(
      const std::vector<drake::geometry::SignedDistancePair<T>> &sdps) const;

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

  void UpdateJacobianRows(const drake::systems::Context<T> *context_plant,
                          const drake::multibody::BodyIndex &body_idx,
                          const drake::VectorX<T> &pC_Body,
                          const drake::VectorX<T> &n_W,
                          const drake::MatrixX<T> &d_W, size_t i_c, size_t n_d,
                          size_t i_f_start,
                          drake::EigenPtr<drake::MatrixX<T>> Jn_ptr,
                          drake::EigenPtr<drake::MatrixX<T>> Jf_ptr) const;

  const drake::multibody::MultibodyPlant<T> *plant_{nullptr};
  const drake::geometry::SceneGraph<T> *sg_{nullptr};

  // MBP.
  const std::set<drake::multibody::ModelInstanceIndex> &models_all_;

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

template class ContactComputer<double>;
template class ContactComputer<drake::AutoDiffXd>;
