#include "contact_computer.h"

using drake::AutoDiffXd;
using drake::Matrix3X;
using drake::MatrixX;
using drake::Vector3;
using drake::Vector4;
using drake::VectorX;
using drake::multibody::ModelInstanceIndex;
using std::vector;

template <class T>
ContactComputer<T>::ContactComputer(
    const drake::systems::Diagram<T> *diagram,
    const std::set<drake::multibody::ModelInstanceIndex> &models_all)
    : models_all_(models_all) {
  plant_ = dynamic_cast<const drake::multibody::MultibodyPlant<T> *>(
      &diagram->GetSubsystemByName(kMultiBodyPlantName));
  sg_ = dynamic_cast<const drake::geometry::SceneGraph<T> *>(
      &diagram->GetSubsystemByName(kSceneGraphName));

  DRAKE_THROW_UNLESS(plant_ != nullptr);
  DRAKE_THROW_UNLESS(sg_ != nullptr);

  // body_indices.
  for (const auto &model : models_all_) {
    const auto body_indices = plant_->GetBodyIndices(model);
    bodies_indices_[model].insert(body_indices.begin(), body_indices.end());
  }

  // friction coefficients.
  const auto &inspector = sg_->model_inspector();
  const auto cc = inspector.GetCollisionCandidates();
  for (const auto &[g_idA, g_idB] : cc) {
    const double mu = GetFrictionCoefficientForSignedDistancePair(g_idA, g_idB);
    friction_coefficients_[g_idA][g_idB] = mu;
    friction_coefficients_[g_idB][g_idA] = mu;
  }
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

template <class T>
void ContactComputer<T>::CalcContactJacobians(
    const std::vector<drake::geometry::SignedDistancePair<T>> &sdps) const {
  const auto n_c = sdps.size();
  const int n_v = plant_->num_velocities();
  const auto &inspector = sg_->model_inspector();

  for (int i_c = 0; i_c < n_c; i_c++) {
    const auto &sdp = sdps[i_c];
  }
}

/*
 * Returns nullptr if body_idx is not in any of the values of bodies_indices_;
 * Otherwise returns the model instance to which body_idx belongs.
 */
template <class T>
std::unique_ptr<ModelInstanceIndex> ContactComputer<T>::FindModelForBody(
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
double ContactComputer<T>::GetFrictionCoefficientForSignedDistancePair(
    drake::geometry::GeometryId id_A, drake::geometry::GeometryId id_B) const {
  const auto &inspector = sg_->model_inspector();
  const auto props_A = inspector.GetProximityProperties(id_A);
  const auto props_B = inspector.GetProximityProperties(id_B);
  const auto &geometryA_friction =
      props_A->template GetProperty<drake::multibody::CoulombFriction<double>>(
          "material", "coulomb_friction");
  const auto &geometryB_friction =
      props_B->template GetProperty<drake::multibody::CoulombFriction<double>>(
          "material", "coulomb_friction");
  auto cf = drake::multibody::CalcContactFrictionFromSurfaceProperties(
      geometryA_friction, geometryB_friction);
  return cf.static_friction();
}

template <class T>
drake::multibody::BodyIndex ContactComputer<T>::GetMbpBodyFromGeometry(
    drake::geometry::GeometryId g_id) const {
  const auto &inspector = sg_->model_inspector();
  return plant_->GetBodyFromFrameId(inspector.GetFrameId(g_id))->index();
}

template <class T>
void ContactComputer<T>::UpdateJacobianRows(
    const drake::systems::Context<T> *context_plant,
    const drake::multibody::BodyIndex &body_idx,
    const drake::VectorX<T> &pC_Body, const drake::VectorX<T> &n_W,
    const drake::MatrixX<T> &d_W, size_t i_c, size_t n_d, size_t i_f_start,
    drake::EigenPtr<drake::MatrixX<T>> Jn_ptr,
    drake::EigenPtr<drake::MatrixX<T>> Jf_ptr) const {
  drake::Matrix3X<T> Ji(3, plant_->num_velocities());
  const auto &frameB = plant_->get_body(body_idx).body_frame();
  plant_->CalcJacobianTranslationalVelocity(
      *context_plant, drake::multibody::JacobianWrtVariable::kV, frameB,
      pC_Body, plant_->world_frame(), plant_->world_frame(), &Ji);

  (*Jn_ptr).row(i_c) += n_W.transpose() * Ji;
  for (size_t i = 0; i < n_d; i++) {
    (*Jf_ptr).row(i + i_f_start) += d_W.col(i).transpose() * Ji;
  }
}

template <class T>
void ContactComputer<T>::CalcJacobianAndPhi(
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
  const int n_v = plant_->num_velocities();
  const auto n_f = n_d * n_c;

  phi.resize(n_c);
  Jn.resize(n_c, n_v);
  Jn.setZero();

  VectorX<T> U(n_c);
  MatrixX<T> Jf(n_f, n_v);
  Jf.setZero();
  const auto &inspector = sg_->model_inspector();

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
      UpdateJacobianRows(context_plant, bodyA_idx, p_ACa_A, n_A_W, d_A_W, i_c,
                         n_d, i_f_start, &Jn, &Jf);
      UpdateJacobianRows(context_plant, bodyB_idx, p_BCb_B, n_B_W, d_B_W, i_c,
                         n_d, i_f_start, &Jn, &Jf);
    } else if (model_A_ptr) {
      const auto n_A_W = sdp.nhat_BA_W;
      const auto d_A_W = CalcTangentVectors<T>(n_A_W, n_d);
      UpdateJacobianRows(context_plant, bodyA_idx, p_ACa_A, n_A_W, d_A_W, i_c,
                         n_d, i_f_start, &Jn, &Jf);
    } else if (model_B_ptr) {
      const auto n_B_W = -sdp.nhat_BA_W;
      const auto d_B_W = CalcTangentVectors<T>(n_B_W, n_d);
      UpdateJacobianRows(context_plant, bodyB_idx, p_BCb_B, n_B_W, d_B_W, i_c,
                         n_d, i_f_start, &Jn, &Jf);
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
