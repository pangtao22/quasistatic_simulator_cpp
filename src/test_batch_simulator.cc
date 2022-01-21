#include <iostream>
#include <random>
#include <thread>

#include <gtest/gtest.h>

#include "batch_quasistatic_simulator.h"
#include "get_model_paths.h"

using drake::multibody::ModelInstanceIndex;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::string;

MatrixXd CreateRandomMatrix(int n_rows, int n_cols, std::mt19937 &gen) {
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  return MatrixXd::NullaryExpr(n_rows, n_cols, [&]() { return dis(gen); });
}

class TestBatchQuasistaticSimulator : public ::testing::Test {
protected:
  void SetUp() override {
    n_tasks_ = std::thread::hardware_concurrency() * 20 + 1;
  }

  void SetUpPlanarHand() {
    const string kObjectSdfPath =
        GetQsimModelsPath() / "sphere_yz_rotation_r_0.25m.sdf";

    const string kModelDirectivePath = GetQsimModelsPath() / "planar_hand.yml";

    QuasistaticSimParameters sim_params;
    sim_params.gravity = Vector3d(0, 0, -10);
    sim_params.nd_per_contact = 2;
    sim_params.contact_detection_tolerance = 1.0;
    sim_params.is_quasi_dynamic = true;
    sim_params.gradient_from_active_constraints = true;

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

    q_sim_batch_ = std::make_unique<BatchQuasistaticSimulator>(
        kModelDirectivePath, robot_stiffness_dict, object_sdf_dict, sim_params);

    // Make sure that n_tasks_ is not divisible by hardware_concurrency.
    auto &q_sim = q_sim_batch_->get_q_sim();
    const auto name_to_idx_map = q_sim.GetModelInstanceNameToIndexMap();
    const auto idx_l = name_to_idx_map.at(robot_l_name);
    const auto idx_r = name_to_idx_map.at(robot_r_name);
    const auto idx_o = name_to_idx_map.at(object_name);

    ModelInstanceIndexToVecMap q0_dict = {{idx_o, Vector3d(0, 0.316, 0)},
                                          {idx_l, Vector2d(-0.775, -0.785)},
                                          {idx_r, Vector2d(0.775, 0.785)}};

    VectorXd q0 = q_sim.GetQFromQdict(q0_dict);
    VectorXd u0 = q_sim.GetQaCmdFromQaCmdDict(q0_dict);

    SampleUBatch(u0, 0.1);
    SetXBatch(q0);
  }

  void SampleUBatch(const Eigen::Ref<const Eigen::VectorXd>& u0,
                    double interval_size) {
    std::mt19937 gen(2007);
    u_batch_ =
        interval_size * CreateRandomMatrix(n_tasks_, u0.size(), gen);
    u_batch_.rowwise() += u0.transpose();
  }

  void SetXBatch(const Eigen::Ref<const Eigen::VectorXd>& x0) {
    x_batch_.resize(n_tasks_, x0.size());
    x_batch_.setZero();
    x_batch_.rowwise() += x0.transpose();
  }

  void CompareIsValid(const std::vector<bool>& is_valid_batch_1,
                      const std::vector<bool>& is_valid_batch_2) const {
    EXPECT_EQ(n_tasks_, is_valid_batch_1.size());
    EXPECT_EQ(n_tasks_, is_valid_batch_2.size());
    for (int i = 0; i < is_valid_batch_1.size(); i++) {
      EXPECT_EQ(is_valid_batch_1[i], is_valid_batch_2[i]);
      ASSERT_TRUE(is_valid_batch_1[i]);
    }
  }

  void CompareXNext(const Eigen::Ref<const MatrixXd>& x_next_batch_1,
                    const Eigen::Ref<const MatrixXd>& x_next_batch_2)
                    const {
    EXPECT_EQ(n_tasks_, x_next_batch_1.rows());
    EXPECT_EQ(n_tasks_, x_next_batch_2.rows());
    const double avg_diff = (x_next_batch_2 - x_next_batch_1)
        .matrix()
        .rowwise()
        .norm()
        .sum() /
        n_tasks_;
    EXPECT_LT(avg_diff, 1e-6);
  }

  void CompareB(const std::vector<MatrixXd>& B_batch_1,
                const std::vector<MatrixXd>& B_batch_2) const {
    EXPECT_EQ(n_tasks_, B_batch_1.size());
    EXPECT_EQ(n_tasks_, B_batch_2.size());
    for (int i = 0; i < n_tasks_; i++) {
      double err = (B_batch_1[i] - B_batch_2[i]).norm();
      EXPECT_LT(err, 1e-6);
    }
  }

  int n_tasks_{0};
  double h_{0.1};
  MatrixXd u_batch_, x_batch_;
  std::unique_ptr<BatchQuasistaticSimulator> q_sim_batch_;
};

TEST_F(TestBatchQuasistaticSimulator, TestForwardDynamics) {
  SetUpPlanarHand();
  auto [x_next_batch_parallel, B_batch_parallel, is_valid_batch_parallel] =
      q_sim_batch_->CalcDynamicsParallel(x_batch_, u_batch_, h_,
                                         GradientMode::kNone);

  auto [x_next_batch_serial, B_batch_serial, is_valid_batch_serial] =
      q_sim_batch_->CalcDynamicsSingleThread(x_batch_, u_batch_, h_,
                                             GradientMode::kNone);
  // is_valid.
  CompareIsValid(is_valid_batch_parallel, is_valid_batch_serial);

  // x_next.
  CompareXNext(x_next_batch_parallel, x_next_batch_serial);

  // B.
  EXPECT_EQ(B_batch_parallel.size(), 0);
  EXPECT_EQ(B_batch_serial.size(), 0);
}

TEST_F(TestBatchQuasistaticSimulator, TestGradient) {
  SetUpPlanarHand();
  auto [x_next_batch_parallel, B_batch_parallel, is_valid_batch_parallel] =
  q_sim_batch_->CalcDynamicsParallel(x_batch_, u_batch_, h_,
                                     GradientMode::kBOnly);

  auto [x_next_batch_serial, B_batch_serial, is_valid_batch_serial] =
  q_sim_batch_->CalcDynamicsSingleThread(x_batch_, u_batch_, h_,
                                         GradientMode::kBOnly);

  // is_valid.
  CompareIsValid(is_valid_batch_parallel, is_valid_batch_serial);

  // x_next.
  CompareXNext(x_next_batch_parallel, x_next_batch_serial);

  // B.
  CompareB(B_batch_parallel, B_batch_serial);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
