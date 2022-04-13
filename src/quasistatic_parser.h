#include <iostream>
#include <unordered_map>

#include <Eigen/Dense>

#include "quasistatic_simulator.h"

class QuasistaticParser {
public:
  explicit QuasistaticParser(const std::string &q_model_path);
  [[nodiscard]] std::unique_ptr<QuasistaticSimulator> MakeSimulator() const;

private:
  std::string model_directive_path_;
  std::unordered_map<std::string, Eigen::VectorXd> robot_stiffness_;
  std::unordered_map<std::string, std::string> object_sdf_paths_;
  QuasistaticSimParameters sim_params_;
};
