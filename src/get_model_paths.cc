#include "get_model_paths.h"

using std::filesystem::current_path;
using std::filesystem::path;

std::filesystem::path GetPyPackagesPath() {
  auto file_path = path(__FILE__);
  return file_path.parent_path() / path("../../..") / path("PycharmProjects");
}

std::filesystem::path GetQsimModelsPath() {
  return GetPyPackagesPath() / path("quasistatic_simulator/models");
}

std::filesystem::path GetRoboticsUtilitiesModelsPath() {
  return GetPyPackagesPath() / path("robotics_utilities/models");
}
