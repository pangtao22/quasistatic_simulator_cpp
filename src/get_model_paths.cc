#include "get_model_paths.h"

using std::filesystem::current_path;
using std::filesystem::path;

/*
 * It is assumed that this executable is located at
 *   ${HOME}/ClionProjects/quasistatic_simulator_cpp/cmake-build-release/src
 */
std::filesystem::path GetPyPackagesPath() {
  return current_path() / path("../../../..") / path("PycharmProjects");
}

std::filesystem::path GetQsimModelsPath() {
  return GetPyPackagesPath() / path("quasistatic_simulator/models");
}

std::filesystem::path GetRoboticsUtilitiesModelsPath() {
  return GetPyPackagesPath() / path("robotics_utilities/models");
}
