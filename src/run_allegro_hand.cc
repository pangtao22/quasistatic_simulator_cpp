#include <vector>

#include "get_model_paths.h"
#include "quasistatic_parser.h"

using std::cout;
using std::endl;

int main() {
  auto q_model_path =
      GetQsimModelsPath() / "q_sys" / "allegro_hand_and_sphere.yml";
  cout << q_model_path << endl;
  auto parser = QuasistaticParser(q_model_path);
  auto q_sim = parser.MakeSimulator();
  return 0;
}