#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "batch_quasistatic_simulator.h"
#include "quasistatic_simulator.h"

namespace py = pybind11;

PYBIND11_MODULE(qsim_cpp, m) {
  py::enum_<GradientMode>(m, "GradientMode")
      .value("kNone", GradientMode::kNone)
      .value("kBOnly", GradientMode::kBOnly)
      .value("kAB", GradientMode::kAB);

  py::enum_<ForwardDynamicsMode>(m, "ForwardDynamicsMode")
  .value("kQpMp", ForwardDynamicsMode::kQpMp)
  .value("kQpCvx", ForwardDynamicsMode::kQpCvx)
  .value("kSocpMp", ForwardDynamicsMode::kSocpMp)
  .value("kLogPyramidMp", ForwardDynamicsMode::kLogPyramidMp)
  .value("kLogPyramidCvx", ForwardDynamicsMode::kLogPyramidCvx)
  .value("kLogIcecreamMp", ForwardDynamicsMode::kLogIcecreamMp)
  .value("kLogIcecreamCvx", ForwardDynamicsMode::kLogIcecreamCvx);

  {
    using Class = QuasistaticSimParameters;
    py::class_<Class>(m, "QuasistaticSimParametersCpp")
        .def(py::init<>())
        .def_readwrite("h", &Class::h)
        .def_readwrite("gravity", &Class::gravity)
        .def_readwrite("nd_per_contact", &Class::nd_per_contact)
        .def_readwrite("contact_detection_tolerance",
                       &Class::contact_detection_tolerance)
        .def_readwrite("is_quasi_dynamic", &Class::is_quasi_dynamic)
        .def_readwrite("forward_mode", &Class::forward_mode)
        .def_readwrite("gradient_mode", &Class::gradient_mode)
        .def_readwrite("log_barrier_weight", &Class::log_barrier_weight)
        .def_readwrite("unactuated_mass_scale", &Class::unactuated_mass_scale)
        .def_readwrite("gradient_lstsq_tolerance",
                       &Class::gradient_lstsq_tolerance);
  }

  {
    using Class = QuasistaticSimulator;
    py::class_<Class>(m, "QuasistaticSimulatorCpp")
        .def(py::init<std::string,
                      const std::unordered_map<std::string, Eigen::VectorXd> &,
                      const std::unordered_map<std::string, std::string> &,
                      QuasistaticSimParameters>(),
             py::arg("model_directive_path"), py::arg("robot_stiffness_str"),
             py::arg("object_sdf_paths"), py::arg("sim_params"))
        .def("update_mbp_positions",
             py::overload_cast<const ModelInstanceIndexToVecMap &>(
                 &Class::UpdateMbpPositions))
        .def("get_mbp_positions", &Class::GetMbpPositions)
        .def("get_positions", &Class::GetPositions)
        .def("step",
             py::overload_cast<const ModelInstanceIndexToVecMap &,
                               const ModelInstanceIndexToVecMap &,
                               const QuasistaticSimParameters &>(
                 &Class::Step),
             py::arg("q_a_cmd_dict"), py::arg("tau_ext_dict"),
             py::arg("params"))
        .def(
            "step_default",
            py::overload_cast<const ModelInstanceIndexToVecMap &,
                              const ModelInstanceIndexToVecMap &>(
                &Class::Step),
            py::arg("q_a_cmd_dict"), py::arg("tau_ext_dict"))
        .def("calc_tau_ext", &Class::CalcTauExt)
        .def("get_model_instance_name_to_index_map",
             &Class::GetModelInstanceNameToIndexMap)
        .def("get_all_models", &Class::get_all_models)
        .def("get_actuated_models", &Class::get_actuated_models)
        .def("get_unactuated_models", &Class::get_unactuated_models)
        .def("get_query_object", &Class::get_query_object,
             py::return_value_policy::reference_internal)
        .def("get_plant", &Class::get_plant,
             py::return_value_policy::reference_internal)
        .def("get_scene_graph", &Class::get_scene_graph,
             py::return_value_policy::reference_internal)
        .def("get_contact_results", &Class::get_contact_results,
             py::return_value_policy::reference_internal)
        .def("get_sim_params", &Class::get_sim_params,
             py::return_value_policy::reference_internal)
        .def("num_actuated_dofs", &Class::num_actuated_dofs)
        .def("num_unactuated_dofs", &Class::num_unactuated_dofs)
        .def("get_Dq_nextDq", &Class::get_Dq_nextDq)
        .def("get_Dq_nextDqa_cmd", &Class::get_Dq_nextDqa_cmd)
        .def("get_velocity_indices", &Class::GetVelocityIndices)
        .def("get_position_indices", &Class::GetPositionIndices)
        .def("get_v_dict_from_vec", &Class::GetVdictFromVec)
        .def("get_q_dict_from_vec", &Class::GetQDictFromVec)
        .def("get_q_vec_from_dict", &Class::GetQVecFromDict)
        .def("get_q_a_cmd_vec_from_dict", &Class::GetQaCmdVecFromDict)
        .def("get_q_a_cmd_dict_from_vec", &Class::GetQaCmdDictFromVec);
  }

  {
    using Class = BatchQuasistaticSimulator;
    py::class_<Class>(m, "BatchQuasistaticSimulator")
        .def(py::init<std::string,
                      const std::unordered_map<std::string, Eigen::VectorXd> &,
                      const std::unordered_map<std::string, std::string> &,
                      QuasistaticSimParameters>(),
             py::arg("model_directive_path"), py::arg("robot_stiffness_str"),
             py::arg("object_sdf_paths"), py::arg("sim_params"))
        .def("calc_dynamics_parallel", &Class::CalcDynamicsParallel)
        .def("calc_bundled_B_trj", &Class::CalcBundledBTrj)
        .def("calc_bundled_B_trj_direct", &Class::CalcBundledBTrjDirect)
        .def("sample_gaussian_matrix", &Class::SampleGaussianMatrix)
        .def("get_num_max_parallel_executions",
             &Class::get_num_max_parallel_executions)
        .def("set_num_max_parallel_executions",
             &Class::set_num_max_parallel_executions);
  }
}
