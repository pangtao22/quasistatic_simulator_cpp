#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quasistatic_simulator.h"

namespace py = pybind11;

PYBIND11_MODULE(quasistatic_simulator_py, m) {
  //  py::module::import("pydrake.multibody");
  //  py::module::import("pydrake.geometry");

  {
    using Class = QuasistaticSimParameters;
    py::class_<Class>(m, "QuasistaticSimParametersCpp")
        .def(py::init<>())
        .def_readwrite("gravity", &Class::gravity)
        .def_readwrite("nd_per_contact", &Class::nd_per_contact)
        .def_readwrite("contact_detection_tolerance",
                       &Class::contact_detection_tolerance)
        .def_readwrite("is_quasi_dynamic", &Class::is_quasi_dynamic)
        .def_readwrite("requires_grad", &Class::requires_grad);
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
        .def("update_mbp_positions", &Class::UpdateMbpPositions)
        .def("get_mbp_positions", &Class::GetMbpPositions)
        .def("get_positions", &Class::GetPositions)
        .def("step",
             py::overload_cast<const ModelInstanceToVecMap &,
                               const ModelInstanceToVecMap &, const double,
                               const double, const bool>(&Class::Step),
             py::arg("q_a_cmd_dict"), py::arg("tau_ext_dict"), py::arg("h"),
             py::arg("contact_detection_tolerance"), py::arg("requires_grad"))
        .def("step_default",
             py::overload_cast<const ModelInstanceToVecMap &,
                               const ModelInstanceToVecMap &, const double>(
                 &Class::Step),
             py::arg("q_a_cmd_dict"), py::arg("tau_ext_dict"), py::arg("h"))
        .def("calc_tau_ext", &Class::CalcTauExt)
        .def("get_models_all", &Class::get_models_all)
        .def("get_query_object", &Class::get_query_object,
             py::return_value_policy::reference_internal)
        .def("get_plant", &Class::get_plant,
             py::return_value_policy::reference_internal)
        .def("get_contact_results", &Class::get_contact_results,
             py::return_value_policy::reference_internal);
  }
}
