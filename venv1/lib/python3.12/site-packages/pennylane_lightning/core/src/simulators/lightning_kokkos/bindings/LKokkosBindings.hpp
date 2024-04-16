// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "BindingsBase.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"
#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "TypeList.hpp"
#include "Util.hpp" // exp2

/// @cond DEV
namespace {
using namespace Pennylane::Bindings;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using Kokkos::InitializationSettings;
using Pennylane::LightningKokkos::StateVectorKokkos;
using Pennylane::Util::exp2;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningKokkos {
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorKokkos<float>,
                              StateVectorKokkos<double>, void>;

/**
 * @brief Get a gate kernel map for a statevector.
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT = typename StateVectorT::ComplexT;
    using ParamT = PrecisionT; // Parameter's data precision
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;

    registerGatesForStateVector<StateVectorT>(pyclass);

    pyclass
        .def(py::init([](std::size_t num_qubits) {
            return new StateVectorT(num_qubits);
        }))
        .def(py::init([](std::size_t num_qubits,
                         const InitializationSettings &kokkos_args) {
            return new StateVectorT(num_qubits, kokkos_args);
        }))
        .def("resetStateVector", &StateVectorT::resetStateVector)
        .def(
            "setBasisState",
            [](StateVectorT &sv, const size_t index) {
                sv.setBasisState(index);
            },
            "Create Basis State on Device.")
        .def(
            "setStateVector",
            [](StateVectorT &sv, const std::vector<std::size_t> &indices,
               const np_arr_c &state) {
                const auto buffer = state.request();
                std::vector<Kokkos::complex<ParamT>> state_kok;
                if (buffer.size) {
                    const auto ptr =
                        static_cast<const Kokkos::complex<ParamT> *>(
                            buffer.ptr);
                    state_kok = std::vector<Kokkos::complex<ParamT>>{
                        ptr, ptr + buffer.size};
                }
                sv.setStateVector(indices, state_kok);
            },
            "Set State Vector on device with values and their corresponding "
            "indices for the state vector on device")
        .def(
            "DeviceToHost",
            [](StateVectorT &device_sv, np_arr_c &host_sv) {
                py::buffer_info numpyArrayInfo = host_sv.request();
                auto *data_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
                if (host_sv.size()) {
                    device_sv.DeviceToHost(data_ptr, host_sv.size());
                }
            },
            "Synchronize data from the GPU device to host.")
        .def("HostToDevice",
             py::overload_cast<ComplexT *, size_t>(&StateVectorT::HostToDevice),
             "Synchronize data from the host device to GPU.")
        .def(
            "HostToDevice",
            [](StateVectorT &device_sv, const np_arr_c &host_sv) {
                const py::buffer_info numpyArrayInfo = host_sv.request();
                auto *data_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
                const auto length =
                    static_cast<size_t>(numpyArrayInfo.shape[0]);
                if (length) {
                    device_sv.HostToDevice(data_ptr, length);
                }
            },
            "Synchronize data from the host device to GPU.")
        .def(
            "apply",
            [](StateVectorT &sv, const std::string &str,
               const std::vector<size_t> &wires, bool inv,
               [[maybe_unused]] const std::vector<std::vector<ParamT>> &params,
               [[maybe_unused]] const np_arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<Kokkos::complex<ParamT>> conv_matrix;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const Kokkos::complex<ParamT> *>(
                            m_buffer.ptr);
                    conv_matrix = std::vector<Kokkos::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                sv.applyOperation(str, wires, inv, std::vector<ParamT>{},
                                  conv_matrix);
            },
            "Apply operation via the gate matrix");
}

/**
 * @brief Register backend specific measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Pybind11's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificMeasurements(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type
    using ParamT = PrecisionT;           // Parameter's data precision

    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    using sparse_index_type = std::size_t;
    using np_arr_sparse_ind =
        py::array_t<sparse_index_type,
                    py::array::c_style | py::array::forcecast>;

    pyclass
        .def("expval",
             static_cast<PrecisionT (Measurements<StateVectorT>::*)(
                 const std::string &, const std::vector<size_t> &)>(
                 &Measurements<StateVectorT>::expval),
             "Expected value of an operation by name.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const np_arr_c &matrix,
               const std::vector<size_t> &wires) {
                const std::size_t matrix_size = exp2(2 * wires.size());
                auto matrix_data =
                    static_cast<ComplexT *>(matrix.request().ptr);
                std::vector<ComplexT> matrix_v{matrix_data,
                                               matrix_data + matrix_size};
                return M.expval(matrix_v, wires);
            },
            "Expected value of a Hermitian observable.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const np_arr_sparse_ind &row_map,
               const np_arr_sparse_ind &entries, const np_arr_c &values) {
                return M.expval(
                    static_cast<sparse_index_type *>(row_map.request().ptr),
                    static_cast<sparse_index_type>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<ComplexT *>(values.request().ptr),
                    static_cast<sparse_index_type>(values.request().size));
            },
            "Expected value of a sparse Hamiltonian.")
        .def("var",
             [](Measurements<StateVectorT> &M, const std::string &operation,
                const std::vector<size_t> &wires) {
                 return M.var(operation, wires);
             })
        .def("var",
             static_cast<PrecisionT (Measurements<StateVectorT>::*)(
                 const std::string &, const std::vector<size_t> &)>(
                 &Measurements<StateVectorT>::var),
             "Variance of an operation by name.")
        .def(
            "var",
            [](Measurements<StateVectorT> &M, const np_arr_sparse_ind &row_map,
               const np_arr_sparse_ind &entries, const np_arr_c &values) {
                return M.var(
                    static_cast<sparse_index_type *>(row_map.request().ptr),
                    static_cast<sparse_index_type>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<ComplexT *>(values.request().ptr),
                    static_cast<sparse_index_type>(values.request().size));
            },
            "Variance of a sparse Hamiltonian.");
}

/**
 * @brief Register observable classes.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificObservables(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ParamT = PrecisionT;             // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;

    std::string class_name;

    class_name = "SparseHamiltonianC" + bitsize;
    py::class_<SparseHamiltonian<StateVectorT>,
               std::shared_ptr<SparseHamiltonian<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init([](const np_arr_c &data,
                         const std::vector<std::size_t> &indices,
                         const std::vector<std::size_t> &indptr,
                         const std::vector<std::size_t> &wires) {
            using ComplexT = typename StateVectorT::ComplexT;
            const py::buffer_info buffer_data = data.request();
            const auto *data_ptr = static_cast<ComplexT *>(buffer_data.ptr);

            return SparseHamiltonian<StateVectorT>{
                std::vector<ComplexT>({data_ptr, data_ptr + data.size()}),
                indices, indptr, wires};
        }))
        .def("__repr__", &SparseHamiltonian<StateVectorT>::getObsName)
        .def("get_wires", &SparseHamiltonian<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const SparseHamiltonian<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<SparseHamiltonian<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<SparseHamiltonian<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");
}

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithms([[maybe_unused]] py::module_ &m) {}

/**
 * @brief Provide backend information.
 */
auto getBackendInfo() -> py::dict {
    using namespace py::literals;

    return py::dict("NAME"_a = "lightning.kokkos");
}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Pybind11 module.
 */
void registerBackendSpecificInfo(py::module_ &m) {
    m.def("kokkos_initialize", []() { Kokkos::initialize(); });
    m.def("kokkos_initialize",
          [](const InitializationSettings &args) { Kokkos::initialize(args); });
    m.def("kokkos_finalize", []() { Kokkos::finalize(); });
    m.def("kokkos_is_initialized", []() { return Kokkos::is_initialized(); });
    m.def("kokkos_is_finalized", []() { return Kokkos::is_finalized(); });
    m.def("backend_info", &getBackendInfo, "Backend-specific information.");
    m.def(
        "print_configuration",
        []() {
            std::ostringstream buffer;
            Kokkos::print_configuration(buffer, true);
            return buffer.str();
        },
        "Kokkos configurations query.");

    py::class_<InitializationSettings>(m, "InitializationSettings")
        .def(py::init([]() {
            return InitializationSettings()
                .set_num_threads(0)
                .set_device_id(0)
                .set_map_device_id_by("")
                .set_disable_warnings(0)
                .set_print_configuration(0)
                .set_tune_internals(0)
                .set_tools_libs("")
                .set_tools_help(0)
                .set_tools_args("");
        }))
        .def("get_num_threads", &InitializationSettings::get_num_threads,
             "Number of threads to use with the host parallel backend. Must be "
             "greater than zero.")
        .def("get_device_id", &InitializationSettings::get_device_id,
             "Device to use with the device parallel backend. Valid IDs are "
             "zero to number of GPU(s) available for execution minus one.")
        .def(
            "get_map_device_id_by",
            &InitializationSettings::get_map_device_id_by,
            "Strategy to select a device automatically from the GPUs available "
            "for execution. Must be either mpi_rank"
            "for round-robin assignment based on the local MPI rank or random.")
        .def("get_disable_warnings",
             &InitializationSettings::get_disable_warnings,
             "Whether to disable warning messages.")
        .def("get_print_configuration",
             &InitializationSettings::get_print_configuration,
             "Whether to print the configuration after initialization.")
        .def("get_tune_internals", &InitializationSettings::get_tune_internals,
             "Whether to allow autotuning internals instead of using "
             "heuristics.")
        .def("get_tools_libs", &InitializationSettings::get_tools_libs,
             "Which tool dynamic library to load. Must either be the full path "
             "to library or the name of library if the path is present in the "
             "runtime library search path (e.g. LD_LIBRARY_PATH)")
        .def("get_tools_help", &InitializationSettings::get_tools_help,
             "Query the loaded tool for its command-line options support.")
        .def("get_tools_args", &InitializationSettings::get_tools_args,
             "Options to pass to the loaded tool as command-line arguments.")
        .def("has_num_threads", &InitializationSettings::has_num_threads,
             "Number of threads to use with the host parallel backend. Must be "
             "greater than zero.")
        .def("has_device_id", &InitializationSettings::has_device_id,
             "Device to use with the device parallel backend. Valid IDs are "
             "zero "
             "to number of GPU(s) available for execution minus one.")
        .def(
            "has_map_device_id_by",
            &InitializationSettings::has_map_device_id_by,
            "Strategy to select a device automatically from the GPUs available "
            "for execution. Must be either mpi_rank"
            "for round-robin assignment based on the local MPI rank or random.")
        .def("has_disable_warnings",
             &InitializationSettings::has_disable_warnings,
             "Whether to disable warning messages.")
        .def("has_print_configuration",
             &InitializationSettings::has_print_configuration,
             "Whether to print the configuration after initialization.")
        .def("has_tune_internals", &InitializationSettings::has_tune_internals,
             "Whether to allow autotuning internals instead of using "
             "heuristics.")
        .def("has_tools_libs", &InitializationSettings::has_tools_libs,
             "Which tool dynamic library to load. Must either be the full path "
             "to "
             "library or the name of library if the path is present in the "
             "runtime library search path (e.g. LD_LIBRARY_PATH)")
        .def("has_tools_help", &InitializationSettings::has_tools_help,
             "Query the loaded tool for its command-line options support.")
        .def("has_tools_args", &InitializationSettings::has_tools_args,
             "Options to pass to the loaded tool as command-line arguments.")
        .def("set_num_threads", &InitializationSettings::set_num_threads,
             "Number of threads to use with the host parallel backend. Must be "
             "greater than zero.")
        .def("set_device_id", &InitializationSettings::set_device_id,
             "Device to use with the device parallel backend. Valid IDs are "
             "zero to number of GPU(s) available for execution minus one.")
        .def(
            "set_map_device_id_by",
            &InitializationSettings::set_map_device_id_by,
            "Strategy to select a device automatically from the GPUs available "
            "for execution. Must be either mpi_rank"
            "for round-robin assignment based on the local MPI rank or random.")
        .def("set_disable_warnings",
             &InitializationSettings::set_disable_warnings,
             "Whether to disable warning messages.")
        .def("set_print_configuration",
             &InitializationSettings::set_print_configuration,
             "Whether to print the configuration after initialization.")
        .def("set_tune_internals", &InitializationSettings::set_tune_internals,
             "Whether to allow autotuning internals instead of using "
             "heuristics.")
        .def("set_tools_libs", &InitializationSettings::set_tools_libs,
             "Which tool dynamic library to load. Must either be the full path "
             "to library or the name of library if the path is present in the "
             "runtime library search path (e.g. LD_LIBRARY_PATH)")
        .def("set_tools_help", &InitializationSettings::set_tools_help,
             "Query the loaded tool for its command-line options support.")
        .def("set_tools_args", &InitializationSettings::set_tools_args,
             "Options to pass to the loaded tool as command-line arguments.")
        .def("__repr__", [](const InitializationSettings &args) {
            std::ostringstream args_stream;
            args_stream << "InitializationSettings:\n";
            args_stream << "num_threads = " << args.get_num_threads() << '\n';
            args_stream << "device_id = " << args.get_device_id() << '\n';
            args_stream << "map_device_id_by = " << args.get_map_device_id_by()
                        << '\n';
            args_stream << "disable_warnings = " << args.get_disable_warnings()
                        << '\n';
            args_stream << "print_configuration = "
                        << args.get_print_configuration() << '\n';
            args_stream << "tune_internals = " << args.get_tune_internals()
                        << '\n';
            args_stream << "tools_libs = " << args.get_tools_libs() << '\n';
            args_stream << "tools_help = " << args.get_tools_help() << '\n';
            args_stream << "tools_args = " << args.get_tools_args();
            return args_stream.str();
        });
}
} // namespace Pennylane::LightningKokkos
