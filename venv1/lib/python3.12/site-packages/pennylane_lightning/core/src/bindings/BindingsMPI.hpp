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

/**
 * @file Bindings.hpp
 * Defines device-agnostic operations to export to Python and other utility
 * functions interfacing with Pybind11.
 */

#pragma once
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "CPUMemoryModel.hpp" // CPUMemoryModel, getMemoryModel, bestCPUMemoryModel, getAlignment
#include "JacobianData.hpp"
#include "Macros.hpp" // CPUArch
#include "Memory.hpp" // alignedAlloc
#include "Observables.hpp"
#include "Util.hpp" // for_each_enum

#ifdef _ENABLE_PLGPU
#include "AdjointJacobianGPUMPI.hpp"
#include "JacobianDataMPI.hpp"
#include "LGPUBindingsMPI.hpp"
#include "MeasurementsGPUMPI.hpp"
#include "ObservablesGPUMPI.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Observables;
using namespace Pennylane::LightningGPU::Measures;
} // namespace
  /// @endcond

#else

static_assert(false, "Backend not found.");

#endif

namespace py = pybind11;

namespace Pennylane {
/**
 * @brief Register observable classes.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT> void registerObservablesMPI(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type.
    using ParamT = PrecisionT;           // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;
    using np_arr_r = py::array_t<ParamT, py::array::c_style>;
    using np_arr_sparse_ind = typename std::conditional<
        std::is_same<ParamT, float>::value,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast>>::type;

    std::string class_name;

    class_name = "ObservableMPIC" + bitsize;
    py::class_<Observable<StateVectorT>,
               std::shared_ptr<Observable<StateVectorT>>>(m, class_name.c_str(),
                                                          py::module_local());

    class_name = "NamedObsMPIC" + bitsize;
    py::class_<NamedObsMPI<StateVectorT>,
               std::shared_ptr<NamedObsMPI<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const std::string &name, const std::vector<size_t> &wires) {
                return NamedObsMPI<StateVectorT>(name, wires);
            }))
        .def("__repr__", &NamedObsMPI<StateVectorT>::getObsName)
        .def("get_wires", &NamedObsMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObsMPI<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<NamedObsMPI<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<NamedObsMPI<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HermitianObsMPIC" + bitsize;
    py::class_<HermitianObsMPI<StateVectorT>,
               std::shared_ptr<HermitianObsMPI<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const np_arr_c &matrix, const std::vector<size_t> &wires) {
                auto buffer = matrix.request();
                const auto *ptr = static_cast<ComplexT *>(buffer.ptr);
                return HermitianObsMPI<StateVectorT>(
                    std::vector<ComplexT>(ptr, ptr + buffer.size), wires);
            }))
        .def("__repr__", &HermitianObsMPI<StateVectorT>::getObsName)
        .def("get_wires", &HermitianObsMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HermitianObsMPI<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HermitianObsMPI<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HermitianObsMPI<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "TensorProdObsMPIC" + bitsize;
    py::class_<TensorProdObsMPI<StateVectorT>,
               std::shared_ptr<TensorProdObsMPI<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const std::vector<std::shared_ptr<Observable<StateVectorT>>>
                   &obs) { return TensorProdObsMPI<StateVectorT>(obs); }))
        .def("__repr__", &TensorProdObsMPI<StateVectorT>::getObsName)
        .def("get_wires", &TensorProdObsMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const TensorProdObsMPI<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<TensorProdObsMPI<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<TensorProdObsMPI<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HamiltonianMPIC" + bitsize;
    using ObsPtr = std::shared_ptr<Observable<StateVectorT>>;
    py::class_<HamiltonianMPI<StateVectorT>,
               std::shared_ptr<HamiltonianMPI<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const np_arr_r &coeffs, const std::vector<ObsPtr> &obs) {
                auto buffer = coeffs.request();
                const auto ptr = static_cast<const ParamT *>(buffer.ptr);
                return HamiltonianMPI<StateVectorT>{
                    std::vector(ptr, ptr + buffer.size), obs};
            }))
        .def("__repr__", &HamiltonianMPI<StateVectorT>::getObsName)
        .def("get_wires", &HamiltonianMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HamiltonianMPI<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HamiltonianMPI<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HamiltonianMPI<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");
#ifdef _ENABLE_PLGPU
    class_name = "SparseHamiltonianMPIC" + bitsize;
    using SpIDX = typename SparseHamiltonianMPI<StateVectorT>::IdxT;
    py::class_<SparseHamiltonianMPI<StateVectorT>,
               std::shared_ptr<SparseHamiltonianMPI<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init([](const np_arr_c &data, const np_arr_sparse_ind &indices,
                         const np_arr_sparse_ind &offsets,
                         const std::vector<std::size_t> &wires) {
            const py::buffer_info buffer_data = data.request();
            const auto *data_ptr = static_cast<ComplexT *>(buffer_data.ptr);

            const py::buffer_info buffer_indices = indices.request();
            const auto *indices_ptr = static_cast<SpIDX *>(buffer_indices.ptr);

            const py::buffer_info buffer_offsets = offsets.request();
            const auto *offsets_ptr = static_cast<SpIDX *>(buffer_offsets.ptr);

            return SparseHamiltonianMPI<StateVectorT>{
                std::vector<ComplexT>({data_ptr, data_ptr + data.size()}),
                std::vector<SpIDX>({indices_ptr, indices_ptr + indices.size()}),
                std::vector<SpIDX>({offsets_ptr, offsets_ptr + offsets.size()}),
                wires};
        }))
        .def("__repr__", &SparseHamiltonianMPI<StateVectorT>::getObsName)
        .def("get_wires", &SparseHamiltonianMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const SparseHamiltonianMPI<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<SparseHamiltonianMPI<StateVectorT>>(
                        other)) {
                    return false;
                }
                auto other_cast =
                    other.cast<SparseHamiltonianMPI<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");
#endif
}

/**
 * @brief Register agnostic measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Pybind11's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendAgnosticMeasurementsMPI(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ParamT = PrecisionT;             // Parameter's data precision

    pyclass
        .def("probs",
             [](MeasurementsMPI<StateVectorT> &M,
                const std::vector<size_t> &wires) {
                 return py::array_t<ParamT>(py::cast(M.probs(wires)));
             })
        .def("probs",
             [](MeasurementsMPI<StateVectorT> &M) {
                 return py::array_t<ParamT>(py::cast(M.probs()));
             })
        .def(
            "expval",
            [](MeasurementsMPI<StateVectorT> &M,
               const std::shared_ptr<Observable<StateVectorT>> &ob) {
                return M.expval(*ob);
            },
            "Expected value of an observable object.")
        .def(
            "var",
            [](MeasurementsMPI<StateVectorT> &M,
               const std::shared_ptr<Observable<StateVectorT>> &ob) {
                return M.var(*ob);
            },
            "Variance of an observable object.")
        .def("generate_samples", [](MeasurementsMPI<StateVectorT> &M,
                                    size_t num_wires, size_t num_shots) {
            auto &&result = M.generate_samples(num_shots);
            const size_t ndim = 2;
            const std::vector<size_t> shape{num_shots, num_wires};
            constexpr auto sz = sizeof(size_t);
            const std::vector<size_t> strides{sz * num_wires, sz};
            // return 2-D NumPy array
            return py::array(py::buffer_info(
                result.data(), /* data as contiguous array  */
                sz,            /* size of one scalar        */
                py::format_descriptor<size_t>::format(), /* data type */
                ndim,   /* number of dimensions      */
                shape,  /* shape of the matrix       */
                strides /* strides for each axis     */
                ));
        });
}

/**
 * @brief Register the adjoint Jacobian method.
 */
template <class StateVectorT>
auto registerAdjointJacobianMPI(
    AdjointJacobianMPI<StateVectorT> &adjoint_jacobian, const StateVectorT &sv,
    const std::vector<std::shared_ptr<Observable<StateVectorT>>> &observables,
    const OpsData<StateVectorT> &operations,
    const std::vector<size_t> &trainableParams)
    -> py::array_t<typename StateVectorT::PrecisionT> {
    using PrecisionT = typename StateVectorT::PrecisionT;
    std::vector<PrecisionT> jac(observables.size() * trainableParams.size(),
                                PrecisionT{0.0});
    const JacobianDataMPI<StateVectorT> jd{operations.getTotalNumParams(), sv,
                                           observables, operations,
                                           trainableParams};
    adjoint_jacobian.adjointJacobian(std::span{jac}, jd, sv);
    return py::array_t<PrecisionT>(py::cast(jac));
}

/**
 * @brief Register agnostic algorithms.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendAgnosticAlgorithmsMPI(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type
    using ParamT = PrecisionT;           // Parameter's data precision

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    std::string class_name;

    //***********************************************************************//
    //                              Operations
    //***********************************************************************//

    class_name = "OpsStructMPIC" + bitsize;
    py::class_<OpsData<StateVectorT>>(m, class_name.c_str(), py::module_local())
        .def(py::init<const std::vector<std::string> &,
                      const std::vector<std::vector<ParamT>> &,
                      const std::vector<std::vector<size_t>> &,
                      const std::vector<bool> &,
                      const std::vector<std::vector<ComplexT>> &>())
        .def("__repr__", [](const OpsData<StateVectorT> &ops) {
            using namespace Pennylane::Util;
            std::ostringstream ops_stream;
            for (size_t op = 0; op < ops.getSize(); op++) {
                ops_stream << "{'name': " << ops.getOpsName()[op];
                ops_stream << ", 'params': " << ops.getOpsParams()[op];
                ops_stream << ", 'inv': " << ops.getOpsInverses()[op];
                ops_stream << "}";
                if (op < ops.getSize() - 1) {
                    ops_stream << ",";
                }
            }
            return "Operations: [" + ops_stream.str() + "]";
        });

    /**
     * Create operation list.
     */
    std::string function_name = "create_ops_listMPIC" + bitsize;
    m.def(
        function_name.c_str(),
        [](const std::vector<std::string> &ops_name,
           const std::vector<std::vector<PrecisionT>> &ops_params,
           const std::vector<std::vector<size_t>> &ops_wires,
           const std::vector<bool> &ops_inverses,
           const std::vector<np_arr_c> &ops_matrices,
           const std::vector<std::vector<size_t>> &ops_controlled_wires,
           const std::vector<std::vector<bool>> &ops_controlled_values) {
            std::vector<std::vector<ComplexT>> conv_matrices(
                ops_matrices.size());
            for (size_t op = 0; op < ops_name.size(); op++) {
                const auto m_buffer = ops_matrices[op].request();
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const ComplexT *>(m_buffer.ptr);
                    conv_matrices[op] =
                        std::vector<ComplexT>{m_ptr, m_ptr + m_buffer.size};
                }
            }
            return OpsData<StateVectorT>{ops_name,
                                         ops_params,
                                         ops_wires,
                                         ops_inverses,
                                         conv_matrices,
                                         ops_controlled_wires,
                                         ops_controlled_values};
        },
        "Create a list of operations from data.");

    //***********************************************************************//
    //                            Adjoint Jacobian
    //***********************************************************************//
    class_name = "AdjointJacobianMPIC" + bitsize;
    py::class_<AdjointJacobianMPI<StateVectorT>>(m, class_name.c_str(),
                                                 py::module_local())
        .def(py::init<>())
        .def(
            "batched",
            [](AdjointJacobianMPI<StateVectorT> &adjoint_jacobian,
               const StateVectorT &sv,
               const std::vector<std::shared_ptr<Observable<StateVectorT>>>
                   &observables,
               const OpsData<StateVectorT> &operations,
               const std::vector<size_t> &trainableParams) {
                using PrecisionT = typename StateVectorT::PrecisionT;
                std::vector<PrecisionT> jac(observables.size() *
                                                trainableParams.size(),
                                            PrecisionT{0.0});
                const JacobianDataMPI<StateVectorT> jd{
                    operations.getTotalNumParams(), sv, observables, operations,
                    trainableParams};
                adjoint_jacobian.adjointJacobian_serial(std::span{jac}, jd);
                return py::array_t<PrecisionT>(py::cast(jac));
            },
            "Batch Adjoint Jacobian method.")
        .def("__call__", &registerAdjointJacobianMPI<StateVectorT>,
             "Adjoint Jacobian method.");
}

/**
 * @brief Templated class to build lightning class bindings.
 *
 * @tparam StateVectorT State vector type
 * @param m Pybind11 module.
 */
template <class StateVectorT> void lightningClassBindingsMPI(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    //***********************************************************************//
    //                              StateVector
    //***********************************************************************//
    std::string class_name = "StateVectorMPIC" + bitsize;
    auto pyclass =
        py::class_<StateVectorT>(m, class_name.c_str(), py::module_local());
    pyclass.def_property_readonly("size", &StateVectorT::getLength);

    registerBackendClassSpecificBindingsMPI<StateVectorT>(pyclass);

    //***********************************************************************//
    //                              Observables
    //***********************************************************************//

    py::module_ obs_submodule =
        m.def_submodule("observablesMPI", "Submodule for observables classes.");
    registerObservablesMPI<StateVectorT>(obs_submodule);

    //***********************************************************************//
    //                             Measurements
    //***********************************************************************//

    class_name = "MeasurementsMPIC" + bitsize;
    auto pyclass_measurements = py::class_<MeasurementsMPI<StateVectorT>>(
        m, class_name.c_str(), py::module_local());

    pyclass_measurements.def(py::init<StateVectorT &>());
    registerBackendAgnosticMeasurementsMPI<StateVectorT>(pyclass_measurements);
    registerBackendSpecificMeasurementsMPI<StateVectorT>(pyclass_measurements);

    //***********************************************************************//
    //                           Algorithms
    //***********************************************************************//

    py::module_ alg_submodule = m.def_submodule(
        "algorithmsMPI", "Submodule for the algorithms functionality.");
    registerBackendAgnosticAlgorithmsMPI<StateVectorT>(alg_submodule);
    registerBackendSpecificAlgorithmsMPI<StateVectorT>(alg_submodule);
}

template <typename TypeList>
void registerLightningClassBindingsMPI(py::module_ &m) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        lightningClassBindingsMPI<StateVectorT>(m);
        registerLightningClassBindingsMPI<typename TypeList::Next>(m);
        py::register_local_exception<Pennylane::Util::LightningException>(
            m, "LightningExceptionMPI");
    }
}
} // namespace Pennylane
