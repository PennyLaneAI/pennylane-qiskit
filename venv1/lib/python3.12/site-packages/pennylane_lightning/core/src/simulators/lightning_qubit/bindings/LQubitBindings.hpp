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
 * @file BindingsLQubit.hpp
 * Defines LightningQubit-specific operations to export to Python, other utility
 * functions interfacing with Pybind11 and support to agnostic bindings.
 */

#pragma once
#include "BindingsBase.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "DynamicDispatcher.hpp"
#include "GateOperation.hpp"
#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "TypeList.hpp"
#include "VectorJacobianProduct.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Bindings;
using namespace Pennylane::LightningQubit::Algorithms;
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::LightningQubit::Observables;
using Pennylane::LightningQubit::StateVectorLQubitManaged;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningQubit {
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorLQubitManaged<float>,
                              StateVectorLQubitManaged<double>, void>;

/**
 * @brief Get a gate kernel map for a statevector.
 */
template <class StateVectorT>
auto svKernelMap(const StateVectorT &sv) -> py::dict {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    py::dict res_map;
    namespace Constant = Pennylane::Gates::Constant;
    using Pennylane::Util::lookup;

    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto [GateKernelMap, GeneratorKernelMap, MatrixKernelMap,
          ControlledGateKernelMap, ControlledGeneratorKernelMap,
          ControlledMatrixKernelMap] = sv.getSupportedKernels();

    for (const auto &[gate_op, kernel] : GateKernelMap) {
        const auto key = std::string(lookup(Constant::gate_names, gate_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[gen_op, kernel] : GeneratorKernelMap) {
        const auto key = std::string(lookup(Constant::generator_names, gen_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : MatrixKernelMap) {
        const auto key = std::string(lookup(Constant::matrix_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : ControlledGateKernelMap) {
        const auto key =
            std::string(lookup(Constant::controlled_gate_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : ControlledGeneratorKernelMap) {
        const auto key =
            std::string(lookup(Constant::controlled_generator_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : ControlledMatrixKernelMap) {
        const auto key =
            std::string(lookup(Constant::controlled_matrix_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    return res_map;
}

/**
 * @brief Register controlled matrix kernel.
 */
template <class StateVectorT>
void applyControlledMatrix(
    StateVectorT &st,
    const py::array_t<std::complex<typename StateVectorT::PrecisionT>,
                      py::array::c_style | py::array::forcecast> &matrix,
    const std::vector<size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<size_t> &wires, bool inverse = false) {
    using ComplexT = typename StateVectorT::ComplexT;
    st.applyControlledMatrix(
        static_cast<const ComplexT *>(matrix.request().ptr), controlled_wires,
        controlled_values, wires, inverse);
}
template <class StateVectorT, class PyClass>
void registerControlledGate(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ParamT = PrecisionT;             // Parameter's data precision

    using Pennylane::Gates::ControlledGateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Pennylane::Gates::Constant;

    for_each_enum<ControlledGateOperation>(
        [&pyclass](ControlledGateOperation gate_op) {
            using Pennylane::Util::lookup;
            const auto gate_name =
                std::string(lookup(Constant::controlled_gate_names, gate_op));
            const std::string doc = "Apply the " + gate_name + " gate.";
            auto func = [gate_name = gate_name](
                            StateVectorT &sv,
                            const std::vector<size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<size_t> &wires, bool inverse,
                            const std::vector<ParamT> &params) {
                sv.applyOperation(gate_name, controlled_wires,
                                  controlled_values, wires, inverse, params);
            };
            pyclass.def(gate_name.c_str(), func, doc.c_str());
        });
}

/**
 * @brief Get a controlled matrix and kernel map for a statevector.
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
    registerControlledGate<StateVectorT>(pyclass);

    pyclass
        .def(py::init([](std::size_t num_qubits) {
            return new StateVectorT(num_qubits);
        }))
        .def("resetStateVector", &StateVectorT::resetStateVector)
        .def(
            "setBasisState",
            [](StateVectorT &sv, const size_t index) {
                sv.setBasisState(index);
            },
            "Create Basis State.")
        .def(
            "setStateVector",
            [](StateVectorT &sv, const std::vector<std::size_t> &indices,
               const np_arr_c &state) {
                const auto buffer = state.request();
                std::vector<ComplexT> state_in;
                if (buffer.size) {
                    const auto ptr = static_cast<const ComplexT *>(buffer.ptr);
                    state_in = std::vector<ComplexT>{ptr, ptr + buffer.size};
                }
                sv.setStateVector(indices, state_in);
            },
            "Set State Vector with values and their corresponding indices")
        .def(
            "getState",
            [](const StateVectorT &sv, np_arr_c &state) {
                py::buffer_info numpyArrayInfo = state.request();
                auto *data_ptr =
                    static_cast<std::complex<PrecisionT> *>(numpyArrayInfo.ptr);
                if (state.size()) {
                    std::copy(sv.getData(), sv.getData() + sv.getLength(),
                              data_ptr);
                }
            },
            "Copy StateVector data into a Numpy array.")
        .def(
            "UpdateData",
            [](StateVectorT &device_sv, const np_arr_c &state) {
                const py::buffer_info numpyArrayInfo = state.request();
                auto *data_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
                const auto length =
                    static_cast<size_t>(numpyArrayInfo.shape[0]);
                if (length) {
                    device_sv.updateData(data_ptr, length);
                }
            },
            "Copy StateVector data into a Numpy array.")
        .def("applyControlledMatrix", &applyControlledMatrix<StateVectorT>,
             "Apply controlled operation")
        .def("kernel_map", &svKernelMap<StateVectorT>,
             "Get internal kernels for operations");
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
    using ParamT = PrecisionT;             // Parameter's data precision

    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    using sparse_index_type = size_t;
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
            [](Measurements<StateVectorT> &M, const np_arr_sparse_ind &row_map,
               const np_arr_sparse_ind &entries, const np_arr_c &values) {
                return M.expval(
                    static_cast<sparse_index_type *>(row_map.request().ptr),
                    static_cast<sparse_index_type>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<std::complex<PrecisionT> *>(
                        values.request().ptr),
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
                    static_cast<std::complex<PrecisionT> *>(
                        values.request().ptr),
                    static_cast<sparse_index_type>(values.request().size));
            },
            "Variance of a sparse Hamiltonian.")
        .def("generate_mcmc_samples",
             [](Measurements<StateVectorT> &M, size_t num_wires,
                const std::string &kernelname, size_t num_burnin,
                size_t num_shots) {
                 std::vector<size_t> &&result = M.generate_samples_metropolis(
                     kernelname, num_burnin, num_shots);

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
 * @brief Register backend specific observables.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificObservables([[maybe_unused]] py::module_ &m) {
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
 * @brief Register Vector Jacobian Product.
 */
template <class StateVectorT, class np_arr_c>
auto registerVJP(VectorJacobianProduct<StateVectorT> &calculate_vjp,
                 const StateVectorT &sv,
                 const OpsData<StateVectorT> &operations, const np_arr_c &dy,
                 const std::vector<size_t> &trainableParams)
    -> py::array_t<std::complex<typename StateVectorT::PrecisionT>> {
    /* Do not cast non-conforming array. Argument trainableParams
     * should only contain indices for operations.
     */
    using PrecisionT = typename StateVectorT::PrecisionT;
    std::vector<std::complex<PrecisionT>> vjp(trainableParams.size(),
                                              std::complex<PrecisionT>{});

    const JacobianData<StateVectorT> jd{operations.getTotalNumParams(),
                                        sv.getLength(),
                                        sv.getData(),
                                        {},
                                        operations,
                                        trainableParams};

    const auto buffer = dy.request();

    calculate_vjp(
        std::span{vjp}, jd,
        std::span{static_cast<const std::complex<PrecisionT> *>(buffer.ptr),
                  static_cast<size_t>(buffer.size)});

    return py::array_t<std::complex<PrecisionT>>(py::cast(vjp));
}

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithms(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ParamT = PrecisionT;             // Parameter's data precision

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    std::string class_name;

    //***********************************************************************//
    //                        Vector Jacobian Product
    //***********************************************************************//
    class_name = "VectorJacobianProductC" + bitsize;
    py::class_<VectorJacobianProduct<StateVectorT>>(m, class_name.c_str(),
                                                    py::module_local())
        .def(py::init<>())
        .def("__call__", &registerVJP<StateVectorT, np_arr_c>,
             "Vector Jacobian Product method.");
}

/**
 * @brief Provide backend information.
 */
auto getBackendInfo() -> py::dict {
    using namespace py::literals;

    return py::dict("NAME"_a = "lightning.qubit");
}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Pybind11 module.
 */
void registerBackendSpecificInfo(py::module_ &m) {
    m.def("backend_info", &getBackendInfo, "Backend-specific information.");
}

} // namespace Pennylane::LightningQubit
