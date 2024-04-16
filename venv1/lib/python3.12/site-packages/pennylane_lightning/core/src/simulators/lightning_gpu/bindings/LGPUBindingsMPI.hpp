// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
#include <tuple>
#include <variant>
#include <vector>

#include "cuda.h"

#include "BindingsBase.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "Error.hpp"
#include "MPIManager.hpp"
#include "MeasurementsGPUMPI.hpp"
#include "ObservablesGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "TypeList.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Bindings;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
using Pennylane::LightningGPU::StateVectorCudaMPI;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningGPU {
using StateVectorMPIBackends =
    Pennylane::Util::TypeList<StateVectorCudaMPI<float>,
                              StateVectorCudaMPI<double>, void>;

/**
 * @brief Get a gate kernel map for a statevector.
 */

template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindingsMPI(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using CFP_t =
        typename StateVectorT::CFP_t; // Statevector's complex precision
    using ParamT = PrecisionT;        // Parameter's data precision
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    using np_arr_sparse_ind = typename std::conditional<
        std::is_same<ParamT, float>::value,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast>>::type;

    registerGatesForStateVector<StateVectorT>(pyclass);

    pyclass
        .def(
            py::init([](MPIManager &mpi_manager, const DevTag<int> devtag_local,
                        std::size_t mpi_buf_size, std::size_t num_global_qubits,
                        std::size_t num_local_qubits) {
                return new StateVectorT(mpi_manager, devtag_local, mpi_buf_size,
                                        num_global_qubits, num_local_qubits);
            })) // qubits, device
        .def(py::init(
            [](const DevTag<int> devtag_local, std::size_t mpi_buf_size,
               std::size_t num_global_qubits, std::size_t num_local_qubits) {
                return new StateVectorT(devtag_local, mpi_buf_size,
                                        num_global_qubits, num_local_qubits);
            })) // qubits, device
        .def(
            "setBasisState",
            [](StateVectorT &sv, const size_t index, const bool use_async) {
                const std::complex<PrecisionT> value(1, 0);
                sv.setBasisState(value, index, use_async);
            },
            "Create Basis State on GPU.")
        .def(
            "setStateVector",
            [](StateVectorT &sv, const np_arr_sparse_ind &indices,
               const np_arr_c &state, const bool use_async) {
                using index_type = typename std::conditional<
                    std::is_same<ParamT, float>::value, int32_t, int64_t>::type;

                sv.template setStateVector<index_type>(
                    static_cast<index_type>(indices.request().size),
                    static_cast<std::complex<PrecisionT> *>(
                        state.request().ptr),
                    static_cast<index_type *>(indices.request().ptr),
                    use_async);
            },
            "Set State Vector on GPU with values and their corresponding "
            "indices for the state vector on device")
        .def(
            "DeviceToDevice",
            [](StateVectorT &sv, const StateVectorT &other, bool async) {
                sv.updateData(other, async);
            },
            "Synchronize data from another GPU device to current device.")
        .def("DeviceToHost",
             py::overload_cast<std::complex<PrecisionT> *, size_t, bool>(
                 &StateVectorT::CopyGpuDataToHost, py::const_),
             "Synchronize data from the GPU device to host.")
        .def(
            "DeviceToHost",
            [](const StateVectorT &gpu_sv, np_arr_c &cpu_sv, bool) {
                py::buffer_info numpyArrayInfo = cpu_sv.request();
                auto *data_ptr =
                    static_cast<std::complex<PrecisionT> *>(numpyArrayInfo.ptr);
                if (cpu_sv.size()) {
                    gpu_sv.CopyGpuDataToHost(data_ptr, cpu_sv.size());
                }
            },
            "Synchronize data from the GPU device to host.")
        .def("HostToDevice",
             py::overload_cast<const std::complex<PrecisionT> *, size_t, bool>(
                 &StateVectorT::CopyHostDataToGpu),
             "Synchronize data from the host device to GPU.")
        .def("HostToDevice",
             py::overload_cast<const std::vector<std::complex<PrecisionT>> &,
                               bool>(&StateVectorT::CopyHostDataToGpu),
             "Synchronize data from the host device to GPU.")
        .def(
            "HostToDevice",
            [](StateVectorT &gpu_sv, const np_arr_c &cpu_sv, bool async) {
                const py::buffer_info numpyArrayInfo = cpu_sv.request();
                const auto *data_ptr =
                    static_cast<std::complex<PrecisionT> *>(numpyArrayInfo.ptr);
                const auto length =
                    static_cast<size_t>(numpyArrayInfo.shape[0]);
                if (length) {
                    gpu_sv.CopyHostDataToGpu(data_ptr, length, async);
                }
            },
            "Synchronize data from the host device to GPU.")
        .def("GetNumGPUs", &getGPUCount, "Get the number of available GPUs.")
        .def("getCurrentGPU", &getGPUIdx,
             "Get the GPU index for the statevector data.")
        .def("numQubits", &StateVectorT::getNumQubits)
        .def("dataLength", &StateVectorT::getLength)
        .def("resetGPU", &StateVectorT::initSV)
        .def(
            "apply",
            [](StateVectorT &sv, const std::string &str,
               const std::vector<size_t> &wires, bool inv,
               [[maybe_unused]] const std::vector<std::vector<ParamT>> &params,
               [[maybe_unused]] const np_arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<CFP_t> matrix_cu;
                if (m_buffer.size) {
                    const auto m_ptr = static_cast<const CFP_t *>(m_buffer.ptr);
                    matrix_cu =
                        std::vector<CFP_t>{m_ptr, m_ptr + m_buffer.size};
                }
                sv.applyOperation(str, wires, inv, std::vector<ParamT>{},
                                  matrix_cu);
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
void registerBackendSpecificMeasurementsMPI(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type
    using ParamT = PrecisionT;           // Parameter's data precision

    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    using sparse_index_type =
        typename std::conditional<std::is_same<ParamT, float>::value, int32_t,
                                  int64_t>::type;
    using np_arr_sparse_ind = typename std::conditional<
        std::is_same<ParamT, float>::value,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast>>::type;

    pyclass
        .def("expval",
             static_cast<PrecisionT (MeasurementsMPI<StateVectorT>::*)(
                 const std::string &, const std::vector<size_t> &)>(
                 &MeasurementsMPI<StateVectorT>::expval),
             "Expected value of an operation by name.")
        .def(
            "expval",
            [](MeasurementsMPI<StateVectorT> &M,
               const np_arr_sparse_ind &row_map,
               const np_arr_sparse_ind &entries, const np_arr_c &values) {
                return M.expval(
                    static_cast<sparse_index_type *>(row_map.request().ptr),
                    static_cast<int64_t>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<ComplexT *>(values.request().ptr),
                    static_cast<int64_t>(values.request().size));
            },
            "Expected value of a sparse Hamiltonian.")
        .def(
            "expval",
            [](MeasurementsMPI<StateVectorT> &M,
               const std::vector<std::string> &pauli_words,
               const std::vector<std::vector<size_t>> &target_wires,
               const np_arr_c &coeffs) {
                return M.expval(pauli_words, target_wires,
                                static_cast<ComplexT *>(coeffs.request().ptr));
            },
            "Expected value of Hamiltonian represented by Pauli words.")
        .def(
            "expval",
            [](MeasurementsMPI<StateVectorT> &M, const np_arr_c &matrix,
               const std::vector<size_t> &wires) {
                const std::size_t matrix_size = exp2(2 * wires.size());
                auto matrix_data =
                    static_cast<ComplexT *>(matrix.request().ptr);
                std::vector<ComplexT> matrix_v{matrix_data,
                                               matrix_data + matrix_size};
                return M.expval(matrix_v, wires);
            },
            "Expected value of a Hermitian observable.")
        .def("var",
             [](MeasurementsMPI<StateVectorT> &M, const std::string &operation,
                const std::vector<size_t> &wires) {
                 return M.var(operation, wires);
             })
        .def("var",
             static_cast<PrecisionT (MeasurementsMPI<StateVectorT>::*)(
                 const std::string &, const std::vector<size_t> &)>(
                 &MeasurementsMPI<StateVectorT>::var),
             "Variance of an operation by name.")
        .def(
            "var",
            [](MeasurementsMPI<StateVectorT> &M,
               const np_arr_sparse_ind &row_map,
               const np_arr_sparse_ind &entries, const np_arr_c &values) {
                return M.var(
                    static_cast<sparse_index_type *>(row_map.request().ptr),
                    static_cast<int64_t>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<ComplexT *>(values.request().ptr),
                    static_cast<int64_t>(values.request().size));
            },
            "Variance of a sparse Hamiltonian.");
}

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithmsMPI([[maybe_unused]] py::module_ &m) {}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Pybind11 module.
 */
void registerBackendSpecificInfoMPI(py::module_ &m) {
    using np_arr_c64 = py::array_t<std::complex<float>,
                                   py::array::c_style | py::array::forcecast>;
    using np_arr_c128 = py::array_t<std::complex<double>,
                                    py::array::c_style | py::array::forcecast>;
    py::class_<MPIManager>(m, "MPIManager")
        .def(py::init<>())
        .def(py::init<MPIManager &>())
        .def("Barrier", &MPIManager::Barrier)
        .def("getRank", &MPIManager::getRank)
        .def("getSize", &MPIManager::getSize)
        .def("getSizeNode", &MPIManager::getSizeNode)
        .def("getTime", &MPIManager::getTime)
        .def("getVendor", &MPIManager::getVendor)
        .def("getVersion", &MPIManager::getVersion)
        .def(
            "Scatter",
            [](MPIManager &mpi_manager, np_arr_c64 &sendBuf,
               np_arr_c64 &recvBuf, int root) {
                auto send_ptr =
                    static_cast<std::complex<float> *>(sendBuf.request().ptr);
                auto recv_ptr =
                    static_cast<std::complex<float> *>(recvBuf.request().ptr);
                mpi_manager.template Scatter<std::complex<float>>(
                    send_ptr, recv_ptr, recvBuf.request().size, root);
            },
            "MPI Scatter.")
        .def(
            "Scatter",
            [](MPIManager &mpi_manager, np_arr_c128 &sendBuf,
               np_arr_c128 &recvBuf, int root) {
                auto send_ptr =
                    static_cast<std::complex<double> *>(sendBuf.request().ptr);
                auto recv_ptr =
                    static_cast<std::complex<double> *>(recvBuf.request().ptr);
                mpi_manager.template Scatter<std::complex<double>>(
                    send_ptr, recv_ptr, recvBuf.request().size, root);
            },
            "MPI Scatter.");
}
} // namespace Pennylane::LightningGPU
  /// @endcond