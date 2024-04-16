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
#include "MeasurementsGPU.hpp"
#include "ObservablesGPU.hpp"
#include "StateVectorCudaManaged.hpp"
#include "TypeList.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Bindings;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
using Pennylane::LightningGPU::StateVectorCudaManaged;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningGPU {
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorCudaManaged<float>,
                              StateVectorCudaManaged<double>, void>;

/**
 * @brief Get a gate kernel map for a statevector.
 */

template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
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
        .def(py::init<std::size_t>())              // qubits, device
        .def(py::init<std::size_t, DevTag<int>>()) // qubits, dev-tag
        .def(py::init([](const np_arr_c &arr) {
            py::buffer_info numpyArrayInfo = arr.request();
            const auto *data_ptr =
                static_cast<const std::complex<PrecisionT> *>(
                    numpyArrayInfo.ptr);
            return new StateVectorT(data_ptr,
                                    static_cast<std::size_t>(arr.size()));
        }))
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
               const std::vector<std::vector<ParamT>> &params,
               const np_arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<CFP_t> matrix_cu;
                if (m_buffer.size) {
                    const auto m_ptr = static_cast<const CFP_t *>(m_buffer.ptr);
                    matrix_cu =
                        std::vector<CFP_t>{m_ptr, m_ptr + m_buffer.size};
                }
                if (params.empty()) {
                    sv.applyOperation(str, wires, inv, std::vector<ParamT>{},
                                      matrix_cu);
                } else {
                    PL_ABORT_IF(params.size() != 1,
                                "params should be a List[List[float]].")
                    sv.applyOperation(str, wires, inv, params[0], matrix_cu);
                }
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
    using sparse_index_type =
        typename std::conditional<std::is_same<ParamT, float>::value, int32_t,
                                  int64_t>::type;
    using np_arr_sparse_ind = typename std::conditional<
        std::is_same<ParamT, float>::value,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast>>::type;

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
                    static_cast<int64_t>(
                        row_map.request()
                            .size), // int64_t is required by cusparse
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<ComplexT *>(values.request().ptr),
                    static_cast<int64_t>(
                        values.request()
                            .size)); // int64_t is required by cusparse
            },
            "Expected value of a sparse Hamiltonian.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M,
               const std::vector<std::string> &pauli_words,
               const std::vector<std::vector<size_t>> &target_wires,
               const np_arr_c &coeffs) {
                return M.expval(pauli_words, target_wires,
                                static_cast<ComplexT *>(coeffs.request().ptr));
            },
            "Expected value of Hamiltonian represented by Pauli words.")
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
                    static_cast<int64_t>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<ComplexT *>(values.request().ptr),
                    static_cast<int64_t>(values.request().size));
            },
            "Variance of a sparse Hamiltonian.");
}

/**
 * @brief Register backend specific observables.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificObservables(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type.
    using ParamT = PrecisionT;           // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;

    std::string class_name;

    class_name = "SparseHamiltonianC" + bitsize;
    using np_arr_sparse_ind = typename std::conditional<
        std::is_same<ParamT, float>::value,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
        py::array_t<int64_t, py::array::c_style | py::array::forcecast>>::type;
    using IdxT = typename SparseHamiltonian<StateVectorT>::IdxT;
    py::class_<SparseHamiltonian<StateVectorT>,
               std::shared_ptr<SparseHamiltonian<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init([](const np_arr_c &data, const np_arr_sparse_ind &indices,
                         const np_arr_sparse_ind &offsets,
                         const std::vector<std::size_t> &wires) {
            const py::buffer_info buffer_data = data.request();
            const auto *data_ptr = static_cast<ComplexT *>(buffer_data.ptr);

            const py::buffer_info buffer_indices = indices.request();
            const auto *indices_ptr =
                static_cast<std::size_t *>(buffer_indices.ptr);

            const py::buffer_info buffer_offsets = offsets.request();
            const auto *offsets_ptr =
                static_cast<std::size_t *>(buffer_offsets.ptr);

            return SparseHamiltonian<StateVectorT>{
                std::vector<ComplexT>({data_ptr, data_ptr + data.size()}),
                std::vector<IdxT>({indices_ptr, indices_ptr + indices.size()}),
                std::vector<IdxT>({offsets_ptr, offsets_ptr + offsets.size()}),
                wires};
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

    return py::dict("NAME"_a = "lightning.gpu");
}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Pybind11 module.
 */
void registerBackendSpecificInfo(py::module_ &m) {
    m.def("backend_info", &getBackendInfo, "Backend-specific information.");
    m.def("device_reset", &deviceReset, "Reset all GPU devices and contexts.");
    m.def("allToAllAccess", []() {
        for (int i = 0; i < static_cast<int>(getGPUCount()); i++) {
            cudaDeviceEnablePeerAccess(i, 0);
        }
    });

    m.def("is_gpu_supported", &isCuQuantumSupported,
          py::arg("device_number") = 0,
          "Checks if the given GPU device meets the minimum architecture "
          "support for the PennyLane-Lightning-GPU device.");

    m.def("get_gpu_arch", &getGPUArch, py::arg("device_number") = 0,
          "Returns the given GPU major and minor GPU support.");
    py::class_<DevicePool<int>>(m, "DevPool")
        .def(py::init<>())
        .def("getActiveDevices", &DevicePool<int>::getActiveDevices)
        .def("isActive", &DevicePool<int>::isActive)
        .def("isInactive", &DevicePool<int>::isInactive)
        .def("acquireDevice", &DevicePool<int>::acquireDevice)
        .def("releaseDevice", &DevicePool<int>::releaseDevice)
        .def("syncDevice", &DevicePool<int>::syncDevice)
        .def_static("getTotalDevices", &DevicePool<int>::getTotalDevices)
        .def_static("getDeviceUIDs", &DevicePool<int>::getDeviceUIDs)
        .def_static("setDeviceID", &DevicePool<int>::setDeviceIdx);

    py::class_<DevTag<int>>(m, "DevTag")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init([](int device_id, void *stream_id) {
            // Note, streams must be handled externally for now.
            // Binding support provided through void* conversion to cudaStream_t
            return new DevTag<int>(device_id,
                                   static_cast<cudaStream_t>(stream_id));
        }))
        .def(py::init<const DevTag<int> &>())
        .def("getDeviceID", &DevTag<int>::getDeviceID)
        .def("getStreamID",
             [](DevTag<int> &dev_tag) {
                 // default stream points to nullptr, so just return void* as
                 // type
                 return static_cast<void *>(dev_tag.getStreamID());
             })
        .def("refresh", &DevTag<int>::refresh);
}

} // namespace Pennylane::LightningGPU
  /// @endcond
