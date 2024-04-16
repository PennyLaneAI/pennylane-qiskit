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

#ifdef _ENABLE_PLQUBIT

#include "AdjointJacobianLQubit.hpp"
#include "LQubitBindings.hpp" // StateVectorBackends, registerBackendClassSpecificBindings, registerBackendSpecificMeasurements, registerBackendSpecificAlgorithms
#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Algorithms;
using namespace Pennylane::LightningQubit::Observables;
using namespace Pennylane::LightningQubit::Measures;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1

#include "AdjointJacobianKokkos.hpp"
#include "LKokkosBindings.hpp" // StateVectorBackends, registerBackendClassSpecificBindings, registerBackendSpecificMeasurements, registerBackendSpecificAlgorithms
#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::LightningKokkos::Measures;
} // namespace
  /// @endcond

#elif _ENABLE_PLGPU == 1
#include "AdjointJacobianGPU.hpp"
#include "LGPUBindings.hpp"
#include "MeasurementsGPU.hpp"
#include "ObservablesGPU.hpp"

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

/// @cond DEV
namespace {
using Pennylane::Util::bestCPUMemoryModel;
using Pennylane::Util::CPUMemoryModel;
using Pennylane::Util::getMemoryModel;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane {
/**
 * @brief Create a State Vector From a 1D Numpy Data object.
 *
 * @tparam StateVectorT
 * @param numpyArray inout data
 * @return StateVectorT
 */
template <class StateVectorT>
auto createStateVectorFromNumpyData(
    const py::array_t<std::complex<typename StateVectorT::PrecisionT>>
        &numpyArray) -> StateVectorT {
    using ComplexT = typename StateVectorT::ComplexT;
    py::buffer_info numpyArrayInfo = numpyArray.request();
    if (numpyArrayInfo.ndim != 1) {
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    }
    if (numpyArrayInfo.itemsize != sizeof(ComplexT)) {
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    }
    auto *data_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
    return StateVectorT(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.shape[0])});
}

/**
 * @brief Get memory alignment of a given numpy array.
 *
 * @param numpyArray Pybind11's numpy array type.
 * @return CPUMemoryModel Memory model describing alignment
 */
auto getNumpyArrayAlignment(const py::array &numpyArray) -> CPUMemoryModel {
    return getMemoryModel(numpyArray.request().ptr);
}

/**
 * @brief Create an aligned numpy array for a given type, memory model and array
 * size.
 *
 * @tparam T Datatype of numpy array to create
 * @param memory_model Memory model to use
 * @param size Size of the array to create
 * @return Numpy array
 */
template <typename T>
auto alignedNumpyArray(CPUMemoryModel memory_model, size_t size,
                       bool zeroInit = false) -> py::array {
    using Pennylane::Util::alignedAlloc;
    if (getAlignment<T>(memory_model) > alignof(std::max_align_t)) {
        void *ptr = alignedAlloc(getAlignment<T>(memory_model),
                                 sizeof(T) * size, zeroInit);
        auto capsule = py::capsule(ptr, &Util::alignedFree);
        return py::array{py::dtype::of<T>(), {size}, {sizeof(T)}, ptr, capsule};
    }
    void *ptr = static_cast<void *>(new T[size]);
    auto capsule =
        py::capsule(ptr, [](void *p) { delete static_cast<T *>(p); });
    return py::array{py::dtype::of<T>(), {size}, {sizeof(T)}, ptr, capsule};
}
/**
 * @brief Create a numpy array whose underlying data is allocated by
 * lightning.
 *
 * See https://github.com/pybind/pybind11/issues/1042#issuecomment-325941022
 * for capsule usage.
 *
 * @param size Size of the array to create
 * @param dt Pybind11's datatype object
 */
auto allocateAlignedArray(size_t size, const py::dtype &dt,
                          bool zeroInit = false) -> py::array {
    // TODO: Move memset operations to here to reduce zeroInit pass-throughs.
    auto memory_model = bestCPUMemoryModel();

    if (dt.is(py::dtype::of<float>())) {
        return alignedNumpyArray<float>(memory_model, size, zeroInit);
    }
    if (dt.is(py::dtype::of<double>())) {
        return alignedNumpyArray<double>(memory_model, size, zeroInit);
    }
    if (dt.is(py::dtype::of<std::complex<float>>())) {
        return alignedNumpyArray<std::complex<float>>(memory_model, size,
                                                      zeroInit);
    }
    if (dt.is(py::dtype::of<std::complex<double>>())) {
        return alignedNumpyArray<std::complex<double>>(memory_model, size,
                                                       zeroInit);
    }
    throw py::type_error("Unsupported datatype.");
}

/**
 * @brief Register functionality for numpy array memory alignment.
 *
 * @param m Pybind module
 */
void registerArrayAlignmentBindings(py::module_ &m) {
    /* Add CPUMemoryModel enum class */
    py::enum_<CPUMemoryModel>(m, "CPUMemoryModel", py::module_local())
        .value("Unaligned", CPUMemoryModel::Unaligned)
        .value("Aligned256", CPUMemoryModel::Aligned256)
        .value("Aligned512", CPUMemoryModel::Aligned512);

    /* Add array alignment functionality */
    m.def("get_alignment", &getNumpyArrayAlignment,
          "Get alignment of an underlying data for a numpy array.");
    m.def("allocate_aligned_array", &allocateAlignedArray,
          "Get numpy array whose underlying data is aligned.");
    m.def("best_alignment", &bestCPUMemoryModel,
          "Best memory alignment. for the simulator.");
}

/**
 * @brief Return basic information of the compiled binary.
 */
auto getCompileInfo() -> py::dict {
    using namespace Pennylane::Util;
    using namespace py::literals;

    const std::string_view cpu_arch_str = [] {
        switch (cpu_arch) {
        case CPUArch::X86_64:
            return "x86_64";
        case CPUArch::PPC64:
            return "PPC64";
        case CPUArch::ARM:
            return "ARM";
        default:
            return "Unknown";
        }
    }();

    const std::string_view compiler_name_str = [] {
        switch (compiler) {
        case Compiler::GCC:
            return "GCC";
        case Compiler::Clang:
            return "Clang";
        case Compiler::MSVC:
            return "MSVC";
        case Compiler::NVCC:
            return "NVCC";
        case Compiler::NVHPC:
            return "NVHPC";
        default:
            return "Unknown";
        }
    }();

    const auto compiler_version_str = getCompilerVersion<compiler>();

    return py::dict("cpu.arch"_a = cpu_arch_str,
                    "compiler.name"_a = compiler_name_str,
                    "compiler.version"_a = compiler_version_str,
                    "AVX2"_a = use_avx2, "AVX512F"_a = use_avx512f);
}

/**
 * @brief Return basic information of runtime environment.
 */
auto getRuntimeInfo() -> py::dict {
    using Pennylane::Util::RuntimeInfo;
    using namespace py::literals;

    return py::dict("AVX"_a = RuntimeInfo::AVX(),
                    "AVX2"_a = RuntimeInfo::AVX2(),
                    "AVX512F"_a = RuntimeInfo::AVX512F());
}

/**
 * @brief Register bindings for general info.
 *
 * @param m Pybind11 module.
 */
void registerInfo(py::module_ &m) {
    /* Add compile info */
    m.def("compile_info", &getCompileInfo, "Compiled binary information.");

    /* Add runtime info */
    m.def("runtime_info", &getRuntimeInfo, "Runtime information.");
}

/**
 * @brief Register observable classes.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendAgnosticObservables(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type.
    using ParamT = PrecisionT;           // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;
    using np_arr_r = py::array_t<ParamT, py::array::c_style>;

    std::string class_name;

    class_name = "ObservableC" + bitsize;
    py::class_<Observable<StateVectorT>,
               std::shared_ptr<Observable<StateVectorT>>>(m, class_name.c_str(),
                                                          py::module_local());

    class_name = "NamedObsC" + bitsize;
    py::class_<NamedObs<StateVectorT>, std::shared_ptr<NamedObs<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const std::string &name, const std::vector<size_t> &wires) {
                return NamedObs<StateVectorT>(name, wires);
            }))
        .def("__repr__", &NamedObs<StateVectorT>::getObsName)
        .def("get_wires", &NamedObs<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObs<StateVectorT> &self, py::handle other) -> bool {
                if (!py::isinstance<NamedObs<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<NamedObs<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HermitianObsC" + bitsize;
    py::class_<HermitianObs<StateVectorT>,
               std::shared_ptr<HermitianObs<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const np_arr_c &matrix, const std::vector<size_t> &wires) {
                auto buffer = matrix.request();
                const auto *ptr = static_cast<ComplexT *>(buffer.ptr);
                return HermitianObs<StateVectorT>(
                    std::vector<ComplexT>(ptr, ptr + buffer.size), wires);
            }))
        .def("__repr__", &HermitianObs<StateVectorT>::getObsName)
        .def("get_wires", &HermitianObs<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HermitianObs<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HermitianObs<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HermitianObs<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "TensorProdObsC" + bitsize;
    py::class_<TensorProdObs<StateVectorT>,
               std::shared_ptr<TensorProdObs<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const std::vector<std::shared_ptr<Observable<StateVectorT>>>
                   &obs) { return TensorProdObs<StateVectorT>(obs); }))
        .def("__repr__", &TensorProdObs<StateVectorT>::getObsName)
        .def("get_wires", &TensorProdObs<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const TensorProdObs<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<TensorProdObs<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<TensorProdObs<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HamiltonianC" + bitsize;
    using ObsPtr = std::shared_ptr<Observable<StateVectorT>>;
    py::class_<Hamiltonian<StateVectorT>,
               std::shared_ptr<Hamiltonian<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const np_arr_r &coeffs, const std::vector<ObsPtr> &obs) {
                auto buffer = coeffs.request();
                const auto ptr = static_cast<const ParamT *>(buffer.ptr);
                return Hamiltonian<StateVectorT>{
                    std::vector(ptr, ptr + buffer.size), obs};
            }))
        .def("__repr__", &Hamiltonian<StateVectorT>::getObsName)
        .def("get_wires", &Hamiltonian<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const Hamiltonian<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<Hamiltonian<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<Hamiltonian<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");
}

/**
 * @brief Register agnostic measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Pybind11's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendAgnosticMeasurements(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ParamT = PrecisionT;             // Parameter's data precision

    pyclass
        .def("probs",
             [](Measurements<StateVectorT> &M,
                const std::vector<size_t> &wires) {
                 return py::array_t<ParamT>(py::cast(M.probs(wires)));
             })
        .def("probs",
             [](Measurements<StateVectorT> &M) {
                 return py::array_t<ParamT>(py::cast(M.probs()));
             })
        .def(
            "expval",
            [](Measurements<StateVectorT> &M,
               const std::shared_ptr<Observable<StateVectorT>> &ob) {
                return M.expval(*ob);
            },
            "Expected value of an observable object.")
        .def(
            "var",
            [](Measurements<StateVectorT> &M,
               const std::shared_ptr<Observable<StateVectorT>> &ob) {
                return M.var(*ob);
            },
            "Variance of an observable object.")
        .def("generate_samples", [](Measurements<StateVectorT> &M,
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
auto registerAdjointJacobian(
    AdjointJacobian<StateVectorT> &adjoint_jacobian, const StateVectorT &sv,
    const std::vector<std::shared_ptr<Observable<StateVectorT>>> &observables,
    const OpsData<StateVectorT> &operations,
    const std::vector<size_t> &trainableParams)
    -> py::array_t<typename StateVectorT::PrecisionT> {
    using PrecisionT = typename StateVectorT::PrecisionT;
    std::vector<PrecisionT> jac(observables.size() * trainableParams.size(),
                                PrecisionT{0.0});
    const JacobianData<StateVectorT> jd{operations.getTotalNumParams(),
                                        sv.getLength(),
                                        sv.getData(),
                                        observables,
                                        operations,
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
void registerBackendAgnosticAlgorithms(py::module_ &m) {
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

    class_name = "OpsStructC" + bitsize;
    py::class_<OpsData<StateVectorT>>(m, class_name.c_str(), py::module_local())
        .def(py::init<const std::vector<std::string> &,
                      const std::vector<std::vector<ParamT>> &,
                      const std::vector<std::vector<size_t>> &,
                      const std::vector<bool> &,
                      const std::vector<std::vector<ComplexT>> &>())
        .def(py::init<const std::vector<std::string> &,
                      const std::vector<std::vector<ParamT>> &,
                      const std::vector<std::vector<size_t>> &,
                      const std::vector<bool> &,
                      const std::vector<std::vector<ComplexT>> &,
                      const std::vector<std::vector<size_t>> &,
                      const std::vector<std::vector<bool>> &>())
        .def("__repr__", [](const OpsData<StateVectorT> &ops) {
            using namespace Pennylane::Util;
            std::ostringstream ops_stream;
            for (size_t op = 0; op < ops.getSize(); op++) {
                ops_stream << "{'name': " << ops.getOpsName()[op];
                ops_stream << ", 'params': " << ops.getOpsParams()[op];
                ops_stream << ", 'inv': " << ops.getOpsInverses()[op];
                ops_stream << ", 'controlled_wires': "
                           << ops.getOpsControlledWires()[op];
                ops_stream << ", 'controlled_values': "
                           << ops.getOpsControlledValues()[op];
                ops_stream << ", 'wires': " << ops.getOpsWires()[op];
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
    std::string function_name = "create_ops_listC" + bitsize;
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
    class_name = "AdjointJacobianC" + bitsize;
    py::class_<AdjointJacobian<StateVectorT>>(m, class_name.c_str(),
                                              py::module_local())
        .def(py::init<>())
#ifdef _ENABLE_PLGPU
        .def(
            "batched",
            [](AdjointJacobian<StateVectorT> &adjoint_jacobian,
               const StateVectorT &sv,
               const std::vector<std::shared_ptr<Observable<StateVectorT>>>
                   &observables,
               const OpsData<StateVectorT> &operations,
               const std::vector<size_t> &trainableParams) {
                using PrecisionT = typename StateVectorT::PrecisionT;
                std::vector<PrecisionT> jac(observables.size() *
                                                trainableParams.size(),
                                            PrecisionT{0.0});
                const JacobianData<StateVectorT> jd{
                    operations.getTotalNumParams(),
                    sv.getLength(),
                    sv.getData(),
                    observables,
                    operations,
                    trainableParams};
                adjoint_jacobian.batchAdjointJacobian(std::span{jac}, jd);
                return py::array_t<PrecisionT>(py::cast(jac));
            },
            "Batch Adjoint Jacobian method.")
#endif
        .def("__call__", &registerAdjointJacobian<StateVectorT>,
             "Adjoint Jacobian method.");
}

/**
 * @brief Templated class to build lightning class bindings.
 *
 * @tparam StateVectorT State vector type
 * @param m Pybind11 module.
 */
template <class StateVectorT> void lightningClassBindings(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    //***********************************************************************//
    //                              StateVector
    //***********************************************************************//
    std::string class_name = "StateVectorC" + bitsize;
    auto pyclass =
        py::class_<StateVectorT>(m, class_name.c_str(), py::module_local());
    pyclass.def(py::init(&createStateVectorFromNumpyData<StateVectorT>))
        .def_property_readonly("size", &StateVectorT::getLength);

    registerBackendClassSpecificBindings<StateVectorT>(pyclass);

    //***********************************************************************//
    //                              Observables
    //***********************************************************************//
    /* Observables submodule */
    py::module_ obs_submodule =
        m.def_submodule("observables", "Submodule for observables classes.");
    registerBackendAgnosticObservables<StateVectorT>(obs_submodule);
    registerBackendSpecificObservables<StateVectorT>(obs_submodule);

    //***********************************************************************//
    //                             Measurements
    //***********************************************************************//
    class_name = "MeasurementsC" + bitsize;
    auto pyclass_measurements = py::class_<Measurements<StateVectorT>>(
        m, class_name.c_str(), py::module_local());

#ifdef _ENABLE_PLGPU
    pyclass_measurements.def(py::init<StateVectorT &>());
#else
    pyclass_measurements.def(py::init<const StateVectorT &>());
#endif
    registerBackendAgnosticMeasurements<StateVectorT>(pyclass_measurements);
    registerBackendSpecificMeasurements<StateVectorT>(pyclass_measurements);

    //***********************************************************************//
    //                           Algorithms
    //***********************************************************************//
    /* Algorithms submodule */
    py::module_ alg_submodule = m.def_submodule(
        "algorithms", "Submodule for the algorithms functionality.");
    registerBackendAgnosticAlgorithms<StateVectorT>(alg_submodule);
    registerBackendSpecificAlgorithms<StateVectorT>(alg_submodule);
}

template <typename TypeList>
void registerLightningClassBindings(py::module_ &m) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        lightningClassBindings<StateVectorT>(m);
        registerLightningClassBindings<typename TypeList::Next>(m);
        py::register_local_exception<Pennylane::Util::LightningException>(
            m, "LightningException");
    }
}
} // namespace Pennylane
