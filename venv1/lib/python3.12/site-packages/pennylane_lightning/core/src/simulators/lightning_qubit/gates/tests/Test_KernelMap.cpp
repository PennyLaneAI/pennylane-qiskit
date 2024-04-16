// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"           // LightningException
#include "IntegerInterval.hpp" // IntegerInterval, full_domain
#include "KernelMap.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp" // for_each_enum

#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::KernelMap;
using namespace Pennylane::Util;

using Catch::Matchers::Contains;
using Pennylane::Util::for_each_enum;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEST_CASE("Test PriorityDispatchSet", "[PriorityDispatchSet]") {
    auto pds = PriorityDispatchSet();
    pds.emplace(Pennylane::Gates::KernelType::PI, 10U,
                Util::IntegerInterval<size_t>(10, 20));

    SECTION("Test conflict") {
        /* If two elements has the same priority but integer intervals overlap,
         * they conflict. */
        REQUIRE(pds.conflict(10U, Util::IntegerInterval<size_t>(19, 23)));
    }

    SECTION("Get Kernel") {
        REQUIRE(pds.getKernel(15) == Pennylane::Gates::KernelType::PI);
        PL_CHECK_THROWS_MATCHES(pds.getKernel(30), LightningException,
                                "Cannot find a kernel");
    }
}

TEST_CASE("Test default kernels for gates are well defined", "[KernelMap]") {
    auto &instance =
        OperationKernelMap<Pennylane::Gates::GateOperation>::getInstance();
    for_each_enum<Threading, CPUMemoryModel>(
        [&instance](Threading threading, CPUMemoryModel memory_model) {
            for (size_t num_qubits = 1; num_qubits < 27; num_qubits++) {
                REQUIRE_NOTHROW(
                    instance.getKernelMap(num_qubits, threading, memory_model));
            }
        });
}

TEST_CASE("Test default kernels for generators are well defined",
          "[KernelMap]") {
    auto &instance =
        OperationKernelMap<Pennylane::Gates::GeneratorOperation>::getInstance();
    for_each_enum<Threading, CPUMemoryModel>(
        [&instance](Threading threading, CPUMemoryModel memory_model) {
            for (size_t num_qubits = 1; num_qubits < 27; num_qubits++) {
                REQUIRE_NOTHROW(
                    instance.getKernelMap(num_qubits, threading, memory_model));
            }
        });
}

TEST_CASE("Test default kernels for matrix operation are well defined",
          "[KernelMap]") {
    auto &instance =
        OperationKernelMap<Pennylane::Gates::MatrixOperation>::getInstance();
    for_each_enum<Threading, CPUMemoryModel>(
        [&instance](Threading threading, CPUMemoryModel memory_model) {
            for (size_t num_qubits = 1; num_qubits < 27; num_qubits++) {
                REQUIRE_NOTHROW(
                    instance.getKernelMap(num_qubits, threading, memory_model));
            }
        });
}

TEST_CASE("Test unallowed kernel", "[KernelMap]") {
    using Pennylane::Gates::GateOperation;
    using Pennylane::Gates::KernelType;
    auto &instance =
        OperationKernelMap<Pennylane::Gates::GateOperation>::getInstance();
    REQUIRE_THROWS(instance.assignKernelForOp(
        Pennylane::Gates::GateOperation::PauliX, Threading::SingleThread,
        CPUMemoryModel::Unaligned, 0, Util::full_domain<size_t>(),
        KernelType::None));

    REQUIRE_THROWS(instance.assignKernelForOp(
        Pennylane::Gates::GateOperation::PauliX, Threading::SingleThread,
        CPUMemoryModel::Unaligned, 0, Util::full_domain<size_t>(),
        KernelType::AVX2));
}

TEST_CASE("Test several limiting cases of default kernels", "[KernelMap]") {
    auto &instance =
        OperationKernelMap<Pennylane::Gates::GateOperation>::getInstance();
    SECTION("Single thread, large number of qubits") {
        // For large N, single thread calls "LM" for all single- and two-qubit
        // gates.
        auto gate_map = instance.getKernelMap(28, Threading::SingleThread,
                                              CPUMemoryModel::Unaligned);
        for_each_enum<Pennylane::Gates::GateOperation>(
            [&gate_map](Pennylane::Gates::GateOperation gate_op) {
                INFO(lookup(Pennylane::Gates::Constant::gate_names, gate_op));
                if (gate_op == Pennylane::Gates::GateOperation::GlobalPhase ||
                    gate_op == Pennylane::Gates::GateOperation::MultiRZ) {
                    REQUIRE(gate_map[gate_op] ==
                            Pennylane::Gates::KernelType::LM);
                } else if (lookup(Pennylane::Gates::Constant::gate_wires,
                                  gate_op) <= 2) {
                    REQUIRE(gate_map[gate_op] ==
                            Pennylane::Gates::KernelType::LM);
                }
            });
    }
}

TEST_CASE("Test KernelMap functionalities", "[KernelMap]") {
    using Pennylane::Gates::GateOperation;
    using Pennylane::Gates::KernelType;
    auto &instance =
        OperationKernelMap<Pennylane::Gates::GateOperation>::getInstance();

    SECTION("Test priority works") {
        auto original_kernel = instance.getKernelMap(
            24, Threading::SingleThread,
            CPUMemoryModel::Unaligned)[Pennylane::Gates::GateOperation::PauliX];

        instance.assignKernelForOp(Pennylane::Gates::GateOperation::PauliX,
                                   Threading::SingleThread,
                                   CPUMemoryModel::Unaligned, 100,
                                   Util::full_domain<size_t>(), KernelType::PI);

        REQUIRE(instance.getKernelMap(24, Threading::SingleThread,
                                      CPUMemoryModel::Unaligned)
                    [Pennylane::Gates::GateOperation::PauliX] ==
                KernelType::PI);

        instance.removeKernelForOp(Pennylane::Gates::GateOperation::PauliX,
                                   Threading::SingleThread,
                                   CPUMemoryModel::Unaligned, 100);
        REQUIRE(instance.getKernelMap(24, Threading::SingleThread,
                                      CPUMemoryModel::Unaligned)
                    [Pennylane::Gates::GateOperation::PauliX] ==
                original_kernel);
    }
    SECTION("Test remove non-existing element") {
        PL_CHECK_THROWS_MATCHES(
            instance.removeKernelForOp(Pennylane::Gates::GateOperation::PauliX,
                                       Threading::END,
                                       CPUMemoryModel::Unaligned, 100),
            LightningException, "does not exist");
    }
}

TEST_CASE("Test KernelMap is consistent in extreme usecase", "[KernelMap]") {
    using Pennylane::Gates::GateOperation;
    using Pennylane::Gates::KernelType;
    using EnumKernelMap =
        OperationKernelMap<Pennylane::Gates::GateOperation>::EnumKernelMap;
    auto &instance =
        OperationKernelMap<Pennylane::Gates::GateOperation>::getInstance();

    const auto num_qubits = std::vector<size_t>{4, 6, 8, 10, 12, 14, 16};
    const auto threadings =
        std::vector<Threading>{Threading::SingleThread, Threading::MultiThread};
    const auto memory_models = std::vector<CPUMemoryModel>{
        CPUMemoryModel::Unaligned, CPUMemoryModel::Aligned256,
        CPUMemoryModel::Aligned512};

    std::random_device rd;

    std::vector<EnumKernelMap> records;

    records.push_back(instance.getKernelMap(12, Threading::SingleThread,
                                            CPUMemoryModel::Aligned256));

    constexpr size_t num_iter = 8096;

#ifdef _OPENMP
#pragma omp parallel default(none)                                             \
    shared(instance, records, rd, num_qubits, threadings, memory_models)       \
    firstprivate(num_iter)
#endif
    {
        std::mt19937 re;

#ifdef _OPENMP
#pragma omp critical
#endif
        { re.seed(rd()); }

        std::uniform_int_distribution<size_t> num_qubit_dist(
            0, num_qubits.size() - 1);
        std::uniform_int_distribution<size_t> threading_dist(
            0, threadings.size() - 1);
        std::uniform_int_distribution<size_t> memory_model_dist(
            0, memory_models.size() - 1);

        std::vector<EnumKernelMap> res;

#ifdef _OPENMP
#pragma omp for
#endif
        for (size_t i = 0; i < num_iter; i++) {
            const auto num_qubit = num_qubits[num_qubit_dist(re)];
            const auto threading = threadings[threading_dist(re)];
            const auto memory_model = memory_models[memory_model_dist(re)];

            res.push_back(
                instance.getKernelMap(num_qubit, threading, memory_model));
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        { records.insert(records.end(), res.begin(), res.end()); }
    }
    records.push_back(instance.getKernelMap(12, Threading::SingleThread,
                                            CPUMemoryModel::Aligned256));

    REQUIRE(records.front() == records.back());
}
