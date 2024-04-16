// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVectorCudaManaged.hpp"
#include "TestHelpers.hpp"

using namespace Pennylane::LightningGPU;
// NOTE: the scaling factors are implicitly included in the Adjoint Jacobian
// evaluation, so excluded from the matrices here.

TEST_CASE("Generators::applyGeneratorRX", "[GateGenerators]") {
    // grad(RX) = grad(e^{-i*0.5*PauliX*a}) => -i*0.5*PauliX
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        cuGates::getPauliX<typename StateVectorCudaManaged<double>::CFP_t>()};
    std::mt19937 re{1337U};
    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (std::size_t applied_qubit = 0; applied_qubit < num_qubits;
             applied_qubit++) {
            auto init_state =
                createRandomStateVectorData<double>(re, num_qubits);

            StateVectorCudaManaged<double> psi(init_state.data(),
                                               init_state.size());
            StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                      init_state.size());
            StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                        init_state.size());

            std::string cache_gate_name = "DirectGenRX" +
                                          std::to_string(applied_qubit) + "_" +
                                          std::to_string(num_qubits);

            psi.applyGeneratorRX({applied_qubit}, false);
            psi_direct.applyOperation(cache_gate_name, {applied_qubit}, false,
                                      {0.0}, matrix);
            psi_dispatch.applyGenerator({"RX"}, {applied_qubit}, false);

            CHECK(psi.getDataVector() == psi_direct.getDataVector());
            CHECK(psi_dispatch.getDataVector() == psi_direct.getDataVector());
        }
    }
}

TEST_CASE("Generators::applyGeneratorRY", "[GateGenerators]") {
    // grad(RY) = grad(e^{-i*0.5*PauliY*a}) => -i*0.5*PauliY
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        cuGates::getPauliY<typename StateVectorCudaManaged<double>::CFP_t>()};
    std::mt19937 re{1337U};
    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (std::size_t applied_qubit = 0; applied_qubit < num_qubits;
             applied_qubit++) {
            auto init_state =
                createRandomStateVectorData<double>(re, num_qubits);

            StateVectorCudaManaged<double> psi(init_state.data(),
                                               init_state.size());
            StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                      init_state.size());
            StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                        init_state.size());

            std::string cache_gate_name = "DirectGenRY" +
                                          std::to_string(applied_qubit) + "_" +
                                          std::to_string(num_qubits);

            psi.applyGeneratorRY({applied_qubit}, false);
            psi_direct.applyOperation(cache_gate_name, {applied_qubit}, false,
                                      {0.0}, matrix);
            psi_dispatch.applyGenerator({"RY"}, {applied_qubit}, false);

            CHECK(psi.getDataVector() == psi_direct.getDataVector());
            CHECK(psi_dispatch.getDataVector() == psi_direct.getDataVector());
        }
    }
}

TEST_CASE("Generators::applyGeneratorRZ", "[GateGenerators]") {
    // grad(RZ) = grad(e^{-i*0.5*PauliZ*a}) => -i*0.5*PauliZ
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        cuGates::getPauliZ<typename StateVectorCudaManaged<double>::CFP_t>()};
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (std::size_t applied_qubit = 0; applied_qubit < num_qubits;
             applied_qubit++) {
            auto init_state =
                createRandomStateVectorData<double>(re, num_qubits);

            StateVectorCudaManaged<double> psi(init_state.data(),
                                               init_state.size());
            StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                      init_state.size());
            StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                        init_state.size());

            std::string cache_gate_name = "DirectGenRZ" +
                                          std::to_string(applied_qubit) + "_" +
                                          std::to_string(num_qubits);

            psi.applyGeneratorRZ({applied_qubit}, false);
            psi_direct.applyOperation(cache_gate_name, {applied_qubit}, false,
                                      {0.0}, matrix);
            psi_dispatch.applyGenerator({"RZ"}, {applied_qubit}, false);

            CHECK(psi.getDataVector() == psi_direct.getDataVector());
            CHECK(psi_dispatch.getDataVector() == psi_direct.getDataVector());
        }
    }
}

TEST_CASE("Generators::applyGeneratorPhaseShift", "[GateGenerators]") {
    // grad(PhaseShift) = grad(e^{i*0.5*a}*e^{-i*0.5*PauliZ*a}) => -i|1><1|
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        for (std::size_t applied_qubit = 0; applied_qubit < num_qubits;
             applied_qubit++) {
            auto init_state =
                createRandomStateVectorData<double>(re, num_qubits);

            StateVectorCudaManaged<double> psi(init_state.data(),
                                               init_state.size());
            StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                      init_state.size());
            StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                        init_state.size());

            std::string cache_gate_name = "DirectGenPhaseShift" +
                                          std::to_string(applied_qubit) + "_" +
                                          std::to_string(num_qubits);

            psi.applyGeneratorPhaseShift({applied_qubit}, false);
            psi_direct.applyOperation(cache_gate_name, {applied_qubit}, false,
                                      {0.0}, matrix);
            psi_dispatch.applyGenerator({"PhaseShift"}, {applied_qubit}, false);

            CHECK(psi.getDataVector() == psi_direct.getDataVector());
            CHECK(psi_dispatch.getDataVector() == psi_direct.getDataVector());
        }
    }
}

TEST_CASE("Generators::applyGeneratorIsingXX", "[GateGenerators]") {
    // grad(IsingXX)() = e^{-i*0.5*a*(kron(X, X))}) => -0.5*i*(kron(X, X))
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenIsingXX" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorIsingXX({applied_qubit, applied_qubit + 1},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"IsingXX"}, {applied_qubit, applied_qubit + 1}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenIsingXX" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorIsingXX({applied_qubit + 1, applied_qubit},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"IsingXX"}, {applied_qubit + 1, applied_qubit}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorIsingXY", "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenIsingXY" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorIsingXY({applied_qubit, applied_qubit + 1},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"IsingXY"}, {applied_qubit, applied_qubit + 1}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenIsingXY" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorIsingXY({applied_qubit + 1, applied_qubit},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"IsingXY"}, {applied_qubit + 1, applied_qubit}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorIsingYY", "[GateGenerators]") {
    // grad(IsingXX)() = e^{-i*0.5*a*(kron(Y, Y))}) => -0.5*i*(kron(Y, Y))
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenIsingYY" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorIsingYY({applied_qubit, applied_qubit + 1},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"IsingYY"}, {applied_qubit, applied_qubit + 1}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenIsingYY" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorIsingYY({applied_qubit + 1, applied_qubit},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"IsingYY"}, {applied_qubit + 1, applied_qubit}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorIsingZZ", "[GateGenerators]") {
    // grad(IsingXX)() = e^{-i*0.5*a*(kron(Z, Z))}) => -0.5*i*(kron(Z, Z))
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenIsingZZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorIsingZZ({applied_qubit, applied_qubit + 1},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"IsingZZ"}, {applied_qubit, applied_qubit + 1}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenIsingZZ" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorIsingZZ({applied_qubit + 1, applied_qubit},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"IsingZZ"}, {applied_qubit + 1, applied_qubit}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorCRX", "[GateGenerators]") {
    // grad(CRX) = grad(kron(|0><0|, I(2)) + kron(|1><1|,
    // e^{-i*0.5*(PauliX)*a})) => -i*0.5*kron(|1><1|, PauliX)
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenCRX" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorCRX({applied_qubit, applied_qubit + 1},
                                      false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"CRX"}, {applied_qubit, applied_qubit + 1}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenCRX" + std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorCRX({applied_qubit + 1, applied_qubit},
                                      false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"CRX"}, {applied_qubit + 1, applied_qubit}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorCRY_GPU", "[GateGenerators]") {
    // grad(CRY) = grad(kron(|0><0|, I(2)) + kron(|1><1|,
    // e^{-i*0.5*(PauliY)*a})) => -i*0.5*kron(|1><1|, PauliY)
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenCRY" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorCRY({applied_qubit, applied_qubit + 1},
                                      false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"CRY"}, {applied_qubit, applied_qubit + 1}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenCRY" + std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorCRY({applied_qubit + 1, applied_qubit},
                                      false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"CRY"}, {applied_qubit + 1, applied_qubit}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorCRZ", "[GateGenerators]") {
    // grad(CRZ) = grad(kron(|0><0|, I(2)) + kron(|1><1|,
    // e^{-i*0.5*(PauliZ)*a})) => -i*0.5*kron(|1><1|, PauliZ)
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenCRZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorCRZ({applied_qubit, applied_qubit + 1},
                                      false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"CRZ"}, {applied_qubit, applied_qubit + 1}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenCRZ" + std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorCRZ({applied_qubit + 1, applied_qubit},
                                      false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator(
                    {"CRZ"}, {applied_qubit + 1, applied_qubit}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorControlledPhaseShift",
          "[GateGenerators]") {
    // grad(ControlledPhaseShift) = grad(kron(|0><0|, I(2)) +  kron(|1><1|,
    // e^{i*0.5*a}*e^{-i*0.5*PauliZ*a} )) => -i|11><11|
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenControlledPhaseShift" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorControlledPhaseShift(
                    {applied_qubit, applied_qubit + 1}, false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"ControlledPhaseShift"},
                                            {applied_qubit, applied_qubit + 1},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenControlledPhaseShift" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorControlledPhaseShift(
                    {applied_qubit + 1, applied_qubit}, false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"ControlledPhaseShift"},
                                            {applied_qubit + 1, applied_qubit},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorSingleExcitation", "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenSingleExcitation" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorSingleExcitation(
                    {applied_qubit, applied_qubit + 1}, false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"SingleExcitation"},
                                            {applied_qubit, applied_qubit + 1},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenSingleExcitation" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorSingleExcitation(
                    {applied_qubit + 1, applied_qubit}, false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"SingleExcitation"},
                                            {applied_qubit + 1, applied_qubit},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorSingleExcitationMinus",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0,-1.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenSingleExcitationMinus" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorSingleExcitationMinus(
                    {applied_qubit, applied_qubit + 1}, false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"SingleExcitationMinus"},
                                            {applied_qubit, applied_qubit + 1},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenSingleExcitationMinus" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorSingleExcitationMinus(
                    {applied_qubit + 1, applied_qubit}, false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"SingleExcitationMinus"},
                                            {applied_qubit + 1, applied_qubit},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorSingleExcitationPlus",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {-1.0, 0.0},{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0,-1.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 2; num_qubits <= 5; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenSingleExcitationPlus" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorSingleExcitationPlus(
                    {applied_qubit, applied_qubit + 1}, false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"SingleExcitationPlus"},
                                            {applied_qubit, applied_qubit + 1},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenSingleExcitationPlus" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorSingleExcitationPlus(
                    {applied_qubit + 1, applied_qubit}, false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"SingleExcitationPlus"},
                                            {applied_qubit + 1, applied_qubit},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorDoubleExcitation_GPU",
          "[GateGenerators]") {
    // clang-format off
    /* For convenience, the DoubleExcitation* matrices were generated from PennyLane and formatted as follows:
        ```python
            mat = qml.matrix(qml.DoubleExcitation(a, wires=[0,1,2,3]).generator())
            def cpp_format(arr):
                s = ""
                for i in arr:
                    s += f"{{{np.real(i) if np.real(i) != 0 else 0}, {np.imag(i) if np.imag(i) != 0 else 0}}},"
                return s
            output = ""

            for i in range(mat.shape[0]):
                out += cpp_format(mat[i][:])
                out += "\n"
            print(output)
        ```
    */
    // clang-format on

    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, -1.0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 1.0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 4; num_qubits <= 8; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenDoubleExcitation" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorDoubleExcitation(
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1,
                                           applied_qubit + 2,
                                           applied_qubit + 3},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"DoubleExcitation"},
                                            {applied_qubit, applied_qubit + 1,
                                             applied_qubit + 2,
                                             applied_qubit + 3},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenDoubleExcitation" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorDoubleExcitation(
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 3, applied_qubit + 2,
                                           applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"DoubleExcitation"},
                                            {applied_qubit + 3,
                                             applied_qubit + 2,
                                             applied_qubit + 1, applied_qubit},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorDoubleExcitationMinus_GPU",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, -1.0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 1.0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 4; num_qubits <= 8; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenDoubleExcitationMinus" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorDoubleExcitationMinus(
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1,
                                           applied_qubit + 2,
                                           applied_qubit + 3},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"DoubleExcitationMinus"},
                                            {applied_qubit, applied_qubit + 1,
                                             applied_qubit + 2,
                                             applied_qubit + 3},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenDoubleExcitationMinus" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorDoubleExcitationMinus(
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 3, applied_qubit + 2,
                                           applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"DoubleExcitationMinus"},
                                            {applied_qubit + 3,
                                             applied_qubit + 2,
                                             applied_qubit + 1, applied_qubit},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorDoubleExcitationPlus_GPU",
          "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix{
        // clang-format off
        {-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, -1.0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 1.0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 4; num_qubits <= 8; num_qubits++) {
        SECTION("Increasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenDoubleExcitationPlus" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorDoubleExcitationPlus(
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2,
                     applied_qubit + 3},
                    false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1,
                                           applied_qubit + 2,
                                           applied_qubit + 3},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"DoubleExcitationPlus"},
                                            {applied_qubit, applied_qubit + 1,
                                             applied_qubit + 2,
                                             applied_qubit + 3},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenDoubleExcitationPlus" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorDoubleExcitationPlus(
                    {applied_qubit + 3, applied_qubit + 2, applied_qubit + 1,
                     applied_qubit},
                    false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 3, applied_qubit + 2,
                                           applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix);
                psi_dispatch.applyGenerator({"DoubleExcitationPlus"},
                                            {applied_qubit + 3,
                                             applied_qubit + 2,
                                             applied_qubit + 1, applied_qubit},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEST_CASE("Generators::applyGeneratorMultiRZ", "[GateGenerators]") {
    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix2{
        // clang-format off
        {1.0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{-1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{-1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{1.0, 0}
        // clang-format on
    };

    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix3{
        // clang-format off
        {1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0}
        // clang-format on
    };

    std::vector<typename StateVectorCudaManaged<double>::CFP_t> matrix4{
        // clang-format off
        {1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0},{0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{-1.0, 0},{0, 0},
        {0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{0, 0},{1.0, 0}
        // clang-format on
    };
    std::mt19937 re{1337U};

    for (std::size_t num_qubits = 4; num_qubits <= 8; num_qubits++) {
        SECTION("Increasing qubit indices, MultiRZ 2") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorMultiRZ({applied_qubit, applied_qubit + 1},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1},
                                          false, {0.0}, matrix2);
                psi_dispatch.applyGenerator(
                    {"MultiRZ"}, {applied_qubit, applied_qubit + 1}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices, MultiRZ 2") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 1;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit + 1) +
                    "_" + std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorMultiRZ({applied_qubit + 1, applied_qubit},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix2);
                psi_dispatch.applyGenerator(
                    {"MultiRZ"}, {applied_qubit + 1, applied_qubit}, false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }

        SECTION("Increasing qubit indices, MultiRZ 3") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 2;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorMultiRZ(
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2},
                    false);
                psi_direct.applyOperation(
                    cache_gate_name,
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2},
                    false, {0.0}, matrix3);
                psi_dispatch.applyGenerator(
                    {"MultiRZ"},
                    {applied_qubit, applied_qubit + 1, applied_qubit + 2},
                    false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices, MultiRZ 3") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 2;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit + 2) +
                    "_" + std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorMultiRZ(
                    {applied_qubit + 2, applied_qubit + 1, applied_qubit},
                    false);
                psi_direct.applyOperation(
                    cache_gate_name,
                    {applied_qubit + 2, applied_qubit + 1, applied_qubit},
                    false, {0.0}, matrix3);
                psi_dispatch.applyGenerator(
                    {"MultiRZ"},
                    {applied_qubit + 2, applied_qubit + 1, applied_qubit},
                    false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }

        SECTION("Increasing qubit indices, MultiRZ 4") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 3) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorMultiRZ({applied_qubit, applied_qubit + 1,
                                           applied_qubit + 2,
                                           applied_qubit + 3},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit, applied_qubit + 1,
                                           applied_qubit + 2,
                                           applied_qubit + 3},
                                          false, {0.0}, matrix4);
                psi_dispatch.applyGenerator({"MultiRZ"},
                                            {applied_qubit, applied_qubit + 1,
                                             applied_qubit + 2,
                                             applied_qubit + 3},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
        SECTION("Decreasing qubit indices") {
            for (std::size_t applied_qubit = 0; applied_qubit < num_qubits - 3;
                 applied_qubit++) {
                auto init_state =
                    createRandomStateVectorData<double>(re, num_qubits);

                StateVectorCudaManaged<double> psi(init_state.data(),
                                                   init_state.size());
                StateVectorCudaManaged<double> psi_direct(init_state.data(),
                                                          init_state.size());
                StateVectorCudaManaged<double> psi_dispatch(init_state.data(),
                                                            init_state.size());

                std::string cache_gate_name =
                    "DirectGenMultiRZ" + std::to_string(applied_qubit + 3) +
                    "_" + std::to_string(applied_qubit + 2) + "_" +
                    std::to_string(applied_qubit + 1) + "_" +
                    std::to_string(applied_qubit) + "_" +
                    std::to_string(num_qubits);

                psi.applyGeneratorMultiRZ({applied_qubit + 3, applied_qubit + 2,
                                           applied_qubit + 1, applied_qubit},
                                          false);
                psi_direct.applyOperation(cache_gate_name,
                                          {applied_qubit + 3, applied_qubit + 2,
                                           applied_qubit + 1, applied_qubit},
                                          false, {0.0}, matrix4);
                psi_dispatch.applyGenerator({"MultiRZ"},
                                            {applied_qubit + 3,
                                             applied_qubit + 2,
                                             applied_qubit + 1, applied_qubit},
                                            false);

                CHECK(psi.getDataVector() == psi_direct.getDataVector());
                CHECK(psi_dispatch.getDataVector() ==
                      psi_direct.getDataVector());
            }
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorCudaManaged::applyGeneratorGlobalPhase",
                   "[StateVectorCudaManaged_Generator]", float, double) {
    const bool inverse = GENERATE(true, false);
    const std::string gate_name = "GlobalPhase";
    {
        using ComplexT = StateVectorCudaManaged<TestType>::ComplexT;
        const size_t num_qubits = 4;
        const TestType ep = 1e-3;
        const TestType EP = 1e-4;

        std::vector<ComplexT> ini_st{
            ComplexT{0.267462841882, 0.010768564798},
            ComplexT{0.228575129706, 0.010564590956},
            ComplexT{0.099492749900, 0.260849823392},
            ComplexT{0.093690204310, 0.189847108173},
            ComplexT{0.033390732374, 0.203836830144},
            ComplexT{0.226979395737, 0.081852150975},
            ComplexT{0.031235505729, 0.176933497281},
            ComplexT{0.294287602843, 0.145156781198},
            ComplexT{0.152742706049, 0.111628061129},
            ComplexT{0.012553863703, 0.120027860480},
            ComplexT{0.237156555364, 0.154658769755},
            ComplexT{0.117001120872, 0.228059505033},
            ComplexT{0.041495873225, 0.065934827444},
            ComplexT{0.089653239407, 0.221581340372},
            ComplexT{0.217892322429, 0.291261296999},
            ComplexT{0.292993251871, 0.186570798697},
        };

        StateVectorCudaManaged<TestType> gntr_sv{ini_st.data(), ini_st.size()};
        StateVectorCudaManaged<TestType> gate_svp{ini_st.data(), ini_st.size()};
        StateVectorCudaManaged<TestType> gate_svm{ini_st.data(), ini_st.size()};

        auto scale = gntr_sv.applyGenerator(gate_name, {0}, inverse);
        if (inverse) {
            gate_svp.applyOperation(gate_name, {0}, inverse, {-ep});
            gate_svm.applyOperation(gate_name, {0}, inverse, {ep});
        } else {
            gate_svp.applyOperation(gate_name, {0}, inverse, {ep});
            gate_svm.applyOperation(gate_name, {0}, inverse, {-ep});
        }

        auto result_gntr_sv = gntr_sv.getDataVector();
        auto result_gate_svp = gate_svp.getDataVector();
        auto result_gate_svm = gate_svm.getDataVector();

        for (size_t j = 0; j < exp2(num_qubits); j++) {
            CHECK(-scale * imag(result_gntr_sv[j]) ==
                  Approx(0.5 *
                         (real(result_gate_svp[j]) - real(result_gate_svm[j])) /
                         ep)
                      .margin(EP));
            CHECK(scale * real(result_gntr_sv[j]) ==
                  Approx(0.5 *
                         (imag(result_gate_svp[j]) - imag(result_gate_svm[j])) /
                         ep)
                      .margin(EP));
        }
    }
}
