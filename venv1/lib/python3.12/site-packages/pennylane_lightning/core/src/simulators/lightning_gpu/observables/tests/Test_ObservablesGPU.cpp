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
#include "ObservablesGPU.hpp"
#include "TestHelpers.hpp"

#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU::Observables;
using Pennylane::Util::LightningException;
} // namespace
/// @endcond

TEMPLATE_PRODUCT_TEST_CASE("NamedObs", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using NamedObsT = NamedObs<StateVectorT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<NamedObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<NamedObsT, std::string,
                                        std::vector<size_t>>);
    }

    SECTION("Constructibility - optional parameters") {
        REQUIRE(
            std::is_constructible_v<NamedObsT, std::string, std::vector<size_t>,
                                    std::vector<PrecisionT>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<NamedObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<NamedObsT>);
    }

    SECTION("NamedObs only accepts correct arguments") {
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("PauliX", {0, 3}), LightningException);

        REQUIRE_THROWS_AS(NamedObsT("RX", {0}), LightningException);
        REQUIRE_THROWS_AS(NamedObsT("RX", {0, 1, 2, 3}), LightningException);
        REQUIRE_THROWS_AS(
            NamedObsT("RX", {0}, std::vector<PrecisionT>{0.3, 0.4}),
            LightningException);
        REQUIRE_NOTHROW(
            NamedObsT("Rot", {0}, std::vector<PrecisionT>{0.3, 0.4, 0.5}));
    }
}

TEMPLATE_PRODUCT_TEST_CASE("HermitianObs", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using ComplexT = typename StateVectorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;
    using HermitianObsT = HermitianObs<StateVectorT>;

    SECTION("Non-Default constructibility") {
        REQUIRE(!std::is_constructible_v<HermitianObsT>);
    }

    SECTION("Constructibility") {
        REQUIRE(std::is_constructible_v<HermitianObsT, MatrixT,
                                        std::vector<size_t>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<HermitianObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<HermitianObsT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("TensorProdObs", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using TensorProdObsT = TensorProdObs<StateVectorT>;
    using NamedObsT = NamedObs<StateVectorT>;
    using HermitianObsT = HermitianObs<StateVectorT>;

    SECTION("Constructibility - NamedObs") {
        REQUIRE(
            std::is_constructible_v<TensorProdObsT,
                                    std::vector<std::shared_ptr<NamedObsT>>>);
    }

    SECTION("Constructibility - HermitianObs") {
        REQUIRE(std::is_constructible_v<
                TensorProdObsT, std::vector<std::shared_ptr<HermitianObsT>>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<TensorProdObsT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<TensorProdObsT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Hamiltonian", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using TensorProdObsT = TensorProdObs<StateVectorT>;
    using NamedObsT = NamedObs<StateVectorT>;
    using HermitianObsT = HermitianObs<StateVectorT>;
    using HamiltonianT = Hamiltonian<StateVectorT>;

    SECTION("Constructibility - NamedObs") {
        REQUIRE(
            std::is_constructible_v<HamiltonianT, std::vector<PrecisionT>,
                                    std::vector<std::shared_ptr<NamedObsT>>>);
    }

    SECTION("Constructibility - HermitianObs") {
        REQUIRE(std::is_constructible_v<
                HamiltonianT, std::vector<PrecisionT>,
                std::vector<std::shared_ptr<HermitianObsT>>>);
    }

    SECTION("Constructibility - TensorProdObsT") {
        REQUIRE(std::is_constructible_v<
                HamiltonianT, std::vector<PrecisionT>,
                std::vector<std::shared_ptr<TensorProdObsT>>>);
    }

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<HamiltonianT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<HamiltonianT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("SparseHamiltonian", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using SparseHamiltonianT = SparseHamiltonian<StateVectorT>;

    SECTION("Copy constructibility") {
        REQUIRE(std::is_copy_constructible_v<SparseHamiltonianT>);
    }

    SECTION("Move constructibility") {
        REQUIRE(std::is_move_constructible_v<SparseHamiltonianT>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Observables::HermitianHasher", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using ComplexT = typename StateVectorT::ComplexT;
    using TensorProdObsT = TensorProdObs<StateVectorT>;
    using NamedObsT = NamedObs<StateVectorT>;
    using HermitianT = HermitianObs<StateVectorT>;
    using HamiltonianT = Hamiltonian<StateVectorT>;

    std::vector<ComplexT> hermitian_h{{0.7071067811865475, 0},
                                      {0.7071067811865475, 0},
                                      {0.7071067811865475, 0},
                                      {-0.7071067811865475, 0}};

    auto obs1 =
        std::make_shared<HermitianT>(hermitian_h, std::vector<size_t>{0});
    auto obs2 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{2});
    auto obs3 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{3});

    auto tp_obs1 = std::make_shared<TensorProdObsT>(obs1, obs2);
    auto tp_obs2 = std::make_shared<TensorProdObsT>(obs2, obs3);

    auto ham_1 =
        HamiltonianT::create({0.165, 0.13, 0.5423}, {obs1, obs2, obs2});
    auto ham_2 = HamiltonianT::create({0.8545, 0.3222}, {tp_obs1, tp_obs2});

    SECTION("HamiltonianGPU<TestType>::obsName") {
        std::ostringstream res1, res2;
        res1 << "Hamiltonian: { 'coeffs' : [0.165, 0.13, 0.5423], "
                "'observables' : [Hermitian"
             << MatrixHasher()(hermitian_h) << ", PauliX[2], PauliX[2]]}";
        res2 << "Hamiltonian: { 'coeffs' : [0.8545, 0.3222], 'observables' : "
                "[Hermitian"
             << MatrixHasher()(hermitian_h)
             << " @ PauliX[2], PauliX[2] @ PauliX[3]]}";

        CHECK(ham_1->getObsName() == res1.str());
        CHECK(ham_2->getObsName() == res2.str());
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Hamiltonian::ApplyInPlace", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using TensorProdObsT = TensorProdObs<StateVectorT>;
    using NamedObsT = NamedObs<StateVectorT>;
    using HamiltonianT = Hamiltonian<StateVectorT>;

    const auto h = PrecisionT{0.809}; // half of the golden ratio

    auto zz = std::make_shared<TensorProdObsT>(
        std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{0}),
        std::make_shared<NamedObsT>("PauliZ", std::vector<size_t>{1}));

    auto x1 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{0});
    auto x2 = std::make_shared<NamedObsT>("PauliX", std::vector<size_t>{1});

    auto ham = HamiltonianT::create({PrecisionT{1.0}, h, h}, {zz, x1, x2});

    SECTION("ApplyInPlace", "[Apply Method]") {
        SECTION("Hamiltonian applies correctly to |+->") {
            auto st_data = createProductState<PrecisionT>("+-");
            std::vector<ComplexT> data_(st_data.data(),
                                        st_data.data() + st_data.size());
            StateVectorT state_vector(data_.data(), data_.size());

            ham->applyInPlace(state_vector);

            auto expected = std::vector<ComplexT>{
                0.5,
                0.5,
                -0.5,
                -0.5,
            };

            REQUIRE(isApproxEqual(state_vector.getDataVector().data(),
                                  state_vector.getDataVector().size(),
                                  expected.data(), expected.size()));
        }

        SECTION("Hamiltonian applies correctly to |01>") {
            auto st_data = createProductState<PrecisionT>("01");
            std::vector<ComplexT> data_(st_data.data(),
                                        st_data.data() + st_data.size());
            StateVectorT state_vector(data_.data(), data_.size());

            ham->applyInPlace(state_vector);

            auto expected = std::vector<ComplexT>{
                h,
                -1.0,
                0.0,
                h,
            };

            REQUIRE(isApproxEqual(state_vector.getDataVector().data(),
                                  state_vector.getDataVector().size(),
                                  expected.data(), expected.size()));
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("SparseHamiltonian::ApplyInPlace", "[Observables]",
                           (StateVectorCudaManaged), (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    const std::size_t num_qubits = 3;
    std::mt19937 re{1337};

    auto sparseH = SparseHamiltonian<StateVectorT>::create(
        {ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
         ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0},
         ComplexT{1.0, 0.0}, ComplexT{1.0, 0.0}},
        {7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2});

    auto init_state = createRandomStateVectorData<PrecisionT>(re, num_qubits);

    StateVectorT state_vector(init_state.data(), init_state.size());

    sparseH->applyInPlace(state_vector);

    std::reverse(init_state.begin(), init_state.end());

    REQUIRE(isApproxEqual(state_vector.getDataVector().data(),
                          state_vector.getDataVector().size(),
                          init_state.data(), init_state.size()));
}