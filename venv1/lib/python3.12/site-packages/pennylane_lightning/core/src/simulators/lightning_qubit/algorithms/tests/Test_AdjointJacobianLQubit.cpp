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
#include <algorithm>
#include <limits>
#include <vector>

#include <catch2/catch.hpp>

#include "AdjointJacobianLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TestHelpers.hpp" // randomIntVector

// using namespace Pennylane;
/// @cond DEV
namespace {
// using namespace Pennylane;
using namespace Pennylane::LightningQubit::Algorithms;
using namespace Pennylane::LightningQubit::Observables;

using Pennylane::Util::randomIntVector;
// using namespace Pennylane::Simulators;
} // namespace
  /// @endcond

#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif
TEMPLATE_PRODUCT_TEST_CASE(
    "Algorithms::adjointJacobian with exceedingly complicated Hamiltonian",
    "[Algorithms]", (StateVectorLQubitManaged, StateVectorLQubitRaw),
    (float, double)) {
    using StateVectorT = TestType;
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using namespace std::literals;
    using Pennylane::LightningQubit::Observables::detail::
        HamiltonianApplyInPlace;

    std::vector<PrecisionT> param{-M_PI / 7, M_PI / 5, 2 * M_PI / 3};
    std::vector<size_t> t_params{0, 2};

    std::mt19937 re{1337};
    const size_t num_qubits = 8;
    const size_t n_terms = 1024;

    std::array<std::string_view, 4> pauli_strs = {""sv, "PauliX"sv, "PauliY"sv,
                                                  "PauliZ"sv};

    std::vector<PrecisionT> coeffs;
    std::vector<std::shared_ptr<Observable<StateVectorT>>> terms;

    std::uniform_real_distribution<PrecisionT> dist(-1.0, 1.0);

    for (size_t k = 0; k < n_terms; k++) {
        auto term_pauli = randomIntVector(re, num_qubits, 0, 3);

        std::vector<std::shared_ptr<Observable<StateVectorT>>> term_comp;
        for (size_t i = 0; i < num_qubits; i++) {
            if (term_pauli[i] == 0) {
                continue;
            }
            auto wires = std::vector<size_t>();
            wires.emplace_back(i);
            auto ob = std::make_shared<NamedObs<StateVectorT>>(
                std::string{pauli_strs[term_pauli[i]]}, wires);
            term_comp.push_back(std::move(ob));
        }

        coeffs.emplace_back(dist(re));
        terms.emplace_back(TensorProdObs<StateVectorT>::create(term_comp));
    }
    std::vector<ComplexT> psi(size_t{1} << num_qubits);
    std::normal_distribution<PrecisionT> ndist;
    for (auto &e : psi) {
        e = ndist(re);
    }
    std::vector<ComplexT> phi = psi;

    StateVectorT sv1(psi.data(), psi.size());
    StateVectorT sv2(phi.data(), phi.size());

    HamiltonianApplyInPlace<StateVectorT, false>::run(coeffs, terms, sv1);
    HamiltonianApplyInPlace<StateVectorT, true>::run(coeffs, terms, sv2);

    PrecisionT eps = std::numeric_limits<PrecisionT>::epsilon() * 1e4;
    REQUIRE(isApproxEqual(sv1.getData(), sv1.getLength(), sv2.getData(),
                          sv2.getLength(), eps));
}