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

#include <cmath>
#include <complex>
#include <vector>

#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace cuUtil;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::cuGates {

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of PauliX data.
 */
template <class CFP_t>
static constexpr auto getIdentity() -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of PauliX data.
 */
template <class CFP_t> static constexpr auto getPauliX() -> std::vector<CFP_t> {
    return {cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(), cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the PauliY gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of PauliY data.
 */
template <class CFP_t> static constexpr auto getPauliY() -> std::vector<CFP_t> {
    return {cuUtil::ZERO<CFP_t>(), -cuUtil::IMAG<CFP_t>(),
            cuUtil::IMAG<CFP_t>(), cuUtil::ZERO<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the PauliZ gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of PauliZ data.
 */
template <class CFP_t> static constexpr auto getPauliZ() -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            -cuUtil::ONE<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the Hadamard gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of Hadamard data.
 */
template <class CFP_t>
static constexpr auto getHadamard() -> std::vector<CFP_t> {
    return {cuUtil::INVSQRT2<CFP_t>(), cuUtil::INVSQRT2<CFP_t>(),
            cuUtil::INVSQRT2<CFP_t>(), -cuUtil::INVSQRT2<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the S gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of S gate data.
 */
template <class CFP_t> static constexpr auto getS() -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::IMAG<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the T gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of T gate data.
 */
template <class CFP_t> static constexpr auto getT() -> std::vector<CFP_t> {
    return {
        cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ConstMultSC(
            cuUtil::SQRT2<decltype(cuUtil::ONE<CFP_t>().x)>() / 2,
            cuUtil::ConstSum(cuUtil::ONE<CFP_t>(), -cuUtil::IMAG<CFP_t>()))};
}

/**
 * @brief Create a matrix representation of the CNOT gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of CNOT gate data.
 */
template <class CFP_t> static constexpr auto getCNOT() -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the SWAP gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of SWAP gate data.
 */
template <class CFP_t> static constexpr auto getSWAP() -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the CZ gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of SWAP gate data.
 */
template <class CFP_t> static constexpr auto getCY() -> std::vector<CFP_t> {
    return {
        cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), -cuUtil::IMAG<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::IMAG<CFP_t>(),
        cuUtil::ZERO<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the CZ gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of SWAP gate data.
 */
template <class CFP_t> static constexpr auto getCZ() -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            -cuUtil::ONE<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the CSWAP gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of CSWAP gate data.
 */
template <class CFP_t> static constexpr auto getCSWAP() -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the Toffoli gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<CFP_t> Return constant expression
 * of Toffoli gate data.
 */
template <class CFP_t>
static constexpr auto getToffoli() -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return Phase-shift gate
 * data.
 */
template <class CFP_t, class U = double>
static auto getPhaseShift(U angle) -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            {std::cos(angle), std::sin(angle)}};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return Phase-shift gate
 * data.
 */
template <class CFP_t, class U = double>
static auto getPhaseShift(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getPhaseShift<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return RX gate data.
 */
template <class CFP_t, class U = double>
static auto getRX(U angle) -> std::vector<CFP_t> {
    const CFP_t c{std::cos(angle / 2), 0};
    const CFP_t js{0, -std::sin(angle / 2)};
    return {c, js, js, c};
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return RX gate data.
 */
template <class CFP_t, class U = double>
static auto getRX(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getRX<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return RY gate data.
 */
template <class CFP_t, class U = double>
static auto getRY(U angle) -> std::vector<CFP_t> {
    const CFP_t c{std::cos(angle / 2), 0};
    const CFP_t s{std::sin(angle / 2), 0};
    return {c, -s, s, c};
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return RY gate data.
 */
template <class CFP_t, class U = double>
static auto getRY(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getRY<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return RZ gate data.
 */
template <class CFP_t, class U = double>
static auto getRZ(U angle) -> std::vector<CFP_t> {
    return {{std::cos(-angle / 2), std::sin(-angle / 2)},
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            {std::cos(angle / 2), std::sin(angle / 2)}};
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return RZ gate data.
 */
template <class CFP_t, class U = double>
static auto getRZ(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getRZ<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the Rot gate data in row-major
format.
 *
 * The gate is defined as:
 * \f$\begin{split}Rot(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)=
\begin{bmatrix}
e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
\end{bmatrix}.\end{split}\f$
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param phi \f$\phi\f$ shift angle.
 * @param theta \f$\theta\f$ shift angle.
 * @param omega \f$\omega\f$ shift angle.
 * @return std::vector<CFP_t> Return Rot gate data.
 */
template <class CFP_t, class U = double>
static auto getRot(U phi, U theta, U omega) -> std::vector<CFP_t> {
    const U c = std::cos(theta / 2);
    const U s = std::sin(theta / 2);
    const U p{phi + omega};
    const U m{phi - omega};
    /*
        return {CFP_t{std::cos(p / 2), -std::sin(p / 2)} * c,
                -CFP_t{std::cos(m / 2), std::sin(m / 2)} * s,
                CFP_t{std::cos(m / 2), -std::sin(m / 2)} * s,
                CFP_t{std::cos(p / 2), std::sin(p / 2)} * c};*/
    return {ConstMultSC(c, CFP_t{std::cos(p / 2), -std::sin(p / 2)}),
            ConstMultSC(s, -CFP_t{std::cos(m / 2), std::sin(m / 2)}),
            ConstMultSC(s, CFP_t{std::cos(m / 2), -std::sin(m / 2)}),
            ConstMultSC(c, CFP_t{std::cos(p / 2), std::sin(p / 2)})};
}

/**
 * @brief Create a matrix representation of the Rot gate data in row-major
format.
 *
 * The gate is defined as:
 * \f$\begin{split}Rot(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)=
\begin{bmatrix}
e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
\end{bmatrix}.\end{split}\f$
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of gate data. Values are expected in order of
\f$[\phi, \theta, \omega]\f$.
 * @return std::vector<CFP_t> Return Rot gate data.
 */
template <class CFP_t, class U = double>
static auto getRot(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getRot<CFP_t>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return RX gate data.
 */
template <class CFP_t, class U = double>
static auto getCRX(U angle) -> std::vector<CFP_t> {
    const auto rx{getRX<CFP_t>(angle)};
    return {cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            rx[0],
            rx[1],
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            rx[2],
            rx[3]};
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return RX gate data.
 */
template <class CFP_t, class U = double>
static auto getCRX(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getCRX<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return RY gate data.
 */
template <class CFP_t, class U = double>
static auto getCRY(U angle) -> std::vector<CFP_t> {
    const auto ry{getRY<CFP_t>(angle)};
    return {cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            ry[0],
            ry[1],
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            ry[2],
            ry[3]};
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return RY gate data.
 */
template <class CFP_t, class U = double>
static auto getCRY(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getCRY<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return RZ gate data.
 */
template <class CFP_t, class U = double>
static auto getCRZ(U angle) -> std::vector<CFP_t> {
    const CFP_t first{std::cos(-angle / 2), std::sin(-angle / 2)};
    const CFP_t second{std::cos(angle / 2), std::sin(angle / 2)};
    return {cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            first,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            second};
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return RZ gate data.
 */
template <class CFP_t, class U = double>
static auto getCRZ(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getCRZ<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(U phi, U theta, U omega)`.
 */
template <class CFP_t, class U = double>
static auto getCRot(U phi, U theta, U omega) -> std::vector<CFP_t> {
    const auto rot{std::move(getRot<CFP_t>(phi, theta, omega))};
    return {cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            rot[0],
            rot[1],
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            rot[2],
            rot[3]};
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(const std::vector<U> &params)`.
 */
template <class CFP_t, class U = double>
static auto getCRot(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getCRot<CFP_t>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate
data in row-major format.
 *
 * @see `getPhaseShift<T,U>(U angle)`.
 */
template <class CFP_t, class U = double>
static auto getControlledPhaseShift(U angle) -> std::vector<CFP_t> {
    return {cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), {std::cos(angle), std::sin(angle)}};
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate
data in row-major format.
 *
 * @see `getPhaseShift<T,U>(const std::vector<U> &params)`.
 */
template <class CFP_t, class U = double>
static auto getControlledPhaseShift(const std::vector<U> &params)
    -> std::vector<CFP_t> {
    return getControlledPhaseShift<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return single excitation rotation
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getSingleExcitation(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t c{std::cos(p2), 0};
    const CFP_t s{std::sin(p2), 0};
    return {cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            c,
            s,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            -s,
            c,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return single excitation rotation
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getSingleExcitation(const std::vector<U> &params)
    -> std::vector<CFP_t> {
    return getSingleExcitation<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation
 * generator data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorSingleExcitation() -> std::vector<CFP_t> {
    return {
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::IMAG<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), -cuUtil::IMAG<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
    };
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <class CFP_t, class U = double>
static auto getSingleExcitationMinus(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, -p2)));
    const CFP_t c{std::cos(p2), 0};
    const CFP_t s{std::sin(p2), 0};
    return {e,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            c,
            s,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            -s,
            c,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            e};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <class CFP_t, class U = double>
static auto getSingleExcitationMinus(const std::vector<U> &params)
    -> std::vector<CFP_t> {
    return getSingleExcitationMinus<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation Minus
 * generator data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorSingleExcitationMinus()
    -> std::vector<CFP_t> {
    return {
        cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::IMAG<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), -cuUtil::IMAG<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
    };
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <class CFP_t, class U = double>
static auto getSingleExcitationPlus(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, p2)));
    const CFP_t c{std::cos(p2), 0};
    const CFP_t s{std::sin(p2), 0};
    return {e,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            c,
            s,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            -s,
            c,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            e};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <class CFP_t, class U = double>
static auto getSingleExcitationPlus(const std::vector<U> &params)
    -> std::vector<CFP_t> {
    return getSingleExcitationPlus<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation Plus
 * generator data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorSingleExcitationPlus() -> std::vector<CFP_t> {
    return {
        -cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::IMAG<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), -cuUtil::IMAG<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), -cuUtil::ONE<CFP_t>(),
    };
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return double excitation rotation
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getDoubleExcitation(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t c{std::cos(p2), 0};
    const CFP_t s{std::sin(p2), 0};
    std::vector<CFP_t> mat(256, cuUtil::ZERO<CFP_t>());
    mat[0] = cuUtil::ONE<CFP_t>();
    mat[17] = cuUtil::ONE<CFP_t>();
    mat[34] = cuUtil::ONE<CFP_t>();
    mat[51] = c;
    mat[60] = s;
    mat[68] = cuUtil::ONE<CFP_t>();
    mat[85] = cuUtil::ONE<CFP_t>();
    mat[102] = cuUtil::ONE<CFP_t>();
    mat[119] = cuUtil::ONE<CFP_t>();
    mat[136] = cuUtil::ONE<CFP_t>();
    mat[153] = cuUtil::ONE<CFP_t>();
    mat[170] = cuUtil::ONE<CFP_t>();
    mat[187] = cuUtil::ONE<CFP_t>();
    mat[195] = -s;
    mat[204] = c;
    mat[221] = cuUtil::ONE<CFP_t>();
    mat[238] = cuUtil::ONE<CFP_t>();
    mat[255] = cuUtil::ONE<CFP_t>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return double excitation rotation
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getDoubleExcitation(const std::vector<U> &params)
    -> std::vector<CFP_t> {
    return getDoubleExcitation<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation
 * generator data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorDoubleExcitation() -> std::vector<CFP_t> {
    std::vector<CFP_t> mat(256, cuUtil::ZERO<CFP_t>());
    mat[60] = cuUtil::IMAG<CFP_t>();
    mat[195] = -cuUtil::IMAG<CFP_t>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <class CFP_t, class U = double>
static auto getDoubleExcitationMinus(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, -p2)));
    const CFP_t c{std::cos(p2), 0};
    const CFP_t s{std::sin(p2), 0};
    std::vector<CFP_t> mat(256, cuUtil::ZERO<CFP_t>());
    mat[0] = e;
    mat[17] = e;
    mat[34] = e;
    mat[51] = c;
    mat[60] = s;
    mat[68] = e;
    mat[85] = e;
    mat[102] = e;
    mat[119] = e;
    mat[136] = e;
    mat[153] = e;
    mat[170] = e;
    mat[187] = e;
    mat[195] = -s;
    mat[204] = c;
    mat[221] = e;
    mat[238] = e;
    mat[255] = e;
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <class CFP_t, class U = double>
static auto getDoubleExcitationMinus(const std::vector<U> &params)
    -> std::vector<CFP_t> {
    return getDoubleExcitationMinus<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation Minus
 * generator data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorDoubleExcitationMinus()
    -> std::vector<CFP_t> {
    std::vector<CFP_t> mat(256, cuUtil::ZERO<CFP_t>());
    mat[0] = cuUtil::ONE<CFP_t>();
    mat[17] = cuUtil::ONE<CFP_t>();
    mat[34] = cuUtil::ONE<CFP_t>();
    mat[60] = cuUtil::IMAG<CFP_t>();
    mat[68] = cuUtil::ONE<CFP_t>();
    mat[85] = cuUtil::ONE<CFP_t>();
    mat[102] = cuUtil::ONE<CFP_t>();
    mat[119] = cuUtil::ONE<CFP_t>();
    mat[136] = cuUtil::ONE<CFP_t>();
    mat[153] = cuUtil::ONE<CFP_t>();
    mat[170] = cuUtil::ONE<CFP_t>();
    mat[187] = cuUtil::ONE<CFP_t>();
    mat[195] = -cuUtil::IMAG<CFP_t>();
    mat[221] = cuUtil::ONE<CFP_t>();
    mat[238] = cuUtil::ONE<CFP_t>();
    mat[255] = cuUtil::ONE<CFP_t>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <class CFP_t, class U = double>
static auto getDoubleExcitationPlus(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, p2)));
    const CFP_t c{std::cos(p2), 0};
    const CFP_t s{std::sin(p2), 0};
    std::vector<CFP_t> mat(256, cuUtil::ZERO<CFP_t>());
    mat[0] = e;
    mat[17] = e;
    mat[34] = e;
    mat[51] = c;
    mat[60] = s;
    mat[68] = e;
    mat[85] = e;
    mat[102] = e;
    mat[119] = e;
    mat[136] = e;
    mat[153] = e;
    mat[170] = e;
    mat[187] = e;
    mat[195] = -s;
    mat[204] = c;
    mat[221] = e;
    mat[238] = e;
    mat[255] = e;
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <class CFP_t, class U = double>
static auto getDoubleExcitationPlus(const std::vector<U> &params)
    -> std::vector<CFP_t> {
    return getDoubleExcitationPlus<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation Plus
 * generator data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorDoubleExcitationPlus() -> std::vector<CFP_t> {
    std::vector<CFP_t> mat(256, cuUtil::ZERO<CFP_t>());
    mat[0] = -cuUtil::ONE<CFP_t>();
    mat[17] = -cuUtil::ONE<CFP_t>();
    mat[34] = -cuUtil::ONE<CFP_t>();
    mat[60] = cuUtil::IMAG<CFP_t>();
    mat[68] = -cuUtil::ONE<CFP_t>();
    mat[85] = -cuUtil::ONE<CFP_t>();
    mat[102] = -cuUtil::ONE<CFP_t>();
    mat[119] = -cuUtil::ONE<CFP_t>();
    mat[136] = -cuUtil::ONE<CFP_t>();
    mat[153] = -cuUtil::ONE<CFP_t>();
    mat[170] = -cuUtil::ONE<CFP_t>();
    mat[187] = -cuUtil::ONE<CFP_t>();
    mat[195] = -cuUtil::IMAG<CFP_t>();
    mat[221] = -cuUtil::ONE<CFP_t>();
    mat[238] = -cuUtil::ONE<CFP_t>();
    mat[255] = -cuUtil::ONE<CFP_t>();
    return mat;
}

/**
 * @brief Create a matrix representation of the Ising XX coupling
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return Ising XX coupling
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getIsingXX(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t c{std::cos(p2), 0};
    const CFP_t neg_is{0, -std::sin(p2)};
    return {c,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            neg_is,
            cuUtil::ZERO<CFP_t>(),
            c,
            neg_is,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            neg_is,
            c,
            cuUtil::ZERO<CFP_t>(),
            neg_is,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            c};
}

/**
 * @brief Create a matrix representation of the Ising XX coupling
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return Ising XX coupling
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getIsingXX(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getIsingXX<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising XX generator
 * data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorIsingXX() -> std::vector<CFP_t> {
    return {
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising YY coupling
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return Ising YY coupling
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getIsingYY(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t c{std::cos(p2), 0};
    const CFP_t pos_is{0, std::sin(p2)};
    const CFP_t neg_is{0, -std::sin(p2)};
    return {c,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            pos_is,
            cuUtil::ZERO<CFP_t>(),
            c,
            neg_is,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            neg_is,
            c,
            cuUtil::ZERO<CFP_t>(),
            pos_is,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            c};
}

/**
 * @brief Create a matrix representation of the Ising YY coupling
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return Ising YY coupling
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getIsingYY(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getIsingYY<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising YY generator
 * data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorIsingYY() -> std::vector<CFP_t> {
    return {
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), -cuUtil::ONE<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        -cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising ZZ coupling
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return Ising ZZ coupling
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getIsingZZ(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t neg_e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, -p2)));
    const CFP_t pos_e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, p2)));
    return {neg_e,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            pos_e,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            pos_e,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            neg_e};
}

/**
 * @brief Create a matrix representation of the Ising ZZ coupling
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return Ising ZZ coupling
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getIsingZZ(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getIsingZZ<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising ZZ generator
 * data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorIsingZZ() -> std::vector<CFP_t> {
    return {
        cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), -cuUtil::ONE<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        -cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising XY coupling
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<CFP_t> Return Ising XY coupling
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getIsingXY(U angle) -> std::vector<CFP_t> {
    const U p2 = angle / 2;
    const CFP_t c{std::cos(p2), 0};
    const CFP_t pos_is{0, std::sin(p2)};
    return {cuUtil::ONE<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            c,
            pos_is,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            pos_is,
            c,
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}

/**
 * @brief Create a matrix representation of the Ising XY coupling
 * gate data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<CFP_t> Return Ising XY coupling
 * gate data.
 */
template <class CFP_t, class U = double>
static auto getIsingXY(const std::vector<U> &params) -> std::vector<CFP_t> {
    return getIsingXY<CFP_t>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising XY generator
 * data in row-major format.
 *
 * @tparam CFP_t Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <class CFP_t, class U = double>
static constexpr auto getGeneratorIsingXY() -> std::vector<CFP_t> {
    return {
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ONE<CFP_t>(),  cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),

        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
        cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
    };
}

template <class CFP_t> static constexpr auto getP11_CU() -> std::vector<CFP_t> {
    return {cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}

template <class CFP_t>
static constexpr auto getP1111_CU() -> std::vector<CFP_t> {
    return {cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
            cuUtil::ONE<CFP_t>()};
}

} // namespace Pennylane::LightningGPU::cuGates
