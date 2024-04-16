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
 * @file
 * Defines helper methods for PennyLane Lightning.
 */
#pragma once

#include <complex>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "CPUMemoryModel.hpp" // getBestAllocator
#include "Error.hpp"          // PL_ABORT
#include "Memory.hpp"         // AlignedAllocator
#include "TypeTraits.hpp"
#include "Util.hpp" // INVSQRT2

namespace Pennylane::Util {
template <class T, class Alloc = std::allocator<T>> struct PLApprox {
    const std::vector<T, Alloc> &comp_;

    explicit PLApprox(const std::vector<T, Alloc> &comp) : comp_{comp} {}

    Util::remove_complex_t<T> margin_{};
    Util::remove_complex_t<T> epsilon_ =
        std::numeric_limits<float>::epsilon() * 100;

    template <class AllocA>
    [[nodiscard]] bool compare(const std::vector<T, AllocA> &lhs) const {
        if (lhs.size() != comp_.size()) {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); i++) {
            if constexpr (Util::is_complex_v<T>) {
                if (lhs[i].real() != Approx(comp_[i].real())
                                         .epsilon(epsilon_)
                                         .margin(margin_) ||
                    lhs[i].imag() != Approx(comp_[i].imag())
                                         .epsilon(epsilon_)
                                         .margin(margin_)) {
                    return false;
                }
            } else {
                if (lhs[i] !=
                    Approx(comp_[i]).epsilon(epsilon_).margin(margin_)) {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] std::string describe() const {
        std::ostringstream ss;
        ss << "is Approx to {";
        for (const auto &elem : comp_) {
            ss << elem << ", ";
        }
        ss << "}" << std::endl;
        return ss.str();
    }

    PLApprox &epsilon(Util::remove_complex_t<T> eps) {
        epsilon_ = eps;
        return *this;
    }
    PLApprox &margin(Util::remove_complex_t<T> m) {
        margin_ = m;
        return *this;
    }
};

/**
 * @brief Simple helper for PLApprox for the cases when the class template
 * deduction does not work well.
 */
template <typename T, class Alloc>
PLApprox<T, Alloc> approx(const std::vector<T, Alloc> &vec) {
    return PLApprox<T, Alloc>(vec);
}

template <typename T, class Alloc>
std::ostream &operator<<(std::ostream &os, const PLApprox<T, Alloc> &approx) {
    os << approx.describe();
    return os;
}
template <class T, class AllocA, class AllocB>
bool operator==(const std::vector<T, AllocA> &lhs,
                const PLApprox<T, AllocB> &rhs) {
    return rhs.compare(lhs);
}
template <class T, class AllocA, class AllocB>
bool operator!=(const std::vector<T, AllocA> &lhs,
                const PLApprox<T, AllocB> &rhs) {
    return !rhs.compare(lhs);
}

template <class PrecisionT> struct PLApproxComplex {
    const std::complex<PrecisionT> comp_;

    explicit PLApproxComplex(const std::complex<PrecisionT> &comp)
        : comp_{comp} {}

    PrecisionT margin_{};
    PrecisionT epsilon_ = std::numeric_limits<float>::epsilon() * 100;

    [[nodiscard]] bool compare(const std::complex<PrecisionT> &lhs) const {
        return (lhs.real() ==
                Approx(comp_.real()).epsilon(epsilon_).margin(margin_)) &&
               (lhs.imag() ==
                Approx(comp_.imag()).epsilon(epsilon_).margin(margin_));
    }
    [[nodiscard]] std::string describe() const {
        std::ostringstream ss;
        ss << "is Approx to " << comp_;
        return ss.str();
    }
    PLApproxComplex &epsilon(PrecisionT eps) {
        epsilon_ = eps;
        return *this;
    }
    PLApproxComplex &margin(PrecisionT m) {
        margin_ = m;
        return *this;
    }
};

template <class T>
bool operator==(const std::complex<T> &lhs, const PLApproxComplex<T> &rhs) {
    return rhs.compare(lhs);
}
template <class T>
bool operator!=(const std::complex<T> &lhs, const PLApproxComplex<T> &rhs) {
    return !rhs.compare(lhs);
}

template <typename T> PLApproxComplex<T> approx(const std::complex<T> &val) {
    return PLApproxComplex<T>{val};
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const PLApproxComplex<T> &approx) {
    os << approx.describe();
    return os;
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t, class AllocA, class AllocB>
inline bool
isApproxEqual(const std::vector<Data_t, AllocA> &data1,
              const std::vector<Data_t, AllocB> &data2,
              const typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    return data1 == PLApprox<Data_t, AllocB>(data2).epsilon(eps);
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data 1.
 * @param data2 StateVector data 2.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool
isApproxEqual(const Data_t &data1, const Data_t &data2,
              typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    return !(data1.real() != Approx(data2.real()).epsilon(eps) ||
             data1.imag() != Approx(data2.imag()).epsilon(eps));
}

/**
 * @brief Utility function to compare complex statevector data.
 *
 * @tparam Data_t Floating point data-type.
 * @param data1 StateVector data array pointer 1.
 * @param length1 StateVector data array pointer 1.
 * @param data2 StateVector data array pointer 2.
 * @param length2 StateVector data array pointer 1.
 * @return true Data are approximately equal.
 * @return false Data are not approximately equal.
 */
template <class Data_t>
inline bool
isApproxEqual(const Data_t *data1, const size_t length1, const Data_t *data2,
              const size_t length2,
              typename Data_t::value_type eps =
                  std::numeric_limits<typename Data_t::value_type>::epsilon() *
                  100) {
    if (data1 == data2) {
        return true;
    }

    if (length1 != length2) {
        return false;
    }

    for (size_t idx = 0; idx < length1; idx++) {
        if (!isApproxEqual(data1[idx], data2[idx], eps)) {
            return false;
        }
    }
    return true;
}

template <class PrecisionT> struct PrecisionToName;

template <> struct PrecisionToName<float> {
    constexpr static auto value = "float";
};
template <> struct PrecisionToName<double> {
    constexpr static auto value = "double";
};

template <typename T> using TestVector = std::vector<T, AlignedAllocator<T>>;

/**
 * @brief Multiplies every value in a dataset by a given complex scalar value.
 *
 * @tparam Data_t Precision of complex data type. Supports float and double
 * data.
 * @param data Data to be scaled.
 * @param scalar Scalar value.
 */
template <class Data_t, class Alloc>
void scaleVector(std::vector<std::complex<Data_t>, Alloc> &data,
                 std::complex<Data_t> scalar) {
    std::transform(
        data.begin(), data.end(), data.begin(),
        [scalar](const std::complex<Data_t> &c) { return c * scalar; });
}

/**
 * @brief Multiplies every value in a dataset by a given complex scalar value.
 *
 * @tparam Data_t Precision of complex data type. Supports float and double
 * data.
 * @param data Data to be scaled.
 * @param scalar Scalar value.
 */
template <class Data_t, class Alloc>
void scaleVector(std::vector<std::complex<Data_t>, Alloc> &data,
                 Data_t scalar) {
    std::transform(
        data.begin(), data.end(), data.begin(),
        [scalar](const std::complex<Data_t> &c) { return c * scalar; });
}

/**
 * @brief create |0>^N
 */
template <typename ComplexT>
auto createZeroState(size_t num_qubits) -> TestVector<ComplexT> {
    TestVector<ComplexT> res(size_t{1U} << num_qubits, {0.0, 0.0},
                             getBestAllocator<ComplexT>());
    res[0] = ComplexT{1.0, 0.0};
    return res;
}

/**
 * @brief create |+>^N
 */
template <typename PrecisionT>
auto createPlusState(size_t num_qubits)
    -> TestVector<std::complex<PrecisionT>> {
    TestVector<std::complex<PrecisionT>> res(
        size_t{1U} << num_qubits, 1.0,
        getBestAllocator<std::complex<PrecisionT>>());
    for (auto &elem : res) {
        elem /= std::sqrt(1U << num_qubits);
    }
    return res;
}

/**
 * @brief create a random state
 */
template <typename PrecisionT, class RandomEngine>
auto createRandomStateVectorData(RandomEngine &re, size_t num_qubits)
    -> TestVector<std::complex<PrecisionT>> {
    TestVector<std::complex<PrecisionT>> res(
        size_t{1U} << num_qubits, 0.0,
        getBestAllocator<std::complex<PrecisionT>>());
    std::uniform_real_distribution<PrecisionT> dist;
    for (size_t idx = 0; idx < (size_t{1U} << num_qubits); idx++) {
        res[idx] = {dist(re), dist(re)};
    }

    scaleVector(res, std::complex<PrecisionT>{1.0, 0.0} /
                         std::sqrt(squaredNorm(res.data(), res.size())));
    return res;
}

/**
 * @brief Create an arbitrary product state in X- or Z-basis.
 *
 * Example: createProductState("+01") will produce |+01> state.
 * Note that the wire index starts from the left.
 */
template <typename PrecisionT, typename ComplexT = std::complex<PrecisionT>>
auto createProductState(std::string_view str) -> TestVector<ComplexT> {
    using Pennylane::Util::INVSQRT2;
    TestVector<ComplexT> st(getBestAllocator<ComplexT>());
    st.resize(1U << str.length());

    std::vector<PrecisionT> zero{1.0, 0.0};
    std::vector<PrecisionT> one{0.0, 1.0};

    std::vector<PrecisionT> plus{INVSQRT2<PrecisionT>(),
                                 INVSQRT2<PrecisionT>()};
    std::vector<PrecisionT> minus{INVSQRT2<PrecisionT>(),
                                  -INVSQRT2<PrecisionT>()};

    for (size_t k = 0; k < (size_t{1U} << str.length()); k++) {
        PrecisionT elem = 1.0;
        for (size_t n = 0; n < str.length(); n++) {
            char c = str[n];
            const size_t wire = str.length() - 1 - n;
            switch (c) {
            case '0':
                elem *= zero[(k >> wire) & 1U];
                break;
            case '1':
                elem *= one[(k >> wire) & 1U];
                break;
            case '+':
                elem *= plus[(k >> wire) & 1U];
                break;
            case '-':
                elem *= minus[(k >> wire) & 1U];
                break;
            default:
                PL_ABORT("Unknown character in the argument.");
            }
        }
        st[k] = elem;
    }
    return st;
}

/**
 * @brief Create non-trivial statevector data using the provided StateVectorT.
 *
 * @tparam StateVectorT Backend used to generate data
 * @param num_qubits number of qubits
 * @return std::vector<typename StateVectorT::ComplexT>>
 */
template <class StateVectorT>
auto createNonTrivialState(size_t num_qubits = 3) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    size_t data_size = Util::exp2(num_qubits);

    std::vector<ComplexT> arr(data_size, ComplexT{0, 0});
    arr[0] = ComplexT{1, 0};
    StateVectorT Measured_StateVector(arr.data(), data_size);

    std::vector<std::string> gates;
    std::vector<std::vector<size_t>> wires;
    std::vector<bool> inv_op(num_qubits * 2, false);
    std::vector<std::vector<PrecisionT>> phase;

    PrecisionT initial_phase = 0.7;
    for (size_t n_qubit = 0; n_qubit < num_qubits; n_qubit++) {
        gates.emplace_back("RX");
        gates.emplace_back("RY");

        wires.push_back({n_qubit});
        wires.push_back({n_qubit});

        phase.push_back({initial_phase});
        phase.push_back({initial_phase});
        initial_phase -= 0.2;
    }
    Measured_StateVector.applyOperations(gates, wires, inv_op, phase);

    return Measured_StateVector.getDataVector();
}

/**
 * @brief Fills the empty vectors with the CSR (Compressed Sparse Row) sparse
 * matrix representation for a tri-diagonal + periodic boundary conditions
 * Hamiltonian.
 *
 * @tparam PrecisionT data float point precision.
 * @tparam IndexT integer type used as indices of the sparse matrix.
 * @param row_map the j element encodes the total number of non-zeros above
 * row j.
 * @param entries column indices.
 * @param values  matrix non-zero elements.
 * @param numRows matrix number of rows.
 */
template <class ComplexT, class IndexT>
void write_CSR_vectors(std::vector<IndexT> &row_map,
                       std::vector<IndexT> &entries,
                       std::vector<ComplexT> &values, IndexT numRows) {
    const ComplexT SC_ONE = 1.0;

    row_map.resize(numRows + 1);
    for (IndexT rowIdx = 1; rowIdx < static_cast<IndexT>(row_map.size());
         ++rowIdx) {
        row_map[rowIdx] = row_map[rowIdx - 1] + 3;
    };
    const IndexT numNNZ = row_map[numRows];

    entries.resize(numNNZ);
    values.resize(numNNZ);
    for (IndexT rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        size_t idx = row_map[rowIdx];
        if (rowIdx == 0) {
            entries[0] = rowIdx;
            entries[1] = rowIdx + 1;
            entries[2] = numRows - 1;

            values[0] = SC_ONE;
            values[1] = -SC_ONE;
            values[2] = -SC_ONE;
        } else if (rowIdx == numRows - 1) {
            entries[idx] = 0;
            entries[idx + 1] = rowIdx - 1;
            entries[idx + 2] = rowIdx;

            values[idx] = -SC_ONE;
            values[idx + 1] = -SC_ONE;
            values[idx + 2] = SC_ONE;
        } else {
            entries[idx] = rowIdx - 1;
            entries[idx + 1] = rowIdx;
            entries[idx + 2] = rowIdx + 1;

            values[idx] = -SC_ONE;
            values[idx + 1] = SC_ONE;
            values[idx + 2] = -SC_ONE;
        }
    }
};

/**
 * @brief Compare std::vectors with same elements data type but different
 * allocators.
 *
 * @tparam T Element data type.
 * @tparam AllocA Allocator for the first vector.
 * @tparam AllocB Allocator for the second vector.
 * @param lhs First vector
 * @param rhs Second vector
 * @return true
 * @return false
 */
template <class T, class AllocA, class AllocB>
bool operator==(const std::vector<T, AllocA> &lhs,
                const std::vector<T, AllocB> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t idx = 0; idx < lhs.size(); idx++) {
        if (lhs[idx] != rhs[idx]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Define linearly spaced data [start, end]
 *
 * @tparam T Data type.
 * @param start Start position.
 * @param end End position.
 * @param num_points Number of data-points in range.
 * @return std::vector<T>
 */
template <class T>
auto linspace(T start, T end, size_t num_points) -> std::vector<T> {
    std::vector<T> data(num_points);
    T step = (end - start) / (num_points - 1);
    for (size_t i = 0; i < num_points; i++) {
        data[i] = start + (step * i);
    }
    return data;
}

template <typename RandomEngine>
std::vector<int> randomIntVector(RandomEngine &re, size_t size, int min,
                                 int max) {
    std::uniform_int_distribution<int> dist(min, max);
    std::vector<int> res;

    res.reserve(size);
    for (size_t i = 0; i < size; i++) {
        res.emplace_back(dist(re));
    }
    return res;
}

/**
 * @brief Generate random unitary matrix
 *
 * @tparam PrecisionT Floating point type
 * @tparam RandomEngine Random engine type
 * @param re Random engine instance
 * @param num_qubits Number of qubits
 * @return Generated unitary matrix in row-major format
 */
template <typename PrecisionT, class RandomEngine>
auto randomUnitary(RandomEngine &re, size_t num_qubits)
    -> std::vector<std::complex<PrecisionT>> {
    using ComplexT = std::complex<PrecisionT>;
    const size_t dim = (1U << num_qubits);
    std::vector<ComplexT> res(dim * dim, ComplexT{});

    std::normal_distribution<PrecisionT> dist;

    auto generator = [&dist, &re]() -> ComplexT {
        return ComplexT{dist(re), dist(re)};
    };

    std::generate(res.begin(), res.end(), generator);

    // Simple algorithm to make rows orthogonal with Gram-Schmidt
    // This algorithm is unstable but works for a small matrix.
    // Use QR decomposition when we have LAPACK support.

    for (size_t row2 = 0; row2 < dim; row2++) {
        ComplexT *row2_p = res.data() + row2 * dim;
        for (size_t row1 = 0; row1 < row2; row1++) {
            const ComplexT *row1_p = res.data() + row1 * dim;
            ComplexT dot12 = std::inner_product(
                row1_p, row1_p + dim, row2_p, std::complex<PrecisionT>(),
                ConstSum<PrecisionT>, ConstMultConj<PrecisionT>);

            ComplexT dot11 = squaredNorm(row1_p, dim);

            // orthogonalize row2
            std::transform(
                row2_p, row2_p + dim, row1_p, row2_p,
                [scale = dot12 / dot11](auto &elem2, const auto &elem1) {
                    return elem2 - scale * elem1;
                });
        }
    }

    // Normalize each row
    for (size_t row = 0; row < dim; row++) {
        ComplexT *row_p = res.data() + row * dim;
        PrecisionT norm2 = std::sqrt(squaredNorm(row_p, dim));

        // normalize row2
        std::transform(row_p, row_p + dim, row_p, [norm2](const auto c) {
            return (static_cast<PrecisionT>(1.0) / norm2) * c;
        });
    }
    return res;
}

#define PL_REQUIRE_THROWS_MATCHES(expr, type, message_match)                   \
    REQUIRE_THROWS_AS(expr, type);                                             \
    REQUIRE_THROWS_WITH(expr, Catch::Matchers::Contains(message_match));
#define PL_CHECK_THROWS_MATCHES(expr, type, message_match)                     \
    CHECK_THROWS_AS(expr, type);                                               \
    CHECK_THROWS_WITH(expr, Catch::Matchers::Contains(message_match));

} // namespace Pennylane::Util