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
 * @file LinearAlgebraKokkos.hpp
 * Contains linear algebra utility functions.
 */

#pragma once

#include <Kokkos_Core.hpp>

#include "Error.hpp"

namespace Pennylane::LightningKokkos::Util {
/**
 * @brief @rst
 * Kokkos functor for :math:`y+=\alpha*x` operation.
 * @endrst
 */
template <class PrecisionT> struct axpy_KokkosFunctor {
    Kokkos::complex<PrecisionT> alpha;
    Kokkos::View<Kokkos::complex<PrecisionT> *> x;
    Kokkos::View<Kokkos::complex<PrecisionT> *> y;
    axpy_KokkosFunctor(Kokkos::complex<PrecisionT> alpha_,
                       Kokkos::View<Kokkos::complex<PrecisionT> *> x_,
                       Kokkos::View<Kokkos::complex<PrecisionT> *> y_) {
        alpha = alpha_;
        x = x_;
        y = y_;
    }
    KOKKOS_INLINE_FUNCTION void operator()(const size_t k) const {
        y[k] += alpha * x[k];
    }
};

/**
 * @brief @rst
 * Kokkos implementation of the :math:`y+=\alpha*x` operation.
 * @endrst
 * @param alpha Scalar to scale x
 * @param x Vector to add
 * @param y Vector to be added
 * @param length number of elements in x
 * */
template <class PrecisionT>
inline auto axpy_Kokkos(Kokkos::complex<PrecisionT> alpha,
                        Kokkos::View<Kokkos::complex<PrecisionT> *> x,
                        Kokkos::View<Kokkos::complex<PrecisionT> *> y,
                        size_t length) {
    Kokkos::parallel_for(length, axpy_KokkosFunctor<PrecisionT>(alpha, x, y));
}

/**
 * @brief @rst
 * Sparse matrix vector multiply functor :math: `y=A*x`.
 * @endrst
 */
template <class PrecisionT> struct SparseMV_KokkosFunctor {
    using KokkosVector = Kokkos::View<Kokkos::complex<PrecisionT> *>;
    using KokkosSizeTVector = Kokkos::View<size_t *>;

    KokkosVector x;
    KokkosVector y;
    KokkosVector data;
    KokkosSizeTVector indices;
    KokkosSizeTVector indptr;

    SparseMV_KokkosFunctor(KokkosVector x_, KokkosVector y_,
                           const KokkosVector data_,
                           const KokkosSizeTVector indices_,
                           const KokkosSizeTVector indptr_) {
        x = x_;
        y = y_;
        data = data_;
        indices = indices_;
        indptr = indptr_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t row) const {
        Kokkos::complex<PrecisionT> tmp = {0.0, 0.0};
        for (size_t j = indptr[row]; j < indptr[row + 1]; j++) {
            tmp += data[j] * x[indices[j]];
        }
        y[row] = tmp;
    }
};

/**
 * @brief @rst
 * Sparse matrix vector multiply :math: `y=A*x`.
 * @endrst
 * @param x Input vector
 * @param y Result vector
 * @param row_map_ptr   row_map array pointer.
 *                      The j element encodes the number of non-zeros
 above
 * row j.
 * @param row_map_size  row_map array size.
 * @param entries_ptr   pointer to an array with column indices of the
 * non-zero elements.
 * @param values_ptr    pointer to an array with the non-zero elements.
 * @param numNNZ        number of non-zero elements.
 */
template <class PrecisionT, class ComplexT>
void SparseMV_Kokkos(Kokkos::View<ComplexT *> x, Kokkos::View<ComplexT *> y,
                     const size_t *row_map, const size_t row_map_size,
                     const size_t *entries_ptr, const ComplexT *values_ptr,
                     const size_t numNNZ) {
    using ConstComplexHostView =
        Kokkos::View<const ComplexT *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ConstSizeTHostView =
        Kokkos::View<const size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using KokkosSizeTVector = Kokkos::View<size_t *>;
    using KokkosVector = Kokkos::View<ComplexT *>;

    KokkosVector kok_data("kokkos_sparese_matrix_vals", numNNZ);
    KokkosSizeTVector kok_entries_ptr("kokkos_entries_ptr", numNNZ);
    KokkosSizeTVector kok_row_map("kokkos_offsets", row_map_size);

    Kokkos::deep_copy(kok_data, ConstComplexHostView(values_ptr, numNNZ));

    Kokkos::deep_copy(kok_entries_ptr, ConstSizeTHostView(entries_ptr, numNNZ));
    Kokkos::deep_copy(kok_row_map, ConstSizeTHostView(row_map, row_map_size));

    Kokkos::parallel_for(row_map_size - 1,
                         SparseMV_KokkosFunctor<PrecisionT>(
                             x, y, kok_data, kok_entries_ptr, kok_row_map));
}

/**
 * @brief @rst
 * Kokkos functor of the :math:`real(conj(x)*y)` operation.
 * @endrst
 */
template <class PrecisionT> struct getRealOfComplexInnerProductFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> x;
    Kokkos::View<Kokkos::complex<PrecisionT> *> y;

    getRealOfComplexInnerProductFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> x_,
        Kokkos::View<Kokkos::complex<PrecisionT> *> y_) {
        x = x_;
        y = y_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k, PrecisionT &inner) const {
        inner += real(x[k]) * real(y[k]) + imag(x[k]) * imag(y[k]);
    }
};

/**
 * @brief @rst
 * Kokkos implementation of the :math:`real(conj(x)*y)` operation.
 * @endrst
 * @param x Input vector
 * @param y Input vector
 * @return :math:`real(conj(x)*y)`
 */
template <class PrecisionT>
inline auto
getRealOfComplexInnerProduct(Kokkos::View<Kokkos::complex<PrecisionT> *> x,
                             Kokkos::View<Kokkos::complex<PrecisionT> *> y)
    -> PrecisionT {
    PL_ASSERT(x.size() == y.size());
    PrecisionT inner = 0;
    Kokkos::parallel_reduce(
        x.size(), getRealOfComplexInnerProductFunctor<PrecisionT>(x, y), inner);
    return inner;
}

/**
 * @brief @rstimagine
 * Kokkos functor of the :math:`imag(conj(x)*y)` operation.
 * @endrst
 */
template <class PrecisionT> struct getImagOfComplexInnerProductFunctor {
    Kokkos::View<Kokkos::complex<PrecisionT> *> x;
    Kokkos::View<Kokkos::complex<PrecisionT> *> y;

    getImagOfComplexInnerProductFunctor(
        Kokkos::View<Kokkos::complex<PrecisionT> *> x_,
        Kokkos::View<Kokkos::complex<PrecisionT> *> y_) {
        x = x_;
        y = y_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_t k, PrecisionT &inner) const {
        inner += real(x[k]) * imag(y[k]) - imag(x[k]) * real(y[k]);
    }
};

/**
 * @brief @rst
 * Kokkos implementation of the :math:`imag(conj(x)*y)` operation.
 * @endrst
 * @param x Input vector
 * @param y Input vector
 * @return :math:`imag(conj(x)*y)`
 */
template <class PrecisionT>
inline auto
getImagOfComplexInnerProduct(Kokkos::View<Kokkos::complex<PrecisionT> *> x,
                             Kokkos::View<Kokkos::complex<PrecisionT> *> y)
    -> PrecisionT {
    PL_ASSERT(x.size() == y.size());
    PrecisionT inner = 0;
    Kokkos::parallel_reduce(
        x.size(), getImagOfComplexInnerProductFunctor<PrecisionT>(x, y), inner);
    return inner;
}

} // namespace Pennylane::LightningKokkos::Util
