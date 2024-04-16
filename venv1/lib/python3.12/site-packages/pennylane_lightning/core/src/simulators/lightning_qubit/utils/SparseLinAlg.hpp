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
 * @file SparseLinAlg.hpp
 * @brief Contains sparse linear algebra utility functions.
 */

#pragma once
#include <complex>
#include <vector>

namespace Pennylane::LightningQubit::Util {
/**
 * @brief Apply a sparse matrix to a vector.
 *
 * @tparam fp_precision data float point precision.
 * @tparam index_type integer type used as indices of the sparse matrix.
 * @param vector_ptr    pointer to the vector.
 * @param vector_size   size of the vector.
 * @param row_map_ptr   Pointer to the row_map array. Elements of this array
 * return the number of non-zero terms in all rows before it.
 * @param row_map_size  number of elements in the row_map.
 * @param entries_ptr   pointer to the column indices of the non-zero elements.
 * @param values_ptr    non-zero elements.
 * @param numNNZ        number of non-zero elements.
 * @return result       result of the matrix vector multiplication.
 */
template <class fp_precision, class index_type>
std::vector<std::complex<fp_precision>>
apply_Sparse_Matrix(const std::complex<fp_precision> *vector_ptr,
                    const index_type vector_size, const index_type *row_map_ptr,
                    [[maybe_unused]] const index_type row_map_size,
                    const index_type *entries_ptr,
                    const std::complex<fp_precision> *values_ptr,
                    [[maybe_unused]] const index_type numNNZ) {
    std::vector<std::complex<fp_precision>> result;
    result.resize(vector_size);
    size_t count = 0;
    for (index_type i = 0; i < vector_size; i++) {
        result[i] = 0.0;
        for (index_type j = 0; j < row_map_ptr[i + 1] - row_map_ptr[i]; j++) {
            result[i] += values_ptr[count] * vector_ptr[entries_ptr[count]];
            count++;
        }
    }
    return result;
};
} // namespace Pennylane::LightningQubit::Util
