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
#include "GateIndices.hpp"
#include "ConstantUtil.hpp"
#include "Util.hpp" // exp2, maxDecimalForQubit

namespace Pennylane::LightningQubit::Gates {
auto getIndicesAfterExclusion(const std::vector<size_t> &indicesToExclude,
                              size_t num_qubits) -> std::vector<size_t> {
    std::set<size_t> indices;
    for (size_t i = 0; i < num_qubits; i++) {
        indices.emplace(i);
    }
    for (const size_t &excludedIndex : indicesToExclude) {
        indices.erase(excludedIndex);
    }
    return {indices.begin(), indices.end()};
}

auto generateBitPatterns(const std::vector<size_t> &qubitIndices,
                         size_t num_qubits) -> std::vector<size_t> {
    std::vector<size_t> indices;
    indices.reserve(Pennylane::Util::exp2(qubitIndices.size()));
    indices.emplace_back(0);

    // NOLINTNEXTLINE(modernize-loop-convert)
    for (auto index_it = qubitIndices.rbegin(); index_it != qubitIndices.rend();
         index_it++) {
        const size_t value =
            Pennylane::Util::maxDecimalForQubit(*index_it, num_qubits);
        const size_t currentSize = indices.size();
        for (size_t j = 0; j < currentSize; j++) {
            indices.emplace_back(indices[j] + value);
        }
    }
    return indices;
}
} // namespace Pennylane::LightningQubit::Gates
