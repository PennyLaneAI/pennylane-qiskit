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
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace Pennylane::LightningGPU::MPI {

enum WireStatus { Default, Target, Control };

enum WiresSwapStatus : std::size_t { Local, Swappable, UnSwappable };

/**
 * @brief Create wire pairs for bit index swap and transform all control and
 * target wires to local ones.
 *
 * @param numLocalQubits Number of local qubits.
 * @param numTotalQubits Number of total qubits.
 * @param ctrls Vector of control wires.
 * @param tgts Vector of target wires.
 * @return wirePairs Wire pairs to be passed to SV bit index swap worker.
 */
inline std::vector<int2> createWirePairs(const int numLocalQubits,
                                         const int numTotalQubits,
                                         std::vector<int> &ctrls,
                                         std::vector<int> &tgts,
                                         std::vector<int> &statusWires) {
    std::vector<int2> wirePairs;
    int localbit = numLocalQubits - 1, globalbit = numLocalQubits;
    while (localbit >= 0 && globalbit < numTotalQubits) {
        if (statusWires[localbit] == WireStatus::Default &&
            statusWires[globalbit] != WireStatus::Default) {
            int2 wirepair = make_int2(localbit, globalbit);
            wirePairs.push_back(wirepair);
            if (statusWires[globalbit] == WireStatus::Control) {
                for (size_t k = 0; k < ctrls.size(); k++) {
                    if (ctrls[k] == globalbit) {
                        ctrls[k] = localbit;
                    }
                }
            } else {
                for (size_t k = 0; k < tgts.size(); k++) {
                    if (tgts[k] == globalbit) {
                        tgts[k] = localbit;
                    }
                }
            }
            std::swap(statusWires[localbit], statusWires[globalbit]);
        } else {
            if (statusWires[localbit] != WireStatus::Default) {
                localbit--;
            }
            if (statusWires[globalbit] == WireStatus::Default) {
                globalbit++;
            }
        }
    }
    return wirePairs;
}

/**
 * @brief Create wire pairs for bit index swap and transform all target wires to
 * local ones.
 *
 * @param numLocalQubits Number of local qubits.
 * @param numTotalQubits Number of total qubits.
 * @param tgts Vector of target wires.
 * @return wirePairs Wire pairs to be passed to SV bit index swap worker.
 */
inline std::vector<int2> createWirePairs(int numLocalQubits, int numTotalQubits,
                                         std::vector<int> &tgts,
                                         std::vector<int> &statusWires) {
    std::vector<int> ctrls;
    return createWirePairs(numLocalQubits, numTotalQubits, ctrls, tgts,
                           statusWires);
}

/**
 * @brief Create wire pairs for bit index swap and transform all target wires to
 * local ones for a vector of targets.
 *
 * @param numLocalQubits Number of local qubits.
 * @param numTotalQubits Number of total qubits.
 * @param tgts Vector of target wires vector.
 * @param localTgts Vector of local target wires vector.
 * @param tgtsSwapStatus Vector of swap status.
 * @param tgtswirePairs Vector of wire pairs for MPI operation.
 */

inline void tgtsVecProcess(const size_t numLocalQubits,
                           const size_t numTotalQubits,
                           const std::vector<std::vector<std::size_t>> &tgts,
                           std::vector<std::vector<size_t>> &localTgts,
                           std::vector<std::size_t> &tgtsSwapStatus,
                           std::vector<std::vector<int2>> &tgtswirePairs) {
    std::vector<std::vector<size_t>> tgtsIntTrans;
    tgtsIntTrans.reserve(tgts.size());

    for (const auto &vec : tgts) {
        std::vector<size_t> tmpVecInt(
            vec.size()); // Reserve memory for efficiency

        std::transform(vec.begin(), vec.end(), tmpVecInt.begin(),
                       [&](std::size_t x) { return numTotalQubits - 1 - x; });
        tgtsIntTrans.push_back(std::move(tmpVecInt));
    }

    for (const auto &vec : tgtsIntTrans) {
        std::vector<int> statusWires(numTotalQubits, WireStatus::Default);

        for (auto &v : vec) {
            statusWires[v] = WireStatus::Target;
        }

        size_t StatusGlobalWires = std::reduce(
            statusWires.begin() + numLocalQubits, statusWires.end());

        if (!StatusGlobalWires) {
            tgtsSwapStatus.push_back(WiresSwapStatus::Local);
            localTgts.push_back(vec);
        } else {
            size_t counts_global_wires = std::count_if(
                statusWires.begin(), statusWires.begin() + numLocalQubits,
                [](int i) { return i != WireStatus::Default; });
            size_t counts_local_wires_avail =
                numLocalQubits - (vec.size() - counts_global_wires);
            // Check if there are sufficent number of local wires for bit
            // swap
            if (counts_global_wires <= counts_local_wires_avail) {
                tgtsSwapStatus.push_back(WiresSwapStatus::Swappable);

                std::vector<int> localVec(vec.size());
                std::transform(vec.begin(), vec.end(), localVec.begin(),
                               [&](size_t x) { return static_cast<int>(x); });
                auto wirePairs = createWirePairs(numLocalQubits, numTotalQubits,
                                                 localVec, statusWires);
                std::vector<size_t> localVecSizeT(localVec.size());
                std::transform(localVec.begin(), localVec.end(),
                               localVecSizeT.begin(),
                               [&](int x) { return static_cast<size_t>(x); });
                localTgts.push_back(localVecSizeT);
                tgtswirePairs.push_back(wirePairs);
            } else {
                tgtsSwapStatus.push_back(WiresSwapStatus::UnSwappable);
                localTgts.push_back(vec);
            }
        }
    }
}
} // namespace Pennylane::LightningGPU::MPI