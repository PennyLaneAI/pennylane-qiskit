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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "Gates.hpp"
#include "cuda.h"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace cuUtil;

} // namespace
/// @endcond

namespace Pennylane::LightningGPU {

/**
 * @brief Represents a cache for gate data to be accessible on the device.
 *
 * @tparam fp_t Floating point precision.
 */
template <class fp_t> class GateCache {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(fp_t{}));
    using gate_id = std::pair<std::string, fp_t>;

    GateCache() = delete;
    GateCache(const GateCache &other) = delete;
    GateCache(GateCache &&other) = delete;
    GateCache(bool populate, int device_id = 0, cudaStream_t stream_id = 0)
        : device_tag_(device_id, stream_id), total_alloc_bytes_{0} {
        if (populate) {
            defaultPopulateCache();
        }
    }
    GateCache(bool populate, const DevTag<int> &device_tag)
        : device_tag_{device_tag}, total_alloc_bytes_{0} {
        if (populate) {
            defaultPopulateCache();
        }
    }
    virtual ~GateCache(){};

    /**
     * @brief Add a default gate-set to the given cache. Assumes
     * initializer-list evaluated gates for "PauliX", "PauliY", "PauliZ",
     * "Hadamard", "S", "T", "SWAP", with "CNOT" and "CZ" represented as their
     * single-qubit values.
     *
     */
    void defaultPopulateCache() {
        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"Identity"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>()}));
        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"PauliX"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>()}));
        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"PauliY"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ZERO<CFP_t>(), -cuUtil::IMAG<CFP_t>(),
                cuUtil::IMAG<CFP_t>(), cuUtil::ZERO<CFP_t>()}));
        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"PauliZ"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), -cuUtil::ONE<CFP_t>()}));
        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"Hadamard"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::INVSQRT2<CFP_t>(), cuUtil::INVSQRT2<CFP_t>(),
                cuUtil::INVSQRT2<CFP_t>(), -cuUtil::INVSQRT2<CFP_t>()}));
        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"S"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), cuUtil::IMAG<CFP_t>()}));
        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"T"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(),
                cuUtil::ConstMultSC(cuUtil::SQRT2<fp_t>() / 2.0,
                                    cuUtil::ConstSum(cuUtil::ONE<CFP_t>(),
                                                     cuUtil::IMAG<CFP_t>()))}));
        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"SWAP"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>()}));

        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"CNOT"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>()}));

        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"Toffoli"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ZERO<CFP_t>(), cuUtil::ONE<CFP_t>(),
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>()}));

        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"CY"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ZERO<CFP_t>(), -cuUtil::IMAG<CFP_t>(),
                cuUtil::IMAG<CFP_t>(), cuUtil::ZERO<CFP_t>()}));

        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"CZ"}, 0.0)),
            std::forward_as_tuple(std::vector<CFP_t>{
                cuUtil::ONE<CFP_t>(), cuUtil::ZERO<CFP_t>(),
                cuUtil::ZERO<CFP_t>(), -cuUtil::ONE<CFP_t>()}));

        host_gates_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::make_pair(std::string{"CSWAP"}, 0.0)),
            std::forward_as_tuple(
                host_gates_.at(std::make_pair(std::string{"SWAP"}, 0.0))));

        for (const auto &[h_gate_k, h_gate_v] : host_gates_) {
            device_gates_.emplace(
                std::piecewise_construct, std::forward_as_tuple(h_gate_k),
                std::forward_as_tuple(h_gate_v.size(), device_tag_));
            device_gates_.at(h_gate_k).CopyHostDataToGpu(h_gate_v.data(),
                                                         h_gate_v.size());
            total_alloc_bytes_ += (sizeof(CFP_t) * h_gate_v.size());
        }
    }

    /**
     * @brief Check for the existence of a given gate.
     *
     * @param gate_id std::pair of gate_name and given parameter value.
     * @return true Gate exists in cache.
     * @return false Gate does not exist in cache.
     */
    bool gateExists(const gate_id &gate) {
        return ((host_gates_.find(gate) != host_gates_.end()) &&
                (device_gates_.find(gate)) != device_gates_.end());
    }
    /**
     * @brief Check for the existence of a given gate.
     *
     * @param gate_name String of gate name.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @return true Gate exists in cache.
     * @return false Gate does not exist in cache.
     */
    bool gateExists(const std::string &gate_name, fp_t gate_param) {
        return (host_gates_.find(std::make_pair(gate_name, gate_param)) !=
                host_gates_.end()) &&
               (device_gates_.find(std::make_pair(gate_name, gate_param)) !=
                device_gates_.end());
    }

    /**
     * @brief Add gate numerical value to the cache, indexed by the gate name
     * and parameter value.
     *
     * @param gate_name String representing the name of the given gate.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @param host_data Vector of the gate values in row-major order.
     */
    void add_gate(const std::string &gate_name, fp_t gate_param,
                  std::vector<CFP_t> host_data) {
        const auto gate_key = std::make_pair(gate_name, gate_param);
        host_gates_[gate_key] = std::move(host_data);
        auto &gate = host_gates_[gate_key];

        device_gates_.emplace(std::piecewise_construct,
                              std::forward_as_tuple(gate_key),
                              std::forward_as_tuple(gate.size(), device_tag_));
        device_gates_.at(gate_key).CopyHostDataToGpu(gate.data(), gate.size());

        this->total_alloc_bytes_ += (sizeof(CFP_t) * gate.size());
    }

    /**
     * @brief see `void add_gate(const std::string &gate_name, fp_t gate_param,
                  const std::vector<CFP_t> &host_data)`
     *
     * @param gate_key
     * @param host_data
     */
    void add_gate(const gate_id &gate_key, std::vector<CFP_t> host_data) {
        host_gates_[gate_key] = std::move(host_data);
        auto &gate = host_gates_[gate_key];

        device_gates_.emplace(std::piecewise_construct,
                              std::forward_as_tuple(gate_key),
                              std::forward_as_tuple(gate.size(), device_tag_));
        device_gates_.at(gate_key).CopyHostDataToGpu(gate.data(), gate.size());

        total_alloc_bytes_ += (sizeof(CFP_t) * gate.size());
    }

    /**
     * @brief Returns a pointer to the GPU device memory where the gate is
     * stored.
     *
     * @param gate_name String representing the name of the given gate.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @return const CFP_t* Pointer to gate values on device.
     */
    const CFP_t *get_gate_device_ptr(const std::string &gate_name,
                                     fp_t gate_param) {
        return device_gates_.at(std::make_pair(gate_name, gate_param))
            .getData();
    }
    const CFP_t *get_gate_device_ptr(const gate_id &gate_key) {
        return device_gates_.at(gate_key).getData();
    }
    auto get_gate_host(const std::string &gate_name, fp_t gate_param) {
        return host_gates_.at(std::make_pair(gate_name, gate_param));
    }
    auto get_gate_host(const gate_id &gate_key) {
        return host_gates_.at(gate_key);
    }

  private:
    const DevTag<int> device_tag_;
    std::size_t total_alloc_bytes_;

    struct gate_id_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &pair) const {
            return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
        }
    };

    std::unordered_map<gate_id, DataBuffer<CFP_t>, gate_id_hash> device_gates_;
    std::unordered_map<gate_id, std::vector<CFP_t>, gate_id_hash> host_gates_;
};
} // namespace Pennylane::LightningGPU
