// Copyright 2022-2023 Xanadu Quantum Technologies Inc. and contributors.

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

#include <functional>
#include <vector>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "LinearAlg.hpp"
#include "Observables.hpp"
#include "StateVectorCudaManaged.hpp"
#include "Util.hpp"
#include "cuError.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::Observables;
using namespace Pennylane::LightningGPU::Util;
using Pennylane::LightningGPU::StateVectorCudaManaged;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::Observables {

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <typename StateVectorT>
class NamedObs final : public NamedObsBase<StateVectorT> {
  private:
    using BaseType = NamedObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObs(std::string obs_name, std::vector<size_t> wires,
             std::vector<PrecisionT> params = {})
        : BaseType{obs_name, wires, params} {
        using Pennylane::Gates::Constant::gate_names;
        using Pennylane::Gates::Constant::gate_num_params;
        using Pennylane::Gates::Constant::gate_wires;

        const auto gate_op = lookup(reverse_pairs(gate_names),
                                    std::string_view{this->obs_name_});
        PL_ASSERT(lookup(gate_wires, gate_op) == this->wires_.size());
        PL_ASSERT(lookup(gate_num_params, gate_op) == this->params_.size());
    }
};

/**
 * @brief Class models arbitrary Hermitian observables
 *
 */

template <class StateVectorT>
class HermitianObs final : public HermitianObsBase<StateVectorT> {
  private:
    using BaseType = HermitianObsBase<StateVectorT>;
    inline static const MatrixHasher mh;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using MatrixT = std::vector<std::complex<PrecisionT>>;
    using ComplexT = typename StateVectorT::ComplexT;

    /**
     * @brief Create Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObs(MatrixT matrix, std::vector<size_t> wires)
        : BaseType{matrix, wires} {}

    auto getObsName() const -> std::string final {
        // To avoid collisions on cached GPU data, use matrix elements to
        // uniquely identify Hermitian
        // TODO: Replace with a performant hash function
        std::ostringstream obs_stream;
        obs_stream << "Hermitian" << mh(this->matrix_);
        return obs_stream.str();
    }
};

/**
 * @brief Class models Tensor product observables
 */
template <class StateVectorT>
class TensorProdObs final : public TensorProdObsBase<StateVectorT> {
  private:
    using BaseType = TensorProdObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObs(Ts &&...arg) : BaseType{arg...} {}

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     */
    static auto
    create(std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObs<StateVectorT>> {
        return std::shared_ptr<TensorProdObs<StateVectorT>>{
            new TensorProdObs(std::move(obs))};
    }

    static auto
    create(std::vector<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObs<StateVectorT>> {
        return std::shared_ptr<TensorProdObs<StateVectorT>>{
            new TensorProdObs(std::move(obs))};
    }
};

/**
 * @brief General Hamiltonian as a sum of observables.
 *
 */
template <class StateVectorT>
class Hamiltonian final : public HamiltonianBase<StateVectorT> {
  private:
    using BaseType = HamiltonianBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    template <typename T1, typename T2>
    explicit Hamiltonian(T1 &&coeffs, T2 &&obs) : BaseType{coeffs, obs} {}

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param coeffs Argument to construct coefficients
     * @param obs Argument to construct terms
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<Hamiltonian<StateVectorT>> {
        return std::shared_ptr<Hamiltonian<StateVectorT>>(
            new Hamiltonian<StateVectorT>{std::move(coeffs), std::move(obs)});
    }

    // to work with
    void applyInPlace(StateVectorT &sv) const override {
        using CFP_t = typename StateVectorT::CFP_t;
        std::unique_ptr<DataBuffer<CFP_t>> buffer =
            std::make_unique<DataBuffer<CFP_t>>(sv.getDataBuffer().getLength(),
                                                sv.getDataBuffer().getDevTag());
        buffer->zeroInit();

        for (size_t term_idx = 0; term_idx < this->coeffs_.size(); term_idx++) {
            StateVectorT tmp(sv);
            this->obs_[term_idx]->applyInPlace(tmp);
            scaleAndAddC_CUDA(
                std::complex<PrecisionT>{this->coeffs_[term_idx], 0.0},
                tmp.getData(), buffer->getData(), tmp.getLength(),
                tmp.getDataBuffer().getDevTag().getDeviceID(),
                tmp.getDataBuffer().getDevTag().getStreamID(),
                tmp.getCublasCaller());
        }
        sv.updateData(std::move(buffer));
    }
};

/**
 * @brief Sparse representation of Hamiltonian<StateVectorT>
 *
 */
template <class StateVectorT>
class SparseHamiltonian final : public SparseHamiltonianBase<StateVectorT> {
  private:
    using BaseType = SparseHamiltonianBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    // cuSparse required index type
    using IdxT = typename BaseType::IdxT;

    /**
     * @brief Create a SparseHamiltonian from data, indices and offsets in CSR
     * format.
     *
     * @param data Arguments to construct data
     * @param indices Arguments to construct indices
     * @param offsets Arguments to construct offsets
     * @param wires Arguments to construct wires
     */
    template <typename T1, typename T2, typename T3 = T2, typename T4>
    explicit SparseHamiltonian(T1 &&data, T2 &&indices, T3 &&offsets,
                               T4 &&wires)
        : BaseType{data, indices, offsets, wires} {}

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param data Argument to construct data
     * @param indices Argument to construct indices
     * @param offsets Argument to construct ofsets
     * @param wires Argument to construct wires
     */
    static auto create(std::initializer_list<ComplexT> data,
                       std::initializer_list<IdxT> indices,
                       std::initializer_list<IdxT> offsets,
                       std::initializer_list<std::size_t> wires)
        -> std::shared_ptr<SparseHamiltonian<StateVectorT>> {
        return std::shared_ptr<SparseHamiltonian<StateVectorT>>(
            new SparseHamiltonian<StateVectorT>{
                std::move(data), std::move(indices), std::move(offsets),
                std::move(wires)});
    }

    /**
     * @brief Updates the statevector SV:->SV', where SV' = a*H*SV, and where H
     * is a sparse Hamiltonian.
     *
     */
    void applyInPlace(StateVectorT &sv) const override {
        PL_ABORT_IF_NOT(this->wires_.size() == sv.getNumQubits(),
                        "SparseH wire count does not match state-vector size");
        using CFP_t = typename StateVectorT::CFP_t;

        const std::size_t nIndexBits = sv.getNumQubits();
        const std::size_t length = std::size_t{1} << nIndexBits;

        auto device_id = sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = sv.getDataBuffer().getDevTag().getStreamID();

        cusparseHandle_t handle = sv.getCusparseHandle();

        std::unique_ptr<DataBuffer<CFP_t>> d_sv_prime =
            std::make_unique<DataBuffer<CFP_t>>(length, device_id, stream_id,
                                                true);

        SparseMV_cuSparse<IdxT, PrecisionT, CFP_t>(
            this->offsets_.data(), static_cast<int64_t>(this->offsets_.size()),
            this->indices_.data(), this->data_.data(),
            static_cast<int64_t>(this->data_.size()), sv.getData(),
            d_sv_prime->getData(), device_id, stream_id, handle);
        sv.updateData(std::move(d_sv_prime));
    }
};

} // namespace Pennylane::LightningGPU::Observables
