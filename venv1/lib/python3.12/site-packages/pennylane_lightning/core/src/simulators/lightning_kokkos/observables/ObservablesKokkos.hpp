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
#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Error.hpp"
#include "LinearAlgebraKokkos.hpp"
#include "Observables.hpp"
#include "StateVectorKokkos.hpp"
#include "Util.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::Observables;
using Pennylane::LightningKokkos::StateVectorKokkos;
using Pennylane::LightningKokkos::Util::SparseMV_Kokkos;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Observables {
/**
 * @brief Final class for named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
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
 * @brief Final class for Hermitian observables
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class HermitianObs final : public HermitianObsBase<StateVectorT> {
  private:
    using BaseType = HermitianObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;

    /**
     * @brief Create an Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObs(MatrixT matrix, std::vector<size_t> wires)
        : BaseType{matrix, wires} {}
};

/**
 * @brief Final class for TensorProdObs observables
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class TensorProdObs final : public TensorProdObsBase<StateVectorT> {
  private:
    using BaseType = TensorProdObsBase<StateVectorT>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

    /**
     * @brief Create an Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
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
 * @brief Final class for a general Hamiltonian representation as a sum of
 * observables.
 *
 * @tparam StateVectorT State vector class.
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
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<Hamiltonian<StateVectorT>> {
        return std::shared_ptr<Hamiltonian<StateVectorT>>(
            new Hamiltonian<StateVectorT>{std::move(coeffs), std::move(obs)});
    }

    /**
     * @brief Updates the statevector sv:->sv'.
     * @param sv The statevector to update
     */
    void applyInPlace(StateVectorT &sv) const override {
        StateVectorT buffer{sv.getNumQubits()};
        buffer.initZeros();
        for (size_t term_idx = 0; term_idx < this->coeffs_.size(); term_idx++) {
            StateVectorT tmp{sv};
            this->obs_[term_idx]->applyInPlace(tmp);
            LightningKokkos::Util::axpy_Kokkos<PrecisionT>(
                ComplexT{this->coeffs_[term_idx], 0.0}, tmp.getView(),
                buffer.getView(), tmp.getLength());
        }
        sv.updateData(buffer);
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
        StateVectorT d_sv_prime(sv.getNumQubits());

        SparseMV_Kokkos<PrecisionT, ComplexT>(
            sv.getView(), d_sv_prime.getView(), this->offsets_.data(),
            this->offsets_.size(), this->indices_.data(), this->data_.data(),
            this->data_.size());

        sv.updateData(d_sv_prime);
    }
};

/// @cond DEV
namespace detail {
using Pennylane::LightningKokkos::Util::axpy_Kokkos;
// Default implementation
template <class StateVectorT, bool use_openmp> struct HamiltonianApplyInPlace {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using KokkosVector = typename StateVectorT::KokkosVector;
    static void
    run(const std::vector<PrecisionT> &coeffs,
        const std::vector<std::shared_ptr<Observable<StateVectorT>>> &terms,
        StateVectorT &sv) {
        KokkosVector res("results", sv.getLength());
        Kokkos::deep_copy(res, ComplexT{0.0, 0.0});
        for (size_t term_idx = 0; term_idx < coeffs.size(); term_idx++) {
            StateVectorT tmp{sv};
            terms[term_idx]->applyInPlace(tmp);
            LightningKokkos::Util::axpy_Kokkos<PrecisionT>(
                ComplexT{coeffs[term_idx], 0.0}, tmp.getView(), res,
                tmp.getLength());
        }
        sv.updateData(res);
    }
};

} // namespace detail
/// @endcond

} // namespace Pennylane::LightningKokkos::Observables
