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
#pragma once

#include <complex>
#include <exception>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "CPUMemoryModel.hpp" // getAllocator
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Error.hpp"
#include "LinearAlgebra.hpp" // scaleAndAdd
#include "Macros.hpp"        // use_openmp
#include "Observables.hpp"
#include "SparseLinAlg.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "Util.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::Observables;

using Pennylane::LightningQubit::StateVectorLQubitManaged;
using Pennylane::LightningQubit::StateVectorLQubitRaw;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Observables {
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

/// @cond DEV
namespace detail {
using Pennylane::LightningQubit::Util::scaleAndAdd;

// Default implementation
template <class StateVectorT, bool use_openmp> struct HamiltonianApplyInPlace {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    static void
    run(const std::vector<PrecisionT> &coeffs,
        const std::vector<std::shared_ptr<Observable<StateVectorT>>> &terms,
        StateVectorT &sv) {
        if constexpr (std::is_same_v<typename StateVectorT::MemoryStorageT,
                                     MemoryStorageLocation::Internal>) {
            auto allocator = sv.allocator();
            std::vector<ComplexT, decltype(allocator)> res(
                sv.getLength(), ComplexT{0.0, 0.0}, allocator);
            for (size_t term_idx = 0; term_idx < coeffs.size(); term_idx++) {
                StateVectorT tmp(sv);
                terms[term_idx]->applyInPlace(tmp);
                scaleAndAdd(tmp.getLength(), ComplexT{coeffs[term_idx], 0.0},
                            tmp.getData(), res.data());
            }
            sv.updateData(res);
        } else if constexpr (std::is_same_v<
                                 typename StateVectorT::MemoryStorageT,
                                 MemoryStorageLocation::External>) {
            std::vector<ComplexT> res(sv.getLength(), ComplexT{0.0, 0.0});
            for (size_t term_idx = 0; term_idx < coeffs.size(); term_idx++) {
                std::vector<ComplexT> tmp_data_storage(
                    sv.getData(), sv.getData() + sv.getLength());
                StateVectorT tmp(tmp_data_storage.data(),
                                 tmp_data_storage.size());
                terms[term_idx]->applyInPlace(tmp);
                scaleAndAdd(tmp.getLength(), ComplexT{coeffs[term_idx], 0.0},
                            tmp.getData(), res.data());
            }
            sv.updateData(res);
        } else {
            /// LCOV_EXCL_START
            PL_ABORT("Undefined memory storage location for StateVectorT.");
            /// LCOV_EXCL_STOP
        }
    }
};

#if defined(_OPENMP)
template <class PrecisionT>
struct HamiltonianApplyInPlace<StateVectorLQubitManaged<PrecisionT>, true> {
    using ComplexT = std::complex<PrecisionT>;
    static void
    run(const std::vector<PrecisionT> &coeffs,
        const std::vector<
            std::shared_ptr<Observable<StateVectorLQubitManaged<PrecisionT>>>>
            &terms,
        StateVectorLQubitManaged<PrecisionT> &sv) {
        std::exception_ptr ex = nullptr;
        const size_t length = sv.getLength();
        auto allocator = sv.allocator();

        std::vector<ComplexT, decltype(allocator)> sum(length, ComplexT{},
                                                       allocator);

#pragma omp parallel default(none) firstprivate(length, allocator)             \
    shared(coeffs, terms, sv, sum, ex)
        {
            StateVectorLQubitManaged<PrecisionT> tmp(sv.getNumQubits());

            std::vector<ComplexT, decltype(allocator)> local_sv(
                length, ComplexT{}, allocator);

#pragma omp for
            for (size_t term_idx = 0; term_idx < terms.size(); term_idx++) {
                try {
                    tmp.updateData(sv.getDataVector());
                    terms[term_idx]->applyInPlace(tmp);
                } catch (...) {
#pragma omp critical
                    ex = std::current_exception();
#pragma omp cancel for
                }
                scaleAndAdd(length, ComplexT{coeffs[term_idx], 0.0},
                            tmp.getData(), local_sv.data());
            }
            if (ex) {
#pragma omp cancel parallel
                std::rethrow_exception(ex);
            } else {
#pragma omp critical
                scaleAndAdd(length, ComplexT{1.0, 0.0}, local_sv.data(),
                            sum.data());
            }
        }

        sv.updateData(sum);
    }
};

template <class PrecisionT>
struct HamiltonianApplyInPlace<StateVectorLQubitRaw<PrecisionT>, true> {
    using ComplexT = std::complex<PrecisionT>;
    static void run(const std::vector<PrecisionT> &coeffs,
                    const std::vector<std::shared_ptr<
                        Observable<StateVectorLQubitRaw<PrecisionT>>>> &terms,
                    StateVectorLQubitRaw<PrecisionT> &sv) {
        std::exception_ptr ex = nullptr;
        const size_t length = sv.getLength();
        std::vector<ComplexT> sum(length, ComplexT{});

#pragma omp parallel default(none) firstprivate(length)                        \
    shared(coeffs, terms, sv, sum, ex)
        {
            std::unique_ptr<std::vector<ComplexT>> tmp_data_storage{nullptr};
            std::unique_ptr<StateVectorLQubitRaw<PrecisionT>> tmp{nullptr};
            std::unique_ptr<std::vector<ComplexT>> local_sv{nullptr};

            try {
                tmp_data_storage.reset(new std::vector<ComplexT>(
                    sv.getData(), sv.getData() + sv.getLength()));
                tmp.reset(new StateVectorLQubitRaw<PrecisionT>(
                    tmp_data_storage->data(), tmp_data_storage->size()));
                local_sv.reset(new std::vector<ComplexT>(length, ComplexT{}));
            } catch (...) {
#pragma omp critical
                ex = std::current_exception();
            }
            if (ex) {
#pragma omp cancel parallel
                std::rethrow_exception(ex);
            }

#pragma omp for
            for (size_t term_idx = 0; term_idx < terms.size(); term_idx++) {
                std::copy(sv.getData(), sv.getData() + sv.getLength(),
                          tmp_data_storage->data());
                try {
                    terms[term_idx]->applyInPlace(*tmp);
                } catch (...) {
#pragma omp critical
                    ex = std::current_exception();
#pragma omp cancel for
                }
                scaleAndAdd(length, ComplexT{coeffs[term_idx], 0.0},
                            tmp->getData(), local_sv->data());
            }
            if (ex) {
#pragma omp cancel parallel
                std::rethrow_exception(ex);
            } else {
#pragma omp critical
                scaleAndAdd(length, ComplexT{1.0, 0.0}, local_sv->data(),
                            sum.data());
            }
        }

        sv.updateData(sum);
    }
};

#endif

} // namespace detail
/// @endcond

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

    void applyInPlace(StateVectorT &sv) const override {
        detail::HamiltonianApplyInPlace<
            StateVectorT, Pennylane::Util::use_openmp>::run(this->coeffs_,
                                                            this->obs_, sv);
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
        // NOLINTBEGIN(*-move-const-arg)
        return std::shared_ptr<SparseHamiltonian<StateVectorT>>(
            new SparseHamiltonian<StateVectorT>{
                std::move(data), std::move(indices), std::move(offsets),
                std::move(wires)});
        // NOLINTEND(*-move-const-arg)
    }

    /**
     * @brief Updates the statevector SV:->SV', where SV' = a*H*SV, and where H
     * is a sparse Hamiltonian.
     *
     */
    void applyInPlace(StateVectorT &sv) const override {
        PL_ABORT_IF_NOT(this->wires_.size() == sv.getNumQubits(),
                        "SparseH wire count does not match state-vector size");
        auto operator_vector = Util::apply_Sparse_Matrix(
            sv.getData(), sv.getLength(), this->offsets_.data(),
            this->offsets_.size(), this->indices_.data(), this->data_.data(),
            this->data_.size());

        sv.updateData(operator_vector);
    }
};

} // namespace Pennylane::LightningQubit::Observables