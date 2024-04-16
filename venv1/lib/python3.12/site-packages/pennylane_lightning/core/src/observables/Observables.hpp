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

#include <algorithm>
#include <complex>
#include <memory>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include "Error.hpp"
#include "Util.hpp"

#ifdef PL_USE_LAPACK
#include "UtilLinearAlg.hpp"
#endif

namespace Pennylane::Observables {
/**
 * @brief A base class (CRTP) for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT> class Observable {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

  protected:
    Observable() = default;
    Observable(const Observable &) = default;
    Observable(Observable &&) noexcept = default;
    Observable &operator=(const Observable &) = default;
    Observable &operator=(Observable &&) noexcept = default;

  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<StateVectorT> to
     * compare.
     */
    [[nodiscard]] virtual bool
    isEqual(const Observable<StateVectorT> &other) const = 0;

  public:
    virtual ~Observable() = default;

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    virtual void applyInPlace(StateVectorT &sv) const = 0;

    /**
     * @brief Apply unitaries of an observable to the given statevector in
     * place.
     *
     * @param sv Reference to StateVector object.
     * @param eigenValues Eigenvalues of an observable.
     * @param ob_wires Reference to a std::vector object which stores wires of
     * the observable.
     */
    virtual void
    applyInPlaceShots(StateVectorT &sv,
                      std::vector<std::vector<PrecisionT>> &eigenValues,
                      std::vector<size_t> &ob_wires) const = 0;

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<size_t> = 0;

    /**
     * @brief Get the observable data.
     *
     */
    [[nodiscard]] virtual auto getObs() const
        -> std::vector<std::shared_ptr<Observable<StateVectorT>>> {
        return {};
    };

    /**
     * @brief Get the coefficients of a Hamiltonian observable.
     */
    [[nodiscard]] virtual auto getCoeffs() const -> std::vector<PrecisionT> {
        return {};
    };

    /**
     * @brief Test whether this object is equal to another object
     */
    [[nodiscard]] auto operator==(const Observable<StateVectorT> &other) const
        -> bool {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] auto operator!=(const Observable<StateVectorT> &other) const
        -> bool {
        return !(*this == other);
    }
};

/**
 * @brief Base class for named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class NamedObsBase : public Observable<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

  protected:
    std::string obs_name_;
    std::vector<size_t> wires_;
    std::vector<PrecisionT> params_;

  private:
    [[nodiscard]] auto isEqual(const Observable<StateVectorT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const NamedObsBase<StateVectorT> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /**
     * @brief Construct a NamedObsBase object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObsBase(std::string obs_name, std::vector<size_t> wires,
                 std::vector<PrecisionT> params = {})
        : obs_name_{std::move(obs_name)}, wires_{std::move(wires)},
          params_{std::move(params)} {}

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    void applyInPlace(StateVectorT &sv) const override {
        sv.applyOperation(obs_name_, wires_, false, params_);
    }

    void applyInPlaceShots(StateVectorT &sv,
                           std::vector<std::vector<PrecisionT>> &eigenValues,
                           std::vector<size_t> &ob_wires) const override {
        ob_wires.clear();
        eigenValues.clear();
        ob_wires.push_back(wires_[0]);

        if (obs_name_ == "PauliX") {
            sv.applyOperation("Hadamard", wires_, false);
        } else if (obs_name_ == "PauliY") {
            sv.applyOperations({"PauliZ", "S", "Hadamard"},
                               {wires_, wires_, wires_}, {false, false, false});
        } else if (obs_name_ == "Hadamard") {
            const PrecisionT theta = -M_PI / 4.0;
            sv.applyOperation("RY", wires_, false, {theta});
        } else if (obs_name_ == "PauliZ") {
        } else if (obs_name_ == "Identity") {
        } else {
            PL_ABORT("Provided NamedObs does not support shot measurement.");
        }

        if (obs_name_ == "Identity") {
            eigenValues.push_back({1, 1});
        } else {
            eigenValues.push_back({1, -1});
        }
    }
};

/**
 * @brief Base class for Hermitian observables
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class HermitianObsBase : public Observable<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;

  protected:
    MatrixT matrix_;
    std::vector<size_t> wires_;

#ifdef PL_USE_LAPACK

  private:
    std::vector<PrecisionT> eigenVals_;
    MatrixT unitary_;
#endif

  private:
    [[nodiscard]] auto isEqual(const Observable<StateVectorT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const HermitianObsBase<StateVectorT> &>(other);

        return (matrix_ == other_cast.matrix_) && (wires_ == other_cast.wires_);
    }

  public:
    /**
     * @brief Create an Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObsBase(MatrixT matrix, std::vector<size_t> wires)
        : matrix_{std::move(matrix)}, wires_{std::move(wires)} {
        PL_ASSERT(matrix_.size() == Util::exp2(2 * wires_.size()));

#ifdef PL_USE_LAPACK
        std::vector<std::complex<PrecisionT>> mat(matrix_.size());

        std::transform(matrix_.begin(), matrix_.end(), mat.begin(),
                       [](ComplexT value) {
                           return static_cast<std::complex<PrecisionT>>(value);
                       });

        std::vector<std::complex<PrecisionT>> unitary(matrix_.size());

        Pennylane::Util::compute_diagonalizing_gates<PrecisionT>(
            Util::exp2(wires_.size()), Util::exp2(wires_.size()), mat,
            eigenVals_, unitary);

        unitary_.resize(unitary.size());
        std::transform(
            unitary.begin(), unitary.end(), unitary_.begin(),
            [](ComplexT value) { return static_cast<ComplexT>(value); });
#endif
    }

    [[nodiscard]] auto getMatrix() const -> const MatrixT & { return matrix_; }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        return "Hermitian";
    }

    void applyInPlace(StateVectorT &sv) const override {
        sv.applyMatrix(matrix_, wires_);
    }

    void applyInPlaceShots(
        [[maybe_unused]] StateVectorT &sv,
        [[maybe_unused]] std::vector<std::vector<PrecisionT>> &eigenValues,
        [[maybe_unused]] std::vector<size_t> &ob_wires) const override {
#ifdef PL_USE_LAPACK
        std::vector<std::complex<PrecisionT>> mat(matrix_.size());

        std::transform(matrix_.begin(), matrix_.end(), mat.begin(),
                       [](ComplexT value) {
                           return static_cast<std::complex<PrecisionT>>(value);
                       });

        PL_ABORT_IF_NOT(
            Pennylane::Util::is_Hermitian<PrecisionT>(Util::exp2(wires_.size()),
                                                      Util::exp2(wires_.size()),
                                                      mat) == true,
            "The matrix passed to HermitianObs is not a Hermitian matrix.");

        eigenValues.clear();
        ob_wires = wires_;
        sv.applyMatrix(unitary_, wires_);
        eigenValues.push_back(eigenVals_);
#else
        PL_ABORT("Hermitian observables do not support shot measurement. "
                 "Please link against Lapack.");
#endif
    }
};

/**
 * @brief Base class for a tensor product of observables.
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class TensorProdObsBase : public Observable<StateVectorT> {
  protected:
    std::vector<std::shared_ptr<Observable<StateVectorT>>> obs_;
    std::vector<size_t> all_wires_;

  private:
    [[nodiscard]] auto isEqual(const Observable<StateVectorT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const TensorProdObsBase<StateVectorT> &>(other);

        if (obs_.size() != other_cast.obs_.size()) {
            return false;
        }

        for (size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObsBase(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
        if (obs_.size() == 1 &&
            obs_[0]->getObsName().find('@') != std::string::npos) {
            // This would prevent the misuse of this constructor for creating
            // TensorProdObsBase(TensorProdObsBase).
            PL_ABORT("A new TensorProdObsBase observable cannot be created "
                     "from a single TensorProdObsBase.");
        }

        std::unordered_set<size_t> wires;
        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                PL_ABORT_IF(wires.contains(wire),
                            "All wires in observables must be disjoint.");
                wires.insert(wire);
            }
        }
        all_wires_ = std::vector<size_t>(wires.begin(), wires.end());
        std::sort(all_wires_.begin(), all_wires_.end());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     * @return std::shared_ptr<TensorProdObsBase<StateVectorT>>
     */
    static auto
    create(std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObsBase<StateVectorT>> {
        return std::shared_ptr<TensorProdObsBase<StateVectorT>>{
            new TensorProdObsBase(std::move(obs))};
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     * @return std::shared_ptr<TensorProdObsBase<StateVectorT>>
     */
    static auto
    create(std::vector<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObsBase<StateVectorT>> {
        return std::shared_ptr<TensorProdObsBase<StateVectorT>>{
            new TensorProdObsBase(std::move(obs))};
    }

    /**
     * @brief Get the number of operations in observable.
     *
     * @return size_t
     */
    [[nodiscard]] auto getSize() const -> size_t { return obs_.size(); }

    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return all_wires_;
    }

    void applyInPlace(StateVectorT &sv) const override {
        for (const auto &ob : obs_) {
            ob->applyInPlace(sv);
        }
    }

    /**
     * @brief Get the observable.
     */
    [[nodiscard]] auto getObs() const
        -> std::vector<std::shared_ptr<Observable<StateVectorT>>> override {
        return obs_;
    };

    void applyInPlaceShots(StateVectorT &sv,
                           std::vector<std::vector<PrecisionT>> &eigenValues,
                           std::vector<size_t> &ob_wires) const override {
        for (const auto &ob : obs_) {
            if (ob->getObsName().find("Hamiltonian") != std::string::npos) {
                PL_ABORT("Hamiltonian observables as a term of an TensorProd "
                         "observable do not "
                         "support shot measurement.");
            }
        }

        eigenValues.clear();
        ob_wires.clear();
        for (const auto &ob : obs_) {
            std::vector<std::vector<PrecisionT>> eigenVals;
            std::vector<size_t> ob_wire;
            ob->applyInPlaceShots(sv, eigenVals, ob_wire);
            ob_wires.push_back(ob_wire[0]);
            eigenValues.push_back(eigenVals[0]);
        }
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Util::operator<<;
        std::ostringstream obs_stream;
        const auto obs_size = obs_.size();
        for (size_t idx = 0; idx < obs_size; idx++) {
            obs_stream << obs_[idx]->getObsName();
            if (idx != obs_size - 1) {
                obs_stream << " @ ";
            }
        }
        return obs_stream.str();
    }
};

/**
 * @brief Base class for a general Hamiltonian representation as a sum of
 * observables.
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class HamiltonianBase : public Observable<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

  protected:
    std::vector<PrecisionT> coeffs_;
    std::vector<std::shared_ptr<Observable<StateVectorT>>> obs_;

  private:
    [[nodiscard]] bool
    isEqual(const Observable<StateVectorT> &other) const override {
        const auto &other_cast =
            static_cast<const HamiltonianBase<StateVectorT> &>(other);

        if (coeffs_ != other_cast.coeffs_) {
            return false;
        }

        for (size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    template <typename T1, typename T2>
    HamiltonianBase(T1 &&coeffs, T2 &&obs)
        : coeffs_{std::forward<T1>(coeffs)}, obs_{std::forward<T2>(obs)} {
        PL_ASSERT(coeffs_.size() == obs_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     * @return std::shared_ptr<HamiltonianBase<StateVectorT>>
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<HamiltonianBase<StateVectorT>> {
        return std::shared_ptr<HamiltonianBase<StateVectorT>>(
            new HamiltonianBase<StateVectorT>{std::move(coeffs),
                                              std::move(obs)});
    }

    void applyInPlace([[maybe_unused]] StateVectorT &sv) const override {
        PL_ABORT("For Hamiltonian Observables, the applyInPlace method must be "
                 "defined at the backend level.");
    }

    void applyInPlaceShots(
        [[maybe_unused]] StateVectorT &sv,
        [[maybe_unused]] std::vector<std::vector<PrecisionT>> &eigenValues,
        [[maybe_unused]] std::vector<size_t> &ob_wires) const override {
        PL_ABORT("Hamiltonian observables as a term of an observable do not "
                 "support shot measurement.");
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            wires.insert(ob_wires.begin(), ob_wires.end());
        }
        auto all_wires = std::vector<size_t>(wires.begin(), wires.end());
        std::sort(all_wires.begin(), all_wires.end());
        return all_wires;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Util::operator<<;
        std::ostringstream ss;
        ss << "Hamiltonian: { 'coeffs' : " << coeffs_ << ", 'observables' : [";
        const auto term_size = coeffs_.size();
        for (size_t t = 0; t < term_size; t++) {
            ss << obs_[t]->getObsName();
            if (t != term_size - 1) {
                ss << ", ";
            }
        }
        ss << "]}";
        return ss.str();
    }

    /**
     * @brief Get the observable.
     */
    [[nodiscard]] auto getObs() const
        -> std::vector<std::shared_ptr<Observable<StateVectorT>>> override {
        return obs_;
    };

    /**
     * @brief Get the coefficients of the observable.
     */
    [[nodiscard]] auto getCoeffs() const -> std::vector<PrecisionT> override {
        return coeffs_;
    };
};

/**
 * @brief Sparse representation of SparseHamiltonian<T>
 *
 * @tparam T Floating-point precision.
 */
template <class StateVectorT>
class SparseHamiltonianBase : public Observable<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
#ifdef _ENABLE_PLGPU
    using IdxT =
        typename std::conditional<std::is_same<PrecisionT, float>::value,
                                  int32_t, int64_t>::type;
#else
    using IdxT = std::size_t;
#endif

  protected:
    std::vector<ComplexT> data_;
    std::vector<IdxT> indices_;
    std::vector<IdxT> offsets_;
    std::vector<std::size_t> wires_;

  private:
    [[nodiscard]] bool
    isEqual(const Observable<StateVectorT> &other) const override {
        const auto &other_cast =
            static_cast<const SparseHamiltonianBase<StateVectorT> &>(other);
        return data_ == other_cast.data_ && indices_ == other_cast.indices_ &&
               offsets_ == other_cast.offsets_ && (wires_ == other_cast.wires_);
    }

  public:
    /**
     * @brief Create a SparseHamiltonianBase from data, indices and offsets in
     * CSR format.
     *
     * @param data Arguments to construct data
     * @param indices Arguments to construct indices
     * @param offsets Arguments to construct offsets
     * @param wires Arguments to construct wires
     */
    template <typename T1, typename T2, typename T3 = T2,
              typename T4 = std::vector<std::size_t>>
    SparseHamiltonianBase(T1 &&data, T2 &&indices, T3 &&offsets, T4 &&wires)
        : data_{std::forward<T1>(data)}, indices_{std::forward<T2>(indices)},
          offsets_{std::forward<T3>(offsets)}, wires_{std::forward<T4>(wires)} {
        PL_ASSERT(data_.size() == indices_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param data Argument to construct data
     * @param indices Argument to construct indices
     * @param offsets Argument to construct offsets
     * @param wires Argument to construct wires
     */
    static auto create(std::initializer_list<ComplexT> data,
                       std::initializer_list<IdxT> indices,
                       std::initializer_list<IdxT> offsets,
                       std::initializer_list<std::size_t> wires)
        -> std::shared_ptr<SparseHamiltonianBase<StateVectorT>> {
        // NOLINTBEGIN(*-move-const-arg)
        return std::shared_ptr<SparseHamiltonianBase<StateVectorT>>(
            new SparseHamiltonianBase<StateVectorT>{
                std::move(data), std::move(indices), std::move(offsets),
                std::move(wires)});
        // NOLINTEND(*-move-const-arg)
    }

    void applyInPlace([[maybe_unused]] StateVectorT &sv) const override {
        PL_ABORT("For SparseHamiltonian Observables, the applyInPlace method "
                 "must be "
                 "defined at the backend level.");
    }

    void applyInPlaceShots(
        [[maybe_unused]] StateVectorT &sv,
        [[maybe_unused]] std::vector<std::vector<PrecisionT>> &eigenValues,
        [[maybe_unused]] std::vector<size_t> &ob_wires) const override {
        PL_ABORT(
            "SparseHamiltonian observables do not support shot measurement.");
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream ss;
        ss << "SparseHamiltonian: {\n'data' : \n";
        for (const auto &d : data_) {
            ss << "{" << d.real() << ", " << d.imag() << "}, ";
        }
        ss << ",\n'indices' : \n";
        for (const auto &i : indices_) {
            ss << i << ", ";
        }
        ss << ",\n'offsets' : \n";
        for (const auto &o : offsets_) {
            ss << o << ", ";
        }
        ss << "\n}";
        return ss.str();
    }
    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    };
};

} // namespace Pennylane::Observables