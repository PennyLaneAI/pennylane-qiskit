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

#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"

using namespace Pennylane::LightningQubit;

template class Observables::NamedObs<StateVectorLQubitRaw<float>>;
template class Observables::NamedObs<StateVectorLQubitRaw<double>>;

template class Observables::NamedObs<StateVectorLQubitManaged<float>>;
template class Observables::NamedObs<StateVectorLQubitManaged<double>>;

template class Observables::HermitianObs<StateVectorLQubitRaw<float>>;
template class Observables::HermitianObs<StateVectorLQubitRaw<double>>;

template class Observables::HermitianObs<StateVectorLQubitManaged<float>>;
template class Observables::HermitianObs<StateVectorLQubitManaged<double>>;

template class Observables::TensorProdObs<StateVectorLQubitRaw<float>>;
template class Observables::TensorProdObs<StateVectorLQubitRaw<double>>;

template class Observables::TensorProdObs<StateVectorLQubitManaged<float>>;
template class Observables::TensorProdObs<StateVectorLQubitManaged<double>>;

template class Observables::Hamiltonian<StateVectorLQubitRaw<float>>;
template class Observables::Hamiltonian<StateVectorLQubitRaw<double>>;

template class Observables::Hamiltonian<StateVectorLQubitManaged<float>>;
template class Observables::Hamiltonian<StateVectorLQubitManaged<double>>;

template class Observables::SparseHamiltonian<StateVectorLQubitRaw<float>>;
template class Observables::SparseHamiltonian<StateVectorLQubitRaw<double>>;

template class Observables::SparseHamiltonian<StateVectorLQubitManaged<float>>;
template class Observables::SparseHamiltonian<StateVectorLQubitManaged<double>>;
