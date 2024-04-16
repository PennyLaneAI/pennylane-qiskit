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

#include "AdjointJacobianGPUMPI.hpp"
#include "JacobianData.hpp"
#include "JacobianDataMPI.hpp"
#include "MPIManager.hpp"

// using namespace Pennylane;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::MPI;
using Pennylane::LightningGPU::StateVectorCudaMPI;

// explicit instantiation
template class Pennylane::Algorithms::OpsData<StateVectorCudaMPI<float>>;
template class Pennylane::Algorithms::OpsData<StateVectorCudaMPI<double>>;

template class Pennylane::Algorithms::JacobianDataMPI<
    StateVectorCudaMPI<float>>;
template class Pennylane::Algorithms::JacobianDataMPI<
    StateVectorCudaMPI<double>>;

// explicit instantiation
template class Algorithms::AdjointJacobianMPI<StateVectorCudaMPI<float>>;
template class Algorithms::AdjointJacobianMPI<StateVectorCudaMPI<double>>;