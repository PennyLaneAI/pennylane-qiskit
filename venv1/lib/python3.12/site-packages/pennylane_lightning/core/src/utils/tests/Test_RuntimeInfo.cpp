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
#include <catch2/catch.hpp>

#include "Macros.hpp"
#include "RuntimeInfo.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEST_CASE("Runtime information is correct", "[Test_RuntimeInfo]") {
    INFO("RuntimeInfo::AVX " << RuntimeInfo::AVX());
    INFO("RuntimeInfo::AVX2 " << RuntimeInfo::AVX2());
    INFO("RuntimeInfo::AVX512F " << RuntimeInfo::AVX512F());
    INFO("RuntimeInfo::vendor " << RuntimeInfo::vendor());
    INFO("RuntimeInfo::brand " << RuntimeInfo::brand());
    REQUIRE(true);
}
