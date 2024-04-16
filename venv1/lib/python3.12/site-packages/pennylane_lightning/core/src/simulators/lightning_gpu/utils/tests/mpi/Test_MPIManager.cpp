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
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "MPIManager.hpp"

using namespace Pennylane;
using namespace Pennylane::LightningGPU::MPI;

TEST_CASE("MPIManager ctor", "[MPIManager]") {
    SECTION("Default constructibility") {
        REQUIRE(std::is_constructible_v<MPIManager>);
    }

    SECTION("Construct with MPI_Comm") {
        REQUIRE(std::is_constructible_v<MPIManager, MPI_Comm>);
    }

    SECTION("Construct with args") {
        REQUIRE(std::is_constructible_v<MPIManager, int, char **>);
    }

    SECTION("MPIManager {MPIManager&}") {
        REQUIRE(std::is_copy_constructible_v<MPIManager>);
    }
}

TEMPLATE_TEST_CASE("MPIManager::Scatter", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply scatter") {
        std::vector<cp_t> sendBuf(size);
        int root = 0;
        cp_t result(2.0 * rank, 2.0 * rank + 1);
        if (rank == root) {
            for (size_t i = 0; i < sendBuf.size(); i++) {
                cp_t data(2.0 * i, 2.0 * i + 1);
                sendBuf[i] = data;
            }
        }

        auto recvBuf = mpi_manager.scatter<cp_t>(sendBuf, root);
        CHECK(recvBuf[0].real() == result.real());
        CHECK(recvBuf[0].imag() == result.imag());
    }

    SECTION("Apply Scatter") {
        std::vector<cp_t> sendBuf(size);
        std::vector<cp_t> recvBuf(1);
        int root = 0;
        cp_t result(2.0 * rank, 2.0 * rank + 1);
        if (rank == root) {
            for (size_t i = 0; i < sendBuf.size(); i++) {
                cp_t data(2.0 * i, 2.0 * i + 1);
                sendBuf[i] = data;
            }
        }

        mpi_manager.Scatter<cp_t>(sendBuf, recvBuf, root);
        CHECK(recvBuf[0].real() == result.real());
        CHECK(recvBuf[0].imag() == result.imag());
    }
}

TEMPLATE_TEST_CASE("MPIManager::Allgather", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply Allgather scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        std::vector<cp_t> recvBuf(size);

        mpi_manager.Allgather<cp_t>(sendBuf, recvBuf);

        for (size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply Allgather vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        std::vector<cp_t> recvBuf(mpi_manager.getSize());

        mpi_manager.Allgather<cp_t>(sendBuf, recvBuf);

        for (size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply allgather scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};

        auto recvBuf = mpi_manager.allgather<cp_t>(sendBuf);
        for (size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Apply allgather vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        auto recvBuf = mpi_manager.allgather<cp_t>(sendBuf);

        for (size_t i = 0; i < recvBuf.size(); i++) {
            CHECK(recvBuf[i].real() == static_cast<PrecisionT>(i));
            CHECK(recvBuf[i].imag() == static_cast<PrecisionT>(0));
        }
    }
}

TEMPLATE_TEST_CASE("MPIManager::Reduce", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply Reduce vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        std::vector<cp_t> recvBuf(1, {0, 0});

        mpi_manager.Reduce<cp_t>(sendBuf, recvBuf, 0, "sum");

        if (mpi_manager.getRank() == 0) {
            CHECK(recvBuf[0].real() ==
                  static_cast<PrecisionT>((size - 1) * size / 2));
            CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
        }
    }

    SECTION("Catch failures caused by unsupported ops") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        std::vector<cp_t> recvBuf(1, {0, 0});
        REQUIRE_THROWS_WITH(
            mpi_manager.Reduce<cp_t>(sendBuf, recvBuf, 0, "SUM"),
            Catch::Matchers::Contains("Op not supported"));
    }

    SECTION("Catch failures caused by unsupported ops") {
        std::vector<std::string> sendBuf(1, "test");
        std::vector<std::string> recvBuf(1, "test");
        REQUIRE_THROWS_WITH(
            mpi_manager.Reduce<std::string>(sendBuf, recvBuf, 0, "SUM"),
            Catch::Matchers::Contains("Type not supported"));
    }
}

TEMPLATE_TEST_CASE("MPIManager::Allreduce", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    SECTION("Apply Allreduce scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        cp_t recvBuf;

        mpi_manager.Allreduce<cp_t>(sendBuf, recvBuf, "sum");
        CHECK(recvBuf.real() == static_cast<PrecisionT>((size - 1) * size / 2));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply allreduce scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        auto recvBuf = mpi_manager.allreduce<cp_t>(sendBuf, "sum");

        CHECK(recvBuf.real() == static_cast<PrecisionT>((size - 1) * size / 2));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Allreduce vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        std::vector<cp_t> recvBuf(1);

        mpi_manager.Allreduce<cp_t>(sendBuf, recvBuf, "sum");

        CHECK(recvBuf[0].real() ==
              static_cast<PrecisionT>((size - 1) * size / 2));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply allreduce vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        auto recvBuf = mpi_manager.allreduce<cp_t>(sendBuf, "sum");

        CHECK(recvBuf[0].real() ==
              static_cast<PrecisionT>((size - 1) * size / 2));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEMPLATE_TEST_CASE("MPIManager::Bcast", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();

    SECTION("Apply Bcast scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0};
        mpi_manager.Bcast<cp_t>(sendBuf, 0);
        CHECK(sendBuf.real() == static_cast<PrecisionT>(0));
        CHECK(sendBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Bcast vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0});
        mpi_manager.Bcast<cp_t>(sendBuf, 0);
        CHECK(sendBuf[0].real() == static_cast<PrecisionT>(0));
        CHECK(sendBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEMPLATE_TEST_CASE("MPIManager::Sendrecv", "[MPIManager]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;

    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int size = mpi_manager.getSize();

    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    SECTION("Apply Sendrecv scalar") {
        cp_t sendBuf = {static_cast<PrecisionT>(rank), 0.0};
        cp_t recvBuf = {-1.0, -1.0};

        mpi_manager.Sendrecv<cp_t>(sendBuf, dest, recvBuf, source);

        CHECK(recvBuf.real() == static_cast<PrecisionT>(source));
        CHECK(recvBuf.imag() == static_cast<PrecisionT>(0));
    }

    SECTION("Apply Sendrecv vector") {
        std::vector<cp_t> sendBuf(1, {static_cast<PrecisionT>(rank), 0.0});
        std::vector<cp_t> recvBuf(1, {-1.0, -1.0});
        mpi_manager.Sendrecv<cp_t>(sendBuf, dest, recvBuf, source);
        CHECK(recvBuf[0].real() == static_cast<PrecisionT>(source));
        CHECK(recvBuf[0].imag() == static_cast<PrecisionT>(0));
    }
}

TEST_CASE("MPIManager::split") {
    MPIManager mpi_manager(MPI_COMM_WORLD);
    REQUIRE(mpi_manager.getSize() == 2);

    int rank = mpi_manager.getRank();
    int color = rank % 2;
    int key = rank;
    auto newComm = mpi_manager.split(color, key);
    CHECK(newComm.getSize() * 2 == mpi_manager.getSize());
}