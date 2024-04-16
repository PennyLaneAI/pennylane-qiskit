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
#include <bit>
#include <complex>
#include <cstring>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "DataBuffer.hpp"
#include "Error.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::MPI {
// LCOV_EXCL_START
inline void errhandler(int errcode, const char *str) {
    char msg[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(errcode, msg, &resultlen);
    fprintf(stderr, "%s: %s\n", str, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}
// LCOV_EXCL_STOP

#define PL_MPI_IS_SUCCESS(fn)                                                  \
    {                                                                          \
        int errcode;                                                           \
        errcode = (fn);                                                        \
        if (errcode != MPI_SUCCESS)                                            \
            errhandler(errcode, #fn);                                          \
    }

template <typename T> auto cppTypeToString() -> const std::string {
    const std::string typestr = std::type_index(typeid(T)).name();
    return typestr;
}

/**
 * @brief MPI operation class. Maintains MPI related operations.
 */
class MPIManager final {
  private:
    bool isExternalComm_;
    size_t rank_;
    size_t size_per_node_;
    size_t size_;
    MPI_Comm communicator_;

    std::string vendor_;
    size_t version_;
    size_t subversion_;

    /**
     * @brief Find C++ data type's corresponding MPI data type.
     *
     * @tparam T C++ data type.
     */
    template <typename T> auto getMPIDatatype() -> MPI_Datatype {
        auto it = cpp_mpi_type_map.find(cppTypeToString<T>());
        if (it != cpp_mpi_type_map.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Type not supported");
        }
    }

    /**
     * @brief Find operation string's corresponding MPI_Op type.
     *
     * @param op_str std::string of MPI_Op name.
     */
    auto getMPIOpType(const std::string &op_str) -> MPI_Op {
        auto it = cpp_mpi_op_map.find(op_str);
        if (it != cpp_mpi_op_map.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Op not supported");
        }
    }

    /**
     * @brief Map of std::string and MPI_Op.
     */
    std::unordered_map<std::string, MPI_Op> cpp_mpi_op_map = {
        {"op_null", MPI_OP_NULL}, {"max", MPI_MAX},
        {"min", MPI_MIN},         {"sum", MPI_SUM},
        {"prod", MPI_PROD},       {"land", MPI_LAND},
        {"band", MPI_BAND},       {"lor", MPI_LOR},
        {"bor", MPI_BOR},         {"lxor", MPI_LXOR},
        {"bxor", MPI_BXOR},       {"minloc", MPI_MINLOC},
        {"maxloc", MPI_MAXLOC},   {"replace", MPI_REPLACE},
    };

    /**
     * @brief Map of std::string and MPI_Datatype.
     */
    std::unordered_map<std::string, MPI_Datatype> cpp_mpi_type_map = {
        {cppTypeToString<char>(), MPI_CHAR},
        {cppTypeToString<signed char>(), MPI_SIGNED_CHAR},
        {cppTypeToString<unsigned char>(), MPI_UNSIGNED_CHAR},
        {cppTypeToString<wchar_t>(), MPI_WCHAR},
        {cppTypeToString<short>(), MPI_SHORT},
        {cppTypeToString<unsigned short>(), MPI_UNSIGNED_SHORT},
        {cppTypeToString<int>(), MPI_INT},
        {cppTypeToString<unsigned int>(), MPI_UNSIGNED},
        {cppTypeToString<long>(), MPI_LONG},
        {cppTypeToString<unsigned long>(), MPI_UNSIGNED_LONG},
        {cppTypeToString<long long>(), MPI_LONG_LONG_INT},
        {cppTypeToString<float>(), MPI_FLOAT},
        {cppTypeToString<double>(), MPI_DOUBLE},
        {cppTypeToString<long double>(), MPI_LONG_DOUBLE},
        {cppTypeToString<int8_t>(), MPI_INT8_T},
        {cppTypeToString<int16_t>(), MPI_INT16_T},
        {cppTypeToString<int32_t>(), MPI_INT32_T},
        {cppTypeToString<int64_t>(), MPI_INT64_T},
        {cppTypeToString<uint8_t>(), MPI_UINT8_T},
        {cppTypeToString<uint16_t>(), MPI_UINT16_T},
        {cppTypeToString<uint32_t>(), MPI_UINT32_T},
        {cppTypeToString<uint64_t>(), MPI_UINT64_T},
        {cppTypeToString<bool>(), MPI_C_BOOL},
        {cppTypeToString<std::complex<float>>(), MPI_C_FLOAT_COMPLEX},
        {cppTypeToString<std::complex<double>>(), MPI_C_DOUBLE_COMPLEX},
        {cppTypeToString<std::complex<long double>>(),
         MPI_C_LONG_DOUBLE_COMPLEX},
        {cppTypeToString<float2>(), MPI_C_FLOAT_COMPLEX},
        {cppTypeToString<cuComplex>(), MPI_C_FLOAT_COMPLEX},
        {cppTypeToString<cuFloatComplex>(), MPI_C_FLOAT_COMPLEX},
        {cppTypeToString<double2>(), MPI_C_DOUBLE_COMPLEX},
        {cppTypeToString<cuDoubleComplex>(), MPI_C_DOUBLE_COMPLEX},
        {cppTypeToString<custatevecIndex_t>(), MPI_INT64_T},
        // cuda related types
        {cppTypeToString<cudaIpcMemHandle_t>(), MPI_UINT8_T},
        {cppTypeToString<cudaIpcEventHandle_t>(), MPI_UINT8_T}};

    /**
     * @brief Set the MPI vendor.
     */
    void setVendor() {
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        int resultlen;

        PL_MPI_IS_SUCCESS(MPI_Get_library_version(version, &resultlen));

        std::string version_str = version;

        if (version_str.find("Open MPI") != std::string::npos) {
            vendor_ = "Open MPI";
        } else if (version_str.find("MPICH") != std::string::npos) {
            vendor_ = "MPICH";
        } else {
            PL_ABORT("Unsupported MPI implementation.\n");
        }
    }

    /**
     * @brief Set the MPI version.
     */
    void setVersion() {
        int version_int, subversion_int;
        PL_MPI_IS_SUCCESS(MPI_Get_version(&version_int, &subversion_int));
        version_ = static_cast<size_t>(version_int);
        subversion_ = static_cast<size_t>(subversion_int);
    }

    /**
     * @brief Set the number of processes per node in the communicator.
     */
    void setNumProcsPerNode() {
        MPI_Comm node_comm;
        int size_per_node_int;
        PL_MPI_IS_SUCCESS(
            MPI_Comm_split_type(this->getComm(), MPI_COMM_TYPE_SHARED,
                                this->getRank(), MPI_INFO_NULL, &node_comm));
        PL_MPI_IS_SUCCESS(MPI_Comm_size(node_comm, &size_per_node_int));
        size_per_node_ = static_cast<size_t>(size_per_node_int);
        int compare;
        PL_MPI_IS_SUCCESS(
            MPI_Comm_compare(MPI_COMM_WORLD, node_comm, &compare));
        if (compare != MPI_IDENT)
            PL_MPI_IS_SUCCESS(MPI_Comm_free(&node_comm));
        this->Barrier();
    }

    /**
     * @brief Check if the MPI configuration meets the cuQuantum.
     */
    void check_mpi_config() {
        // check if number of processes is power of two.
        // This is required by custatevec
        PL_ABORT_IF(std::has_single_bit(
                        static_cast<unsigned int>(this->getSize())) != true,
                    "Processes number is not power of two.");
        PL_ABORT_IF(std::has_single_bit(
                        static_cast<unsigned int>(size_per_node_)) != true,
                    "Number of processes per node is not power of two.");
    }

  public:
    MPIManager() : communicator_(MPI_COMM_WORLD) {
        int status = 0;
        MPI_Initialized(&status);
        if (!status) {
            PL_MPI_IS_SUCCESS(MPI_Init(nullptr, nullptr));
        }

        isExternalComm_ = true;
        int rank_int;
        int size_int;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communicator_, &rank_int));
        PL_MPI_IS_SUCCESS(MPI_Comm_size(communicator_, &size_int));

        rank_ = static_cast<size_t>(rank_int);
        size_ = static_cast<size_t>(size_int);

        setVendor();
        setVersion();
        setNumProcsPerNode();
        check_mpi_config();
    }

    MPIManager(MPI_Comm communicator) : communicator_(communicator) {
        int status = 0;
        MPI_Initialized(&status);
        if (!status) {
            PL_MPI_IS_SUCCESS(MPI_Init(nullptr, nullptr));
        }
        isExternalComm_ = true;
        int rank_int;
        int size_int;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communicator_, &rank_int));
        PL_MPI_IS_SUCCESS(MPI_Comm_size(communicator_, &size_int));

        rank_ = static_cast<size_t>(rank_int);
        size_ = static_cast<size_t>(size_int);

        setVendor();
        setVersion();
        setNumProcsPerNode();
        check_mpi_config();
    }

    MPIManager(int argc, char **argv) {
        int status = 0;
        MPI_Initialized(&status);
        if (!status) {
            PL_MPI_IS_SUCCESS(MPI_Init(&argc, &argv));
        }
        isExternalComm_ = false;
        communicator_ = MPI_COMM_WORLD;
        int rank_int;
        int size_int;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(communicator_, &rank_int));
        PL_MPI_IS_SUCCESS(MPI_Comm_size(communicator_, &size_int));

        rank_ = static_cast<size_t>(rank_int);
        size_ = static_cast<size_t>(size_int);

        setVendor();
        setVersion();
        setNumProcsPerNode();
        check_mpi_config();
    }

    MPIManager(const MPIManager &other) {
        int status = 0;
        MPI_Initialized(&status);
        if (!status) {
            PL_MPI_IS_SUCCESS(MPI_Init(nullptr, nullptr));
        }
        isExternalComm_ = true;
        rank_ = other.rank_;
        size_ = other.size_;
        MPI_Comm_dup(
            other.communicator_,
            &communicator_); // Avoid freeing other.communicator_ in ~MPIManager
        vendor_ = other.vendor_;
        version_ = other.version_;
        subversion_ = other.subversion_;
        size_per_node_ = other.size_per_node_;
    }

    // LCOV_EXCL_START
    ~MPIManager() {
        if (!isExternalComm_) {
            int initflag;
            int finflag;
            PL_MPI_IS_SUCCESS(MPI_Initialized(&initflag));
            PL_MPI_IS_SUCCESS(MPI_Finalized(&finflag));
            if (initflag && !finflag) {
                PL_MPI_IS_SUCCESS(MPI_Finalize());
            }
        } else {
            int compare;
            PL_MPI_IS_SUCCESS(
                MPI_Comm_compare(MPI_COMM_WORLD, communicator_, &compare));
            if (compare != MPI_IDENT)
                PL_MPI_IS_SUCCESS(MPI_Comm_free(&communicator_));
        }
    }
    // LCOV_EXCL_STOP

    // General MPI operations
    /**
     * @brief Get the process rank in the communicator.
     */
    auto getRank() const -> size_t { return rank_; }

    /**
     * @brief Get the process number in the communicator.
     */
    auto getSize() const -> size_t { return size_; }

    /**
     * @brief Get the number of processes per node in the communicator.
     */
    auto getSizeNode() const -> size_t { return size_per_node_; }

    /**
     * @brief Get the communicator.
     */
    MPI_Comm getComm() { return communicator_; }

    /**
     * @brief Get an elapsed time.
     */
    double getTime() { return MPI_Wtime(); }

    /**
     * @brief Get the MPI vendor.
     */
    auto getVendor() const -> const std::string & { return vendor_; }

    /**
     * @brief Get the MPI version.
     */
    auto getVersion() const -> std::tuple<size_t, size_t> {
        return {version_, subversion_};
    }

    /**
     * @brief MPI_Allgather wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param recvBuf Receive buffer vector.
     * @param sendCount Number of elements received from any process.
     */
    template <typename T>
    void Allgather(T &sendBuf, std::vector<T> &recvBuf, size_t sendCount = 1) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        if (sendCount != 1) {
            if (cppTypeToString<T>() != cppTypeToString<cudaIpcMemHandle_t>() &&
                cppTypeToString<T>() !=
                    cppTypeToString<cudaIpcEventHandle_t>()) {
                throw std::runtime_error(
                    "Unsupported MPI DataType implementation.\n");
            }
        }
        PL_ABORT_IF(recvBuf.size() != this->getSize(),
                    "Incompatible size of sendBuf and recvBuf.");

        int sendCountInt = static_cast<int>(sendCount);
        PL_MPI_IS_SUCCESS(MPI_Allgather(&sendBuf, sendCountInt, datatype,
                                        recvBuf.data(), sendCountInt, datatype,
                                        this->getComm()));
    }

    /**
     * @brief MPI_Allgather wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param sendCount Number of elements received from any process.
     * @return recvBuf Vector of receive buffer.
     */
    template <typename T> auto allgather(T &sendBuf) -> std::vector<T> {
        MPI_Datatype datatype = getMPIDatatype<T>();
        std::vector<T> recvBuf(this->getSize());
        PL_MPI_IS_SUCCESS(MPI_Allgather(&sendBuf, 1, datatype, recvBuf.data(),
                                        1, datatype, this->getComm()));
        return recvBuf;
    }

    /**
     * @brief MPI_Allgather wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param recvBuf Receive buffer vector.
     */
    template <typename T>
    void Allgather(std::vector<T> &sendBuf, std::vector<T> &recvBuf) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_ABORT_IF(recvBuf.size() != sendBuf.size() * this->getSize(),
                    "Incompatible size of sendBuf and recvBuf.");
        PL_MPI_IS_SUCCESS(MPI_Allgather(
            sendBuf.data(), sendBuf.size(), datatype, recvBuf.data(),
            sendBuf.size(), datatype, this->getComm()));
    }

    /**
     * @brief MPI_Allgather wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @return recvBuf Vector of receive buffer.
     */
    template <typename T>
    auto allgather(std::vector<T> &sendBuf) -> std::vector<T> {
        MPI_Datatype datatype = getMPIDatatype<T>();
        std::vector<T> recvBuf(sendBuf.size() * this->getSize());
        PL_MPI_IS_SUCCESS(MPI_Allgather(
            sendBuf.data(), sendBuf.size(), datatype, recvBuf.data(),
            sendBuf.size(), datatype, this->getComm()));
        return recvBuf;
    }

    /**
     * @brief MPI_Allreduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param recvBuf Receive buffer.
     * @param op_str String of MPI_Op.
     */
    template <typename T>
    void Allreduce(T &sendBuf, T &recvBuf, const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Allreduce(&sendBuf, &recvBuf, 1, datatype, op,
                                        this->getComm()));
    }

    /**
     * @brief MPI_Allreduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param op_str String of MPI_Op.
     * @return recvBuf Receive buffer.
     */
    template <typename T>
    auto allreduce(T &sendBuf, const std::string &op_str) -> T {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        T recvBuf;
        PL_MPI_IS_SUCCESS(MPI_Allreduce(&sendBuf, &recvBuf, 1, datatype, op,
                                        this->getComm()));
        return recvBuf;
    }

    /**
     * @brief MPI_Allreduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param recvBuf Receive buffer vector.
     * @param op_str String of MPI_Op.
     */
    template <typename T>
    void Allreduce(std::vector<T> &sendBuf, std::vector<T> &recvBuf,
                   const std::string &op_str) {
        PL_ABORT_IF(recvBuf.size() != sendBuf.size(),
                    "Incompatible size of sendBuf and recvBuf.");
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Allreduce(sendBuf.data(), recvBuf.data(),
                                        sendBuf.size(), datatype, op,
                                        this->getComm()));
    }

    /**
     * @brief MPI_Allreduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param op_str String of MPI_Op.
     * @return recvBuf Receive buffer.
     */
    template <typename T>
    auto allreduce(std::vector<T> &sendBuf, const std::string &op_str)
        -> std::vector<T> {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        std::vector<T> recvBuf(sendBuf.size());
        PL_MPI_IS_SUCCESS(MPI_Allreduce(sendBuf.data(), recvBuf.data(),
                                        sendBuf.size(), datatype, op,
                                        this->getComm()));
        return recvBuf;
    }

    /**
     * @brief MPI_Reduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param recvBuf Receive buffer.
     * @param root Rank of root process.
     * @param op_str String of MPI_Op.
     */
    template <typename T>
    void Reduce(T &sendBuf, T &recvBuf, size_t root,
                const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Reduce(&sendBuf, &recvBuf, 1, datatype, op, root,
                                     this->getComm()));
    }

    /**
     * @brief MPI_Reduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param recvBuf Receive buffer vector.
     * @param root Rank of root process.
     * @param op_str String of MPI_Op.
     */
    template <typename T>
    void Reduce(std::vector<T> &sendBuf, std::vector<T> &recvBuf, size_t root,
                const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Reduce(sendBuf.data(), recvBuf.data(),
                                     sendBuf.size(), datatype, op, root,
                                     this->getComm()));
    }

    /**
     * @brief MPI_Reduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer (DataBuffer type).
     * @param recvBuf Receive buffer (DataBuffer type).
     * @param root Rank of root process.
     * @param op_str String of MPI_Op.
     */
    template <typename T>
    void Reduce(DataBuffer<T> &sendBuf, DataBuffer<T> &recvBuf, size_t length,
                size_t root, const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Reduce(sendBuf.getData(), recvBuf.getData(),
                                     length, datatype, op, root,
                                     this->getComm()));
    }

    /**
     * @brief MPI_Reduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param recvBuf Receive buffer vector.
     * @param root Rank of root process.
     */
    template <typename T>
    void Reduce(T *sendBuf, T *recvBuf, size_t length, size_t root,
                const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);
        PL_MPI_IS_SUCCESS(MPI_Reduce(sendBuf, recvBuf, length, datatype, op,
                                     root, this->getComm()));
    }

    template <typename T>
    void Gather(T &sendBuf, std::vector<T> &recvBuf, size_t root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_MPI_IS_SUCCESS(MPI_Gather(&sendBuf, 1, datatype, recvBuf.data(), 1,
                                     datatype, root, this->getComm()));
    }

    /**
     * @brief MPI_Reduce wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param recvBuf Receive buffer vector.
     * @param root Rank of root process.
     */
    template <typename T>
    void Gather(std::vector<T> &sendBuf, std::vector<T> &recvBuf, size_t root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_MPI_IS_SUCCESS(MPI_Gather(sendBuf.data(), sendBuf.size(), datatype,
                                     recvBuf.data(), sendBuf.size(), datatype,
                                     root, this->getComm()));
    }

    /**
     * @brief MPI_Barrier wrapper.
     */
    void Barrier() { PL_MPI_IS_SUCCESS(MPI_Barrier(this->getComm())); }

    /**
     * @brief MPI_Bcast wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param root Rank of broadcast root.
     */
    template <typename T> void Bcast(T &sendBuf, size_t root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        int rootInt = static_cast<int>(root);
        PL_MPI_IS_SUCCESS(
            MPI_Bcast(&sendBuf, 1, datatype, rootInt, this->getComm()));
    }

    /**
     * @brief MPI_Bcast wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param root Rank of broadcast root.
     */
    template <typename T> void Bcast(std::vector<T> &sendBuf, size_t root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        int rootInt = static_cast<int>(root);
        PL_MPI_IS_SUCCESS(MPI_Bcast(sendBuf.data(), sendBuf.size(), datatype,
                                    rootInt, this->getComm()));
    }

    /**
     * @brief MPI_Scatter wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param recvBuf Receive buffer.
     * @param root Rank of scatter root.
     */
    template <typename T>
    void Scatter(T *sendBuf, T *recvBuf, size_t dataSize, size_t root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        int rootInt = static_cast<int>(root);
        PL_MPI_IS_SUCCESS(MPI_Scatter(sendBuf, dataSize, datatype, recvBuf,
                                      dataSize, datatype, rootInt,
                                      this->getComm()));
    }

    /**
     * @brief MPI_Scatter wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param recvBuf Receive buffer vector.
     * @param root Rank of scatter root.
     */
    template <typename T>
    void Scatter(std::vector<T> &sendBuf, std::vector<T> &recvBuf,
                 size_t root) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_ABORT_IF(sendBuf.size() != recvBuf.size() * this->getSize(),
                    "Incompatible size of sendBuf and recvBuf.");
        int rootInt = static_cast<int>(root);
        PL_MPI_IS_SUCCESS(MPI_Scatter(sendBuf.data(), recvBuf.size(), datatype,
                                      recvBuf.data(), recvBuf.size(), datatype,
                                      rootInt, this->getComm()));
    }

    /**
     * @brief MPI_Scatter wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param root Rank of scatter root.
     * @return recvBuf Receive buffer vector.
     */
    template <typename T>
    auto scatter(std::vector<T> &sendBuf, size_t root) -> std::vector<T> {
        MPI_Datatype datatype = getMPIDatatype<T>();
        int recvBufSize;
        if (this->getRank() == root) {
            recvBufSize = sendBuf.size() / this->getSize();
        }
        this->Bcast<int>(recvBufSize, root);
        std::vector<T> recvBuf(recvBufSize);
        int rootInt = static_cast<int>(root);
        PL_MPI_IS_SUCCESS(MPI_Scatter(sendBuf.data(), recvBuf.size(), datatype,
                                      recvBuf.data(), recvBuf.size(), datatype,
                                      rootInt, this->getComm()));
        return recvBuf;
    }

    /**
     * @brief MPI_Send wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param dest Rank of send dest.
     */
    template <typename T> void Send(std::vector<T> &sendBuf, size_t dest) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        const int tag = 6789;

        PL_MPI_IS_SUCCESS(MPI_Send(sendBuf.data(), sendBuf.size(), datatype,
                                   static_cast<int>(dest), tag,
                                   this->getComm()));
    }

    /**
     * @brief MPI_Recv wrapper.
     *
     * @tparam T C++ data type.
     * @param recvBuf Recv buffer vector.
     * @param source Rank of data source.
     */
    template <typename T> void Recv(std::vector<T> &recvBuf, size_t source) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Status status;
        const int tag = MPI_ANY_TAG;

        PL_MPI_IS_SUCCESS(MPI_Recv(recvBuf.data(), recvBuf.size(), datatype,
                                   static_cast<int>(source), tag,
                                   this->getComm(), &status));
    }

    /**
     * @brief MPI_Sendrecv wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer.
     * @param dest Rank of destination.
     * @param recvBuf Receive buffer.
     * @param source Rank of source.
     */
    template <typename T>
    void Sendrecv(T &sendBuf, size_t dest, T &recvBuf, size_t source) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Status status;
        int sendtag = 0;
        int recvtag = 0;
        int destInt = static_cast<int>(dest);
        int sourceInt = static_cast<int>(source);
        PL_MPI_IS_SUCCESS(MPI_Sendrecv(&sendBuf, 1, datatype, destInt, sendtag,
                                       &recvBuf, 1, datatype, sourceInt,
                                       recvtag, this->getComm(), &status));
    }

    /**
     * @brief MPI_Sendrecv wrapper.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer vector.
     * @param dest Rank of destination.
     * @param recvBuf Receive buffer vector.
     * @param source Rank of source.
     */
    template <typename T>
    void Sendrecv(std::vector<T> &sendBuf, size_t dest, std::vector<T> &recvBuf,
                  size_t source) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Status status;
        int sendtag = 0;
        int recvtag = 0;
        int destInt = static_cast<int>(dest);
        int sourceInt = static_cast<int>(source);
        PL_MPI_IS_SUCCESS(MPI_Sendrecv(sendBuf.data(), sendBuf.size(), datatype,
                                       destInt, sendtag, recvBuf.data(),
                                       recvBuf.size(), datatype, sourceInt,
                                       recvtag, this->getComm(), &status));
    }

    template <typename T>
    void Scan(T &sendBuf, T &recvBuf, const std::string &op_str) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        MPI_Op op = getMPIOpType(op_str);

        PL_MPI_IS_SUCCESS(
            MPI_Scan(&sendBuf, &recvBuf, 1, datatype, op, this->getComm()));
    }

    /**
     * @brief Creates new MPIManager based on colors and keys.
     *
     * @param color Processes with the same color are in the same new
     * communicator.
     * @param key Rank assignment control.
     * @return new MPIManager object.
     */
    auto split(size_t color, size_t key) -> MPIManager {
        MPI_Comm newcomm;
        int colorInt = static_cast<int>(color);
        int keyInt = static_cast<int>(key);
        PL_MPI_IS_SUCCESS(
            MPI_Comm_split(this->getComm(), colorInt, keyInt, &newcomm));
        return MPIManager(newcomm);
    }
};
} // namespace Pennylane::LightningGPU::MPI
