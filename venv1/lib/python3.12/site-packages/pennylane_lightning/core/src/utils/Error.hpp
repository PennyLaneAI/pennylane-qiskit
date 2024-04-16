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

#include <exception>
#include <sstream>
#include <string>

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
/**
 * @brief Macro that throws `%LightningException` with given message.
 *
 * @param message string literal describing error
 */
#define PL_ABORT(message)                                                      \
    Pennylane::Util::Abort((message), __FILE__, __LINE__, __func__)
/**
 * @brief Macro that throws `%LightningException` if expression evaluates to
 * true.
 *
 * @param expression an expression
 * @param message string literal describing error
 */
#define PL_ABORT_IF(expression, message)                                       \
    if ((expression)) {                                                        \
        PL_ABORT(message);                                                     \
    }
/**
 * @brief Macro that throws `%LightningException` with error message if
 * expression evaluates to false.
 *
 * @param expression an expression
 * @param message string literal describing error
 */
#define PL_ABORT_IF_NOT(expression, message)                                   \
    if (!(expression)) {                                                       \
        PL_ABORT(message);                                                     \
    }

/**
 * @brief Macro that throws `%LightningException` with the given expression and
 * source location if expression evaluates to false.
 *
 * @param expression an expression
 */
#define PL_ASSERT(expression)                                                  \
    PL_ABORT_IF_NOT(expression, "Assertion failed: " #expression)

// NOLINTEND(cppcoreguidelines-macro-usage)

namespace Pennylane::Util {
/**
 * @brief `%LightningException` is the general exception thrown by PennyLane for
 * runtime errors.
 *
 */
class LightningException : public std::exception {
  private:
    std::string err_msg;

  public:
    /**
     * @brief Constructs a new `%LightningException` exception.
     *
     * @param err_msg Error message explaining the exception condition.
     */
    explicit LightningException(std::string err_msg) noexcept
        : err_msg{std::move(err_msg)} {}

    LightningException(const LightningException &) = default;
    LightningException(LightningException &&) noexcept = default;

    auto operator=(const LightningException &) -> LightningException & = delete;
    auto operator=(LightningException &&) -> LightningException & = delete;

    /**
     * @brief Destroys the `%LightningException` object.
     */
    ~LightningException() override = default;

    /**
     * @brief Returns a string containing the exception message. Overrides
     *        the `std::exception` method.
     *
     * @return Exception message.
     */
    [[nodiscard]] auto what() const noexcept -> const char * override {
        return err_msg.c_str();
    }
};

/**
 * @brief Throws a `%LightningException` with the given error message.
 *
 * This function should not be called directly - use one of the `PL_ASSERT()`
 * or `PL_ABORT()` macros, which provide the source location at compile time.
 *
 * @param message string literal describing the error
 * @param file_name source file where error occured
 * @param line line of source file
 * @param function_name function in which error occured
 */
[[noreturn]] inline void Abort(const char *message, const char *file_name,
                               int line, const char *function_name) {
    std::stringstream err_msg;
    err_msg << "[" << file_name << "][Line:" << line
            << "][Method:" << function_name
            << "]: Error in PennyLane Lightning: " << message;
    throw LightningException(err_msg.str());
}
/**
 * @brief Throws a `%LightningException` with the given error message.
 *
 * This function should not be called directly - use one of the `PL_ASSERT()`
 * or `PL_ABORT()` macros, which provide the source location at compile time.
 *
 * @param message string literal describing the error
 * @param file_name source file where error occurred
 * @param line line of source file
 * @param function_name function in which error occurred
 */
[[noreturn]] inline void Abort(const std::string &message,
                               const char *file_name, int line,
                               const char *function_name) {
    std::stringstream err_msg;
    err_msg << "[" << file_name << "][Line:" << line
            << "][Method:" << function_name
            << "]: Error in PennyLane Lightning: " << message;
    throw LightningException(err_msg.str());
}

} // namespace Pennylane::Util