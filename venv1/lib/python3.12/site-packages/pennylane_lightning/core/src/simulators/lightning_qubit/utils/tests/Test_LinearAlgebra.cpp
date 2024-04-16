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
#include <algorithm>
#include <complex>
#include <vector>

#include <catch2/catch.hpp>

#include "LinearAlgebra.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4305)
#endif

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using Pennylane::Util::randomUnitary;
} // namespace
/// @endcond

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEMPLATE_TEST_CASE("Inner product", "[Util][LinearAlgebra]", float, double) {
    SECTION("innerProd") {
        SECTION("Iterative increment") {
            for (size_t i = 0; i < 12; i++) {
                auto sz = static_cast<size_t>(1U << i);
                std::vector<std::complex<TestType>> data1(sz, {1.0, 1.0});
                std::vector<std::complex<TestType>> data2(sz, {1.0, 1.0});
                std::complex<TestType> expected_result(0,
                                                       size_t{1U} << (i + 1));
                std::complex<TestType> result = Util::innerProd(data1, data2);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<TestType>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::vector<std::complex<TestType>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::complex<TestType> expected_result(0, -0.312985152368);
            std::complex<TestType> result = Util::innerProd(data1, data2);
            CHECK(isApproxEqual(result, expected_result));
        }
    }
    SECTION("innerProd-inline") {
        SECTION("Iterative increment") {
            for (size_t i = 0; i < 12; i++) {
                auto sz = static_cast<size_t>(1U << i);
                std::vector<std::complex<TestType>> data1(sz, {1.0, 1.0});
                std::vector<std::complex<TestType>> data2(sz, {1.0, 1.0});
                std::complex<TestType> expected_result(0,
                                                       size_t{1U} << (i + 1));
                std::complex<TestType> result = Util::innerProd<TestType, 1>(
                    data1.data(), data2.data(), sz);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<TestType>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::vector<std::complex<TestType>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::complex<TestType> expected_result(0, -0.312985152368);
            std::complex<TestType> result =
                Util::innerProd<TestType, 1>(data1.data(), data2.data(), 8);
            CHECK(isApproxEqual(result, expected_result));
        }
    }
    SECTION("innerProdC") {
        SECTION("Iterative increment") {
            for (size_t i = 0; i < 12; i++) {
                auto sz = static_cast<size_t>(1U << i);
                std::vector<std::complex<TestType>> data1(sz, {1.0, 1.0});
                std::vector<std::complex<TestType>> data2(sz, {1.0, 1.0});
                std::complex<TestType> expected_result(size_t{1U} << (i + 1),
                                                       0);
                std::complex<TestType> result = Util::innerProdC(data1, data2);
                CAPTURE(result);
                CAPTURE(expected_result);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<TestType>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::vector<std::complex<TestType>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::complex<TestType> expected_result(0, -4.40916e-7);
            std::complex<TestType> result = Util::innerProdC(data1, data2);
            CAPTURE(result);
            CAPTURE(expected_result);
            CHECK(real(result) == Approx(real(expected_result)).margin(1e-7));
            CHECK(imag(result) == Approx(imag(expected_result)).margin(1e-7));
        }
    }
    SECTION("innerProdC-inline") {
        SECTION("Iterative increment") {
            for (size_t i = 0; i < 12; i++) {
                auto sz = static_cast<size_t>(1U << i);
                std::vector<std::complex<TestType>> data1(sz, {1.0, 1.0});
                std::vector<std::complex<TestType>> data2(sz, {1.0, 1.0});
                std::complex<TestType> expected_result(size_t{1U} << (i + 1),
                                                       0);
                std::complex<TestType> result = Util::innerProdC<TestType, 1>(
                    data1.data(), data2.data(), sz);
                CAPTURE(result);
                CAPTURE(expected_result);
                CHECK(isApproxEqual(result, expected_result));
            }
        }
        SECTION("Random complex") {
            std::vector<std::complex<TestType>> data2{
                {0, -0.479426}, {0, 0}, {2.77556e-17, 0}, {0, 0},
                {0.877583, 0},  {0, 0}, {0, 0},           {0, 0}};
            std::vector<std::complex<TestType>> data1{
                {0.326417, 0},  {-0, 0.343918}, {0, 0.508364}, {-0.53562, -0},
                {0, -0.178322}, {0.187883, -0}, {0.277721, 0}, {-0, 0.292611}};
            std::complex<TestType> expected_result(0, -4.40916e-7);
            std::complex<TestType> result =
                Util::innerProdC<TestType, 1>(data1.data(), data2.data(), 8);
            CAPTURE(result);
            CAPTURE(expected_result);
            CHECK(real(result) == Approx(real(expected_result)).margin(1e-7));
            CHECK(imag(result) == Approx(imag(expected_result)).margin(1e-7));
        }
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEMPLATE_TEST_CASE("Product", "[Util][LinearAlgebra]", float, double) {
    using Util::Trans;
    SECTION("matrixVecProd") {
        SECTION("Simple Iterative with NoTranspose") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<TestType>> mat(m * m, {1.0, 1.0});
                std::vector<std::complex<TestType>> v_in(m, {1.0, 1.0});
                std::vector<std::complex<TestType>> v_expected(
                    m, {0, static_cast<TestType>(2 * m)});
                std::vector<std::complex<TestType>> v_out =
                    Util::matrixVecProd(mat, v_in, m, m, Trans::NoTranspose);
                CAPTURE(v_out);
                CAPTURE(v_expected);

                CHECK(v_out == approx(v_expected).margin(1e-7));
            }
        }
        SECTION("Simple Iterative with Transpose") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<TestType>> mat(m * m, {1.0, 1.0});
                std::vector<std::complex<TestType>> v_in(m, {1.0, 1.0});
                std::vector<std::complex<TestType>> v_expected(
                    m, {0, static_cast<TestType>(2 * m)});
                std::vector<std::complex<TestType>> v_out =
                    Util::matrixVecProd(mat, v_in, m, m, Trans::Transpose);
                CAPTURE(v_out);
                CAPTURE(v_expected);

                CHECK(v_out == approx(v_expected).margin(1e-7));
            }
        }
        SECTION("Random Complex with NoTranspose") {
            std::vector<std::complex<TestType>> mat{
                {0.417876, 0.27448},   {0.601209, 0.723548},
                {0.781624, 0.538222},  {0.0597232, 0.27755},
                {0.0431741, 0.593319}, {0.224124, 0.130335},
                {0.237877, 0.01557},   {0.931634, 0.786367},
                {0.378397, 0.894381},  {0.840747, 0.889789},
                {0.530623, 0.463644},  {0.868736, 0.760685},
                {0.258175, 0.836569},  {0.495012, 0.667726},
                {0.298962, 0.384992},  {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> v_in{{0.417876, 0.27448},
                                                     {0.601209, 0.723548},
                                                     {0.781624, 0.538222},
                                                     {0.0597232, 0.27755}};
            std::vector<std::complex<TestType>> v_expected{
                {0.184998, 1.97393},
                {-0.0894368, 0.946047},
                {-0.219747, 2.55541},
                {-0.305997, 1.83881}};
            std::vector<std::complex<TestType>> v_out =
                Util::matrixVecProd(mat, v_in, 4, 4, Trans::NoTranspose);
            CAPTURE(v_out);

            CHECK(v_out == approx(v_expected).margin(1e-7));
        }
        SECTION("Random Complex with Transpose") {
            std::vector<std::complex<TestType>> mat{
                {0.417876, 0.27448},   {0.601209, 0.723548},
                {0.781624, 0.538222},  {0.0597232, 0.27755},
                {0.0431741, 0.593319}, {0.224124, 0.130335},
                {0.237877, 0.01557},   {0.931634, 0.786367},
                {0.378397, 0.894381},  {0.840747, 0.889789},
                {0.530623, 0.463644},  {0.868736, 0.760685},
                {0.258175, 0.836569},  {0.495012, 0.667726},
                {0.298962, 0.384992},  {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> v_in{{0.417876, 0.27448},
                                                     {0.601209, 0.723548},
                                                     {0.781624, 0.538222},
                                                     {0.0597232, 0.27755}};
            std::vector<std::complex<TestType>> v_expected{{-0.706439, 1.64169},
                                                           {0.115553, 2.03315},
                                                           {0.386844, 1.37488},
                                                           {0.184316, 2.5383}};
            std::vector<std::complex<TestType>> v_out =
                Util::matrixVecProd(mat, v_in, 4, 4, Trans::Transpose);
            CAPTURE(v_out);

            CHECK(v_out == approx(v_expected).margin(1e-7));
        }
        SECTION("Random Complex with Adjoint") {
            std::vector<std::complex<TestType>> v_in{
                {0.643624335855, 0.578590708232},
                {0.538989919338, 0.466635790378},
                {0.618764845639, 0.599437240657},
                {0.357299642534, 0.053014685781},
            };
            std::vector<std::complex<TestType>> mat{
                {0.532146153405, 0.701149723765},
                {0.846447557122, 0.115252356911},
                {0.054946913257, 0.827064042981},
                {0.716903475670, 0.676278287205},
                {0.695253133818, 0.701356633873},
                {0.652528065087, 0.918467022349},
                {0.977379072529, 0.855331203864},
                {0.254556889390, 0.518154071409},
                {0.772606503900, 0.513854930480},
                {0.536592010310, 0.154591920456},
                {0.415274560257, 0.825736069441},
                {0.634991477205, 0.533747170519},
                {0.347378131788, 0.235064960717},
                {0.936386930317, 0.181149920417},
                {0.375506408059, 0.085886487536},
                {0.249630514363, 0.514992726206},
            };
            std::vector<std::complex<TestType>> v_expected{
                {2.372858970605, -0.117375105278},
                {2.160640571918, 0.435930598280},
                {2.330478161419, -0.778246884580},
                {2.061051094071, -0.301369019950},
            };

            std::vector<std::complex<TestType>> v_out =
                Util::matrixVecProd(mat, v_in, 4, 4, Trans::Adjoint);
            CAPTURE(v_out);

            CHECK(v_out == approx(v_expected).margin(1e-7));
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<std::complex<TestType>> mat(2 * 3, {1.0, 1.0});
            std::vector<std::complex<TestType>> v_in(2, {1.0, 1.0});
            CHECK_THROWS_AS(Util::matrixVecProd(mat, v_in, 2, 3),
                            std::invalid_argument);
            CHECK_THROWS_WITH(Util::matrixVecProd(mat, v_in, 2, 3),
                              Contains("Invalid size for the input vector"));
            CHECK_THROWS_AS(Util::matrixVecProd(mat, v_in, 2, 2),
                            std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::matrixVecProd(mat, v_in, 2, 2),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));
        }
        SECTION("nullptr for v_out") {
            std::vector<std::complex<TestType>> mat(2 * 3, {1.0, 1.0});
            std::vector<std::complex<TestType>> v_in(2, {1.0, 1.0});
            auto v_out = nullptr;
            Util::matrixVecProd<TestType>(mat.data(), v_in.data(), v_out, 2, 3,
                                          Trans::NoTranspose);
            CHECK(v_out == nullptr);

            Util::omp_matrixVecProd<TestType>(mat.data(), v_in.data(), v_out, 2,
                                              3, Trans::NoTranspose);
            CHECK(v_out == nullptr);
        }
    }
    SECTION("vecMatrixProd") {
        SECTION("Simple Iterative") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<TestType> mat(m * m, TestType{1.0});
                std::vector<TestType> v_in(m, TestType{1.0});
                std::vector<TestType> v_expected(m, static_cast<TestType>(m));
                std::vector<TestType> v_out =
                    Util::vecMatrixProd(v_in, mat, m, m);

                CAPTURE(v_out);
                CAPTURE(v_expected);

                CHECK(v_out == approx(v_expected).margin(1e-7));
            }
        }
        SECTION("Zero Vector") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<TestType> mat(m * m, 1);
                std::vector<TestType> v_in(m, 0);
                std::vector<TestType> v_expected(m, 0);
                std::vector<TestType> v_out =
                    Util::vecMatrixProd(v_in, mat, m, m);

                CAPTURE(v_out);
                CAPTURE(v_expected);

                CHECK(v_out == approx(v_expected).margin(1e-7));
            }
        }
        SECTION("Random Matrix") {
            std::vector<TestType> v_in{1.0, 2.0, 3.0, 4.0};
            std::vector<TestType> mat{1.0, 0.1,  0.2, 0.2,  0.6,  0.1,
                                      0.4, -0.7, 1.2, -0.5, -0.6, 0.7};
            std::vector<TestType> v_expected{0.6, -3.2, 6.8};
            std::vector<TestType> v_out = Util::vecMatrixProd(v_in, mat, 4, 3);

            CAPTURE(v_out);
            CAPTURE(v_expected);

            CHECK(v_out == approx(v_expected).margin(1e-7));
        }
        SECTION("In Place") {
            std::vector<TestType> v_in{1.0, 2.0, 3.0, 4.0};
            std::vector<TestType> mat{1.0, 0.1,  0.2, 0.2,  0.6,  0.1,
                                      0.4, -0.7, 1.2, -0.5, -0.6, 0.7};
            std::vector<TestType> v_expected{0.6, -3.2, 6.8};
            std::vector<TestType> v_out1 = Util::vecMatrixProd(v_in, mat, 4, 3);

            std::vector<TestType> v_out2(3, TestType{});
            Util::vecMatrixProd(v_out2, v_in, mat, 4, 3);

            CHECK(v_out2 == v_out1);
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<TestType> v_in(4, {1.0});
            std::vector<TestType> mat(8, {1.0});
            CHECK_THROWS_AS(Util::vecMatrixProd(v_in, mat, 2, 3),
                            std::invalid_argument);
            CHECK_THROWS_WITH(Util::vecMatrixProd(v_in, mat, 2, 3),
                              Contains("Invalid size for the input vector"));
            CHECK_THROWS_AS(Util::vecMatrixProd(v_in, mat, 4, 3),
                            std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::vecMatrixProd(v_in, mat, 4, 3),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));

            std::vector<TestType> v_out(3);
            CHECK_THROWS_AS(Util::vecMatrixProd(v_out, v_in, mat, 2, 3),
                            std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::vecMatrixProd(v_out, v_in, mat, 2, 3),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));
            CHECK_THROWS_AS(Util::vecMatrixProd(v_out, v_in, mat, 2, 4),
                            std::invalid_argument);
            CHECK_THROWS_WITH(Util::vecMatrixProd(v_out, v_in, mat, 2, 4),
                              Contains("Invalid size for the input vector"));
            CHECK_THROWS_AS(Util::vecMatrixProd(v_out, v_in, mat, 4, 2),
                            std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::vecMatrixProd(v_out, v_in, mat, 4, 2),
                Contains("Invalid preallocated size for the result"));
        }
        SECTION("nullptr for v_out") {
            std::vector<TestType> v_in(4, {1.0});
            std::vector<TestType> mat{4 * 2, {1.0}};
            auto v_out = nullptr;
            Util::vecMatrixProd<TestType>(v_in.data(), mat.data(), v_out, 4, 2);
            CHECK(v_out == nullptr);
        }
    }
    SECTION("matrixMatProd") {
        SECTION("Simple Iterative (Trans::Transpose)") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<TestType>> m_left(m * m, {1.0, 1.0});
                std::vector<std::complex<TestType>> m_right(m * m, {1.0, 1.0});
                std::vector<std::complex<TestType>> m_out_exp(
                    m * m, {0, static_cast<TestType>(2 * m)});
                std::vector<std::complex<TestType>> m_out = Util::matrixMatProd(
                    m_left, m_right, m, m, m, Trans::Transpose);

                CAPTURE(m_out);
                CAPTURE(m_out_exp);

                CHECK(m_out == approx(m_out_exp));
            }
        }
        SECTION("Simple Iterative (Trans::Adjoint)") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<TestType>> m_left(m * m, {1.0, 1.0});
                std::vector<std::complex<TestType>> m_right(m * m, {1.0, 1.0});
                std::vector<std::complex<TestType>> m_out_exp(
                    m * m, {static_cast<TestType>(2 * m), 0});
                std::vector<std::complex<TestType>> m_out = Util::matrixMatProd(
                    m_left, m_right, m, m, m, Trans::Adjoint);

                CAPTURE(m_out);
                CAPTURE(m_out_exp);

                CHECK(m_out == approx(m_out_exp));
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<TestType>> m_left{
                {0.94007, 0.424517},  {0.256163, 0.0615097},
                {0.505297, 0.343107}, {0.729021, 0.241991},
                {0.860825, 0.633264}, {0.987668, 0.195166},
                {0.606897, 0.144482}, {0.0183697, 0.375071},
                {0.355853, 0.152383}, {0.985341, 0.0888863},
                {0.608352, 0.653375}, {0.268477, 0.58398},
                {0.960381, 0.786669}, {0.498357, 0.185307},
                {0.283511, 0.844801}, {0.269318, 0.792981}};
            std::vector<std::complex<TestType>> m_right{
                {0.94007, 0.424517},  {0.256163, 0.0615097},
                {0.505297, 0.343107}, {0.729021, 0.241991},
                {0.860825, 0.633264}, {0.987668, 0.195166},
                {0.606897, 0.144482}, {0.0183697, 0.375071},
                {0.355853, 0.152383}, {0.985341, 0.0888863},
                {0.608352, 0.653375}, {0.268477, 0.58398},
                {0.960381, 0.786669}, {0.498357, 0.185307},
                {0.283511, 0.844801}, {0.269318, 0.792981}};
            std::vector<std::complex<TestType>> m_right_tp{
                {0.94007, 0.424517},   {0.860825, 0.633264},
                {0.355853, 0.152383},  {0.960381, 0.786669},
                {0.256163, 0.0615097}, {0.987668, 0.195166},
                {0.985341, 0.0888863}, {0.498357, 0.185307},
                {0.505297, 0.343107},  {0.606897, 0.144482},
                {0.608352, 0.653375},  {0.283511, 0.844801},
                {0.729021, 0.241991},  {0.0183697, 0.375071},
                {0.268477, 0.58398},   {0.269318, 0.792981}};
            std::vector<std::complex<TestType>> m_out_exp{
                {1.522375435807200, 2.018315393556500},
                {1.241561065671800, 0.915996420839700},
                {0.561409446565600, 1.834755796266900},
                {0.503973820211400, 1.664651528374090},
                {1.183556828429700, 2.272762769584300},
                {1.643767359748500, 0.987318478828500},
                {0.752063484100700, 1.482770126810700},
                {0.205343773497200, 1.552791421044900},
                {0.977117116888800, 2.092066653216500},
                {1.604565422784600, 1.379671036009100},
                {0.238648365886400, 1.582741563052100},
                {-0.401698027789600, 1.469264325654110},
                {0.487510164243000, 2.939585667799000},
                {0.845207296911400, 1.843583823364000},
                {-0.482010055957000, 2.062995137499000},
                {-0.524094900662100, 1.815727577737900}};
            std::vector<std::complex<TestType>> m_out_1 = Util::matrixMatProd(
                m_left, m_right_tp, 4, 4, 4, Trans::Transpose);
            std::vector<std::complex<TestType>> m_out_2 = Util::matrixMatProd(
                m_left, m_right, 4, 4, 4, Trans::NoTranspose);

            CAPTURE(m_out_1);
            CAPTURE(m_out_2);
            CAPTURE(m_out_exp);

            CHECK(m_out_1 == approx(m_out_2));
            CHECK(m_out_1 == approx(m_out_exp));
        }
        SECTION("Random complex non-square") {
            const size_t m = 4;
            const size_t k = 2;
            const size_t n = 8;
            std::vector<std::complex<TestType>> mat1{
                {-0.08981826740301613, -0.27637263739311546},
                {0.8727813226706924, 0.258896589429058},
                {-0.5949864309922819, 0.18285525036841555},
                {-0.11134824896298667, -0.13897161071967146},
                {-0.1735039889140335, 0.21741315096734315},
                {0.5297588834950917, -0.7710643565719117},
                {-0.8579225641269561, 0.9468802886927876},
                {-0.9547436543380183, -0.42965580917488455},
            }; // m x k

            std::vector<std::complex<TestType>> mat2{
                {-0.2737437848727584, 0.049006595173545886},
                {0.5192576146240857, 0.7301480202235375},
                {0.653209372398236, 0.6769135138733755},
                {0.24089266818363964, -0.31764940703302114},
                {-0.15687970845233856, 0.733905243784096},
                {-0.6359345637324041, 0.539820653560259},
                {-0.6134456192663982, -0.931661049054201},
                {0.24880944149427253, 0.10974806658737091},
                {-0.940232124693291, -0.5232886593299633},
                {0.4843048111315602, 0.35624698471285265},
                {0.2799604418471606, 0.45969907217442474},
                {-0.4750846673676159, 0.8376958296863313},
                {-0.06299227175924127, 0.8246636234326061},
                {0.5033921732959004, -0.21101881518679866},
                {0.005396787998366959, -0.17840690755422184},
                {-0.6585640941981321, 0.14497152454268059},
            }; // k x n

            const std::vector<std::complex<TestType>> expected{
                {-0.6470081137924694, -0.6288858542578596},
                {0.4856151790538221, 0.22722135179683745},
                {0.25373996181718034, 0.2323691717947702},
                {-0.7409477841248294, 0.5700820515045693},
                {-0.05155908637184568, 0.68088168504282},
                {0.7002935368283547, 0.07342239296147188},
                {-0.15148784295130674, 0.09890675818471939},
                {-0.6043313448030839, -0.12259306171417446},
                {0.18588365718252325, 0.10971910126048831},
                {-0.44688090933900626, -0.4464512811494489},
                {-0.47971588770318185, -0.37340483265491176},
                {0.08407177752281807, 0.20579289380315458},
                {0.0807617819372747, -0.5484206530358158},
                {0.19428593398213712, -0.48393058348427054},
                {0.5099565157985027, 0.46126922661113773},
                {-0.07462936679210332, 0.05557688979687145},
                {-0.864744593670385, 0.3797443237699707},
                {0.28241707890402445, -0.19849533255712323},
                {0.24226476870790875, 0.05223126234803932},
                {0.4215023926257107, 0.9175843340352592},
                {0.4701566142679236, 0.3240007768827886},
                {0.09694046297691916, -0.7318584281365454},
                {0.17428641523616784, -0.07039815172433624},
                {-0.304127907507671, 0.6196479698717412},
                {0.8612942331452932, 0.6023362633910544},
                {-1.4461689510687892, -0.6829445852068847},
                {-1.2711372016033236, -0.5214097006435647},
                {0.9076136576558541, -0.09504677750515506},
                {-0.14586677921501573, -1.538457538144058},
                {-0.5368388676391003, -1.080094719371998},
                {1.3266542084028976, 0.38644757259657603},
                {0.3736702325957464, 0.28598265876636564},
            }; // m x n

            const auto m_out = Util::matrixMatProd(mat1, mat2, m, n, k);

            CHECK(m_out == approx(expected));
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<std::complex<TestType>> m_left(2 * 3, {1.0, 1.0});
            std::vector<std::complex<TestType>> m_right(3 * 4, {1.0, 1.0});
            CHECK_THROWS_AS(Util::matrixMatProd(m_left, m_right, 2, 3, 4),
                            std::invalid_argument);
            CHECK_THROWS_WITH(Util::matrixMatProd(m_left, m_right, 2, 3, 4),
                              Contains("Invalid number of rows and columns for "
                                       "the input left matrix"));
            CHECK_THROWS_AS(Util::matrixMatProd(m_left, m_right, 2, 3, 3),
                            std::invalid_argument);
            CHECK_THROWS_WITH(Util::matrixMatProd(m_left, m_right, 2, 3, 3),
                              Contains("Invalid number of rows and columns for "
                                       "the input right matrix"));
        }
        SECTION("nullptr for m_out") {
            std::vector<std::complex<TestType>> m_left(2 * 3, {1.0, 1.0});
            std::vector<std::complex<TestType>> m_right(3 * 4, {1.0, 1.0});
            auto m_out = nullptr;
            Util::matrixMatProd<TestType>(m_left.data(), m_right.data(), m_out,
                                          2, 3, 4, Trans::NoTranspose);
            CHECK(m_out == nullptr);

            Util::omp_matrixMatProd<TestType>(m_left.data(), m_right.data(),
                                              m_out, 2, 3, 4,
                                              Trans::NoTranspose);
            CHECK(m_out == nullptr);
        }
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
TEMPLATE_TEST_CASE("Transpose", "[Util][LinearAlgebra]", float, double) {
    SECTION("CFTranspose") {
        SECTION("Simple Matrix") {
            for (size_t m = 2; m < 10; m++) {
                std::vector<TestType> mat(m * m, {0});
                for (size_t i = 0; i < m; i++) {
                    mat[i * m + i] = 1.0;
                }
                std::vector<TestType> mat_t(m * m);
                Util::CFTranspose<TestType, 4>(mat.data(), mat_t.data(), m, m,
                                               0, m, 0, m);

                CAPTURE(mat_t);
                CAPTURE(mat);

                CHECK(mat_t == approx(mat).margin(1e-7));
            }
        }
        SECTION("Random Complex") {
            std::vector<TestType> mat{
                0.417876, 0.27448,   0.601209, 0.723548, 0.781624,
                0.538222, 0.0597232, 0.27755,  0.836569,
            };
            std::vector<TestType> mat_t_exp{
                0.417876, 0.723548, 0.0597232, 0.27448,  0.781624,
                0.27755,  0.601209, 0.538222,  0.836569,
            };
            std::vector<TestType> mat_t(9);
            Util::CFTranspose<TestType, 2>(mat.data(), mat_t.data(), 3, 3, 0, 3,
                                           0, 3);

            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);

            CHECK(mat_t == approx(mat_t_exp));
        }
        SECTION("Random Complex non-square") {
            std::vector<TestType> mat{
                0.417876, 0.27448,  0.601209,  0.723548,
                0.781624, 0.538222, 0.0597232, 0.27755,
            };
            std::vector<TestType> mat_t_exp{0.417876, 0.781624, 0.27448,
                                            0.538222, 0.601209, 0.0597232,
                                            0.723548, 0.27755};
            std::vector<TestType> mat_t(8);
            Util::CFTranspose<TestType, 2>(mat.data(), mat_t.data(), 2, 4, 0, 2,
                                           0, 4);

            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);

            CHECK(mat_t == approx(mat_t_exp));
        }
    }
    SECTION("Transpose") {
        SECTION("Simple Matrix") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<TestType>> mat(m * m, {0, 0});
                for (size_t i = 0; i < m; i++) {
                    mat[i * m + i] = {1, 1};
                }
                std::vector<std::complex<TestType>> mat_t =
                    Util::Transpose(mat, m, m);

                CAPTURE(mat_t);
                CAPTURE(mat);

                CHECK(mat_t == approx(mat).margin(1e-7));
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<TestType>> mat{
                {0.417876, 0.27448},   {0.601209, 0.723548},
                {0.781624, 0.538222},  {0.0597232, 0.27755},
                {0.0431741, 0.593319}, {0.224124, 0.130335},
                {0.237877, 0.01557},   {0.931634, 0.786367},
                {0.378397, 0.894381},  {0.840747, 0.889789},
                {0.530623, 0.463644},  {0.868736, 0.760685},
                {0.258175, 0.836569},  {0.495012, 0.667726},
                {0.298962, 0.384992},  {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> mat_t_exp{
                {0.417876, 0.27448},  {0.0431741, 0.593319},
                {0.378397, 0.894381}, {0.258175, 0.836569},
                {0.601209, 0.723548}, {0.224124, 0.130335},
                {0.840747, 0.889789}, {0.495012, 0.667726},
                {0.781624, 0.538222}, {0.237877, 0.01557},
                {0.530623, 0.463644}, {0.298962, 0.384992},
                {0.0597232, 0.27755}, {0.931634, 0.786367},
                {0.868736, 0.760685}, {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> mat_t =
                Util::Transpose(mat, 4, 4);

            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);

            CHECK(mat_t == approx(mat_t_exp));
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<TestType> mat(2 * 3, {1.0});
            CHECK_THROWS_AS(
                Util::Transpose(std::span<const TestType>{mat}, 2, 2),
                std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::Transpose(mat, 2, 2),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));
        }
    }
    SECTION("Transpose<complex<T>>") {
        SECTION("Simple Matrix") {
            for (size_t m = 2; m < 8; m++) {
                std::vector<std::complex<TestType>> mat(m * m, {0, 0});
                for (size_t i = 0; i < m; i++) {
                    mat[i * m + i] = {1.0, 1.0};
                }
                std::vector<std::complex<TestType>> mat_t =
                    Util::Transpose(mat, m, m);

                CAPTURE(mat_t);
                CAPTURE(mat);

                CHECK(mat_t == approx(mat).margin(1e-7));
            }
        }
        SECTION("Random Complex") {
            std::vector<std::complex<TestType>> mat{
                {0.417876, 0.27448},   {0.601209, 0.723548},
                {0.781624, 0.538222},  {0.0597232, 0.27755},
                {0.0431741, 0.593319}, {0.224124, 0.130335},
                {0.237877, 0.01557},   {0.931634, 0.786367},
                {0.378397, 0.894381},  {0.840747, 0.889789},
                {0.530623, 0.463644},  {0.868736, 0.760685},
                {0.258175, 0.836569},  {0.495012, 0.667726},
                {0.298962, 0.384992},  {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> mat_t_exp{
                {0.417876, 0.27448},  {0.0431741, 0.593319},
                {0.378397, 0.894381}, {0.258175, 0.836569},
                {0.601209, 0.723548}, {0.224124, 0.130335},
                {0.840747, 0.889789}, {0.495012, 0.667726},
                {0.781624, 0.538222}, {0.237877, 0.01557},
                {0.530623, 0.463644}, {0.298962, 0.384992},
                {0.0597232, 0.27755}, {0.931634, 0.786367},
                {0.868736, 0.760685}, {0.659472, 0.232696}};
            std::vector<std::complex<TestType>> mat_t =
                Util::Transpose(mat, 4, 4);

            CAPTURE(mat_t);
            CAPTURE(mat_t_exp);

            CHECK(mat_t == approx(mat_t_exp));
        }
        SECTION("Invalid Arguments") {
            using namespace Catch::Matchers;
            std::vector<std::complex<TestType>> mat(2 * 3, {1.0, 1.0});
            CHECK_THROWS_AS(Util::Transpose(mat, 2, 2), std::invalid_argument);
            CHECK_THROWS_WITH(
                Util::Transpose(mat, 2, 2),
                Contains(
                    "Invalid number of rows and columns for the input matrix"));
        }
    }
}

TEMPLATE_TEST_CASE("Util::scaleAndAdd", "[Util][LinearAlgebra]", float,
                   double) {
    using PrecisionT = TestType;
    using ComplexT = std::complex<PrecisionT>;

    SECTION("Test result is correct") {
        auto a = ComplexT{0.36572644485147254, 0.4729529811649217};
        std::vector<ComplexT> x{
            ComplexT{0.481941495077, 0.734106237571},
            ComplexT{0.960470937496, 0.880529982024},
            ComplexT{0.135982489400, 0.049663856666},
            ComplexT{0.589227566883, 0.646648171030},
            ComplexT{0.051294350194, 0.013730433456},
            ComplexT{0.716464613724, 0.296251370128},
            ComplexT{0.820197028755, 0.199230854010},
            ComplexT{0.100767632907, 0.745810000609},
            ComplexT{0.603122469037, 0.437680494447},
            ComplexT{0.815084269631, 0.501486284044},
            ComplexT{0.554633849948, 0.437321144284},
            ComplexT{0.822295519809, 0.810051588437},
            ComplexT{0.217638951648, 0.663920104700},
            ComplexT{0.289819402719, 0.839919161595},
            ComplexT{0.498496405040, 0.906874924446},
            ComplexT{0.365971064862, 0.230694150520},
        };
        std::vector<ComplexT> y{
            ComplexT{0.516438479285, 0.970319841313},
            ComplexT{0.085702308539, 0.005302125762},
            ComplexT{0.591955559108, 0.945946312721},
            ComplexT{0.710102120659, 0.410003006045},
            ComplexT{0.171020364152, 0.020935262021},
            ComplexT{0.904267565256, 0.235752839391},
            ComplexT{0.715111137847, 0.402137049186},
            ComplexT{0.590485707389, 0.550485111898},
            ComplexT{0.830734963458, 0.777755725832},
            ComplexT{0.988885576027, 0.541038298049},
            ComplexT{0.375479099161, 0.275849441779},
            ComplexT{0.441329976617, 0.825285998539},
            ComplexT{0.376823807696, 0.896094272876},
            ComplexT{0.558768533750, 0.963077088666},
            ComplexT{0.402000571969, 0.344065008137},
            ComplexT{0.805773653517, 0.316132703093},
        };
        std::vector<ComplexT> expected{
            ComplexT{0.345499495355, 1.466737572567},
            ComplexT{0.020522649889, 0.781592818884},
            ComplexT{0.618199282452, 1.028423022205},
            ComplexT{0.619764043650, 0.925176277047},
            ComplexT{0.183286215053, 0.050216660476},
            ComplexT{1.026184652619, 0.682953874730},
            ComplexT{0.920852054907, 0.862915671020},
            ComplexT{0.274606032358, 0.870905904344},
            ComplexT{0.844310505222, 1.223075626786},
            ComplexT{1.049804015161, 1.109941629077},
            ComplexT{0.371491026381, 0.698105081924},
            ComplexT{0.358948880046, 1.510450403616},
            ComplexT{0.142417134970, 1.241840403433},
            ComplexT{0.267520882141, 1.407328688114},
            ComplexT{0.155404690895, 0.911498511043},
            ComplexT{0.830511463762, 0.573590760757},
        };
        Util::scaleAndAdd(a, x, y);
        REQUIRE(y == approx(expected));
    }
    SECTION("Throws exception when the size mismatches") {
        std::vector<ComplexT> x(8, ComplexT{});
        std::vector<ComplexT> y(4, ComplexT{});

        PL_REQUIRE_THROWS_MATCHES(Util::scaleAndAdd(ComplexT{0.5, 0.4}, x, y),
                                  std::invalid_argument,
                                  "Dimensions of vectors mismatch");
    }
    SECTION("omp_scaleAndAdd uses STD_CROSSOVER") {
        std::vector<ComplexT> x(32);
        std::vector<ComplexT> y(32);
        REQUIRE_NOTHROW(Util::omp_scaleAndAdd<PrecisionT, 16>(
            32, {1.0, 0.0}, x.data(), y.data()));
    }
}

/**
 * @brief Test randomUnitary is correct
 */
TEMPLATE_TEST_CASE("randomUnitary", "[Util][LinearAlgebra]", float, double) {
    using PrecisionT = TestType;

    std::mt19937 re{1337};

    for (size_t num_qubits = 1; num_qubits <= 5; num_qubits++) {
        const size_t dim = (1U << num_qubits);
        const auto unitary = randomUnitary<PrecisionT>(re, num_qubits);

        auto unitary_dagger = Util::Transpose(unitary, dim, dim);
        std::transform(
            unitary_dagger.begin(), unitary_dagger.end(),
            unitary_dagger.begin(),
            [](const std::complex<PrecisionT> &v) { return std::conj(v); });

        std::vector<std::complex<PrecisionT>> mat(dim * dim);
        Util::matrixMatProd(unitary.data(), unitary_dagger.data(), mat.data(),
                            dim, dim, dim);

        std::vector<std::complex<PrecisionT>> identity(
            dim * dim, std::complex<PrecisionT>{});
        for (size_t i = 0; i < dim; i++) {
            identity[i * dim + i] = std::complex<PrecisionT>{1.0, 0.0};
        }

        REQUIRE(mat == approx(identity).margin(1e-5));
    }
}