#include "Backend.h"
#include "Hooks.h"
#include "NC_Arrays.h"
#include "NC_ConstantsVariables.h"
#include "Parallel.h"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace numcalc {
Options Settings;
nlohmann::json Reports = nlohmann::json::array();
namespace {
using Clock = std::chrono::steady_clock;
Clock::time_point AssemblyStart;
double Elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::unique_ptr<Gpu> DeviceOperator;
std::unique_ptr<Operator> ActiveOperator;
size_t Applications{};
double ApplySeconds{};
bool DirectRequested{};

Operator CurrentOperator() {
    Operator op;
    const int n = numRowsOfCoefficientMatrix_;
    op.Near = {n, n, methodFMM_ ? irownea : nullptr, methodFMM_ ? jcolnea : nullptr, zcoefl};
    const bool single = methodFMM_ == 1;
    const int levels = single ? 1 : methodFMM_ == 3 ? numClusterLevels_ :
                                                      0;
    for (int l = 0; l < levels; ++l) {
        const int nt = single ? numIntegrationPointsUnitSphere_ * numOriginalReflectedClusters_ :
                                clulevarry[l].nPoinSpheLv * clulevarry[l].nClustSLv;
        const auto &d = dmtxlev[l];
        op.Levels.push_back({{nt, n, single ? irowtmtx : tmtxlev[l].irowTmxLv, single ? jcoltmtx : tmtxlev[l].jcolTmxLv, single ? ztmtx : tmtxlev[l].zTmxLv}, {d.nRowsD, nt, d.irowDmxLv, d.jcolDmxLv, d.zDmxLv}, {n, d.nRowsD, single ? irowsmtx : smtxlev[l].irowSmxLv, single ? jcolsmtx : smtxlev[l].jcolSmxLv, single ? zsmtx : smtxlev[l].zSmxLv}});
    }
    return op;
}

void Multiply(const Matrix &a, const Complex *x, Complex *y, bool add = false) {
    static_assert(std::is_standard_layout_v<Complex> && sizeof(Complex) == 2 * sizeof(double));
    if (!a.Offsets) {
        const __LAPACK_double_complex one(1, 0), beta(add ? 1 : 0, 0);
        // NumCalc and Accelerate both store complex numbers as interleaved FP64 real/imaginary pairs.
        cblas_zgemv(CblasRowMajor, CblasNoTrans, a.Rows, a.Columns, &one, reinterpret_cast<const __LAPACK_double_complex *>(a.Values), a.Columns, reinterpret_cast<const __LAPACK_double_complex *>(x), 1, &beta, reinterpret_cast<__LAPACK_double_complex *>(y), 1);
    } else {
        ParallelFor(a.Rows, 128, [&](size_t i) {
            Complex sum(0, 0);
            for (int j = a.Offsets[i]; j < a.Offsets[i + 1]; ++j) sum += a.Values[j] * x[a.Indices[j]];
            if (add) y[i] += sum;
            else y[i] = sum;
        });
    }
}

void Direct(const Matrix &matrix, const Complex *rhs) {
    const int n = matrix.Rows;
    std::vector<__LAPACK_double_complex> a(size_t(n) * n), b(n);
    std::vector<__LAPACK_int> pivots(n);
    for (int i = 0; i < n; ++i) {
        b[i] = {rhs[i].re(), rhs[i].im()};
        for (int j = 0; j < n; ++j) {
            const auto z = matrix.Values[size_t(i) * n + j];
            a[size_t(j) * n + i] = {z.re(), z.im()};
        }
    }
    __LAPACK_int size = n, nrhs = 1, info{};
    zgesv_(&size, &nrhs, a.data(), &size, pivots.data(), b.data(), &size, &info);
    if (info) throw std::runtime_error("Accelerate LU failed: " + std::to_string(info));
    for (int i = 0; i < n; ++i) zrhs[i].set(b[i].real(), b[i].imag());
}

double Difference(const std::vector<Complex> &a, const std::vector<Complex> &b) {
    double error{}, norm{};
    for (size_t i = 0; i < a.size(); ++i) {
        error += (a[i] - b[i]).qnorm();
        norm += b[i].qnorm();
    }
    return std::sqrt(error / std::max(norm, 1e-300));
}

nlohmann::json Benchmark(const Operator &op) {
    const int n = op.Near.Rows, repeats = Settings.BenchmarkRepeats;
    std::vector<Complex> x(n), cpu(n), gpu(n);
    for (int i = 0; i < n; ++i) x[i].set(std::sin(i * .17), std::cos(i * .31));
    op.Apply(x.data(), cpu.data());
    Gpu device(op);
    device.Apply(x.data(), gpu.data());
    const double error = Difference(gpu, cpu);
    if (!std::isfinite(error) || error > 1e-10) throw std::runtime_error("Metal operator disagrees with FP64");
    auto start = Clock::now();
    for (int i = 0; i < repeats; ++i) op.Apply(x.data(), cpu.data());
    const double cpuSeconds = Elapsed(start) / repeats;
    start = Clock::now();
    for (int i = 0; i < repeats; ++i) device.Apply(x.data(), gpu.data());
    return {{"repeats", repeats}, {"cpu_seconds", cpuSeconds}, {"metal_seconds", Elapsed(start) / repeats}, {"metal_relative_error", error}, {"scope", "operator application including host/device transfer; warm pipelines"}};
}
} // namespace

void Operator::Apply(const Complex *x, Complex *y) const {
    Multiply(Near, x, y);
    std::vector<Complex> t, d;
    for (const auto &l : Levels) {
        t.resize(l.T.Rows);
        d.resize(l.D.Rows);
        Multiply(l.T, x, t.data());
        Multiply(l.D, t.data(), d.data());
        Multiply(l.S, d.data(), y, true);
    }
}

size_t Operator::Bytes() const {
    auto const bytes = [](const Matrix &a) { return a.Entries() * (sizeof(Complex) + (a.Offsets ? sizeof(int) : 0)) +
                                                 (a.Offsets ? size_t(a.Rows + 1) * sizeof(int) : 0); };
    size_t total = bytes(Near);
    for (const auto &l : Levels) total += bytes(l.T) + bytes(l.D) + bytes(l.S);
    return total;
}
} // namespace numcalc

void NumCalcBeginAssembly() { numcalc::AssemblyStart = numcalc::Clock::now(); }
void NumCalcRequestDirectSolve() { numcalc::DirectRequested = true; }

void NumCalcResetOperator() {
    numcalc::DeviceOperator.reset();
    numcalc::ActiveOperator.reset();
}

void NumCalcMultiply(Vector<Complex> const &x, Vector<Complex> const &y) {
    using namespace numcalc;
    if (!ActiveOperator) {
        ActiveOperator = std::make_unique<Operator>(CurrentOperator());
        if (Settings.Metal) DeviceOperator = std::make_unique<Gpu>(*ActiveOperator);
    }
    const auto start = Clock::now();
    if (DeviceOperator) DeviceOperator->Apply(&x[0], &y[0]);
    else ActiveOperator->Apply(&x[0], &y[0]);
    ApplySeconds += Elapsed(start);
    ++Applications;
}

void NumCalcNativeSolve(std::ofstream &log) {
    using namespace numcalc;
    const double assembly = Elapsed(AssemblyStart);
    NumCalcResetOperator();
    DirectRequested = false;
    auto op = CurrentOperator();
    const int n = op.Near.Rows;
    std::vector<Complex> rhs(zrhs, zrhs + n);
    std::vector<std::vector<Complex>> originals;
    // ILU scales the near and S matrices in place; retain their original values for LU and residual checks.
    if (methodSolver_ != 4 && methodPreconditioner_ == 0) {
        auto const preserve = [&](numcalc::Matrix &a) {
            originals.emplace_back(a.Values, a.Values + a.Entries());
            a.Values = originals.back().data();
        };
        preserve(op.Near);
        for (auto &l : op.Levels) preserve(l.S);
    }
    nlohmann::json report{{"frequency", frequency_}, {"unknowns", n}, {"method", methodFMM_ == 0 ? "dense" : methodFMM_ == 1 ? "slfmm" :
                                                                                                                               "mlfmm"},
                          {"levels", op.Levels.size()},
                          {"operator_bytes", op.Bytes()},
                          {"assembly_seconds", assembly}};
    if (Settings.BenchmarkRepeats) report["benchmark"] = Benchmark(op);
    Applications = 0;
    ApplySeconds = 0;
    double norm{};
    for (const auto &z : rhs) norm += z.qnorm();
    const auto start = Clock::now();
    if (norm == 0) {
        for (int i = 0; i < n; ++i) zrhs[i].set(0, 0);
    } else if (methodSolver_ == 4) {
        if (Settings.Metal) throw std::runtime_error("Direct LU uses Accelerate; select backend cpu");
        Direct(op.Near, rhs.data());
    } else {
        NC_IterativeSolverCGS(log);
        if (DirectRequested) Direct(op.Near, rhs.data());
    }
    report["solver"] = norm == 0 ? "zero" : DirectRequested ? "direct_fallback" :
        methodSolver_ == 4                                  ? "direct" :
                                                              "cgs";
    if (DirectRequested || methodSolver_ == 4) report["direct_backend"] = "accelerate";
    report["solve_seconds"] = Elapsed(start);
    report["operator_seconds"] = ApplySeconds;
    report["operator_applications"] = Applications;
    NumCalcResetOperator();
    std::vector<Complex> actual(n);
    op.Apply(zrhs, actual.data());
    const double residual = Difference(actual, rhs);
    report["relative_residual"] = residual;
    report["residual_limit"] = Settings.ResidualLimit;
    Reports.push_back(report);
    if (!std::isfinite(residual) || residual > Settings.ResidualLimit)
        throw std::runtime_error("True FP64 residual exceeds limit: " + std::to_string(residual));
}
