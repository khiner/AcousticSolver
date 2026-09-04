#include "ImmersedExact.h"
#include "ImmersedGeometry.h"
#include "ImmersedGpu.h"
#include "ImmersedImmittance.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using immersed::RationalImmittance;
using immersed::TrapezoidImmittance;
using json = nlohmann::json;

int Failures = 0;
std::filesystem::path FigureDirectory;
const auto ZeroImmittance = immersed::BoundaryImmittance::Finite(0.);
const auto LimitImmittance = immersed::BoundaryImmittance::Limit();

struct FigureTable {
    std::string Name;
    std::string Paper;
    std::vector<std::string> Columns;
    std::vector<std::string> Rows;
    std::vector<double> Values;
};

std::vector<FigureTable> FigureTables;

void RecordFigure(FigureTable table) {
    if (!FigureDirectory.empty()) FigureTables.push_back(std::move(table));
}

void WriteFigures() {
    std::filesystem::create_directories(FigureDirectory);
    for (const auto &table : FigureTables) {
        if (table.Columns.empty() || table.Values.size() % table.Columns.size() != 0)
            throw std::runtime_error("Immersed figure table has an invalid shape");
        const auto stem = FigureDirectory / table.Name;
        std::ofstream binary{stem.string() + ".bin", std::ios::binary};
        binary.write(reinterpret_cast<const char *>(table.Values.data()), std::streamsize(table.Values.size() * sizeof(double)));
        if (!binary) throw std::runtime_error("Failed to write immersed figure data");
        json metadata{
            {"paper", table.Paper},
            {"columns", table.Columns},
            {"dtype", "float64 little-endian"},
            {"shape", {table.Values.size() / table.Columns.size(), table.Columns.size()}},
        };
        if (!table.Rows.empty()) metadata["rows"] = table.Rows;
        std::ofstream output{stem.string() + ".json"};
        output << metadata.dump(2) << '\n';
        if (!output) throw std::runtime_error("Failed to write immersed figure metadata");
    }
    std::printf("[figures] wrote %zu measured tables to %s\n", FigureTables.size(), FigureDirectory.string().c_str());
}

void AtMost(const char *what, double got, double most) {
    if (got <= most) return;
    std::printf("[gate] FAIL %s: %.6g, want <= %.6g\n", what, got, most);
    ++Failures;
}

void Between(const char *what, double got, double least, double most) {
    if (got >= least && got <= most) return;
    std::printf("[gate] FAIL %s: %.6g, want %.6g to %.6g\n", what, got, least, most);
    ++Failures;
}

struct GridWeights {
    std::vector<int> Index;
    std::vector<double> Weight;
};

GridWeights LagrangeWeights(double grid_coordinate, int node_count, int order) {
    if (order < 1 || order % 2 == 0) throw std::runtime_error("Lagrange order must be positive and odd");
    const int count = order + 1;
    const int first = int(std::floor(grid_coordinate)) - order / 2;
    if (first < 0 || first + count > node_count) throw std::runtime_error("Lagrange support leaves the grid");
    GridWeights result;
    for (int i = 0; i < count; ++i) {
        const int xi = first + i;
        double weight = 1.;
        for (int j = 0; j < count; ++j) {
            if (i == j) continue;
            const int xj = first + j;
            weight *= (grid_coordinate - xj) / double(xi - xj);
        }
        result.Index.push_back(xi);
        result.Weight.push_back(weight);
    }
    return result;
}

double Gather(const std::vector<double> &field, const GridWeights &weights) {
    double value = 0.;
    for (size_t i = 0; i < weights.Index.size(); ++i) value += weights.Weight[i] * field[size_t(weights.Index[i])];
    return value;
}

void Scatter(std::vector<double> &field, const GridWeights &weights, double value) {
    for (size_t i = 0; i < weights.Index.size(); ++i) field[size_t(weights.Index[i])] += weights.Weight[i] * value;
}

double NormSquared(const GridWeights &weights) {
    double value = 0.;
    for (const double w : weights.Weight) value += w * w;
    return value;
}

struct OneDimensional {
    OneDimensional(int half_cells, const RationalImmittance &zv, const RationalImmittance &yp)
        : N(2 * half_cells), H(2. / half_cells), TimeStep(Courant * H / SoundSpeed),
          P(static_cast<size_t>(N) + 1), V(static_cast<size_t>(N)), Jp(LagrangeWeights(2. / H, N + 1, 5)),
          Jv(LagrangeWeights(2. / H - .5, N, 5)), Js(LagrangeWeights(1. / H, N + 1, 5)),
          Left(LagrangeWeights(1.5 / H, N + 1, 5)), Right(LagrangeWeights(2.5 / H, N + 1, 5)),
          Zv(zv, TimeStep), Yp(yp, TimeStep) {}

    void Step(double source, double next_source) {
        const double zv_history = Zv.History() + Zv.PreviousOutput();
        std::vector<double> v_rhs = V;
        for (int i = 0; i < N; ++i) v_rhs[size_t(i)] -= Courant * (P[size_t(i) + 1] - P[size_t(i)]) / Impedance;
        Scatter(v_rhs, Jv, -.5 * Courant * zv_history);
        SolveRankOne(v_rhs, Jv, .5 * Courant * Zv.Feedthrough());
        V = std::move(v_rhs);
        Zv.Step(Gather(V, Jv));

        const double yp_history = Yp.History() + Yp.PreviousOutput();
        std::vector<double> p_rhs = P;
        p_rhs.front() -= Impedance * Courant * V.front();
        for (int i = 1; i < N; ++i)
            p_rhs[size_t(i)] -= Impedance * Courant * (V[size_t(i)] - V[size_t(i - 1)]);
        p_rhs.back() += Impedance * Courant * V.back();
        Scatter(p_rhs, Jp, -.5 * Courant * yp_history);
        Scatter(p_rhs, Js, Impedance * Courant * .5 * (source + next_source));
        SolveRankOne(p_rhs, Jp, .5 * Courant * Yp.Feedthrough());
        P = std::move(p_rhs);
        Yp.Step(Gather(P, Jp));
    }

    static constexpr double SoundSpeed = 343.4;
    static constexpr double Density = 1.2;
    static constexpr double Impedance = SoundSpeed * Density;
    static constexpr double Courant = .9993;
    int N;
    double H, TimeStep;
    std::vector<double> P, V;
    GridWeights Jp, Jv, Js, Left, Right;
    TrapezoidImmittance Zv, Yp;

private:
    static void SolveRankOne(std::vector<double> &field, const GridWeights &weights, double scale) {
        if (scale == 0.) return;
        const double correction = scale * Gather(field, weights) / (1. + scale * NormSquared(weights));
        Scatter(field, weights, -correction);
    }
};

double GaussianPulse(double time, double f60, double maximum) {
    const double sigma = std::sqrt(12. * std::numbers::ln10) / (2. * std::numbers::pi * f60);
    const double t0 = sigma * std::sqrt(2. * std::log(maximum / std::numeric_limits<double>::epsilon()));
    const double x = (time - t0) / sigma;
    return maximum * std::exp(-.5 * x * x);
}

double Gaussian(double time) {
    return GaussianPulse(time, 12000., 1e-3);
}

double GaussianPulseDerivative(double time, double f60, double maximum) {
    const double sigma = std::sqrt(12. * std::numbers::ln10) / (2. * std::numbers::pi * f60);
    const double t0 = sigma * std::sqrt(2. * std::log(maximum / std::numeric_limits<double>::epsilon()));
    return -(time - t0) * GaussianPulse(time, f60, maximum) / (sigma * sigma);
}

double Distance(const immersed::Point &a, const immersed::Point &b = {}) {
    double squared = 0.;
    for (size_t axis = 0; axis < a.size(); ++axis) squared += (a[axis] - b[axis]) * (a[axis] - b[axis]);
    return std::sqrt(squared);
}

immersed::Grid CubicGrid(int cells, double spacing, double courant, int pml_width = 0) {
    immersed::Grid grid;
    grid.Nx = grid.Ny = grid.Nz = cells;
    grid.PmlWidth = pml_width;
    grid.H = spacing;
    grid.Origin = {-.5 * cells * spacing, -.5 * cells * spacing, -.5 * cells * spacing};
    grid.Courant = courant;
    grid.Finalize();
    return grid;
}

immersed::Grid AnalyticGrid(double extent, double sample_rate) {
    immersed::Grid grid;
    const int intervals = int(std::floor(.5754 * extent * sample_rate / grid.C));
    grid.H = extent / intervals;
    grid.Courant = grid.C / (sample_rate * grid.H);
    grid.Nx = grid.Ny = grid.Nz = intervals + 1;
    grid.PmlWidth = 8;
    grid.Origin = {-.5 * (grid.Nx - 1) * grid.H, -.5 * (grid.Ny - 1) * grid.H, -.5 * (grid.Nz - 1) * grid.H};
    grid.Finalize();
    return grid;
}

std::vector<float> GaussianSource(const immersed::Grid &grid, double duration, double f60 = 4000., double maximum = 1e-3) {
    std::vector<float> signal(size_t(std::ceil(duration / grid.TimeStep)) + 1);
    for (size_t step = 0; step < signal.size(); ++step)
        signal[step] = float(GaussianPulse(step * grid.TimeStep, f60, maximum));
    return signal;
}

struct Traces {
    double TimeStep;
    std::vector<double> Left, Right;
};

Traces Simulate(int half_cells, const RationalImmittance &zv, const RationalImmittance &yp, double duration = .0065) {
    OneDimensional solver(half_cells, zv, yp);
    const int steps = int(std::ceil(duration / solver.TimeStep));
    Traces result{solver.TimeStep};
    result.Left.reserve(size_t(steps));
    result.Right.reserve(size_t(steps));
    for (int n = 0; n < steps; ++n) {
        solver.Step(Gaussian(n * solver.TimeStep), Gaussian((n + 1) * solver.TimeStep));
        result.Left.push_back(Gather(solver.P, solver.Left));
        result.Right.push_back(Gather(solver.P, solver.Right));
    }
    return result;
}

double RelativeError(const std::vector<double> &got, const std::vector<double> &want, size_t begin, size_t end) {
    double error = 0., signal = 0.;
    for (size_t i = begin; i < end; ++i) {
        const double d = got[i] - want[i];
        error += d * d;
        signal += want[i] * want[i];
    }
    return std::sqrt(error / signal);
}

double RelativeError(const float *got, size_t stride, const std::vector<double> &want, size_t want_offset, size_t count) {
    double error = 0., signal = 0.;
    for (size_t i = 0; i < count; ++i) {
        const double exact = want[i + want_offset];
        const double delta = got[i * stride] - exact;
        signal += exact * exact;
        error += delta * delta;
    }
    return std::sqrt(error / signal);
}

double FitScale(const std::vector<double> &signal, const std::vector<double> &basis, size_t begin, size_t end) {
    double cross = 0., norm = 0.;
    for (size_t i = begin; i < end; ++i) {
        cross += signal[i] * basis[i];
        norm += basis[i] * basis[i];
    }
    return cross / norm;
}

double Slope(const std::vector<double> &x, const std::vector<double> &y, size_t begin = 0) {
    double sx = 0., sy = 0., sxx = 0., sxy = 0.;
    const double count = double(x.size() - begin);
    for (size_t i = begin; i < x.size(); ++i) {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    return (count * sxy - sx * sy) / (count * sxx - sx * sx);
}

std::vector<double> LogValues(std::vector<double> values) {
    for (double &value : values) value = std::log(value);
    return values;
}

double GatherFloat(const float *field, const immersed::DeltaStencil &stencil) {
    double result = 0.;
    for (size_t i = 0; i < stencil.Index.size(); ++i) result += stencil.Weight[i] * field[stencil.Index[i]];
    return result;
}

double StencilNormSquared(const immersed::DeltaStencil &stencil) {
    double result = 0.;
    for (const double weight : stencil.Weight) result += weight * weight;
    return result;
}

double StencilDot(const immersed::DeltaStencil &a, const immersed::DeltaStencil &b) {
    size_t i = 0, j = 0;
    double result = 0.;
    while (i < a.Index.size() && j < b.Index.size()) {
        if (a.Index[i] < b.Index[j]) ++i;
        else if (b.Index[j] < a.Index[i]) ++j;
        else {
            result += a.Weight[i] * b.Weight[j];
            ++i;
            ++j;
        }
    }
    return result;
}

class LosslessReference {
public:
    LosslessReference(immersed::Grid grid, const std::vector<immersed::Patch> &patches, double mass, double stiffness)
        : Grid(std::move(grid)), Mass(mass), Stiffness(stiffness), P(Grid.NumCells()),
          PPrevious(Grid.NumCells()), Vx(Grid.NumCells()), Vy(Grid.NumCells()), Vz(Grid.NumCells()) {
        const auto shape = std::array{Grid.Nx, Grid.Ny, Grid.Nz};
        Data.reserve(patches.size());
        for (const auto &patch : patches) {
            PatchData data;
            data.Area = patch.Area;
            data.Surface = patch.Area / (Grid.H * Grid.H);
            data.Normal = patch.Normal;
            data.Scale = std::sqrt(.5 * Grid.Courant * data.Surface);
            data.Jx = immersed::LagrangeDelta(patch.Center, Grid.Origin, Grid.H, shape, 5, 0);
            data.Jy = immersed::LagrangeDelta(patch.Center, Grid.Origin, Grid.H, shape, 5, 1);
            data.Jz = immersed::LagrangeDelta(patch.Center, Grid.Origin, Grid.H, shape, 5, 2);
            Data.push_back(std::move(data));
        }
        BuildCholesky();
    }

    void SeedPressure(size_t index, double value) {
        P.at(index) = value;
        PPrevious = P;
    }

    void Step() {
        const double inv_rho = Grid.TimeStep / (Grid.Rho * Grid.H);
        const double rho_cc = Grid.Rho * Grid.C * Grid.C * Grid.TimeStep / Grid.H;
        const int plane = Grid.Nx * Grid.Ny;
        for (int z = 0; z < Grid.Nz; ++z) {
            for (int y = 0; y < Grid.Ny; ++y) {
                for (int x = 0; x < Grid.Nx; ++x) {
                    const size_t i = size_t(x) + size_t(Grid.Nx) * (size_t(y) + size_t(Grid.Ny) * size_t(z));
                    const double center = P[i];
                    Vx[i] = x + 1 < Grid.Nx ? Vx[i] - inv_rho * (P[i + 1] - center) : 0.;
                    Vy[i] = y + 1 < Grid.Ny ? Vy[i] - inv_rho * (P[i + size_t(Grid.Nx)] - center) : 0.;
                    Vz[i] = z + 1 < Grid.Nz ? Vz[i] - inv_rho * (P[i + size_t(plane)] - center) : 0.;
                }
            }
        }

        for (PatchData const &patch : Data) {
            const double history = (Stiffness * Grid.TimeStep - 4. * Mass / Grid.TimeStep) * patch.Vbar -
                2. * Mass * patch.Derivative + 2. * Stiffness * patch.Integral;
            const double forcing = -.5 * Grid.Courant * patch.Surface * (history + patch.PreviousForcing);
            ScatterPatch(patch, forcing);
        }
        std::vector<double> rhs(Data.size());
        for (size_t k = 0; k < Data.size(); ++k) rhs[k] = Data[k].Scale * GatherPatch(Data[k]);
        const std::vector<double> solution = Solve(rhs);
        for (size_t k = 0; k < Data.size(); ++k) ScatterPatch(Data[k], -Data[k].Scale * solution[k]);

        for (PatchData &patch : Data) {
            const double next = GatherPatch(patch);
            const double derivative = 2. / Grid.TimeStep * (next - patch.Vbar) - patch.Derivative;
            const double integral = patch.Integral + .5 * Grid.TimeStep * (next + patch.Vbar);
            patch.Vbar = next;
            patch.Derivative = derivative;
            patch.Integral = integral;
            patch.PreviousForcing = 2. * Mass * derivative + 2. * Stiffness * integral;
        }

        PPrevious = P;
        for (int z = 0; z < Grid.Nz; ++z) {
            for (int y = 0; y < Grid.Ny; ++y) {
                for (int x = 0; x < Grid.Nx; ++x) {
                    const size_t i = size_t(x) + size_t(Grid.Nx) * (size_t(y) + size_t(Grid.Ny) * size_t(z));
                    const double left = x > 0 ? Vx[i - 1] : 0.;
                    const double down = y > 0 ? Vy[i - size_t(Grid.Nx)] : 0.;
                    const double front = z > 0 ? Vz[i - size_t(plane)] : 0.;
                    P[i] -= rho_cc * (Vx[i] - left + Vy[i] - down + Vz[i] - front);
                }
            }
        }
    }

    double Energy() const {
        double velocity = 0., pressure = 0.;
        for (size_t i = 0; i < P.size(); ++i) {
            velocity += Vx[i] * Vx[i] + Vy[i] * Vy[i] + Vz[i] * Vz[i];
            pressure += P[i] * PPrevious[i];
        }
        const double volume = Grid.H * Grid.H * Grid.H;
        double boundary = 0.;
        for (const PatchData &patch : Data)
            boundary += patch.Area * (Mass * patch.Vbar * patch.Vbar + Stiffness * patch.Integral * patch.Integral);
        return .5 * Grid.Rho * volume * velocity + volume / (2. * Grid.Rho * Grid.C * Grid.C) * pressure +
            Grid.Rho * Grid.C * boundary;
    }

private:
    struct PatchData {
        immersed::DeltaStencil Jx, Jy, Jz;
        immersed::Point Normal{};
        double Area{0.}, Surface{0.}, Scale{0.};
        double Vbar{0.}, Derivative{0.}, Integral{0.}, PreviousForcing{0.};
    };

    double GatherPatch(const PatchData &patch) const {
        return patch.Normal[0] * immersed::Gather(Vx, patch.Jx) +
            patch.Normal[1] * immersed::Gather(Vy, patch.Jy) +
            patch.Normal[2] * immersed::Gather(Vz, patch.Jz);
    }

    void ScatterPatch(const PatchData &patch, double value) {
        for (size_t i = 0; i < patch.Jx.Index.size(); ++i)
            Vx[size_t(patch.Jx.Index[i])] += value * patch.Normal[0] * patch.Jx.Weight[i];
        for (size_t i = 0; i < patch.Jy.Index.size(); ++i)
            Vy[size_t(patch.Jy.Index[i])] += value * patch.Normal[1] * patch.Jy.Weight[i];
        for (size_t i = 0; i < patch.Jz.Index.size(); ++i)
            Vz[size_t(patch.Jz.Index[i])] += value * patch.Normal[2] * patch.Jz.Weight[i];
    }

    void BuildCholesky() {
        const int count = int(Data.size());
        Cholesky.assign(size_t(count) * count, 0.);
        const double feedthrough = 4. * Mass / Grid.TimeStep + Stiffness * Grid.TimeStep;
        for (int row = 0; row < count; ++row) {
            for (int column = 0; column <= row; ++column) {
                const auto &a = Data[size_t(row)];
                const auto &b = Data[size_t(column)];
                const double dot = a.Normal[0] * b.Normal[0] * StencilDot(a.Jx, b.Jx) +
                    a.Normal[1] * b.Normal[1] * StencilDot(a.Jy, b.Jy) +
                    a.Normal[2] * b.Normal[2] * StencilDot(a.Jz, b.Jz);
                double value = a.Scale * b.Scale * dot;
                if (row == column) value += 1. / feedthrough;
                for (int k = 0; k < column; ++k)
                    value -= Cholesky[size_t(row) * count + k] * Cholesky[size_t(column) * count + k];
                Cholesky[size_t(row) * count + column] = row == column ? std::sqrt(value) :
                                                                         value / Cholesky[size_t(column) * count + column];
            }
        }
    }

    std::vector<double> Solve(const std::vector<double> &rhs) const {
        const int count = int(rhs.size());
        std::vector<double> result(rhs);
        for (int row = 0; row < count; ++row) {
            for (int column = 0; column < row; ++column)
                result[size_t(row)] -= Cholesky[size_t(row) * count + column] * result[size_t(column)];
            result[size_t(row)] /= Cholesky[size_t(row) * count + row];
        }
        for (int row = count - 1; row >= 0; --row) {
            for (int column = row + 1; column < count; ++column)
                result[size_t(row)] -= Cholesky[size_t(column) * count + row] * result[size_t(column)];
            result[size_t(row)] /= Cholesky[size_t(row) * count + row];
        }
        return result;
    }

    immersed::Grid Grid;
    double Mass, Stiffness;
    std::vector<double> P, PPrevious, Vx, Vy, Vz;
    std::vector<PatchData> Data;
    std::vector<double> Cholesky;
};

void Filters(bool gate) {
    constexpr double time_step = 2e-5;
    const std::vector<RationalImmittance> models{
        RationalImmittance::Constant(3.25),
        RationalImmittance::Series(10., 0., 0.),
        RationalImmittance::Series(0., .001, 0.),
        RationalImmittance::Series(0., 0., 3000.),
        RationalImmittance::Series(0., .002, 50000.),
        RationalImmittance::Series(0., .001, 50000.).Reciprocal(),
    };
    double worst = 0.;
    for (const auto &model : models) {
        const TrapezoidImmittance discrete(model, time_step);
        for (const double frequency : {100., 1000., 5000., 10000.}) {
            const double omega = 2. * std::numbers::pi * frequency;
            const std::complex<double> s{0., 2. / time_step * std::tan(.5 * omega * time_step)};
            const auto want = 2. * model.Evaluate(s);
            worst = std::max(worst, std::abs(discrete.Response(omega) - want) / std::max(std::abs(want), 1e-300));
        }
    }
    std::printf("[filters] bilinear response max relative error %.3e\n", worst);
    if (gate) AtMost("filter response relative error", worst, 1e-12);
}

void OneD(bool gate) {
    const auto zero = RationalImmittance::Constant(0.);
    const auto resistance = RationalImmittance::Constant(10.);
    const Traces free = Simulate(291, zero, zero);
    const Traces barrier = Simulate(291, resistance, zero);
    const double reflection = 10. / 11., transmission = 1. / 11.;
    std::vector<double> reflected(free.Left.size()), reflected_want(free.Left.size()), transmitted_want(free.Left.size());
    for (size_t i = 0; i < free.Left.size(); ++i) {
        reflected[i] = barrier.Left[i] - free.Left[i];
        reflected_want[i] = reflection * free.Right[i];
        transmitted_want[i] = transmission * free.Right[i];
    }
    const size_t begin = size_t(.0045 / free.TimeStep), end = std::min(free.Left.size(), size_t(.0056 / free.TimeStep));
    const double reflection_error = RelativeError(reflected, reflected_want, begin, end);
    const double transmission_error = RelativeError(barrier.Right, transmitted_want, begin, end);
    const double measured_reflection = FitScale(reflected, free.Right, begin, end);
    const double measured_transmission = FitScale(barrier.Right, free.Right, begin, end);

    std::vector<double> steps, leakage;
    for (const int half_cells : {145, 291, 582, 1164}) {
        const auto impedance = Simulate(half_cells, resistance, resistance.Reciprocal());
        const size_t i0 = size_t(.0045 / impedance.TimeStep), i1 = std::min(impedance.Right.size(), size_t(.0056 / impedance.TimeStep));
        double leak_peak = 0.;
        for (size_t i = i0; i < i1; ++i) leak_peak = std::max(leak_peak, std::abs(impedance.Right[i]));
        steps.push_back(impedance.TimeStep);
        leakage.push_back(leak_peak);
    }
    const double leakage_order = Slope(LogValues(steps), LogValues(leakage));
    std::printf("[one-d] a=10 reflection %.6f (want %.6f, waveform rel %.3e) | transmission %.6f "
                "(want %.6f, waveform rel %.3e) | impedance leakage order %.3f\n",
                measured_reflection, reflection, reflection_error, measured_transmission, transmission, transmission_error, leakage_order);
    if (gate) {
        AtMost("1D reflection coefficient error", std::abs(measured_reflection - reflection), .005);
        AtMost("1D transmission coefficient error", std::abs(measured_transmission - transmission), .005);
        Between("1D leakage order", leakage_order, .8, 1.2);
    }
}

void FreeField(bool gate) {
    constexpr double F60 = 3000., Maximum = 1e-3, Duration = .0055;
    const immersed::Grid grid = CubicGrid(96, .02, .99 / std::numbers::sqrt3, 8);
    const auto signal = GaussianSource(grid, Duration, F60, Maximum);
    const int steps = int(signal.size()) - 1;
    const std::vector<immersed::Point> sources{{0., 0., 0.}};
    const std::vector<immersed::Point> receivers{{.23, .04, .02}, {.31, -.07, .05}, {.39, .08, -.03}};

    immersed::Gpu gpu;
    gpu.Init(grid, sources, receivers, steps, signal);
    gpu.RunSteps(steps);
    const float *samples = gpu.Samples();
    double signal_energy = 0., error_energy = 0.;
    for (int n = 0; n < steps; ++n) {
        const double time = (n + 1) * grid.TimeStep;
        for (size_t receiver = 0; receiver < receivers.size(); ++receiver) {
            const double radius = Distance(receivers[receiver]);
            const double truth = grid.Rho * GaussianPulseDerivative(time - radius / grid.C, F60, Maximum) /
                (4. * std::numbers::pi * radius);
            const double error = double(samples[size_t(n) * receivers.size() + receiver]) - truth;
            signal_energy += truth * truth;
            error_energy += error * error;
        }
    }
    const double snr = 10. * std::log10(signal_energy / error_energy);
    std::printf("[free] 3D Gaussian monopole SNR %.2f dB at %zu receivers\n", snr, receivers.size());
    if (gate) AtMost("free-field inverse SNR", -snr, -15.);
}

struct HostState {
    std::vector<double> P, Vx, Vy, Vz;
};

HostState ReferenceStep(const immersed::Grid &grid, const immersed::Patch &patch, const std::vector<float> &p_seed, const std::vector<float> &vx_seed, const std::vector<float> &vy_seed, const std::vector<float> &vz_seed, int steps) {
    HostState state{{p_seed.begin(), p_seed.end()}, {vx_seed.begin(), vx_seed.end()}, {vy_seed.begin(), vy_seed.end()}, {vz_seed.begin(), vz_seed.end()}};
    const std::array shape{grid.Nx, grid.Ny, grid.Nz};
    const auto jx = immersed::LagrangeDelta(patch.Center, grid.Origin, grid.H, shape, 5, 0);
    const auto jy = immersed::LagrangeDelta(patch.Center, grid.Origin, grid.H, shape, 5, 1);
    const auto jz = immersed::LagrangeDelta(patch.Center, grid.Origin, grid.H, shape, 5, 2);
    const auto jp = immersed::LagrangeDelta(patch.Center, grid.Origin, grid.H, shape, 5);
    const std::array velocity{&state.Vx, &state.Vy, &state.Vz};
    const std::array stencil{&jx, &jy, &jz};
    const auto gather_velocity = [&] {
        double value = 0.;
        for (size_t axis = 0; axis < velocity.size(); ++axis)
            value += patch.Normal[axis] * immersed::Gather(*velocity[axis], *stencil[axis]);
        return value;
    };
    const auto scatter_velocity = [&](double value) {
        for (size_t axis = 0; axis < velocity.size(); ++axis)
            for (size_t i = 0; i < stencil[axis]->Index.size(); ++i)
                (*velocity[axis])[size_t(stencil[axis]->Index[i])] += value * patch.Normal[axis] * stencil[axis]->Weight[i];
    };
    const double inv_rho = grid.TimeStep / (grid.Rho * grid.H);
    const double rho_cc = grid.Rho * grid.C * grid.C * grid.TimeStep / grid.H;
    const int plane = grid.Nx * grid.Ny;
    TrapezoidImmittance zv(patch.Zv.Value, grid.TimeStep), yp(patch.Yp.Value, grid.TimeStep);
    for (int step = 0; step < steps; ++step) {
        for (int z = 0; z < grid.Nz; ++z) {
            for (int y = 0; y < grid.Ny; ++y) {
                for (int x = 0; x < grid.Nx; ++x) {
                    const int i = x + grid.Nx * (y + grid.Ny * z);
                    const double center = state.P[size_t(i)];
                    state.Vx[size_t(i)] = x + 1 < grid.Nx ? state.Vx[size_t(i)] - inv_rho * (state.P[size_t(i) + 1] - center) : 0.;
                    state.Vy[size_t(i)] = y + 1 < grid.Ny ? state.Vy[size_t(i)] - inv_rho * (state.P[size_t(i) + size_t(grid.Nx)] - center) : 0.;
                    state.Vz[size_t(i)] = z + 1 < grid.Nz ? state.Vz[size_t(i)] - inv_rho * (state.P[size_t(i) + size_t(plane)] - center) : 0.;
                }
            }
        }
        const double surface = patch.Area / (grid.H * grid.H);
        const double scale = std::sqrt(.5 * grid.Courant * surface);
        const double v_history = zv.History() + zv.PreviousOutput();
        scatter_velocity(-.5 * grid.Courant * surface * v_history);
        const double vbar = gather_velocity();
        double nv_norm = 0.;
        for (size_t axis = 0; axis < velocity.size(); ++axis)
            nv_norm += patch.Normal[axis] * patch.Normal[axis] * StencilNormSquared(*stencil[axis]);
        if (zv.Feedthrough() > 0.) {
            const double v_solution = scale * vbar / (1. / zv.Feedthrough() + scale * scale * nv_norm);
            scatter_velocity(-scale * v_solution);
        }
        zv.Step(gather_velocity());

        for (int z = 0; z < grid.Nz; ++z) {
            for (int y = 0; y < grid.Ny; ++y) {
                for (int x = 0; x < grid.Nx; ++x) {
                    const int i = x + grid.Nx * (y + grid.Ny * z);
                    const double left = x > 0 ? state.Vx[size_t(i - 1)] : 0.;
                    const double down = y > 0 ? state.Vy[size_t(i - grid.Nx)] : 0.;
                    const double front = z > 0 ? state.Vz[size_t(i - plane)] : 0.;
                    state.P[size_t(i)] -= rho_cc * (state.Vx[size_t(i)] - left + state.Vy[size_t(i)] - down + state.Vz[size_t(i)] - front);
                }
            }
        }
        const double p_history = yp.History() + yp.PreviousOutput();
        for (size_t i = 0; i < jp.Index.size(); ++i)
            state.P[size_t(jp.Index[i])] -= .5 * grid.Courant * surface * jp.Weight[i] * p_history;
        const double pbar = immersed::Gather(state.P, jp);
        if (yp.Feedthrough() > 0.) {
            const double p_solution = scale * pbar / (1. / yp.Feedthrough() + scale * scale * StencilNormSquared(jp));
            for (size_t i = 0; i < jp.Index.size(); ++i) state.P[size_t(jp.Index[i])] -= scale * jp.Weight[i] * p_solution;
        }
        yp.Step(immersed::Gather(state.P, jp));
    }
    return state;
}

void Reference(bool gate) {
    const immersed::Grid grid = CubicGrid(40, .04, .5);
    const double inv_norm = 1. / std::sqrt(14.);
    const auto zv = RationalImmittance::Series(.2, .002, 5000.);
    immersed::Patch patch{{.013, -.027, .041}, {inv_norm, 2. * inv_norm, 3. * inv_norm}, grid.H * grid.H, immersed::BoundaryImmittance::Finite(zv), immersed::BoundaryImmittance::Finite(zv.Reciprocal())};
    std::vector<float> p(grid.NumCells()), vx(grid.NumCells()), vy(grid.NumCells()), vz(grid.NumCells());
    for (size_t i = 0; i < p.size(); ++i) {
        p[i] = float(1e-3 * std::sin(.017 * double(i)));
        vx[i] = float(1e-6 * std::cos(.013 * double(i)));
        vy[i] = float(1e-6 * std::sin(.011 * double(i)));
        vz[i] = float(1e-6 * std::cos(.019 * double(i)));
    }
    constexpr int Steps = 12;
    const HostState reference = ReferenceStep(grid, patch, p, vx, vy, vz, Steps);
    immersed::Gpu gpu;
    gpu.Init(grid, {}, {}, Steps, {}, {patch});
    gpu.SeedPressure(p);
    gpu.SeedVelocity(vx, vy, vz);
    gpu.RunSteps(Steps);
    const float *got_p = gpu.Pressure();
    const float *got_vx = gpu.VelocityX();
    const float *got_vy = gpu.VelocityY();
    const float *got_vz = gpu.VelocityZ();
    double signal = 0., error = 0.;
    for (size_t i = 0; i < p.size(); ++i) {
        for (const auto &[truth, got] : {std::pair{reference.P[i], double(got_p[i])}, {reference.Vx[i], double(got_vx[i])}, {reference.Vy[i], double(got_vy[i])}, {reference.Vz[i], double(got_vz[i])}}) {
            signal += truth * truth;
            error += (got - truth) * (got - truth);
        }
    }
    const double relative = std::sqrt(error / signal);

    const auto shape = std::array{grid.Nx, grid.Ny, grid.Nz};
    const auto jp = immersed::LagrangeDelta(patch.Center, grid.Origin, grid.H, shape, 5);
    const auto jx = immersed::LagrangeDelta(patch.Center, grid.Origin, grid.H, shape, 5, 0);
    const auto jy = immersed::LagrangeDelta(patch.Center, grid.Origin, grid.H, shape, 5, 1);
    const auto jz = immersed::LagrangeDelta(patch.Center, grid.Origin, grid.H, shape, 5, 2);

    immersed::Patch rigid = patch;
    rigid.Zv = LimitImmittance;
    rigid.Yp = ZeroImmittance;
    immersed::Gpu rigid_gpu;
    rigid_gpu.Init(grid, {}, {}, 1, {}, {rigid});
    rigid_gpu.SeedPressure(p);
    rigid_gpu.SeedVelocity(vx, vy, vz);
    rigid_gpu.RunSteps(1);
    const double rigid_velocity = patch.Normal[0] * GatherFloat(rigid_gpu.VelocityX(), jx) +
        patch.Normal[1] * GatherFloat(rigid_gpu.VelocityY(), jy) +
        patch.Normal[2] * GatherFloat(rigid_gpu.VelocityZ(), jz);

    immersed::Patch release = patch;
    release.Zv = ZeroImmittance;
    release.Yp = LimitImmittance;
    immersed::Gpu release_gpu;
    release_gpu.Init(grid, {}, {}, 1, {}, {release});
    release_gpu.SeedPressure(p);
    release_gpu.SeedVelocity(vx, vy, vz);
    release_gpu.RunSteps(1);
    const double release_pressure = GatherFloat(release_gpu.Pressure(), jp);

    std::printf("[reference] host vs GPU relative %.3e | rigid vbar %.3e | release pbar %.3e | condition %.3e/%.3e\n", relative, rigid_velocity, release_pressure, rigid_gpu.VelocityCondition(), release_gpu.PressureCondition());
    if (gate) {
        AtMost("host versus GPU relative error", relative, 2e-6);
        AtMost("rigid projected velocity", std::abs(rigid_velocity), 1e-10);
        AtMost("release projected pressure", std::abs(release_pressure), 1e-7);
    }
}

void Null(bool gate) {
    const immersed::Grid grid = CubicGrid(32, .05, .5);
    const auto resonant = RationalImmittance::Series(.2, .001, 3000.);
    const std::vector<immersed::Patch> patches{
        {{-.25, 0., 0.}, {1., 0., 0.}, grid.H * grid.H, immersed::BoundaryImmittance::Finite(resonant), ZeroImmittance},
        {{0., .25, 0.}, {0., 1., 0.}, grid.H * grid.H, LimitImmittance, ZeroImmittance},
        {{.25, 0., 0.}, {0., 0., 1.}, grid.H * grid.H, ZeroImmittance, LimitImmittance},
        {{0., -.25, 0.}, {1., 0., 0.}, grid.H * grid.H, immersed::BoundaryImmittance::Finite(resonant), immersed::BoundaryImmittance::Finite(resonant.Reciprocal())},
    };
    immersed::Gpu gpu;
    gpu.Init(grid, {}, {}, 4, {}, patches);
    gpu.RunSteps(4);
    double zero_maximum = 0.;
    for (const float *field : {gpu.Pressure(), gpu.VelocityX(), gpu.VelocityY(), gpu.VelocityZ()})
        for (size_t i = 0; i < grid.NumCells(); ++i) zero_maximum = std::max(zero_maximum, std::abs(double(field[i])));

    immersed::Gpu constant_gpu;
    constant_gpu.Init(grid, {}, {}, 4, {}, {patches[0], patches[1]});
    constant_gpu.SeedPressure(std::vector<float>(grid.NumCells(), 1.f));
    constant_gpu.RunSteps(4);
    const float *constant_p = constant_gpu.Pressure();
    const float *constant_vx = constant_gpu.VelocityX();
    const float *constant_vy = constant_gpu.VelocityY();
    const float *constant_vz = constant_gpu.VelocityZ();
    double constant_maximum = 0.;
    for (size_t i = 0; i < grid.NumCells(); ++i) {
        constant_maximum = std::max(constant_maximum, std::abs(double(constant_p[i]) - 1.));
        constant_maximum = std::max({constant_maximum, std::abs(double(constant_vx[i])), std::abs(double(constant_vy[i])), std::abs(double(constant_vz[i]))});
    }
    std::printf("[null] mixed finite, rigid and release zero max %.1e | constant-field residual %.1e\n", zero_maximum, constant_maximum);
    if (gate) {
        AtMost("zero-state residual", zero_maximum, 0.);
        AtMost("constant-field residual", constant_maximum, 0.);
    }
}

double GpuLosslessEnergy(const immersed::Grid &grid, const std::vector<immersed::Patch> &patches, const immersed::Gpu &gpu, const std::vector<float> &previous_pressure, const std::vector<double> &integrals, double mass, double stiffness) {
    const float *p = gpu.Pressure();
    const float *vx = gpu.VelocityX();
    const float *vy = gpu.VelocityY();
    const float *vz = gpu.VelocityZ();
    double velocity = 0., pressure = 0., boundary = 0.;
    for (size_t i = 0; i < grid.NumCells(); ++i) {
        velocity += double(vx[i]) * vx[i] + double(vy[i]) * vy[i] + double(vz[i]) * vz[i];
        pressure += double(p[i]) * previous_pressure[i];
    }
    const auto *states = gpu.VelocityStates();
    for (size_t k = 0; k < patches.size(); ++k)
        boundary += patches[k].Area * (mass * double(states[k].X1) * states[k].X1 + stiffness * integrals[k] * integrals[k]);
    const double volume = grid.H * grid.H * grid.H;
    return .5 * grid.Rho * volume * velocity + volume / (2. * grid.Rho * grid.C * grid.C) * pressure +
        grid.Rho * grid.C * boundary;
}

void Energy(bool gate) {
    constexpr double Mass = .0005, Stiffness = 500.;
    const int steps = gate ? 2000 : 20000;
    const immersed::Grid grid = CubicGrid(18, .08, .5);
    const auto zv = immersed::BoundaryImmittance::Finite(RationalImmittance::Series(0., Mass, Stiffness));
    const auto patches = immersed::SpherePatches({.013, -.009, .017}, .2, grid.H * grid.H, zv, ZeroImmittance);
    const size_t impulse = size_t(grid.Nx / 2) + size_t(grid.Nx) * (size_t(grid.Ny / 2) + size_t(grid.Ny) * size_t(grid.Nz / 2));

    LosslessReference host(grid, patches, Mass, Stiffness);
    host.SeedPressure(impulse, 1.);
    const double host_initial = host.Energy();
    double host_drift = 0.;
    for (int step = 0; step < steps; ++step) {
        host.Step();
        if ((step + 1) % 20 == 0 || step + 1 == steps)
            host_drift = std::max(host_drift, std::abs(host.Energy() - host_initial) / host_initial);
    }

    std::vector<float> seed(grid.NumCells());
    seed[impulse] = 1.f;
    immersed::Gpu gpu;
    gpu.Init(grid, {}, {}, steps, {}, patches);
    gpu.SeedPressure(seed);
    std::vector<float> previous_pressure = seed;
    std::vector<double> integrals(patches.size()), last_velocity(patches.size());
    const double gpu_initial = GpuLosslessEnergy(grid, patches, gpu, previous_pressure, integrals, Mass, Stiffness);
    double gpu_drift = 0.;
    for (int step = 0; step < steps; ++step) {
        const bool measure = (step + 1) % 20 == 0 || step + 1 == steps;
        if (measure) {
            const float *pressure = gpu.Pressure();
            previous_pressure.assign(pressure, pressure + grid.NumCells());
        }
        gpu.RunSteps(1);
        const auto *states = gpu.VelocityStates();
        for (size_t k = 0; k < patches.size(); ++k) {
            const double velocity = states[k].X1;
            integrals[k] += .5 * grid.TimeStep * (velocity + last_velocity[k]);
            last_velocity[k] = velocity;
        }
        if (measure)
            gpu_drift = std::max(gpu_drift, std::abs(GpuLosslessEnergy(grid, patches, gpu, previous_pressure, integrals, Mass, Stiffness) - gpu_initial) / gpu_initial);
    }
    std::printf("[energy] lossless sphere %d steps | host drift %.3e | GPU drift %.3e\n", steps, host_drift, gpu_drift);
    RecordFigure({"Fig12Energy", "Bilbao 2022, Figure 12", {"steps", "host_relative_drift", "metal_relative_drift"}, {}, {double(steps), host_drift, gpu_drift}});
    if (gate) {
        AtMost("host lossless energy drift", host_drift, 1e-12);
        AtMost("GPU lossless energy drift", gpu_drift, 1e-4);
    }
}

double StateMagnitude(const immersed::Grid &grid, const immersed::Gpu &gpu) {
    const double impedance = grid.Rho * grid.C;
    const float *p = gpu.Pressure();
    const float *vx = gpu.VelocityX();
    const float *vy = gpu.VelocityY();
    const float *vz = gpu.VelocityZ();
    double norm = 0.;
    for (size_t i = 0; i < grid.NumCells(); ++i) {
        const double pressure = p[i] / impedance;
        norm += pressure * pressure + double(vx[i]) * vx[i] + double(vy[i]) * vy[i] + double(vz[i]) * vz[i];
    }
    return std::sqrt(norm / grid.NumCells());
}

double SoakLeg(const char *name, double courant, const std::vector<immersed::Patch> &patches, int steps) {
    const immersed::Grid grid = CubicGrid(18, .08, courant);
    std::vector<float> p(grid.NumCells()), vx(grid.NumCells()), vy(grid.NumCells()), vz(grid.NumCells());
    uint32_t random = 0x9e3779b9u;
    auto const fill = [&](std::vector<float> &field, double scale) {
        double mean = 0.;
        for (float &value : field) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            value = float(scale * (double(random) / double(UINT32_MAX) - .5));
            mean += value;
        }
        mean /= field.size();
        for (float &value : field) value -= float(mean);
    };
    fill(p, 1e-3);
    fill(vx, 1e-3 / (grid.Rho * grid.C));
    fill(vy, 1e-3 / (grid.Rho * grid.C));
    fill(vz, 1e-3 / (grid.Rho * grid.C));

    immersed::Gpu gpu;
    gpu.Init(grid, {}, {}, steps, {}, patches);
    gpu.SeedPressure(p);
    gpu.SeedVelocity(vx, vy, vz);
    constexpr int Reports = 100;
    const int chunk = std::max(1, steps / Reports);
    std::vector<double> x, y;
    for (int done = 0; done < steps;) {
        const int count = std::min(chunk, steps - done);
        gpu.RunSteps(count);
        done += count;
        x.push_back(done);
        y.push_back(std::log(StateMagnitude(grid, gpu)));
    }
    const double growth = 20. * 1000. / std::numbers::ln10 * Slope(x, y, x.size() / 4);
    std::printf("  %-19s lambda %.6f growth %+.4f dB per 1000\n", name, courant, growth);
    return growth;
}

void Soak(bool gate) {
    const int steps = gate ? 4000 : 40000;
    const double h = .08;
    const auto resonant = immersed::BoundaryImmittance::Finite(RationalImmittance::Series(0., .0005, 500.));
    const auto transmitting = immersed::BoundaryImmittance::Finite(1.);
    const auto rigid_sphere = immersed::SpherePatches({.013, -.009, .017}, .2, h * h, LimitImmittance, ZeroImmittance);
    const auto resonant_sphere = immersed::SpherePatches({.013, -.009, .017}, .2, h * h, resonant, ZeroImmittance);
    const double angle = std::numbers::pi / 6.;
    const immersed::Point u{std::cos(angle), 0., -std::sin(angle)};
    const auto barrier = immersed::SquarePatches({0., 0., 0.}, u, {0., 1., 0.}, .48, h * h, transmitting, ZeroImmittance);
    std::printf("[soak] mean-removed noise, %d steps a leg\n", steps);
    double worst_growth = -std::numeric_limits<double>::infinity();
    for (const double courant : {.999 / std::numbers::sqrt3, .95 / std::numbers::sqrt3}) {
        worst_growth = std::max(worst_growth, SoakLeg("rigid sphere", courant, rigid_sphere, steps));
        worst_growth = std::max(worst_growth, SoakLeg("resonant sphere", courant, resonant_sphere, steps));
        worst_growth = std::max(worst_growth, SoakLeg("transmitting barrier", courant, barrier, steps));
    }
    if (gate) AtMost("noise-soak growth", worst_growth, .05);
}

std::vector<std::complex<double>> Dft(const std::vector<double> &signal, bool inverse) {
    const size_t count = signal.size();
    std::vector<std::complex<double>> result(count);
    const double sign = inverse ? 1. : -1.;
    for (size_t k = 0; k < count; ++k) {
        std::complex<double> sum{};
        for (size_t n = 0; n < count; ++n) {
            const double phase = sign * 2. * std::numbers::pi * double(k * n) / count;
            sum += signal[n] * std::exp(std::complex<double>{0., phase});
        }
        result[k] = sum / (inverse ? double(count) : 1.);
    }
    return result;
}

std::vector<double> InverseDft(const std::vector<std::complex<double>> &spectrum) {
    const size_t count = spectrum.size();
    std::vector<double> result(count);
    for (size_t n = 0; n < count; ++n) {
        std::complex<double> sum{};
        for (size_t k = 0; k < count; ++k) {
            const double phase = 2. * std::numbers::pi * double(k * n) / count;
            sum += spectrum[k] * std::exp(std::complex<double>{0., phase});
        }
        result[n] = sum.real() / count;
    }
    return result;
}

std::vector<std::vector<double>> ExactSphereTraces(const std::vector<float> &source_signal, double sample_rate, const immersed::Grid &grid, double radius, const immersed::Point &source, const std::vector<immersed::Point> &receivers, double alpha) {
    size_t count = 1;
    while (count < 4 * source_signal.size()) count *= 2;
    std::vector<double> padded(count);
    std::copy(source_signal.begin(), source_signal.end(), padded.begin());
    const auto source_spectrum = Dft(padded, false);
    std::vector<std::vector<double>> traces;
    traces.reserve(receivers.size());
    for (const auto &receiver : receivers) {
        std::vector<std::complex<double>> pressure(count);
        for (size_t k = 1; k < count / 2; ++k) {
            const double omega = 2. * std::numbers::pi * k * sample_rate / count;
            pressure[k] = source_spectrum[k] * immersed::SphereTransfer(omega, grid.C, grid.Rho, radius, source, receiver, alpha);
            pressure[count - k] = std::conj(pressure[k]);
        }
        const size_t nyquist = count / 2;
        const auto transfer = immersed::SphereTransfer(std::numbers::pi * sample_rate, grid.C, grid.Rho, radius, source, receiver, alpha);
        pressure[nyquist] = source_spectrum[nyquist].real() * transfer.real();
        traces.push_back(InverseDft(pressure));
    }
    return traces;
}

std::array<double, 3> SphereCase(double alpha, int interpolation_order = 5, double sample_rate = 25000.) {
    const immersed::Point source{0., 0., .4};
    const std::vector<immersed::Point> receivers{{0., 0., .3}, {.3, 0., 0.}, {0., 0., -.3}};
    const immersed::Grid grid = AnalyticGrid(2.2, sample_rate);
    const auto source_signal = GaussianSource(grid, .0045);
    const int steps = int(source_signal.size()) - 1;
    immersed::BoundaryImmittance zv, yp;
    if (std::isinf(alpha)) {
        zv = LimitImmittance;
        yp = ZeroImmittance;
    } else if (alpha == 0.) {
        zv = ZeroImmittance;
        yp = LimitImmittance;
    } else {
        zv = immersed::BoundaryImmittance::Finite(alpha);
        yp = immersed::BoundaryImmittance::Finite(1. / alpha);
    }
    const double patch_size = grid.H;
    const auto patches = immersed::SpherePatches({0., 0., 0.}, .25, patch_size * patch_size, zv, yp);
    immersed::Gpu gpu;
    gpu.Init(grid, {source}, receivers, steps, source_signal, patches, interpolation_order);
    gpu.RunSteps(steps);
    const float *samples = gpu.Samples();
    const auto truth = ExactSphereTraces(source_signal, sample_rate, grid, .25, source, receivers, alpha);
    std::array<double, 3> errors{};
    for (size_t receiver = 0; receiver < receivers.size(); ++receiver)
        errors[receiver] = RelativeError(samples + receiver, receivers.size(), truth[receiver], 1, size_t(steps));
    return errors;
}

void Sphere(bool gate) {
    const double sample_rate = gate ? 25000. : 50000.;
    const auto rigid = SphereCase(std::numeric_limits<double>::infinity(), 5, sample_rate);
    const auto release = SphereCase(0., 5, sample_rate);
    const auto resistive = SphereCase(1., 5, sample_rate);
    std::printf("[sphere] %.0f Hz Eq. 34 relative error at A/B/C\n"
                "  rigid     %.4f %.4f %.4f\n"
                "  release   %.4f %.4f %.4f\n"
                "  resistive %.4f %.4f %.4f\n",
                sample_rate, rigid[0], rigid[1], rigid[2], release[0], release[1], release[2], resistive[0], resistive[1], resistive[2]);
    RecordFigure({"Fig8SphereErrors", "Bilbao 2023, Figure 8", {"receiver_a", "receiver_b", "receiver_c"}, {"rigid", "release", "resistive"}, {rigid[0], rigid[1], rigid[2], release[0], release[1], release[2], resistive[0], resistive[1], resistive[2]}});
    if (gate) {
        AtMost("rigid sphere A error", rigid[0], .05);
        AtMost("rigid sphere B error", rigid[1], .14);
        AtMost("rigid sphere C error", rigid[2], .28);
        AtMost("release sphere A error", release[0], .05);
        AtMost("release sphere B error", release[1], .08);
        AtMost("release sphere C error", release[2], .5);
        AtMost("resistive sphere A error", resistive[0], .06);
        AtMost("resistive sphere B error", resistive[1], .1);
        AtMost("resistive sphere C error", resistive[2], .15);
    }
}

void Converge(bool gate) {
    std::vector<double> rates;
    if (gate) rates = {10000., 20000., 40000.};
    else
        for (int rate = 10000; rate <= 75000; rate += 5000) rates.push_back(rate);
    std::vector<double> time_steps, courants, errors;
    std::printf("[converge] rigid sphere receiver C\n");
    for (const double rate : rates) {
        const immersed::Grid grid = AnalyticGrid(2.2, rate);
        const double error = SphereCase(std::numeric_limits<double>::infinity(), 5, rate)[2];
        time_steps.push_back(grid.TimeStep);
        courants.push_back(grid.Courant);
        errors.push_back(error);
        std::printf("  %.0f Hz  %.6f\n", rate, error);
    }
    const double order = Slope(LogValues(time_steps), LogValues(errors));
    std::printf("  fitted order %.3f\n", order);
    std::vector<double> figure_values;
    figure_values.reserve(rates.size() * 4);
    for (size_t i = 0; i < rates.size(); ++i) {
        figure_values.push_back(rates[i]);
        figure_values.push_back(time_steps[i]);
        figure_values.push_back(courants[i]);
        figure_values.push_back(errors[i]);
    }
    RecordFigure({"Fig9Convergence", "Bilbao 2023, Figure 9", {"sample_rate_hz", "time_step_seconds", "courant", "receiver_c_error"}, {}, std::move(figure_values)});
    if (gate) {
        AtMost("10 kHz sphere error", errors[0], 1.0);
        AtMost("20 kHz sphere error", errors[1], .4);
        AtMost("40 kHz sphere error", errors[2], .16);
        AtMost("10 to 20 kHz error ratio", errors[1] / errors[0], .5);
        AtMost("20 to 40 kHz error ratio", errors[2] / errors[1], .5);
    }
}

void Interpolant(bool gate) {
    const double linear = SphereCase(std::numeric_limits<double>::infinity(), 1)[2];
    const double cubic = SphereCase(std::numeric_limits<double>::infinity(), 3)[2];
    const double quintic = SphereCase(std::numeric_limits<double>::infinity(), 5)[2];
    std::printf("[interp] rigid sphere receiver C error linear/cubic/quintic %.4f/%.4f/%.4f\n", linear, cubic, quintic);
    if (gate) {
        AtMost("cubic interpolant against linear", cubic / linear, .999);
        AtMost("quintic interpolant against cubic", quintic / cubic, .999);
    }
}

void Staircase(bool gate) {
    constexpr double SampleRate = 25000.;
    const immersed::Point source{0., 0., .4}, receiver{0., 0., -.3};
    const immersed::Grid grid = AnalyticGrid(2.2, SampleRate);
    const auto source_signal = GaussianSource(grid, .0045);
    const int steps = int(source_signal.size()) - 1;
    std::vector<uint8_t> solid(grid.NumCells());
    for (int z = 0; z < grid.Nz; ++z) {
        for (int y = 0; y < grid.Ny; ++y) {
            for (int x = 0; x < grid.Nx; ++x) {
                const double px = grid.Origin[0] + x * grid.H;
                const double py = grid.Origin[1] + y * grid.H;
                const double pz = grid.Origin[2] + z * grid.H;
                solid[size_t(x) + size_t(grid.Nx) * (size_t(y) + size_t(grid.Ny) * size_t(z))] = px * px + py * py + pz * pz <= .25 * .25;
            }
        }
    }
    immersed::Gpu gpu;
    gpu.Init(grid, {source}, {receiver}, steps, source_signal, {}, 5, solid);
    gpu.RunSteps(steps);
    const float *samples = gpu.Samples();
    const auto truth = ExactSphereTraces(source_signal, SampleRate, grid, .25, source, {receiver}, std::numeric_limits<double>::infinity()).front();
    const double staircase = RelativeError(samples, 1, truth, 1, size_t(steps));
    const double ib = SphereCase(std::numeric_limits<double>::infinity())[2];
    std::printf("[staircase] rigid sphere receiver C IB %.4f | solid cells %.4f | improvement %.2fx\n", ib, staircase, staircase / ib);
    RecordFigure({"Fig10Staircase", "Bilbao 2022, Figure 10", {"immersed_error", "solid_cell_error"}, {}, {ib, staircase}});
    if (gate) AtMost("IB error against staircase error", ib / staircase, .8);
}

struct BarrierResult {
    immersed::Grid Grid;
    std::vector<float> Samples;
    int Steps{0};
};

BarrierResult BarrierSimulation(double angle, const immersed::BoundaryImmittance &zv, const immersed::BoundaryImmittance &yp, double sample_rate = 25000.) {
    const immersed::Point source{0., 0., .3};
    const std::vector<immersed::Point> receivers{{0., 0., .2}, {.2, 0., .2}, {0., 0., -.3}};
    immersed::Grid grid = AnalyticGrid(2.2, sample_rate);
    const auto source_signal = GaussianSource(grid, .0045);
    const int steps = int(source_signal.size()) - 1;
    const immersed::Point u{std::cos(angle), 0., -std::sin(angle)};
    const auto patches = immersed::SquarePatches({0., 0., 0.}, u, {0., 1., 0.}, 1., grid.H * grid.H, zv, yp);
    immersed::Gpu gpu;
    gpu.Init(grid, {source}, receivers, steps, source_signal, patches);
    gpu.RunSteps(steps);
    const float *samples = gpu.Samples();
    return {std::move(grid), {samples, samples + size_t(steps) * receivers.size()}, steps};
}

void Barrier(bool gate) {
    const double sample_rate = gate ? 25000. : 50000.;
    const immersed::Point source{0., 0., .3}, receiver{0., 0., .2};
    std::array<double, 3> errors{}, leakage{};
    size_t leg = 0;
    for (const double angle : {0., std::numbers::pi / 6., std::numbers::pi / 4.}) {
        const auto result = BarrierSimulation(angle, LimitImmittance, ZeroImmittance, sample_rate);
        const immersed::Point normal{std::sin(angle), 0., std::cos(angle)};
        double signed_distance = 0.;
        for (int d = 0; d < 3; ++d) signed_distance += source[size_t(d)] * normal[size_t(d)];
        immersed::Point image = source;
        for (int d = 0; d < 3; ++d) image[size_t(d)] -= 2. * signed_distance * normal[size_t(d)];
        double signal = 0., error = 0., peak_near = 0., peak_far = 0.;
        const double direct = Distance(receiver, source), reflected = Distance(receiver, image);
        for (int step = 0; step < result.Steps; ++step) {
            const double time = (step + 1) * result.Grid.TimeStep;
            const double truth = result.Grid.Rho / (4. * std::numbers::pi) *
                (GaussianPulseDerivative(time - direct / result.Grid.C, 4000., 1e-3) / direct +
                 GaussianPulseDerivative(time - reflected / result.Grid.C, 4000., 1e-3) / reflected);
            const double got = result.Samples[size_t(step) * 3];
            signal += truth * truth;
            error += (got - truth) * (got - truth);
            peak_near = std::max(peak_near, std::abs(got));
            peak_far = std::max(peak_far, std::abs(double(result.Samples[size_t(step) * 3 + 2])));
        }
        errors[leg] = std::sqrt(error / signal);
        leakage[leg] = peak_far / peak_near;
        ++leg;
    }
    std::printf("[barrier] %.0f Hz rigid image error at 0/30/45 deg %.4f/%.4f/%.4f | prediffraction leakage %.3e/%.3e/%.3e\n", sample_rate, errors[0], errors[1], errors[2], leakage[0], leakage[1], leakage[2]);
    RecordFigure({"Fig5RigidBarrier", "Bilbao 2023, Figure 5", {"angle_degrees", "image_error", "prediffraction_leakage"}, {}, {0., errors[0], leakage[0], 30., errors[1], leakage[1], 45., errors[2], leakage[2]}});
    if (gate) {
        for (double const error : errors) AtMost("rigid barrier image error", error, .06);
        for (double const leak : leakage) AtMost("rigid barrier prediffraction leakage", leak, .03);
    }
}

void Transmit(bool gate) {
    const double sample_rate = gate ? 25000. : 50000.;
    const auto resistance = immersed::BoundaryImmittance::Finite(1.);
    const auto barrier = BarrierSimulation(std::numbers::pi / 6., resistance, ZeroImmittance, sample_rate);
    const auto free = BarrierSimulation(std::numbers::pi / 6., ZeroImmittance, ZeroImmittance, sample_rate);
    std::vector<double> barrier_trace(size_t(barrier.Steps)), free_trace(size_t(barrier.Steps));
    for (int step = 0; step < barrier.Steps; ++step) {
        const double basis = free.Samples[size_t(step) * 3 + 2];
        barrier_trace[size_t(step)] = barrier.Samples[size_t(step) * 3 + 2];
        free_trace[size_t(step)] = basis;
    }
    const auto barrier_spectrum = Dft(barrier_trace, false);
    const auto free_spectrum = Dft(free_trace, false);
    std::complex<double> spectral_cross{};
    double spectral_norm = 0.;
    for (size_t bin = 1; bin < free_spectrum.size() / 2; ++bin) {
        const double frequency = sample_rate * bin / free_spectrum.size();
        if (frequency > 2500.) break;
        spectral_cross += std::conj(free_spectrum[bin]) * barrier_spectrum[bin];
        spectral_norm += std::norm(free_spectrum[bin]);
    }
    const std::complex<double> coefficient = spectral_cross / spectral_norm;
    const double error = std::abs(coefficient - .5);
    std::printf("[transmit] %.0f Hz z=1 transfer below 2.5 kHz %.4f%+.4fi | error %.4f\n", sample_rate, coefficient.real(), coefficient.imag(), error);
    RecordFigure({"Fig6Transmission", "Bilbao 2023, Figure 6", {"target", "real", "imaginary", "magnitude", "complex_error"}, {}, {.5, coefficient.real(), coefficient.imag(), std::abs(coefficient), error}});
    if (gate) AtMost("low-frequency transmission coefficient error", error, .06);
}

void SphereExact(bool gate) {
    constexpr double C = 344., Rho = 1.18, Radius = .25;
    const immersed::Point source{0., 0., .4};
    const std::vector<immersed::Point> receivers{{0., 0., .3}, {.3, 0., 0.}, {0., 0., -.3}};
    double free_worst = 0.;
    for (const double frequency : {100., 500., 1000., 4000., 8000.}) {
        const double omega = 2. * std::numbers::pi * frequency;
        for (const auto &receiver : receivers) {
            const auto closed = immersed::FreeMonopoleTransfer(omega, C, Rho, Distance(receiver, source));
            const auto series = immersed::FreeSphereSeriesTransfer(omega, C, Rho, source, receiver);
            free_worst = std::max(free_worst, std::abs(series - closed) / std::abs(closed));
        }
    }
    double rigid_residual = 0., release_residual = 0.;
    for (const double frequency : {100., 1000., 4000., 8000.}) {
        const double omega = 2. * std::numbers::pi * frequency;
        rigid_residual = std::max(rigid_residual, immersed::SphereBoundaryResidual(omega, C, Radius, .4, std::numeric_limits<double>::infinity(), true));
        release_residual = std::max(release_residual, immersed::SphereBoundaryResidual(omega, C, Radius, .4, 0., false));
    }
    const auto rigid = immersed::SphereTransfer(2. * std::numbers::pi * 1000., C, Rho, Radius, source, receivers.back(), std::numeric_limits<double>::infinity());
    const auto resistive = immersed::SphereTransfer(2. * std::numbers::pi * 1000., C, Rho, Radius, source, receivers.back(), 1.);
    const auto release = immersed::SphereTransfer(2. * std::numbers::pi * 1000., C, Rho, Radius, source, receivers.back(), 0.);
    std::printf("[sphere-exact] free series relative %.3e | rigid derivative %.3e | release pressure %.3e | "
                "1 kHz magnitudes %.3e/%.3e/%.3e\n",
                free_worst, rigid_residual, release_residual, std::abs(rigid), std::abs(resistive), std::abs(release));
    if (gate) {
        AtMost("sphere free-series relative error", free_worst, 1e-10);
        AtMost("sphere rigid boundary residual", rigid_residual, 1e-12);
        AtMost("sphere release boundary residual", release_residual, 1e-12);
    }
}

void Roofline(bool) {
    const immersed::Grid grid = CubicGrid(128, .02, .99 / std::numbers::sqrt3);
    immersed::Gpu gpu;
    gpu.Init(grid, {}, {}, 1, {});
    const auto result = gpu.Roofline(1000);
    std::printf("[roofline] interior step %.3f ms | matched stream %.3f ms, %.1f GB/s | %.1f%% of ceiling\n", 1e3 * result.Step, 1e3 * result.Stream, result.Bytes / result.Stream / 1e9, 100. * result.Stream / result.Step);

    const auto measure = [](const char *name, const immersed::Grid &scene_grid,
                            const std::vector<immersed::Patch> &patches) {
        constexpr int Rounds = 6, Steps = 40;
        immersed::Gpu full, free;
        full.Init(scene_grid, {}, {}, Rounds * Steps, {}, patches);
        free.Init(scene_grid, {}, {}, Rounds * Steps, {});
        std::vector<double> full_samples, free_samples;
        for (int round = 0; round < Rounds; ++round) {
            const auto run_full = [&] { return full.RunTimedSteps(Steps); };
            const auto run_free = [&] { return free.RunTimedSteps(Steps); };
            const double first = round % 2 ? run_free() : run_full();
            const double second = round % 2 ? run_full() : run_free();
            if (round > 0) {
                full_samples.push_back(round % 2 ? second : first);
                free_samples.push_back(round % 2 ? first : second);
            }
        }
        std::ranges::sort(full_samples);
        std::ranges::sort(free_samples);
        const double full_step = full_samples[full_samples.size() / 2];
        const double free_step = free_samples[free_samples.size() / 2];
        std::printf("  %-7s free %.3f ms | immersed %.3f ms | IB overhead %.1f%%\n", name, 1e3 * free_step, 1e3 * full_step, 100. * (full_step / free_step - 1.));
    };

    const double h = 344. / (.5754 * 25000.);
    immersed::Grid sphere_grid;
    sphere_grid.Nx = sphere_grid.Ny = sphere_grid.Nz = 59;
    sphere_grid.H = h;
    sphere_grid.Origin = {-.5 * 58 * h, -.5 * 58 * h, -.5 * 58 * h};
    sphere_grid.Courant = .5754;
    sphere_grid.Finalize();
    const auto sphere = immersed::SpherePatches({0., 0., 0.}, .25, .03 * .03, LimitImmittance, ZeroImmittance);
    measure("sphere", sphere_grid, sphere);

    immersed::Grid barrier_grid;
    barrier_grid.Nx = barrier_grid.Ny = 67;
    barrier_grid.Nz = 59;
    barrier_grid.H = h;
    barrier_grid.Origin = {-.5 * 66 * h, -.5 * 66 * h, -.5 * 58 * h};
    barrier_grid.Courant = .5754;
    barrier_grid.Finalize();
    const auto impedance = RationalImmittance::Series(.1, .0005, 500.);
    const auto barrier = immersed::SquarePatches({0., 0., 0.}, {std::cos(std::numbers::pi / 6.), 0., -std::sin(std::numbers::pi / 6.)}, {0., 1., 0.}, 1., .04 * .04, immersed::BoundaryImmittance::Finite(impedance), immersed::BoundaryImmittance::Finite(impedance.Reciprocal()));
    measure("barrier", barrier_grid, barrier);
}

} // namespace

int main(int argc, char *const *argv) {
    struct Test {
        const char *Option;
        void (*Run)(bool);
        bool Default;
        bool InGate;
        bool Figure;
        bool Selected{false};
    };
    std::array tests{
        Test{"--filters", Filters, true, true, false},
        Test{"--one-d", OneD, true, true, false},
        Test{"--free", FreeField, true, true, false},
        Test{"--null", Null, true, true, false},
        Test{"--reference", Reference, true, true, false},
        Test{"--energy", Energy, false, true, true},
        Test{"--soak", Soak, false, true, false},
        Test{"--sphere", Sphere, false, true, true},
        Test{"--converge", Converge, false, true, true},
        Test{"--interp", Interpolant, false, true, false},
        Test{"--staircase", Staircase, false, true, true},
        Test{"--barrier", Barrier, false, true, true},
        Test{"--transmit", Transmit, false, true, true},
        Test{"--sphere-exact", SphereExact, true, true, false},
        Test{"--roofline", Roofline, false, false, false},
    };
    bool gate = false, any = false, test_selected = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--gate") {
            gate = any = true;
            continue;
        }
        if (arg == "--figures" && i + 1 < argc) {
            FigureDirectory = argv[++i];
            any = true;
            continue;
        }
        auto *const test = std::ranges::find_if(tests, [&](const Test &candidate) { return arg == candidate.Option; });
        if (test == tests.end()) {
            std::printf("unknown arg %s\n", arg.c_str());
            return 1;
        }
        test->Selected = any = test_selected = true;
    }
    if (!FigureDirectory.empty() && !test_selected)
        for (Test &test : tests) test.Selected |= test.Figure;
    for (const Test &test : tests)
        if (test.Selected || (!any && test.Default)) test.Run(false);
    if (gate) {
        for (const Test &test : tests)
            if (test.InGate) test.Run(true);
        std::printf("[gate] %s (%d failures)\n", Failures ? "FAIL" : "PASS", Failures);
    }
    if (!FigureDirectory.empty() && Failures == 0) WriteFigures();
    return Failures ? 1 : 0;
}
