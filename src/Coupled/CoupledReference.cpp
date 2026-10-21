#include "CoupledBoundary.h"

#include "AirboxIndexed.h"
#include "Embedding.h"
#include "GlobalSettings.h"
#include "InputStrike.h"
#include "SettingsManager.h"
#undef real

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace coupled {
namespace {
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
void Require(bool value, const char *message) {
    if (!value) throw std::invalid_argument(message);
}
double Number(const json &j, const char *key, double fallback, double low, double high) {
    const double value = j.value(key, fallback);
    Require(std::isfinite(value) && value >= low && value <= high, key);
    return value;
}
std::array<double, 3> Point(const json &j) {
    auto result = j.get<std::array<double, 3>>();
    for (double const value : result) Require(std::isfinite(value), "Nonfinite coordinate");
    return result;
}
Matrix Copy(Tables &t, const CSRmatrix *m) {
    return t.Store(m->nrow, m->rowStart, m->colIndex, m->values);
}
class Air final : public AirboxIndexed {
public:
    using AirboxIndexed::AirboxIndexed;
    void Export(Tables &t, bool reflecting) {
        reflection = reflecting;
        t.Nx = nx;
        t.Ny = ny;
        t.Nz = nz;
        t.Q = Q;
        t.Gamma = gammabar;
        t.Lambda = lambda;
        t.AirIndex.assign(index, index + ss);
        t.AirCoefficients.assign(coeffs, coeffs + 4096);
    }
};
class Panel final : public PlateEmbedded {
public:
    using PlateEmbedded::PlateEmbedded;
    void Export(Tables &t, Plate &p) {
        Require(ss <= 4096, "Panel exceeds 4096 grid points");
        p.Size = ss;
        p.Stride = bandSize;
        p.Iterations = iterinv;
        p.Nonlinear = !linear;
        p.WScale = 2 * CF;
        p.FScale = BL;
        p.InvH2 = 1 / (h * h);
        p.Alpha = alpha;
        p.H = h;
        p.B = Copy(t, B);
        p.C = Copy(t, C);
        auto *ij = CSR_matrix_multiply(IMat, JMat);
        CSR_scalar_mult(ij, cplfac);
        p.IJ = Copy(t, ij);
        CSR_free(ij);
        if (linear) return;
        p.K = Copy(t, DDDD_F);
        const matrix_3x3_t *derivatives[]{Dyyw3, Dxxw3, DLMM3, DLMP3, DLPM3, DLPP3};
        p.Derivatives = uint32_t(t.Values.size());
        for (const auto *matrix : derivatives) t.Append(matrix->values, 9 * size_t(ss));
        std::vector<double> mask(9 * size_t(ss));
        for (int row = 0; row < ss; ++row) {
            if (isCircular && !std::binary_search(TruePhi, TruePhi + TruePhiSize, row)) continue;
            for (int k = LOPW->rowStart[row]; k < LOPW->rowStart[row + 1]; ++k) {
                const int offset = LOPW->colIndex[k] - row;
                for (int slot = 0; slot < 9; ++slot)
                    if (offset == (slot / 3 - 1) * bandSize + slot % 3 - 1) mask[9 * row + slot] = 1;
            }
        }
        p.Mask = t.Append(mask.data(), mask.size());
        Require(t.Values.size() + size_t(ss) * ss <= 64 * 1024 * 1024, "Operator storage exceeds 512 MiB");
        // Apply the upstream Cholesky preconditioner to basis vectors once, for parallel GPU matvecs.
        std::vector<double> inverse(size_t(ss) * ss), basis(ss), column(ss);
        for (int col = 0; col < ss; ++col) {
            basis[col] = 1;
            pcg->preconditioner(pcg->precondData, basis.data(), column.data(), ss);
            for (int row = 0; row < ss; ++row) {
                Require(std::isfinite(column[row]), "Nonfinite NESS preconditioner");
                inverse[size_t(row) * ss + col] = column[row];
            }
            basis[col] = 0;
        }
        p.Inverse = t.Append(inverse.data(), inverse.size());
    }
};
class Interface final : public Embedding {
public:
    using Embedding::Embedding;
    void Export(Tables &t, Plate &p) {
        p.I = Copy(t, BfIMat);
        p.J = Copy(t, JMat);
        p.Bf = Bf;
        p.LowerSlice = pd;
        p.AirMask.assign(true_Psi, true_Psi + Diff_size);
    }
};
} // namespace

uint32_t Tables::Append(const double *values, size_t count) {
    Require(Values.size() + count <= 64 * 1024 * 1024, "Operator storage exceeds 512 MiB");
    const auto offset = uint32_t(Values.size());
    if (count) Values.insert(Values.end(), values, values + count);
    return offset;
}
Matrix Tables::Store(uint32_t rows, const int *row, const int *column, const double *values) {
    Matrix result{rows, uint32_t(Indices.size()), 0, 0};
    Indices.insert(Indices.end(), row, row + rows + 1);
    result.Column = uint32_t(Indices.size());
    if (row[rows]) Indices.insert(Indices.end(), column, column + row[rows]);
    result.Value = Append(values, row[rows]);
    return result;
}

struct Reference::Impl {
    std::unique_ptr<Air> Field;
    std::vector<std::unique_ptr<Panel>> Panels;
    std::vector<std::unique_ptr<Interface>> Interfaces;
};
Reference::~Reference() = default;
Reference::Reference(const json &config) : State(std::make_unique<Impl>()) {
    auto &t = Data;
    t.SampleRate = Number(config, "sample_rate", 22050, 1000, 192000);
    Require(t.SampleRate == std::floor(t.SampleRate), "Sample rate must be an integer");
    const double seconds = Number(config, "seconds", .25, 1 / t.SampleRate, 3600);
    Require(seconds * t.SampleRate < INT32_MAX, "Too many timesteps");
    t.Steps = uint32_t(std::floor(seconds * t.SampleRate));
    t.Tolerance = Number(config, "solver_tolerance", 1e-9, 1e-14, 1e-3);
    t.MaxIterations = uint32_t(Number(config, "solver_iterations", 64, 1, 512));
    const auto dimensions = Point(config.at("airbox"));
    const double speed = Number(config, "sound_speed", 340, 100, 2000);
    const double density = Number(config, "air_density", 1.21, .01, 2000);
    t.AirDensity = density;
    t.Lossless = config.value("lossless", false);
    t.PressureScale = .5 * density * t.SampleRate;
    const double minQ = std::numbers::sqrt3 * speed / t.SampleRate;
    double cells = 1;
    for (double const size : dimensions) {
        Require(size / minQ >= 6, "Airbox must span at least six grid intervals per axis");
        cells *= 1 + std::floor(size / minQ);
    }
    Require(cells <= 4000000, "Airbox exceeds four million cells; reduce sample rate or dimensions");
    t.Reflecting = config.value("reflecting", false);
    auto *global = GlobalSettings::getInstance();
    global->setSampleRate(t.SampleRate);
    global->setDuration(seconds);
    global->setEnergyOn(false);
    global->setGpuEnabled(false);
    global->setImpulse(false);
    global->setPcgTolerance(t.Tolerance);
    global->setPcgMaxIterations(t.MaxIterations);
    global->setSymmetricSolve(false);
    global->setLossMode(t.Lossless ? 0 : 1);
    State->Field = std::make_unique<Air>("coupled_air", dimensions[0], dimensions[1], dimensions[2], speed, density);
    auto &air = *State->Field;
    air.Export(t, t.Reflecting);
    const auto &panels = config.at("panels");
    Require(panels.is_array() && !panels.empty() && panels.size() <= 32, "Expected 1 to 32 panels");
    static uint64_t serial = 0;
    for (const auto &item : panels) {
        const bool circular = item.contains("radius");
        const bool nonlinear = item.value("nonlinear", true);
        const std::string boundary = item.value("boundary", "ness");
        Require(boundary == "ness" || boundary == "clamped" || boundary == "simply_supported" || boundary == "free", "Unknown plate boundary");
        Require(!circular || boundary == "ness", "Selectable edge conditions require a rectangular plate");
        const double radius = Number(item, "radius", .1, 1e-4, 10);
        auto size = circular ? std::array<double, 2>{2 * radius, 2 * radius} : item.at("size").get<std::array<double, 2>>();
        auto center = Point(item.at("center"));
        const double rho = Number(item, "density", 7800, 1, 30000);
        const double young = Number(item, "young_modulus", 2e11, 1, 1e13);
        const double thickness = Number(item, "thickness", .001, 1e-6, .1);
        const double nu = Number(item, "poisson_ratio", .3, 0, .49);
        const double tension = Number(item, "tension", 0, 0, 1e8);
        Require(boundary != "free" || tension == 0, "A free plate edge cannot support prescribed uniform tension");
        const double t60 = Number(item, "t60", 4, 1e-3, 10000);
        const double sig1 = Number(item, "sigma1", .001, 0, 10);
        const double dt = 1 / t.SampleRate;
        const double rigidity = young * thickness * thickness / (12 * rho * (1 - nu * nu));
        const double c2 = tension / (rho * thickness);
        const double s = c2 * dt * dt + 4 * sig1 * dt;
        const double h = std::sqrt(s + std::sqrt(s * s + 16 * rigidity * dt * dt));
        double panelCells = 1;
        for (int axis = 0; axis < 2; ++axis) {
            Require(std::isfinite(size[axis]) && size[axis] / h >= 5, "Panel must span at least five grid intervals");
            Require(std::abs(center[axis]) + size[axis] / 2 + 2 * t.Q < dimensions[axis] / 2, "Panel needs two air cells of lateral clearance");
            panelCells *= 1 + std::floor(size[axis] / h);
        }
        Require(panelCells <= 4096, "Panel exceeds 4096 grid points; reduce sample rate or size");
        const int lower = int(std::floor((center[2] + dimensions[2] / 2) / t.Q)) - 1;
        Require(lower >= 1 && lower + 3 < int(t.Nz), "Panel needs two air slices above and below");
        for (const auto &p : t.Plates) Require(std::abs(int(p.LowerSlice) - lower) >= 4, "Panels must be separated by at least four air slices");
        const std::string name = "coupled_panel_" + std::to_string(++serial);
        auto *settings = SettingsManager::getInstance();
        settings->putSettingBool(name, "linear", !nonlinear || boundary != "ness");
        settings->putSettingInt(name, "iterinv", int(Number(item, "coupling_iterations", circular ? 8 : 5, 1, 64)));
        auto panel = std::make_unique<Panel>(name, nu, rho, young, thickness, tension, circular ? radius : size[0], size[1], t60, sig1, center[0], center[1], center[2], circular || tension > 0, circular);
        auto interface = std::make_unique<Interface>(&air, panel.get());
        Plate p;
        panel->Export(t, p);
        interface->Export(t, p);
        if (boundary != "ness") PrepareBoundary(t, p, item);
        t.Plates.push_back(std::move(p));
        State->Panels.push_back(std::move(panel));
        State->Interfaces.push_back(std::move(interface));
    }
    if (config.contains("drum_shell")) {
        const auto &shell = config.at("drum_shell");
        const auto bottom = shell.at("bottom").get<size_t>(), top = shell.at("top").get<size_t>();
        Require(bottom < panels.size() && top < panels.size() && bottom != top, "Invalid drum shell panel indices");
        Require(panels[bottom].contains("radius") && panels[top].contains("radius"), "Drum shell requires circular membranes");
        auto a = Point(panels[bottom].at("center")), b = Point(panels[top].at("center"));
        Require(a[0] == 0 && a[1] == 0 && b[0] == 0 && b[1] == 0 && a[2] < b[2], "Drum membranes must be centered and ordered bottom to top");
        const double radius = panels[bottom].at("radius");
        Require(radius == panels[top].at("radius").get<double>(), "Drum membranes must have equal radii");
        air.addDrumShell(int(t.Plates[bottom].LowerSlice) + 1, int(t.Plates[top].LowerSlice), radius);
        air.Export(t, t.Reflecting);
    }
    const auto strikes = config.value("strikes", json::array());
    Require(strikes.is_array() && strikes.size() <= 128, "Expected at most 128 strikes");
    for (const auto &strike : strikes) {
        const auto index = strike.at("panel").get<size_t>();
        Require(index < t.Plates.size(), "Invalid strike panel");
        const auto position = strike.at("position").get<std::array<double, 2>>();
        for (double const x : position) Require(std::isfinite(x) && x > 0 && x < 1, "Strike position must be inside (0,1)");
        auto &panel = *State->Panels[index];
        const double start = Number(strike, "start", 0, 0, seconds);
        const double duration = Number(strike, "duration", .002, 1 / t.SampleRate, seconds);
        const double amplitude = Number(strike, "force", 1, -1e8, 1e8);
        const auto &p = t.Plates[index];
        const int node = p.Adjoint ? int(std::round(position[0] * (p.Size / p.Stride - 1))) * p.Stride + int(std::round(position[1] * (p.Stride - 1))) : panel.getIndexf(position[0], position[1]);
        Require(!p.Adjoint || t.Values[p.MassInverse + node] != 0, "Strike lies on a fixed edge");
        if (panel.getCircular()) Require(std::binary_search(panel.getTruePhi(), panel.getTruePhi() + panel.getTruePhiSize(), node), "Strike lies outside circular membrane");
        t.Strikes.push_back({uint32_t(index), uint32_t(node), uint32_t(std::floor(start * t.SampleRate)), uint32_t(std::floor(duration * t.SampleRate)), amplitude});
        panel.addInput(new InputStrike(&panel, position[0], position[1], 0, start, duration, amplitude, 0, 0));
    }
    const auto &receivers = config.at("receivers");
    Require(receivers.is_array() && !receivers.empty() && receivers.size() <= 64, "Expected 1 to 64 receivers");
    std::vector<std::array<double, 3>> resolved;
    for (const auto &receiver : receivers) {
        const auto point = Point(receiver);
        std::array<int, 3> coord;
        for (int axis = 0; axis < 3; ++axis) {
            Require(std::abs(point[axis]) < dimensions[axis] / 2, "Receiver outside airbox");
            coord[axis] = int(std::round((point[axis] + dimensions[axis] / 2) / t.Q));
        }
        Require(coord[0] < int(t.Nx) && coord[1] < int(t.Ny) && coord[2] < int(t.Nz), "Receiver beyond rounded grid");
        t.Receivers.push_back({-1, uint32_t(air.getIndex(coord[0], coord[1], coord[2]))});
        resolved.push_back({coord[0] * t.Q - dimensions[0] / 2, coord[1] * t.Q - dimensions[1] / 2, coord[2] * t.Q - dimensions[2] / 2});
    }
    for (size_t i = 0; i < t.Plates.size(); ++i) {
        const auto &p = t.Plates[i];
        const uint32_t pickup = p.Adjoint ? ((p.Size / p.Stride) / 2) * p.Stride + p.Stride / 2 : State->Panels[i]->getIndexf(.5, .5);
        t.Receivers.push_back({int(i), pickup});
    }
    Require(size_t(t.Steps) * (t.Receivers.size() + t.Strikes.size()) <= 32 * 1024 * 1024, "Trace and excitation storage exceeds 256 MiB; shorten the render");
    t.Metadata = {{"config", config}, {"air_grid", {t.Nx, t.Ny, t.Nz}}, {"air_spacing", t.Q}, {"sample_rate", t.SampleRate}, {"steps", t.Steps}, {"reference_commit", ACOUSTIC_NESS_COMMIT}, {"receiver_fields", "air pressure in Pa followed by panel center displacements in metres"}, {"pressure_scale", t.PressureScale}, {"resolved_receivers", resolved}, {"pressure_conversion", "rho * sample_rate / 2 times the centered potential difference; NESS OutputPressure omits rho/2"}};
    for (size_t i = 0; i < t.Plates.size(); ++i) {
        const auto &p = t.Plates[i];
        t.Metadata["panels"].push_back({{"points", p.Size}, {"spacing", p.H}, {"lower_air_slice", p.LowerSlice}, {"boundary", panels[i].value("boundary", "ness")}, {"grid", {p.Size / p.Stride, p.Stride}}, {"in_plane_boundary", p.Adjoint ? "traction-free; Airy value and normal derivative zero" : "NESS"}});
    }
    t.Metadata["reference_backend"] = std::ranges::any_of(t.Plates, &Plate::Adjoint) ? "FP64 sparse direct plate solve and NESS air" : "NESS FP64";
}

Trace Reference::Run() {
    Trace trace;
    auto &t = Data;
    auto &air = *State->Field;
    if (std::ranges::any_of(t.Plates, &Plate::Adjoint)) return RunBoundaryReference(t, air);
    const auto start = Clock::now();
    trace.Samples.reserve(size_t(t.Steps) * t.Receivers.size());
    for (uint32_t step = 0; step < t.Steps; ++step) {
        air.runTimestep(step);
        for (auto &panel : State->Panels) panel->runTimestep(step);
        for (auto &interface : State->Interfaces) interface->runTimestep(step);
        for (const auto &r : t.Receivers) {
            double const sample = r.Plate < 0 ? t.PressureScale * (air.getU()[r.Index] - air.getU2()[r.Index]) : State->Panels[r.Plate]->getU()[r.Index];
            if (!std::isfinite(sample)) throw std::runtime_error("NESS reference became nonfinite");
            trace.Samples.push_back(sample);
        }
        air.swapBuffers(step);
        for (auto &panel : State->Panels) panel->swapBuffers(step);
    }
    trace.Seconds = std::chrono::duration<double>(Clock::now() - start).count();
    trace.Air.assign(air.getU1(), air.getU1() + air.getStateSize());
    for (auto &panel : State->Panels) trace.Displacements.insert(trace.Displacements.end(), panel->getU1(), panel->getU1() + panel->getStateSize());
    return trace;
}
} // namespace coupled
