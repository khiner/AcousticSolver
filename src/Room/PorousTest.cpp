#include "Porous.h"
#include "PorousScene.h"
#include "PorousSlab.h"
#include "RoomGpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
using json = nlohmann::json;
int Checks{}, Failures{};

void Check(bool ok, const char *message) {
    ++Checks;
    Failures += !ok;
    std::printf("  %s %s\n", ok ? "PASS" : "FAIL", message);
}

template<class F> void Reject(F &&f, const char *message) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) { rejected = true; }
    Check(rejected, message);
}

json SmallRoom() {
    return {
        {"room", {{"size", {.8, .64, .48}}, {"h", .04}, {"courant", .8}, {"seconds", .04}}},
        {"source", {{"position", {.12, .2, .2}}, {"peak_hz", 800.}}},
        {"receivers", {{.24, .24, .24}, {.48, .4, .24}}},
        {"porous_materials", {{{"flow_resistivity", 10000}, {"fit_hz", {20, 4000}}}}},
        {"porous_volumes", {{{"material", 0}, {"min", {.52, 0., 0.}}, {"max", {.64, .64, .48}}}}}
    };
}

template<class T> double RelativeError(const std::vector<double> &a, const T &b) {
    double error = 0., norm = 0.;
    for (size_t i = 0; i < a.size(); ++i) {
        error += (a[i] - b[i]) * (a[i] - b[i]);
        norm += a[i] * a[i];
    }
    return std::sqrt(error / std::max(norm, 1e-30));
}

void MaterialChecks() {
    using namespace Porous;
    const auto m = PrepareMaterial(10000);
    Check(m.MaxRelativeError <= .0032, "Allard-Champoux fitted material error <= 0.32%");
    const auto other = PrepareMaterial(5000);
    const auto mixed = PrepareLinkInverseDensity(m, other);
    double mixed_error = 0.;
    for (int i = 0; i < 65; ++i) {
        const Complex s(0., 2 * std::numbers::pi * 20 * std::pow(200., i / 64.));
        const Complex exact = 2. / (AllardChampoux(10000, s).Density + AllardChampoux(5000, s).Density);
        mixed_error = std::max(mixed_error, std::abs(mixed.Evaluate(s) - exact) / std::abs(exact));
    }
    Check(mixed_error <= .0032 && std::all_of(mixed.Poles.begin(), mixed.Poles.end(), [](const auto &p) { return p.Residue <= 0.; }), "mixed porous face fits averaged density with passive relaxation weights");
    const double frequency = 1000., x = 1.2 * frequency / 10000.;
    const Complex s(0., 2. * std::numbers::pi * frequency);
    const auto fluid = AllardChampoux(10000, s);
    const Complex density = 1.2 + std::sqrt(Complex(-.0364 / (x * x), -.1144 / x));
    const Complex z = std::sqrt(Complex(2.82 / (x * x), 24.9 / x));
    const Complex bulk = 101320. * (Complex(0., 29.64) + z) / (Complex(0., 21.17) + z);
    Check(std::abs(fluid.Density - density) < 1e-12 && std::abs(fluid.BulkModulus - bulk) < 1e-8, "material evaluation agrees with printed frequency-domain equations");
    const std::vector<Layer> air{{.3, 0.}}, panel{{.1, 10000.}, {.2, 0.}};
    const double angle = std::numbers::pi / 4.;
    const Complex phase = std::exp(Complex(0., -2. * std::numbers::pi * frequency * .3 * std::cos(angle) / 343.));
    const auto open = LayerResponse(air, frequency, angle, false);
    const auto rigid = LayerResponse(air, frequency, angle, true);
    Check(std::abs(open.Reflection) < 1e-12 && std::abs(open.Transmission - phase) < 1e-12 && std::abs(rigid.Reflection - phase * phase) < 1e-12, "analytical air layer amplitude and phase at oblique incidence");
    bool passive = true;
    for (double const f : {20., 100., 500., 1000., 4000.}) {
        const auto r = LayerResponse(panel, f, angle, true);
        const auto t = LayerResponse(std::span(panel).first(1), f, angle, false);
        passive &= r.Absorption >= 0. && r.Absorption <= 1. && t.Absorption >= 0. && t.Absorption <= 1.;
    }
    Check(passive, "backed and freestanding analytical slabs are passive");
    Reject([] { PrepareMaterial(-1.); }, "invalid flow resistivity is rejected");
}

void GeometryChecks(const RoomScene &scene) {
    Check(scene.Porous && !scene.Porous->Interior.empty() && !scene.Porous->Interface.empty() && !scene.Porous->Links.empty(), "native scene builds scalar and staggered porous regions");
    bool reciprocal = true, both_halves = false;
    for (const auto &row : scene.Porous->Interface) {
        both_halves |= Porous::UnfoldedY(scene, row.Node) > scene.Ny;
        for (int const l : row.Links) {
            if (!l) continue;
            const auto &link = scene.Porous->Links[size_t(std::abs(l) - 1)];
            reciprocal &= l > 0 ? row.Node == link.A : row.Node == link.B;
        }
    }
    Check(reciprocal && both_halves, "interface orientation includes both halves of folded FCC grid");
    auto config = SmallRoom();
    config["porous_volumes"].push_back(config["porous_volumes"][0]);
    Reject([&] { Porous::MakeRoom(config); }, "overlapping porous volumes rejected");
    config = SmallRoom();
    config["source"]["position"] = {.6, .2, .2};
    Reject([&] { Porous::MakeRoom(config); }, "source inside absorber rejected");

    const auto path = std::filesystem::temp_directory_path() / "acousticsolver-porous-box.obj";
    {
        std::ofstream file(path);
        for (int i = 0; i < 8; ++i)
            file << "v " << (i & 1 ? .64 : .52) << ' ' << (i & 2 ? .64 : 0.) << ' ' << (i & 4 ? .48 : 0.) << '\n';
        constexpr int faces[][4]{{0, 2, 6, 4}, {1, 5, 7, 3}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 6, 7, 5}};
        for (const auto &f : faces) {
            file << "f " << f[0] + 1 << ' ' << f[2] + 1 << ' ' << f[1] + 1 << '\n';
            file << "f " << f[0] + 1 << ' ' << f[3] + 1 << ' ' << f[2] + 1 << '\n';
        }
    }
    config = SmallRoom();
    config["porous_volumes"] = {{{"material", 0}, {"mesh", path.string()}}};
    const auto mesh = Porous::MakeRoom(config);
    bool same = mesh.Porous->Interior.size() == scene.Porous->Interior.size() &&
        mesh.Porous->Interface.size() == scene.Porous->Interface.size();
    if (same) {
        for (size_t i = 0; i < mesh.Porous->Interior.size(); ++i)
            same &= mesh.Porous->Interior[i].Node == scene.Porous->Interior[i].Node;
        for (size_t i = 0; i < mesh.Porous->Interface.size(); ++i)
            same &= mesh.Porous->Interface[i].Node == scene.Porous->Interface[i].Node;
    }
    Check(same, "closed OBJ volume gives the same node classification as its box");
    std::filesystem::remove(path);
}

void SolverChecks(const RoomScene &scene, bool cpu_only) {
    auto transparent = scene;
    std::vector<int> labels(size_t(scene.NumNodes()), -1);
    for (const auto &n : scene.Porous->Interior) labels[n.Node] = 0;
    for (const auto &n : scene.Porous->Interface)
        if (n.Material) labels[n.Node] = 0;
    transparent.Porous = Porous::Prepare(scene, labels, {Porous::AirMaterial(scene.C, scene.Rho)});
    auto air = scene;
    air.Porous.reset();
    Porous::Cpu base, split, reference;
    base.Init(air);
    split.Init(transparent);
    reference.Init(scene);
    base.RunSteps(scene.Nt);
    split.RunSteps(scene.Nt);
    reference.RunSteps(scene.Nt);
    const double transparency = RelativeError(base.Output(), split.Output());
    std::printf("  transparent FP64 relative error %.3g\n", transparency);
    Check(transparency < 1e-10, "transparent staggered interface recovers scalar FCC");
    Check(std::all_of(reference.Pressure().begin(), reference.Pressure().end(), [](double x) { return std::isfinite(x); }), "porous FP64 reference remains finite after excitation stops");
    const auto norm2 = [](const auto &v) { return std::inner_product(v.begin(), v.end(), v.begin(), 0.); };
    const double treated = norm2(reference.Pressure()), untreated = norm2(base.Pressure());
    std::printf("  final pressure norm squared treated/air %.5f\n", treated / untreated);
    Check(treated < untreated * .8, "absorber reduces late pressure in rigid room");
    if (cpu_only) return;
    RoomGpu gpu;
    gpu.Init(scene);
    gpu.RunSteps(scene.Nt / 2);
    gpu.RunSteps(scene.Nt - scene.Nt / 2);
    const float *samples = gpu.Samples();
    const double error = RelativeError(reference.Output(), samples);
    std::printf("  Metal/FP64 receiver relative error %.3g\n", error);
    Check(error < 2e-4, "Metal pressure agrees with FP64 reference");
    RoomGpu whole;
    whole.Init(scene);
    whole.RunSteps(scene.Nt);
    Check(std::equal(samples, samples + reference.Output().size(), whole.Samples()), "split and whole Metal runs agree exactly");
    for (const auto *variant : {&transparent, &air}) {
        whole.Init(*variant);
        whole.RunSteps(scene.Nt);
        Check(RelativeError(base.Output(), whole.Samples()) < 2e-4, variant == &air ? "native untreated Metal room agrees with scalar FP64" : "Metal transparent interface agrees with scalar FP64");
    }
}
void SlabChecks(bool cpu_only, bool band) {
    json report = json::array();
    const std::vector<double> frequencies = band ? std::vector<double>{20, 40, 80, 125, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000} : std::vector<double>{343 / (std::numbers::sqrt2 * .24)};
    const auto *const destination = band ? "porous/band.json" : "porous/slabs.json";
    std::filesystem::create_directories("porous");
    for (double frequency : frequencies) {
        const double h = !band ? .02 : frequency <= 100 ? .05 :
            frequency <= 500                            ? .02 :
            frequency <= 1500                           ? .01 :
            frequency <= 3000                           ? .005 :
                                                          .0025;
        const double width = band ? std::round(343 / (std::numbers::sqrt2 * frequency * h)) * h : .24;
        for (bool rigid : {true, false}) {
            for (bool oblique : {false, true}) {
                std::printf("  slab %.1f Hz, %s, %s, h %.5f / %.5f\n", frequency, rigid ? "backed" : "freestanding", oblique ? "oblique" : "normal", h, h / 2);
                std::fflush(stdout);
                const auto run = [&](double spacing) { return Porous::RunSlab(spacing, oblique, rigid, !cpu_only, frequency, width, band ? std::max(96., .2 * frequency) : 96); };
                const auto coarse = run(h), fine = run(h / 2);
                std::printf("  R error %.5g -> %.5g, T %.5g -> %.5g; stationarity %.3g, residual %.3g, condition %.3g\n", coarse.ReflectionError, fine.ReflectionError, coarse.TransmissionError, fine.TransmissionError, fine.Stationarity, fine.SpatialResidual, fine.Condition);
                Check(fine.ReflectionError < .03 && fine.ReflectionError < coarse.ReflectionError && (rigid || (fine.TransmissionError < .03 && fine.TransmissionError < coarse.TransmissionError)), "slab scattering agrees with continuum and improves under grid refinement");
                Check(std::max(coarse.Stationarity, fine.Stationarity) < .001 && std::max(coarse.SpatialResidual, fine.SpatialResidual) < .001 && std::max(coarse.Condition, fine.Condition) < (band ? 100 : 10), "slab phasors are stationary and wave decomposition is well conditioned");
                for (const auto &r : {coarse, fine}) {
                    const auto impedance = [](Porous::Complex reflection, double angle) {
                        return (1. + reflection) / ((1. - reflection) * std::cos(angle));
                    };
                    const auto z = impedance(r.Reflection, r.Angle), expected_z = impedance(r.ExpectedReflection, r.Angle);
                    const double z_error = std::abs(z - expected_z) / std::max(1., std::abs(expected_z));
                    const double absorption = 1 - std::norm(r.Reflection) - std::norm(r.Transmission);
                    const double expected_absorption = 1 - std::norm(r.ExpectedReflection) - std::norm(r.ExpectedTransmission);
                    if (band && r.Spacing == fine.Spacing)
                        Check(z_error < .05 && std::abs(absorption - expected_absorption) < .05, "fine-grid normalized impedance and absorption agree with continuum");
                    report.push_back({{"rigid_backing", rigid}, {"angle_degrees", r.Angle * 180 / std::numbers::pi}, {"h", r.Spacing}, {"frequency_hz", r.Frequency}, {"thickness", r.Thickness}, {"air_gap", r.AirGap}, {"nodes", r.Nodes}, {"steps", r.Steps}, {"seconds", r.Seconds}, {"reflection", {r.Reflection.real(), r.Reflection.imag()}}, {"expected_reflection", {r.ExpectedReflection.real(), r.ExpectedReflection.imag()}}, {"transmission", {r.Transmission.real(), r.Transmission.imag()}}, {"expected_transmission", {r.ExpectedTransmission.real(), r.ExpectedTransmission.imag()}}, {"reflection_error", r.ReflectionError}, {"transmission_error", r.TransmissionError}, {"normalized_impedance", {z.real(), z.imag()}}, {"expected_normalized_impedance", {expected_z.real(), expected_z.imag()}}, {"impedance_error", z_error}, {"absorption", absorption}, {"expected_absorption", expected_absorption}, {"stationarity", r.Stationarity}, {"spatial_residual", r.SpatialResidual}, {"condition", r.Condition}});
                }
                std::ofstream(destination) << json{{"backend", cpu_only ? "FP64" : "Metal"}, {"cases", report}}.dump(2) << '\n';
            }
        }
    }
}
} // namespace

int main(int argc, char *const *argv) {
    try {
        bool cpu_only = false, slabs = false, band = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--cpu") cpu_only = true;
            else if (arg == "--slabs") slabs = true;
            else if (arg == "--band") band = true;
            else throw std::runtime_error("Usage: PorousTest [--cpu] [--slabs] [--band]");
        }
        MaterialChecks();
        const auto scene = Porous::MakeRoom(SmallRoom());
        GeometryChecks(scene);
        SolverChecks(scene, cpu_only);
        if (slabs || band) SlabChecks(cpu_only, band);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "PorousTest: %s\n", e.what());
        ++Failures;
    }
    std::printf("%d checks, %d failures\n", Checks, Failures);
    return Failures ? 1 : 0;
}
