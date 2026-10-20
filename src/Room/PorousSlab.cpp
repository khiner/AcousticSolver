#include "PorousSlab.h"
#include "Porous.h"
#include "RoomGpu.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace Porous {
namespace {
constexpr double Pi = std::numbers::pi;

std::vector<Complex> Phasors(const RoomScene &scene, const std::vector<double> &samples, double frequency, double end_cycles) {
    const double begin = (end_cycles - 16) / frequency, end = end_cycles / frequency;
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(3, scene.NumOutputs());
    for (int n = std::max(0, int(std::ceil(begin / scene.Ts)));
         n < std::min(scene.Nt, int(std::floor(end / scene.Ts))); ++n) {
        double t = n * scene.Ts, omega_t = 2 * Pi * frequency * t;
        double const weight = .5 - .5 * std::cos(2 * Pi * (t - begin) / (end - begin));
        Eigen::Vector3d basis(std::cos(omega_t), -std::sin(omega_t), 1.);
        normal += weight * basis * basis.transpose();
        for (int j = 0; j < scene.NumOutputs(); ++j)
            rhs.col(j) += weight * basis * samples[size_t(n) * scene.NumOutputs() + j];
    }
    const Eigen::MatrixXd fit = normal.ldlt().solve(rhs);
    std::vector<Complex> result(scene.NumOutputs());
    for (int j = 0; j < scene.NumOutputs(); ++j) result[j] = {fit(0, j), fit(1, j)};
    return result;
}

struct Waves {
    Complex Forward, Backward;
    double Residual, Condition;
};

Waves Decompose(std::span<const double> positions, std::span<const Complex> pressure, double kx, double reference) {
    Eigen::MatrixXcd basis(positions.size(), 2);
    Eigen::VectorXcd data(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        basis(i, 0) = std::exp(Complex(0, -kx * (positions[i] - reference)));
        basis(i, 1) = std::conj(basis(i, 0));
        data[i] = pressure[i];
    }
    const Eigen::Vector2cd amplitude = basis.colPivHouseholderQr().solve(data);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2cd> const eig(basis.adjoint() * basis);
    return {amplitude[0], amplitude[1], (basis * amplitude - data).norm() / std::max(data.norm(), 1e-30), std::sqrt(eig.eigenvalues()[1] / eig.eigenvalues()[0])};
}
} // namespace

SlabResult RunSlab(double h, bool oblique, bool rigid, bool gpu, double frequency, double width, double cycles) {
    if (!(h > 0 && h <= .05) || std::abs(std::round(.1 / h) * h - .1) > 1e-12 ||
        !(width >= 3 * h) || std::abs(std::round(width / h) * h - width) > 1e-10 || cycles < 48)
        throw std::invalid_argument("Slab spacing must divide thickness and width; at least 48 cycles required");
    RoomScene scene;
    scene.Fcc = true;
    scene.C = 343;
    scene.Rho = 1.2;
    scene.H = h;
    scene.L = .8;
    scene.L2 = scene.L * scene.L;
    scene.Ts = scene.L * h / scene.C;
    scene.Srate = 1 / scene.Ts;
    scene.Nx = int(std::lround((rigid ? 1.2 : 1.8) / h)) + 3;
    scene.Ny = 6; // Uniform transverse y mode; mirrored boundaries make this exact.
    scene.Nz = oblique ? int(std::lround(width / h)) + 3 : 6;
    const double sine_angle = oblique ? scene.C / (2 * width * frequency) : 0;
    if (!(frequency > 0 && frequency < .5 * scene.Srate) || sine_angle >= 1)
        throw std::invalid_argument("Slab frequency must support a propagating transverse mode");
    const double angle = std::asin(sine_angle);
    scene.Nt = int(std::ceil(cycles / (frequency * scene.Ts)));
    const int first = int(std::lround(.9 / h)) + 1;
    const int end = first + int(std::lround(.1 / h));
    const double front = (first - 1.5) * h, back = (end - 1.5) * h;
    const double gap = (scene.Nx - 3) * h - back;
    std::vector<int> labels(size_t(scene.NumNodes()), -1);
    for (int i = 0; i < scene.NumNodes(); ++i) {
        if (!IsInterior(scene, i)) continue;
        int x = i / (scene.Ny * scene.Nz), y = i / scene.Nz % scene.Ny, z = i % scene.Nz;
        if (x == 1 || x == scene.Nx - 2 || y == 1 || z == 1 || z == scene.Nz - 2) {
            scene.BnIxyz.push_back(i);
            scene.AdjBn.push_back(4095);
            scene.MatBn.push_back(-1);
            scene.SafBn.push_back(1);
        }
        if (x >= first && x < end) labels[i] = 0;
        if (x != int(std::lround(.1 / h)) + 1) continue;
        scene.InIxyz.push_back(i);
        const double transverse = oblique ? std::cos(Pi * (z - 1) * h / width) : 1;
        for (int n = 0; n < scene.Nt; ++n) {
            const double cycles = n * scene.Ts * frequency;
            const double ramp = cycles < 8 ? .5 - .5 * std::cos(Pi * cycles / 8) : 1.;
            scene.InSigs.push_back(.01 * transverse * ramp * std::sin(2 * Pi * cycles));
        }
    }
    std::vector<double> positions;
    const auto probes = [&](double start, double finish) {
        // One sample per x plane at z=0; the transverse mode equals one here.
        for (int x = int(std::lround(start / h)) + 1; x <= int(std::lround(finish / h)) + 1; ++x) {
            int z = 1, y = ((x + z + 1) & 1) ? 2 : 1;
            scene.OutIxyz.push_back((x * scene.Ny + y) * scene.Nz + z);
            scene.OutAlpha.push_back(1);
            positions.push_back((x - 1) * h);
        }
    };
    probes(.3, .64);
    const size_t left_count = positions.size();
    if (!rigid) probes(1.24, 1.58);
    scene.NumReceivers = scene.NumOutputs();
    scene.CornersPerReceiver = 1;
    scene.Porous = Prepare(scene, labels, {PrepareMaterial(10000)});

    const auto start = std::chrono::steady_clock::now();
    std::vector<double> output;
    if (gpu) {
        RoomGpu solver;
        solver.Init(scene);
        solver.RunSteps(scene.Nt);
        const float *samples = solver.Samples();
        output.assign(samples, samples + size_t(scene.Nt) * scene.NumOutputs());
    } else {
        Cpu solver;
        solver.Init(scene);
        solver.RunSteps(scene.Nt);
        output = solver.Output();
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto pressure = Phasors(scene, output, frequency, cycles);
    const auto previous = Phasors(scene, output, frequency, cycles - 16);
    double difference = 0, norm = 0;
    for (size_t i = 0; i < pressure.size(); ++i) {
        difference += std::norm(pressure[i] - previous[i]);
        norm += std::norm(pressure[i]);
    }
    // Exact production FCC dispersion with ky=0 and the chosen Neumann kz.
    const double transverse_cos = std::cos(oblique ? Pi * h / width : 0.);
    const double temporal_sine = std::sin(Pi * frequency * scene.Ts);
    const double kx = std::acos((3 - transverse_cos - 4 * temporal_sine * temporal_sine / scene.L2) / (1 + transverse_cos)) / h;
    const auto left = Decompose(std::span(positions).first(left_count), std::span(pressure).first(left_count), kx, front);
    Complex reflection = left.Backward / left.Forward, transmission = 0;
    double residual = left.Residual, condition = left.Condition;
    if (!rigid) {
        const auto right = Decompose(std::span(positions).subspan(left_count), std::span(pressure).subspan(left_count), kx, back);
        // A homogeneous freestanding slab between equal air media has the same
        // reflection from either side. Solve both equations, including the wave
        // returning from the far rigid end, instead of assuming anechoic output.
        Complex const denominator = left.Forward * left.Forward - right.Backward * right.Backward;
        condition = std::max({condition, right.Condition, (std::norm(left.Forward) + std::norm(right.Backward)) / std::abs(denominator)});
        reflection = (left.Backward * left.Forward - right.Forward * right.Backward) / denominator;
        transmission = (right.Forward * left.Forward - left.Backward * right.Backward) / denominator;
        residual = std::max(residual, right.Residual);
    }
    std::vector<Layer> layers{{back - front, 10000}};
    if (rigid) layers.push_back({gap, 0});
    const auto expected = LayerResponse(layers, frequency, angle, rigid, scene.C, scene.Rho);
    return {.Spacing = h, .Frequency = frequency, .Angle = angle, .Thickness = back - front, .AirGap = rigid ? gap : 0, .Nodes = scene.NumNodes(), .Steps = scene.Nt, .Seconds = seconds, .Stationarity = std::sqrt(difference / std::max(norm, 1e-30)), .SpatialResidual = residual, .Condition = condition, .Reflection = reflection, .Transmission = transmission, .ExpectedReflection = expected.Reflection, .ExpectedTransmission = expected.Transmission, .ReflectionError = std::abs(reflection - expected.Reflection), .TransmissionError = std::abs(transmission - expected.Transmission)};
}
} // namespace Porous
