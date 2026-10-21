#include "CoupledBoundary.h"
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>

namespace {
using coupled::Sparse;
using Vector = Eigen::VectorXd;
using Dense = Eigen::MatrixXd;
using json = nlohmann::json;
void Check(bool pass, const char *message) {
    if (!pass) throw std::runtime_error(message);
}
json Modes() {
    json report;
    for (const std::string boundary : {"clamped", "simply_supported", "free"}) {
        // Leissa, NASA SP-160, Tables 4.22 and 4.67; the simply supported value is analytic.
        const double target = boundary == "clamped" ? 35.9866 : boundary == "simply_supported" ? 2 * std::numbers::pi * std::numbers::pi :
                                                                                                 13.473;
        double previousError = 1;
        for (const int intervals : {8, 16, 24}) {
            auto const op = coupled::BuildBoundaryOperators(intervals, intervals, 1. / intervals, .3, boundary);
            const int n = op.Bending.rows();
            Vector const mass = op.InverseMass.unaryExpr([](double x) { return x ? 1 / x : 0; });
            Dense symmetric = mass.cwiseSqrt().asDiagonal() * Dense(op.Bending) * op.InverseMass.cwiseSqrt().asDiagonal();
            Check((symmetric - symmetric.transpose()).norm() < 1e-12 * symmetric.norm(), "Bending operator is not mass self-adjoint");
            Eigen::SelfAdjointEigenSolver<Dense> const eigen(symmetric, Eigen::EigenvaluesOnly);
            Check(eigen.info() == Eigen::Success && eigen.eigenvalues()[0] > -1e-7, "Bending operator is not positive semidefinite");
            double frequency = 0;
            for (double const value : eigen.eigenvalues())
                if (value > 1) {
                    frequency = std::sqrt(value);
                    break;
                }
            const double error = std::abs(frequency / target - 1);
            Check(error < previousError, "Mode frequency does not converge under refinement");
            previousError = error;
            report[boundary].push_back({{"intervals", intervals}, {"frequency_parameter", frequency}, {"relative_error", error}});
            if (intervals == 24) Check(error < .012, "Mode frequency misses independent reference");
            if (boundary == "free") {
                Dense rigid(n, 3);
                for (int x = 0; x <= intervals; ++x)
                    for (int y = 0; y <= intervals; ++y)
                        rigid.row(x * (intervals + 1) + y) << 1, double(x) / intervals, double(y) / intervals;
                Check((op.Bending * rigid).norm() < 1e-12 * op.Bending.norm() * rigid.norm(), "Free plate lost rigid translation/tilt modes");
                Check((eigen.eigenvalues().array().abs() < 1e-6).count() == 3, "Free plate has spurious zero-frequency modes");
            }
        }
    }
    return report;
}
json Isolated(const std::string &boundary) {
    json const config = {{"sample_rate", 22050}, {"seconds", .01}, {"airbox", {.7, .7, .7}}, {"lossless", true}, {"panels", {{{"boundary", boundary}, {"size", {.24, .22}}, {"center", {0, 0, 0}}, {"thickness", .001}, {"density", 7800}, {"young_modulus", 2e11}, {"poisson_ratio", .3}}}}, {"strikes", json::array()}, {"receivers", {{0, 0, .2}}}};
    coupled::Reference const reference(config);
    const auto &t = reference.Data;
    const auto &p = t.Plates[0];
    const int n = p.Size;
    const auto b = coupled::ReadMatrix(t, p.B, n), c = coupled::ReadMatrix(t, p.C, n), k = coupled::ReadMatrix(t, p.K, n);
    const Vector inverseMass = Eigen::Map<const Vector>(t.Values.data() + p.MassInverse, n);
    const Vector mass = inverseMass.unaryExpr([](double x) { return x ? 1 / x : 0; });
    const double dt = 1 / t.SampleRate, rhoH = 7.8, eh = 2e8;
    Sparse identity(n, n);
    identity.setIdentity();
    const Sparse stiffness = (rhoH / (dt * dt)) * mass.asDiagonal() * (2 * identity - b);
    Vector u = Vector::Zero(n), old(n), phi = Vector::Zero(n), oldPhi = phi;
    for (int i = 0; i < n; ++i) old[i] = inverseMass[i] ? -5 * dt * std::sin(.31 * i) : 0;
    Vector a(n), v(n);
    for (int q = 0; q < n; ++q) {
        a[q] = std::sin(.37 * q * q);
        v[q] = std::cos(.19 * q + .02 * q * q);
    }
    const Vector av = coupled::NonlinearOperator(t, p, a) * v, va = coupled::NonlinearOperator(t, p, v) * a;
    Check((av - va).norm() < 1e-12 * std::max(1., av.norm()), "Nonlinear polarization lost symmetry");
    if (boundary == "free") {
        for (int q = 0; q < n; ++q) a[q] = 1 + double(q / p.Stride) + double(q % p.Stride);
        Check(coupled::NonlinearOperator(t, p, a).norm() < 1e-12 * coupled::NonlinearOperator(t, p, v).norm(), "Affine motion generates nonlinear stress");
    }
    const auto j = coupled::ReadMatrix(t, p.J, n), i = coupled::ReadMatrix(t, p.I, t.Nx * t.Ny);
    const Sparse mismatch = (p.H * p.H / p.Bf) * mass.asDiagonal() * i - t.Q * t.Q * Sparse(j.transpose());
    // Fixed displacement nodes cannot do work and are omitted from the interface identity.
    const Vector active = (inverseMass.array() != 0).cast<double>();
    Check(Sparse(active.asDiagonal() * mismatch).norm() < 1e-12 * j.norm(), "Air interpolation does not balance work");
    double minEnergy = 1e100, maxEnergy = 0, maxNonlinear = 0;
    for (int step = 0; step < 400; ++step) {
        const Sparse l = coupled::NonlinearOperator(t, p, u);
        const Sparse f = p.FScale * inverseMass.asDiagonal() * Sparse(l.transpose());
        const Vector right = b * u - c * old + f * oldPhi;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solve;
        solve.compute(k + Sparse(p.WScale * l * f));
        Check(solve.info() == Eigen::Success, "Isolated stress factorization failed");
        const Vector nextPhi = solve.solve(-k * phi - p.WScale * l * right);
        const Vector next = right + f * nextPhi;
        const double nonlinear = (nextPhi.dot(k * nextPhi) + phi.dot(k * phi)) / (4 * eh);
        const double energy = p.H * p.H * (.5 * rhoH / (dt * dt) * ((next - u).array().square() * mass.array()).sum() + .5 * next.dot(stiffness * u) + nonlinear);
        minEnergy = std::min(minEnergy, energy);
        maxEnergy = std::max(maxEnergy, energy);
        maxNonlinear = std::max(maxNonlinear, p.H * p.H * nonlinear);
        old = u;
        u = next;
        oldPhi = phi;
        phi = nextPhi;
    }
    const double drift = (maxEnergy - minEnergy) / maxEnergy;
    Check(minEnergy > 0 && drift < 1e-8, "Lossless nonlinear plate energy drifts");
    Check(maxNonlinear / maxEnergy > .01, "Energy test did not exercise nonlinearity");
    return {{"energy_relative_drift", drift}, {"maximum_nonlinear_energy_fraction", maxNonlinear / maxEnergy}};
}
} // namespace
int main() try {
    json report;
    report["modes"] = Modes();
    for (const std::string boundary : {"clamped", "simply_supported", "free"}) {
        report["nonlinear"][boundary] = Isolated(boundary);
    }
    report["passed"] = true;
    std::cout << report.dump(2) << '\n';
    return 0;
} catch (const std::exception &e) {
    std::cerr << "CoupledBoundaryTest: " << e.what() << '\n';
    return 1;
}
