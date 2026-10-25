#include "Geometry.h"
#include "Thermoviscous.h"
#include <Eigen/QR>
#include <algorithm>
#include <chrono>
#include <functional>
#include <stdexcept>

namespace thermoviscous {
namespace {
using Action = std::function<Vector(const Vector &)>;
Vector Gmres(const Action &apply, const Vector &b, const Settings &s, double tolerance, int &iterations) {
    Vector x = Vector::Zero(b.size());
    const double norm = b.norm();
    if (norm == 0) return x;
    const int restart = std::min<int>(s.Restart, b.size());
    Matrix basis(b.size(), restart + 1), h(restart + 1, restart);
    int performed = 0;
    while (performed < s.MaxIterations) {
        Vector residual = b - apply(x);
        const double beta = residual.norm();
        if (!std::isfinite(beta)) throw std::runtime_error("Nonfinite GMRES residual");
        if (beta <= tolerance * norm) return x;
        basis.col(0) = residual / beta;
        h.setZero();
        Vector y;
        int columns = 0;
        for (int j = 0; j < restart && performed < s.MaxIterations; ++j) {
            Vector w = apply(basis.col(j));
            ++iterations;
            ++performed;
            for (int pass = 0; pass < 2; ++pass)
                for (int k = 0; k <= j; ++k) {
                    const Complex dot = basis.col(k).dot(w);
                    h(k, j) += dot;
                    w -= dot * basis.col(k);
                }
            const double next = w.norm();
            h(j + 1, j) = next;
            if (next > 0) basis.col(j + 1) = w / next;
            Vector target = Vector::Zero(j + 2);
            target[0] = beta;
            y = h.topLeftCorner(j + 2, j + 1).householderQr().solve(target);
            columns = j + 1;
            if ((target - h.topLeftCorner(j + 2, j + 1) * y).norm() <= tolerance * norm || next < 1e-30) break;
        }
        x += basis.leftCols(columns) * y;
    }
    const double residual = (b - apply(x)).norm() / norm;
    if (residual > tolerance || !std::isfinite(residual)) throw std::runtime_error("GMRES did not converge: relative residual " + std::to_string(residual));
    return x;
}
} // namespace

Solution Solve(const Problem &p, const Operators &a) {
    const auto start = std::chrono::steady_clock::now();
    Solution s;
    const auto &o = p.Options;
    const int n = int(a.Gh.rows());
    if (p.Velocity.size() != 3 * n) throw std::invalid_argument("Velocity size differs from assembled system");
    const auto inverse = [&](const auto &m, const Vector &v) { return Gmres([&](const Vector &x) -> Vector { return m * x; }, v, o, o.InnerTolerance, s.InnerIterations); };
    const auto viscous = [&](const Vector &v) {
        Vector result(3 * n);
        for (int axis = 0; axis < 3; ++axis) result.segment(axis * n, n) = inverse(a.Gv, Vector(a.Hv * v.segment(axis * n, n)));
        return result;
    };
    const auto r = [&](const Vector &x) -> Vector { return a.Dr * x - a.Nd.adjoint() * viscous(x); };
    const Action rn = [&](const Vector &x) -> Vector { return r(Vector(a.Nd * x)); };
    const auto inner = [&](const Vector &x) { return Gmres(rn, x, o, o.InnerTolerance, s.InnerIterations); };
    const Complex muh = a.Physics.TauA * a.Physics.PhiH / a.Physics.TauH, mua = a.Physics.PhiA - muh;
    const Action outer = [&](const Vector &x) -> Vector {
        return -a.Physics.PhiA * (a.ApplyH(x)) + a.ApplyG(muh * inverse(a.Gh, Vector(a.Hh * x)) + mua * inner(r(Vector(a.Dc * x))));
    };
    const Vector rhs = a.ApplyG(inner(r(p.Velocity)));
    s.Pa = o.Method == "iterative" ? Gmres(outer, rhs, o, o.Tolerance, s.Iterations) : DirectPressure(p, a);
    const auto pressureDone = std::chrono::steady_clock::now();
    s.PressureSeconds = std::chrono::duration<double>(pressureDone - start).count();
    s.RelativeResidual = rhs.norm() > 0 ? (outer(s.Pa) - rhs).norm() / rhs.norm() : 0;
    s.Ph = -a.Physics.TauA / a.Physics.TauH * s.Pa;
    s.Dpa = -Gmres([&](const Vector &x) { return a.ApplyG(x); }, a.ApplyH(s.Pa), o, o.InnerTolerance, s.InnerIterations);
    s.Dph = -inverse(a.Gh, Vector(a.Hh * s.Ph));
    s.Vv = p.Velocity - a.Physics.PhiA * (a.Dc * s.Pa + a.Nd * s.Dpa) - a.Physics.PhiH * (a.Dc * s.Ph + a.Nd * s.Dph);
    s.Dvv = -viscous(s.Vv);
    const double divergence = (a.Dr * s.Vv + a.Nd.adjoint() * s.Dvv).norm();
    const double scale = (a.Dr * s.Vv).norm() + (a.Nd.adjoint() * s.Dvv).norm();
    s.BoundaryResidual = scale > 0 ? divergence / scale : 0;
    s.Pressure = Evaluate(p, a, s);
    for (const auto *field : {&s.Pa, &s.Dpa, &s.Ph, &s.Dph, &s.Vv, &s.Dvv, &s.Pressure})
        if (!field->allFinite()) throw std::runtime_error("Nonfinite thermoviscous solution");
    s.RecoverySeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - pressureDone).count();
    s.SolveSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return s;
}

Problem Read(const nlohmann::json &j) {
    Problem p;
    for (const auto &v : j.at("vertices")) {
        if (v.size() != 3) throw std::invalid_argument("Vertex needs three coordinates");
        p.Vertices.emplace_back(v[0].get<double>(), v[1].get<double>(), v[2].get<double>());
    }
    for (const char *key : {"triangles", "quadrilaterals"})
        if (j.contains(key))
            for (const auto &t : j[key]) {
                auto nodes = t.get<std::vector<int>>();
                if (Corners(int(nodes.size())) != (std::string(key) == "triangles" ? 3 : 4)) throw std::invalid_argument("Element type does not match connectivity");
                p.Elements.push_back({nodes, {}});
            }
    PrepareMesh(p, j);
    p.Frequency = j.value("frequency", p.Frequency);
    if (j.contains("material")) {
        const auto &m = j["material"];
        p.Fluid.Density = m.value("density", p.Fluid.Density);
        p.Fluid.SoundSpeed = m.value("sound_speed", p.Fluid.SoundSpeed);
        p.Fluid.Viscosity = m.value("viscosity", p.Fluid.Viscosity);
        p.Fluid.BulkViscosity = m.value("bulk_viscosity", p.Fluid.BulkViscosity);
        p.Fluid.HeatCapacityP = m.value("heat_capacity_p", p.Fluid.HeatCapacityP);
        p.Fluid.HeatCapacityV = m.value("heat_capacity_v", p.Fluid.HeatCapacityV);
        p.Fluid.Conductivity = m.value("conductivity", p.Fluid.Conductivity);
    }
    auto &o = p.Options;
    o.Method = j.value("method", o.Method);
    o.DirectMemoryGiB = j.value("direct_memory_gib", o.DirectMemoryGiB);
    if (o.Method != "iterative" && o.Method != "direct1" && o.Method != "direct4" && o.Method != "direct10") throw std::invalid_argument("Unknown solve method");
    if (!(o.DirectMemoryGiB > 0) || !std::isfinite(o.DirectMemoryGiB)) throw std::invalid_argument("Invalid direct_memory_gib");
    o.AcousticBackend = j.value("acoustic_backend", o.AcousticBackend);
    o.CompressionTolerance = j.value("compression_tolerance", o.CompressionTolerance);
    o.FastQuadrature = j.value("fast_quadrature", o.FastQuadrature);
    o.LeafSize = j.value("leaf_size", o.LeafSize);
    o.Exterior = j.value("exterior", o.Exterior);
    o.Quadrature = j.value("quadrature", o.Quadrature);
    o.Restart = j.value("restart", o.Restart);
    o.MaxIterations = j.value("max_iterations", o.MaxIterations);
    o.Tolerance = j.value("tolerance", o.Tolerance);
    o.InnerTolerance = j.value("inner_tolerance", o.InnerTolerance);
    o.QuadratureTolerance = j.value("quadrature_tolerance", o.QuadratureTolerance);
    o.LossDecay = j.value("loss_decay", o.LossDecay);
    const auto neighborhood = j.value("loss_neighborhood", std::string("distance"));
    if (neighborhood != "distance" && neighborhood != "incident") throw std::invalid_argument("loss_neighborhood must be distance or incident");
    o.IncidentLosses = neighborhood == "incident";
    const int n = int(p.Sources.size());
    p.Velocity = Vector::Zero(3 * n);
    const auto complex = [](const nlohmann::json &v) -> Complex {
        if (v.is_number()) return v.get<double>();
        if (!v.is_array() || v.size() != 2) throw std::invalid_argument("Complex value must be a number or [real,imaginary]");
        return {v[0].get<double>(), v[1].get<double>()};
    };
    if (j.contains("nodal_velocity")) {
        const auto &v = j["nodal_velocity"];
        if (v.size() != size_t(n)) throw std::invalid_argument("nodal_velocity must contain one vector per node");
        for (int i = 0; i < n; ++i) {
            if (v[i].size() != 3) throw std::invalid_argument("Velocity needs three components");
            for (int axis = 0; axis < 3; ++axis) p.Velocity[axis * n + i] = complex(v[i][axis]);
        }
    } else {
        const auto v = j.value("velocity", nlohmann::json::array({0, 0, .01}));
        if (v.size() != 3) throw std::invalid_argument("Velocity needs three components");
        for (int axis = 0; axis < 3; ++axis) p.Velocity.segment(axis * n, n).setConstant(complex(v[axis]));
    }
    if (j.contains("receivers"))
        for (const auto &v : j["receivers"]) {
            if (v.size() != 3) throw std::invalid_argument("Receiver needs three coordinates");
            p.Receivers.emplace_back(v[0].get<double>(), v[1].get<double>(), v[2].get<double>());
        }
    return p;
}
} // namespace thermoviscous
