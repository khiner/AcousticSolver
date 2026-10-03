// Formulation: Chew et al., Acoustic Reliefs (2025). See LICENSE.AcousticReliefs.
#include "Bem.h"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <chrono>
#include <cmath>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>

namespace bem {
namespace {
using C = std::complex<double>;
using V = Eigen::Vector3d;
constexpr double Pi = std::numbers::pi;
constexpr double Quadrature[6][3] = {
    {.223381589678011, .445948490915965, .108103018168070},
    {.223381589678011, .445948490915965, .445948490915965},
    {.223381589678011, .108103018168070, .445948490915965},
    {.109951743655322, .091576213509771, .816847572980459},
    {.109951743655322, .091576213509771, .091576213509771},
    {.109951743655322, .816847572980459, .091576213509771}
};
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
struct Face {
    V A, B, C, Center, Normal;
};
std::vector<Face> Geometry(const Problem &p) {
    std::vector<Face> faces;
    for (int i = 0; i < p.Faces.rows(); ++i) {
        const V a = p.Vertices.row(p.Faces(i, 0)), b = p.Vertices.row(p.Faces(i, 1)), c = p.Vertices.row(p.Faces(i, 2));
        faces.push_back({a, b, c, (a + b + c) / 3., (b - a).cross(c - a)});
    }
    return faces;
}
C Incident(double k, const V &r) { return std::exp(C{0., -k * r.norm()}) / (4. * Pi * r.norm()); }
C Integral(double k, const V &point, const Face &f) {
    C sum{};
    for (const auto &q : Quadrature) {
        const V r = point - (f.A + q[1] * (f.B - f.A) + q[2] * (f.C - f.A));
        const double distance = r.norm();
        sum += .5 * q[0] * std::exp(C{0., -k * distance}) * C{1., k * distance} * r.dot(f.Normal) / (4. * Pi * distance * distance * distance);
    }
    return sum;
}
// Directional derivative includes centroid motion, quadrature motion and the
// unnormalized area normal. Differentiating a unit normal alone loses the area term.
C Derivative(double k, const V &point, const Face &f, double pointDy, int sourceCorner) {
    const V ey{0., 1., 0.};
    V dn = V::Zero();
    if (sourceCorner == 0) dn = (-ey).cross(f.C - f.A) + (f.B - f.A).cross(-ey);
    if (sourceCorner == 1) dn = ey.cross(f.C - f.A);
    if (sourceCorner == 2) dn = (f.B - f.A).cross(ey);
    C sum{};
    for (const auto &q : Quadrature) {
        const double sourceDy = sourceCorner == 0 ? 1. - q[1] - q[2] : sourceCorner == 1 ? q[1] :
            sourceCorner == 2                                                            ? q[2] :
                                                                                           0.;
        const V r = point - (f.A + q[1] * (f.B - f.A) + q[2] * (f.C - f.A));
        const V dr = (pointDy - sourceDy) * ey;
        const double d = r.norm(), d2 = d * d;
        const C exponential = std::exp(C{0., -k * d}) / (4. * Pi);
        const C radial = exponential * C{1., k * d} / (d2 * d);
        const C radialDerivative = exponential * (C{k * k * d2 - 3., -3. * k * d}) / (d2 * d2 * d);
        sum += .5 * q[0] * (radial * (dr.dot(f.Normal) + r.dot(dn)) + radialDerivative * r.dot(f.Normal) * r.dot(dr));
    }
    return sum;
}
} // namespace

void Validate(const Problem &p) {
    if (p.Faces.rows() < 4 || p.Faces.rows() > 1000000) throw std::invalid_argument("BEM requires 4..1000000 triangles");
    if (p.Vertices.rows() < 4 || !p.Vertices.allFinite() || p.Listeners.rows() < 2 || !p.Listeners.allFinite() || !p.Source.allFinite()) throw std::invalid_argument("Invalid BEM geometry/source/listeners");
    if (!(p.Frequency > 0.) || !std::isfinite(p.Frequency) || !(p.SoundSpeed > 0.) || !std::isfinite(p.SoundSpeed)) throw std::invalid_argument("Invalid BEM frequency/medium");
    for (int i = 0; i < p.Faces.size(); ++i)
        if (p.Faces.data()[i] < 0 || p.Faces.data()[i] >= p.Vertices.rows()) throw std::invalid_argument("BEM triangle index out of bounds");
    std::map<std::pair<int, int>, std::pair<int, int>> edges;
    double volume = 0.;
    const Eigen::Vector3d origin = p.Vertices.row(0);
    for (int t = 0; t < p.Faces.rows(); ++t) {
        for (int c = 0; c < 3; ++c) {
            const int a = p.Faces(t, c), b = p.Faces(t, (c + 1) % 3);
            auto &edge = edges[std::minmax(a, b)];
            ++edge.first;
            edge.second += a < b ? 1 : -1;
        }
        const Eigen::Vector3d a = p.Vertices.row(p.Faces(t, 0)).transpose() - origin;
        const Eigen::Vector3d b = p.Vertices.row(p.Faces(t, 1)).transpose() - origin;
        const Eigen::Vector3d c = p.Vertices.row(p.Faces(t, 2)).transpose() - origin;
        volume += a.dot(b.cross(c));
    }
    for (const auto &[indices, edge] : edges)
        if (edge.first != 2 || edge.second != 0) throw std::invalid_argument("BEM requires a closed, consistently oriented triangle mesh with shared vertices");
    if (!(volume > 0.)) throw std::invalid_argument("BEM requires outward-oriented triangles");
    std::set<int> unique;
    for (const int v : p.HeightVertices)
        if (v < 0 || v >= p.Vertices.rows() || !unique.insert(v).second) throw std::invalid_argument("Invalid or repeated height vertex");
    for (const auto &f : Geometry(p)) {
        if (f.Normal.norm() < 1e-14) throw std::invalid_argument("Degenerate BEM triangle");
        if ((f.Center - p.Source).norm() < 1e-8) throw std::invalid_argument("Source on a collocation point");
        for (int i = 0; i < p.Listeners.rows(); ++i)
            if ((p.Listeners.row(i).transpose() - f.Center).norm() < 1e-8) throw std::invalid_argument("Listener on a collocation point");
    }
}

void ValidateDense(const Problem &p) {
    Validate(p);
    if (p.Faces.rows() > 2048 || uint64_t(p.Faces.rows()) * uint64_t(p.Listeners.rows()) > 16777216) throw std::invalid_argument("BEM dense oracle exceeds storage limit");
}

Matrix ReferenceRows(const Problem &p, const std::vector<int> &rows, bool boundary, bool transpose) {
    const auto faces = Geometry(p);
    const double k = 2. * Pi * p.Frequency / p.SoundSpeed;
    const int columns = transpose && !boundary ? p.Listeners.rows() : p.Faces.rows();
    Matrix out(rows.size(), columns);
    for (size_t r = 0; r < rows.size(); ++r)
        for (int c = 0; c < columns; ++c) {
            const int i = transpose ? c : rows[r], j = transpose ? rows[r] : c;
            if (j < 0 || j >= p.Faces.rows() || i < 0 || i >= (boundary ? p.Faces.rows() : p.Listeners.rows())) throw std::invalid_argument("BEM reference sample index out of bounds");
            const V point = boundary ? faces[i].Center : V(p.Listeners.row(i));
            out(r, c) = boundary && i == j ? C{-.5, 0.} : Integral(k, point, faces[j]);
        }
    return out;
}

Operators AssembleReference(const Problem &p) {
    ValidateDense(p);
    const auto faces = Geometry(p);
    const int n = p.Faces.rows(), l = p.Listeners.rows();
    const double k = 2. * Pi * p.Frequency / p.SoundSpeed;
    Operators op{Matrix(n, n), Matrix(l, n), Vector(n)};
    for (int i = 0; i < n; ++i) {
        op.Incident[i] = Incident(k, faces[i].Center - p.Source);
        for (int j = 0; j < n; ++j) op.Boundary(i, j) = i == j ? C{-.5, 0.} : Integral(k, faces[i].Center, faces[j]);
    }
    for (int i = 0; i < l; ++i)
        for (int j = 0; j < n; ++j) op.Listener(i, j) = Integral(k, p.Listeners.row(i), faces[j]);
    return op;
}

Vector PressureSensitivity(const Vector &y) {
    const Eigen::ArrayXd power = y.cwiseAbs2();
    const double sum = power.sum(), squared = power.square().sum();
    if (!(squared > 0.) || !std::isfinite(squared)) throw std::runtime_error("BEM diffusion undefined for zero/nonfinite scattered pressure");
    return (4. * (sum * squared - power * sum * sum) / ((y.size() - 1) * squared * squared)) * y.conjugate().array();
}

static Result SolveFields(const Operators &op, const Eigen::PartialPivLU<Matrix> &lu) {
    Result r;
    r.SurfacePressure = lu.solve(-op.Incident);
    r.ForwardResidual = (op.Boundary * r.SurfacePressure + op.Incident).norm() / op.Incident.norm();
    r.ScatteredPressure = op.Listener * r.SurfacePressure;
    const auto power = r.ScatteredPressure.cwiseAbs2().eval();
    const double sum = power.sum(), squared = power.squaredNorm();
    if (!(squared > 0.)) throw std::runtime_error("BEM diffusion undefined for zero pressure");
    r.Diffusion = (sum * sum - squared) / ((power.size() - 1) * squared);
    if (!r.SurfacePressure.allFinite() || !std::isfinite(r.Diffusion) || r.ForwardResidual > 1e-9) throw std::runtime_error("BEM forward solve failed residual check");
    return r;
}

Eigen::VectorXd ReferenceGradient(const Problem &p, const Vector &x, const Vector &vf, const Vector &lambda) {
    const auto faces = Geometry(p);
    const double k = 2. * Pi * p.Frequency / p.SoundSpeed;
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(p.HeightVertices.size());
    for (size_t h = 0; h < p.HeightVertices.size(); ++h) {
        C g{};
        for (int t = 0; t < p.Faces.rows(); ++t)
            for (int corner = 0; corner < 3; ++corner)
                if (p.Faces(t, corner) == p.HeightVertices[h]) {
                    for (int l = 0; l < p.Listeners.rows(); ++l) g += vf[l] * x[t] * Derivative(k, p.Listeners.row(l), faces[t], 0., corner);
                    for (int j = 0; j < p.Faces.rows(); ++j)
                        if (j != t) {
                            g -= lambda[t] * x[j] * Derivative(k, faces[t].Center, faces[j], 1. / 3., -1);
                            g -= lambda[j] * x[t] * Derivative(k, faces[j].Center, faces[t], 0., corner);
                        }
                    const V dr = faces[t].Center - p.Source;
                    const double distance = dr.norm();
                    const C incidentDerivative = Incident(k, dr) * C{-1. / distance, -k} * dr.y() / (3. * distance);
                    g -= lambda[t] * incidentDerivative;
                }
        gradient[h] = g.real();
    }
    return gradient;
}

namespace {
Result Solve(const Problem &p) {
    const auto t = Clock::now();
    Operators op = AssembleReference(p);
    const double assembly = Seconds(t);
    const auto factorStart = Clock::now();
    if (!op.Boundary.allFinite() || !op.Listener.allFinite() || !op.Incident.allFinite()) throw std::runtime_error("Nonfinite BEM operator");
    const Eigen::PartialPivLU<Matrix> lu(op.Boundary);
    Result r = SolveFields(op, lu);
    r.SolveSeconds = Seconds(factorStart);
    r.AssemblySeconds = assembly;
    if (!p.HeightVertices.empty()) {
        const auto solveStart = Clock::now();
        const Vector vf = PressureSensitivity(r.ScatteredPressure);
        const Vector rhs = op.Listener.transpose() * vf;
        const Vector adjoint = lu.transpose().solve(rhs);
        r.AdjointResidual = (op.Boundary.transpose() * adjoint - rhs).norm() / std::max(rhs.norm(), 1e-300);
        if (!adjoint.allFinite() || r.AdjointResidual > 1e-9) throw std::runtime_error("BEM adjoint solve failed residual check");
        r.SolveSeconds += Seconds(solveStart);
        const auto derivativeStart = Clock::now();
        r.HeightGradient = ReferenceGradient(p, r.SurfacePressure, vf, adjoint);
        r.DerivativeSeconds = Seconds(derivativeStart);
        if (!r.HeightGradient.allFinite()) throw std::runtime_error("Nonfinite BEM height gradient");
    }
    return r;
}
} // namespace
Result SolveReference(const Problem &p) { return Solve(p); }
} // namespace bem
