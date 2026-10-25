#include "Acoustic.h"
#include "Geometry.h"
#include "Thermoviscous.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>

namespace thermoviscous {
namespace {
constexpr Complex I{0, 1};
struct Sample {
    UV U;
    double Weight;
};
using Rule = std::vector<Sample>;
const auto Nodes = ReferenceNodes(6);
struct IntegrationPoint {
    Point Position, Normal;
    Shape N;
    double Weight;
};
std::vector<IntegrationPoint> Interpolate(const Coordinates &x, const Element &e, const Rule &rule) {
    std::vector<IntegrationPoint> result;
    result.reserve(rule.size());
    for (const auto &q : rule) {
        const auto b = BasisAt(int(x.cols()), q.U);
        const Point normal = (x * b.U).cross(x * b.V);
        const double jac = normal.norm();
        if (!(jac > 0) || !std::isfinite(jac)) throw std::invalid_argument("Degenerate quadratic element");
        result.push_back({x * b.N, -normal / jac, BasisAt(int(e.Dofs.size()), q.U, e.Inset).N, q.Weight * jac});
    }
    return result;
}
std::pair<Eigen::VectorXd, Eigen::VectorXd> Gauss(int n) {
    Eigen::MatrixXd j = Eigen::MatrixXd::Zero(n, n);
    for (int k = 1; k < n; ++k) j(k - 1, k) = j(k, k - 1) = k / std::sqrt(4.0 * k * k - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(j);
    return {(eig.eigenvalues().array() + 1) / 2, eig.eigenvectors().row(0).array().square()};
}
Rule AcousticRule(int n, int node) {
    const auto [u, wu] = Gauss(n);
    const auto [v, wv] = Gauss(node < 3 ? n : 2 * ((n + 1) / 2));
    Rule rule;
    for (int j = 0; j < v.size(); ++j)
        for (int i = 0; i < u.size(); ++i) {
            const double a = u[i], b = v[j], w = wu[i] * wv[j];
            if (node == 0) rule.push_back({{a * b, (1 - a) * b}, w * b});
            else if (node == 1) rule.push_back({{a, b * (1 - a)}, w * (1 - a)});
            else if (node == 2) rule.push_back({{b * (1 - a), a}, w * (1 - a)});
            else {
                for (const auto &s : std::array<Sample, 2>{{{{a * .5, b * (1 - a)}, w * .5 * (1 - a)}, {{(a - .5) * b + .5, b * (1 - a)}, w * .5 * b}}}) {
                    UV q = s.U;
                    if (node == 4) q = {1 - s.U.x() - s.U.y(), s.U.x()};
                    if (node == 5) q = {s.U.y(), 1 - s.U.x() - s.U.y()};
                    rule.push_back({q, s.Weight});
                }
            }
        }
    return rule;
}
Rule FastRule(int count) {
    if (count == 3) return {{{1. / 6, 1. / 6}, 1. / 6}, {{2. / 3, 1. / 6}, 1. / 6}, {{1. / 6, 2. / 3}, 1. / 6}};
    Rule rule;
    const auto orbit = [&](double a, double b, double weight) {
        rule.push_back({{a, b}, weight});
        rule.push_back({{b, a}, weight});
        rule.push_back({{b, b}, weight});
    };
    if (count == 7) {
        rule.push_back({{1. / 3, 1. / 3}, .1125});
        orbit(.797426985353087, .101286507323456, .0629695902724135);
        orbit(.059715871789770, .470142064105115, .066197076394253);
    } else if (count == 12) {
        orbit(.873821971016996, .063089014491502, .0254224531851035);
        orbit(.501426509658279, .249286745170910, .0583931378631895);
        const double a = .636502499121399, b = .310352451033785, c = .053145049844816, w = .041425537809187;
        for (UV p : std::array<UV, 6>{UV{a, b}, UV{a, c}, UV{b, a}, UV{b, c}, UV{c, a}, UV{c, b}}) rule.push_back({p, w});
    } else throw std::invalid_argument("fast_quadrature must be 3, 7 or 12");
    return rule;
}
// The author's OpenBEM subdivision rule resolves singular and boundary-layer integrals.
void Subdivide(Rule &rule, const Coordinates &x, Point source, UV a, UV b, UV c, double minimumSize, int depth = 0) {
    const UV center = (a + b + c) / 3;
    const std::array<Point, 4> points{x * BasisAt(int(x.cols()), center).N, x * BasisAt(int(x.cols()), a).N, x * BasisAt(int(x.cols()), b).N, x * BasisAt(int(x.cols()), c).N};
    double distance = std::numeric_limits<double>::infinity();
    for (const auto &p : points) distance = std::min(distance, (p - source).norm());
    const double size = std::max({(points[1] - points[2]).norm(), (points[2] - points[3]).norm(), (points[3] - points[1]).norm()});
    if (size >= minimumSize && size > distance / 4) {
        if (depth >= 40) throw std::runtime_error("Surface quadrature subdivision limit exceeded");
        const UV ab = (a + b) / 2, bc = (b + c) / 2, ca = (c + a) / 2;
        Subdivide(rule, x, source, a, ab, ca, minimumSize, depth + 1);
        Subdivide(rule, x, source, ab, b, bc, minimumSize, depth + 1);
        Subdivide(rule, x, source, ca, bc, c, minimumSize, depth + 1);
        Subdivide(rule, x, source, ab, bc, ca, minimumSize, depth + 1);
    } else {
        const double scale = std::min({(a - b).norm(), (b - c).norm(), (c - a).norm()});
        for (const UV q : std::array<UV, 3>{UV{1. / 6, 1. / 6}, UV{2. / 3, 1. / 6}, UV{1. / 6, 2. / 3}}) {
            if (Corners(int(x.cols())) == 4) {
                const UV ab = b - a, ac = c - a;
                rule.push_back({a + q.x() * ab + q.y() * ac, std::abs(ab.x() * ac.y() - ab.y() * ac.x()) / 6});
            } else rule.push_back({center + scale * (q - UV{1. / 3, 1. / 3}), scale * scale / 6});
        }
    }
}
Rule AdaptiveRule(const Coordinates &x, Point source, double minimumSize) {
    const auto corners = ReferenceNodes(Corners(int(x.cols())));
    Rule rule;
    Subdivide(rule, x, source, corners[0], corners[1], corners[2], minimumSize);
    if (corners.size() == 4) Subdivide(rule, x, source, corners[0], corners[2], corners[3], minimumSize);
    return rule;
}
Rule LossRule(const Element &e, int node, double tolerance) {
    const int corners = Corners(int(e.Geometry.size()));
    const auto vertices = ReferenceNodes(int(e.Geometry.size()));
    Coordinates reference(3, vertices.size());
    for (size_t j = 0; j < vertices.size(); ++j) reference.col(j) = Point{corners == 3 ? 1 - vertices[j].sum() : 0, vertices[j].x(), vertices[j].y()};
    const UV uv = ReferenceNodes(int(e.Dofs.size()), e.Inset)[node];
    return AdaptiveRule(reference, Point(reference * BasisAt(int(reference.cols()), uv).N), (corners == 3 ? std::sqrt(2.) : 2 * std::sqrt(2.)) * tolerance);
}
Rule QuadRule(int n) {
    const auto [nodes, weights] = Gauss(n);
    Rule rule;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) rule.push_back({{2 * nodes[i] - 1, 2 * nodes[j] - 1}, 4 * weights[i] * weights[j]});
    return rule;
}
struct Integral {
    Eigen::Matrix<Complex, 1, Eigen::Dynamic, Eigen::RowMajor, 1, 9> H, G;
    double C{};
};
Integral Integrate(const std::vector<IntegrationPoint> &points, Point source, Complex k, bool omitCoincident = false) {
    Integral value;
    value.H.setZero(points.front().N.size());
    value.G.setZero(points.front().N.size());
    for (const auto &p : points) {
        const Point delta = p.Position - source;
        const double r = delta.norm();
        if (r == 0 && omitCoincident) continue;
        if (!(r > 0)) throw std::runtime_error("Receiver lies on a quadrature point");
        const Complex green = std::exp(I * k * r) * (p.Weight / (4 * std::numbers::pi * r));
        const double dn = p.Normal.dot(delta) / (r * r);
        value.G += green * p.N.transpose();
        value.H += green * (I * k * r - 1.) * dn * p.N.transpose();
        value.C -= p.Weight * dn / (4 * std::numbers::pi * r);
    }
    return value;
}
std::pair<Point, double> Bounds(const Coordinates &x) {
    const Point center = x.leftCols(Corners(int(x.cols()))).rowwise().mean();
    double radius = 0;
    for (int j = 0; j < x.cols(); ++j) radius = std::max(radius, (x.col(j) - center).norm());
    return {center, radius};
}
void Validate(const Problem &p) {
    const int n = int(p.Sources.size());
    if (n < 4 || p.Elements.empty() || p.Velocity.size() != 3 * n || !p.Velocity.allFinite()) throw std::invalid_argument("Invalid boundary mesh or velocity");
    const auto &o = p.Options;
    if (o.AcousticBackend != "dense" && o.AcousticBackend != "fmm" && o.AcousticBackend != "hmatrix") throw std::invalid_argument("Unknown acoustic_backend");
    if (!(o.CompressionTolerance >= 1e-14 && o.CompressionTolerance < 1) || o.LeafSize < 4 || o.LeafSize > 1024) throw std::invalid_argument("Invalid compression settings");
    if (o.Quadrature < 2 || o.Quadrature > 32 || o.Restart < 1 || o.MaxIterations < 1 || !(o.Tolerance > 0 && o.Tolerance < 1) || !(o.InnerTolerance > 0 && o.InnerTolerance <= o.Tolerance) || !(o.QuadratureTolerance >= 1e-10 && o.QuadratureTolerance <= 1e-3) || !(o.LossDecay >= 10 && o.LossDecay <= 100)) throw std::invalid_argument("Invalid numerical settings");
    struct Edge {
        int Winding{}, Count{}, Midpoint{-1};
    };
    std::map<std::pair<int, int>, Edge> edges;
    std::vector<int> used(p.Vertices.size());
    for (const auto &v : p.Vertices)
        if (!v.allFinite()) throw std::invalid_argument("Nonfinite mesh vertex");
    for (const auto &v : p.Receivers)
        if (!v.allFinite()) throw std::invalid_argument("Nonfinite receiver");
    for (const auto &e : p.Elements) {
        const auto &t = e.Geometry;
        const int corners = Corners(int(t.size()));
        if (std::set<int>(t.begin(), t.end()).size() != t.size()) throw std::invalid_argument("Element repeats a node");
        for (int vertex : t) {
            if (vertex < 0 || vertex >= int(used.size())) throw std::invalid_argument("Element node out of range");
            ++used[vertex];
        }
        for (int j = 0; j < corners; ++j) {
            const int a = t[j], b = t[(j + 1) % corners], mid = int(t.size()) == corners ? -1 : t[corners + j];
            auto &edge = edges[std::minmax(a, b)];
            if (edge.Count && edge.Midpoint != mid) throw std::invalid_argument("Inconsistent edge midpoint");
            edge.Winding += a < b ? 1 : -1;
            ++edge.Count;
            edge.Midpoint = mid;
        }
    }
    for (const auto &[edge, v] : edges)
        if (v.Winding || v.Count != 2) throw std::invalid_argument("Mesh must be closed with consistent winding");
    if (std::find(used.begin(), used.end(), 0) != used.end()) throw std::invalid_argument("Mesh has unused nodes");
}
} // namespace

Constants WaveConstants(const Material &m, double frequency) {
    for (double x : {m.Density, m.SoundSpeed, m.Viscosity, m.HeatCapacityP, m.HeatCapacityV, m.Conductivity, frequency})
        if (!(x > 0) || !std::isfinite(x)) throw std::invalid_argument("Fluid parameters and frequency must be positive and finite");
    if (!(m.BulkViscosity >= 0) || !std::isfinite(m.BulkViscosity) || m.HeatCapacityP <= m.HeatCapacityV) throw std::invalid_argument("Invalid bulk viscosity or heat capacities");
    const double gamma = m.HeatCapacityP / m.HeatCapacityV, beta = m.Density * (m.HeatCapacityP - m.HeatCapacityV);
    const double lh = m.Conductivity / (m.Density * m.HeatCapacityP * m.SoundSpeed), lv = (m.BulkViscosity + 4 * m.Viscosity / 3) / (m.Density * m.SoundSpeed), lvp = m.Viscosity / (m.Density * m.SoundSpeed);
    const double k = 2 * std::numbers::pi * frequency / m.SoundSpeed, lvh = lv + (gamma - 1) * lh, lvhp = (gamma - 1) * (lh - lv);
    const Complex ka2 = k * k / (1. - I * k * lvh - k * k * lh * lvhp), kh2 = I * k / (lh * (1. + I * k * lvhp)), kv2 = I * k / lvp;
    const auto tau = [&](Complex k2) { return (gamma - 1) / (beta * gamma * (1. + I * lh * k2 / k)); };
    const auto phi = [&](Complex k2) { return -I / (m.Density * 2 * std::numbers::pi * frequency * (1. + I * lv * k2 / k)); };
    return {std::sqrt(ka2), std::sqrt(kh2), std::sqrt(kv2), tau(ka2), tau(kh2), phi(ka2), phi(kh2)};
}

Operators Assemble(const Problem &p) {
    const auto start = std::chrono::steady_clock::now();
    Validate(p);
    Operators a;
    a.Physics = WaveConstants(p.Fluid, p.Frequency);
    const int n = int(p.Sources.size());
    const bool accelerated = p.Options.AcousticBackend != "dense";
    if (!accelerated) {
        a.Ha = Matrix::Zero(n, n);
        a.Ga = Matrix::Zero(n, n);
    }
    FastData fast;
    const Rule triangleRegular = FastRule(p.Options.FastQuadrature);
    const Rule quadRegular = QuadRule(p.Options.Quadrature);
    if (accelerated) {
        size_t samples = 0;
        for (const auto &e : p.Elements) samples += Corners(int(e.Geometry.size())) == 3 ? triangleRegular.size() : quadRegular.size();
        fast.Sources.resize(3, samples);
        fast.Normals.resize(3, samples);
        fast.Targets.resize(3, n);
        for (int i = 0; i < n; ++i) fast.Targets.col(i) = p.Sources[i];
    }
    int sourceOffset = 0;
    Eigen::VectorXd freeTerm = Eigen::VectorXd::Zero(n);
    std::vector<Point> normals(n, Point::Zero());
    std::vector<int> count(n);
    using Triplet = Eigen::Triplet<Complex>;
    std::vector<Triplet> derivatives, hht, ght, hvt, gvt, nt, coefficients, correctionH, correctionG;
    const auto sparse = [](int rows, int cols, const std::vector<Triplet> &entries) {
        Sparse matrix(rows, cols);
        matrix.setFromTriplets(entries.begin(), entries.end());
        matrix.prune(Complex(0));
        return matrix;
    };
    std::map<std::tuple<int, int, double>, std::vector<Rule>> lossRules;
    for (const auto &e : p.Elements) {
        const auto &t = e.Dofs;
        const int size = int(t.size()), geometrySize = int(e.Geometry.size()), corners = Corners(geometrySize);
        const auto x = ElementCoordinates(p, e);
        const auto nodes = ReferenceNodes(size, e.Inset);
        for (int j = 0; j < size; ++j) {
            const auto geometry = BasisAt(geometrySize, nodes[j]), physics = BasisAt(size, nodes[j], e.Inset);
            const Point u = x * geometry.U, v = x * geometry.V, c = u.cross(v);
            if (c.norm() == 0) throw std::invalid_argument("Degenerate element at node");
            normals[t[j]] += c.normalized();
            ++count[t[j]];
            Eigen::Matrix3d cov;
            cov.row(0) = u;
            cov.row(1) = v;
            cov.row(2) = c;
            Coordinates d(3, size);
            d.row(0) = physics.U;
            d.row(1) = physics.V;
            d.row(2).setZero();
            const Coordinates grad = cov.partialPivLu().solve(d);
            for (int axis = 0; axis < 3; ++axis)
                for (int l = 0; l < size; ++l) derivatives.emplace_back(axis * n + t[j], t[l], grad(axis, l));
        }
        auto &rules = lossRules[{geometrySize, size, e.Inset}];
        if (rules.empty())
            for (int j = 0; j < size; ++j) rules.push_back(LossRule(e, j, p.Options.QuadratureTolerance));
        std::vector<std::vector<IntegrationPoint>> aq(size), lq(size);
        for (int j = 0; j < size; ++j) {
            lq[j] = Interpolate(x, e, rules[j]);
            if (!accelerated) aq[j] = (e.Inset > 0 || corners == 4) ? lq[j] : Interpolate(x, e, AcousticRule(p.Options.Quadrature, j));
        }
        const auto fq = accelerated ? Interpolate(x, e, corners == 3 ? triangleRegular : quadRegular) : std::vector<IntegrationPoint>{};
        if (accelerated)
            for (const auto &q : fq) {
                fast.Sources.col(sourceOffset) = q.Position;
                fast.Normals.col(sourceOffset) = q.Normal;
                for (int j = 0; j < size; ++j) coefficients.emplace_back(sourceOffset, t[j], q.Weight * q.N[j]);
                ++sourceOffset;
            }
        const auto [center, radius] = Bounds(x);
        const double cutoff = p.Options.LossDecay / std::min(a.Physics.Kh.imag(), a.Physics.Kv.imag());
        for (int row = 0; row < n; ++row) {
            const bool incident = std::find(t.begin(), t.end(), row) != t.end();
            const double distance = (p.Sources[row] - center).norm();
            const bool near = incident || distance <= 2 * radius;
            const bool losses = incident || (!p.Options.IncidentLosses && distance <= 2 * radius + cutoff);
            if (accelerated && !near && !losses) continue;
            int closest = 0;
            for (int j = 1; j < size; ++j)
                if ((p.Sources[t[j]] - p.Sources[row]).squaredNorm() < (p.Sources[t[closest]] - p.Sources[row]).squaredNorm()) closest = j;
            std::vector<IntegrationPoint> extra;
            if (!incident && (losses || (accelerated && near))) {
                const Rule rule = AdaptiveRule(x, p.Sources[row], 2 * radius * p.Options.QuadratureTolerance);
                extra = Interpolate(x, e, rule);
            }
            const auto &points = incident ? lq[closest] : extra;
            if (!accelerated) {
                const auto av = Integrate(aq[closest], p.Sources[row], a.Physics.Ka);
                for (int j = 0; j < size; ++j) {
                    a.Ha(row, t[j]) += p.Options.Exterior ? av.H[j] : -av.H[j];
                    a.Ga(row, t[j]) += av.G[j];
                }
                freeTerm[row] += av.C;
            } else if (near) {
                const auto exact = Integrate(points, p.Sources[row], a.Physics.Ka);
                const auto raw = Integrate(fq, p.Sources[row], a.Physics.Ka, true);
                freeTerm[row] += exact.C - raw.C;
                for (int j = 0; j < size; ++j) {
                    correctionH.emplace_back(row, t[j], exact.H[j] - raw.H[j]);
                    correctionG.emplace_back(row, t[j], exact.G[j] - raw.G[j]);
                }
            }
            if (!losses) continue;
            const auto hv = Integrate(points, p.Sources[row], a.Physics.Kh), vv = Integrate(points, p.Sources[row], a.Physics.Kv);
            for (int j = 0; j < size; ++j) {
                hht.emplace_back(row, t[j], p.Options.Exterior ? hv.H[j] : -hv.H[j]);
                ght.emplace_back(row, t[j], hv.G[j]);
                hvt.emplace_back(row, t[j], p.Options.Exterior ? vv.H[j] : -vv.H[j]);
                gvt.emplace_back(row, t[j], vv.G[j]);
            }
        }
    }
    if (accelerated) {
        fast.Coefficients = sparse(sourceOffset, n, coefficients);
        fast.CorrectionH = sparse(n, n, correctionH);
        fast.CorrectionG = sparse(n, n, correctionG);
        freeTerm += Fmm(fast, 0, Vector(fast.Coefficients * Vector::Ones(n)), true, p.Options.CompressionTolerance).real();
        fast.FreeTerm.resize(n);
    }
    for (int j = 0; j < n; ++j) {
        // Green identity: a constant interior Laplace field has zero boundary residual.
        const double c = p.Options.Exterior ? 1 - freeTerm[j] : freeTerm[j];
        if (accelerated) fast.FreeTerm[j] = c;
        else a.Ha(j, j) += c;
        hht.emplace_back(j, j, c);
        hvt.emplace_back(j, j, c);
    }
    for (int j = 0; j < n; ++j) {
        if (normals[j].norm() < 1e-8) throw std::invalid_argument("Undefined averaged nodal normal");
        normals[j].normalize();
        for (int axis = 0; axis < 3; ++axis) nt.emplace_back(axis * n + j, j, (p.Options.Exterior ? 1 : -1) * normals[j][axis]);
    }
    for (auto &t : derivatives) t = Triplet(t.row(), t.col(), t.value() / double(count[t.row() % n]));
    a.Dc = sparse(3 * n, n, derivatives);
    for (auto &t : derivatives) t = Triplet(t.row() % n, t.col() + (t.row() / n) * n, t.value());
    a.Dr = sparse(n, 3 * n, derivatives);
    a.Nd = sparse(3 * n, n, nt);
    a.Hh = sparse(n, n, hht);
    a.Gh = sparse(n, n, ght);
    a.Hv = sparse(n, n, hvt);
    a.Gv = sparse(n, n, gvt);
    if (accelerated) a.Acoustic = MakeAcoustic(std::move(fast), a.Physics.Ka, p.Options);
    a.AssemblySeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return a;
}

Vector Evaluate(const Problem &p, const Operators &a, const Solution &s) {
    Vector result = Vector::Zero(p.Receivers.size());
    for (const auto &e : p.Elements) {
        const auto x = ElementCoordinates(p, e);
        const int corners = Corners(int(e.Geometry.size()));
        const Rule rule = corners == 3 ? AcousticRule(std::max(8, p.Options.Quadrature), 0) : QuadRule(std::max(8, p.Options.Quadrature));
        const auto farPoints = Interpolate(x, e, rule);
        const auto [center, radius] = Bounds(x);
        for (size_t j = 0; j < p.Receivers.size(); ++j) {
            std::vector<IntegrationPoint> nearPoints;
            if ((p.Receivers[j] - center).norm() < 2 * radius) nearPoints = Interpolate(x, e, AdaptiveRule(x, p.Receivers[j], 2 * radius * p.Options.QuadratureTolerance));
            const auto &points = nearPoints.empty() ? farPoints : nearPoints;
            const auto ga = Integrate(points, p.Receivers[j], a.Physics.Ka), gh = Integrate(points, p.Receivers[j], a.Physics.Kh);
            for (size_t k = 0; k < e.Dofs.size(); ++k) {
                const int node = e.Dofs[k];
                result[j] += (p.Options.Exterior ? -1. : 1.) * (ga.H[k] * s.Pa[node] + gh.H[k] * s.Ph[node]) - ga.G[k] * s.Dpa[node] - gh.G[k] * s.Dph[node];
            }
        }
    }
    return result;
}
} // namespace thermoviscous
