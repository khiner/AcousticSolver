#include "ElasticRun.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>

namespace elastic {
namespace {
void Require(bool ok, const std::string &message) {
    if (!ok) throw std::invalid_argument(message);
}
Mat Inverse(const Mat &a) {
    Eigen::LLT<Mat> const factor(a);
    Require(a.allFinite() && factor.info() == Eigen::Success, "Nonpositive definite material or mass");
    return factor.solve(Mat::Identity(a.rows(), a.cols()));
}
double Field(const Json &j, const V3 &x) {
    if (j.is_number()) return j.get<double>();
    double value = j.at("value");
    const auto gradient = j.value("gradient", std::array<double, 3>{});
    for (int i = 0; i < 3; ++i) value += gradient[i] * x[i];
    if (j.contains("sinusoid")) {
        const auto &wave = j.at("sinusoid");
        const auto k = wave.at("wavevector").get<std::array<double, 3>>();
        value += double(wave.at("amplitude")) * std::sin(k[0] * x[0] + k[1] * x[1] + k[2] * x[2] + wave.value("phase", 0.));
    }
    return value;
}
V3 Point(const Json &j) {
    Require(j.is_array() && j.size() == 3, "Expected a three-vector");
    return {double(j[0]), double(j[1]), double(j[2])};
}
Mat Points(const Json &j) {
    Mat p(j.size(), 3);
    for (int i = 0; i < p.rows(); ++i) p.row(i) = Point(j[i]);
    Require(p.allFinite(), "Nonfinite coordinates");
    return p;
}
Mat Basis(const Mat &p, int degree) { return dg::reference::Bernstein(p, degree).V; }
Mat ReferenceVertices() {
    Mat v(4, 3);
    v << -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1;
    return v;
}
Mat Constitutive(const Material &m) {
    Mat h = Mat::Zero(m.Fields(), m.Fields());
    h.topLeftCorner(3, 3).diagonal().setConstant(m.Density);
    if (m.Fluid()) h(3, 3) = 1 / m.Bulk;
    else if (!m.RelaxationRates.size()) h.bottomRightCorner(6, 6) = Inverse(m.Stiffness);
    else {
        // Scaled Maxwell branch stresses: H^-1 couples total stress to local memory variables.
        Mat inverse = Mat::Identity(6 + m.RelaxationRates.size(), 6 + m.RelaxationRates.size());
        inverse.topLeftCorner(6, 6) = m.Stiffness;
        inverse.topRightCorner(6, m.RelaxationRates.size()) = m.RelaxationCoupling;
        inverse.bottomLeftCorner(m.RelaxationRates.size(), 6) = m.RelaxationCoupling.transpose();
        h.bottomRightCorner(inverse.rows(), inverse.cols()) = Inverse(inverse);
    }
    return h;
}
Mat Operator(int fields, const V3 &n) {
    Mat a = Mat::Zero(fields, fields);
    if (fields == 4) {
        a.block<3, 1>(0, 3) = n;
        a.block<1, 3>(3, 0) = n.transpose();
    } else {
        a.block(3, 0, 6, 3) = NormalMatrix(n);
        a.block(0, 3, 3, 6) = a.block(3, 0, 6, 3).transpose();
    }
    return a;
}
Mat Mass(const Mat &v, const Vec &w, const std::vector<Mat> &h) {
    const int n = v.cols(), f = h.front().rows();
    Mat m = Mat::Zero(n * f, n * f);
    for (int a = 0; a < f; ++a)
        for (int b = 0; b < f; ++b) {
            Vec weights(w.size());
            for (int q = 0; q < w.size(); ++q) weights[q] = w[q] * h[q](a, b);
            if (weights.cwiseAbs().maxCoeff() > 0) m.block(a * n, b * n, n, n) = v.transpose() * weights.asDiagonal() * v;
        }
    return m;
}
Mat ApplyReference(const Mat &a, const Mat &inverse, int fields) {
    const int n = inverse.rows();
    Mat out = a;
    for (int i = 0; i < fields; ++i)
        for (int j = 0; j < fields; ++j) out.block(i * n, j * n, n, n) = inverse * a.block(i * n, j * n, n, n) * inverse;
    return .5 * (out + out.transpose());
}
Vec BoundaryFlux(const Vec &q, const Material &m, const V3 &n, const std::string &boundary, double penalty) {
    if (boundary == "absorbing") {
        Vec flux = Vec::Zero(q.size());
        if (m.Fluid()) {
            const double velocity = .5 * (q.head<3>().dot(n) - q[3] / m.Impedance);
            flux.head<3>() = -m.Impedance * velocity * n;
            flux[3] = velocity;
        } else {
            const Mat b = NormalMatrix(n);
            Eigen::SelfAdjointEigenSolver<Mat> const eigen(m.Density * b.transpose() * m.Stiffness * b);
            Require(eigen.info() == Eigen::Success && eigen.eigenvalues().minCoeff() > 0, "Invalid characteristic impedance");
            const Mat z = eigen.eigenvectors() * eigen.eigenvalues().cwiseSqrt().asDiagonal() * eigen.eigenvectors().transpose();
            const Vec velocity = .5 * (q.head(3) - z.ldlt().solve(b.transpose() * q.segment(3, 6)));
            flux.head(3) = -z * velocity;
            flux.segment(3, 6) = b * velocity;
        }
        return flux;
    }
    Vec ghost = q;
    if (boundary == "prescribed") return Flux(q, -q, m, m, n, penalty);
    if (m.Fluid()) {
        if (boundary == "rigid") ghost.head<3>() -= 2 * q.head<3>().dot(n) * n;
        else {
            Require(boundary == "pressure_release", "Unknown fluid boundary: " + boundary);
            ghost[3] = -q[3];
        }
    } else {
        if (boundary == "fixed") ghost.head(3) = -q.head(3);
        else {
            Require(boundary == "free", "Unknown solid boundary: " + boundary);
            const Mat b = NormalMatrix(n);
            ghost.segment(3, 6) -= 2 * b * (b.transpose() * b).ldlt().solve(b.transpose() * q.segment(3, 6));
        }
    }
    return Flux(q, ghost, m, m, n, penalty);
}
} // namespace
Model::Geometry Model::Map(const Element &e, const Mat &points, bool positive) const {
    auto b = dg::reference::Bernstein(points, GeometryDegree);
    Geometry result{b.V * GeometryInverse * e.Geometry, {}};
    std::array<Mat, 3> derivative;
    for (int i = 0; i < 3; ++i) derivative[i] = b.D[i] * GeometryInverse * e.Geometry;
    for (int q = 0; q < points.rows(); ++q) {
        Eigen::Matrix3d j;
        for (int i = 0; i < 3; ++i) j.row(i) = derivative[i].row(q);
        Require(j.allFinite() && (!positive || j.determinant() > 0), "Inverted or degenerate tetrahedron");
        result.J.push_back(j);
    }
    return result;
}
double Source::Drive(double time) const {
    const double a = std::numbers::pi * Frequency * (time - Time);
    return (1 - 2 * a * a) * std::exp(-a * a);
}
Material EvaluateMaterial(const Json &j, const V3 &x) {
    Material m;
    m.Density = Field(j.at("density"), x);
    Require(std::isfinite(m.Density) && m.Density > 0, "Density must be finite and positive");
    if (j.at("type") == "fluid") {
        Require(!j.contains("relaxation"), "Solid relaxation is not a fluid absorption model");
        if (j.contains("bulk_modulus")) m.Bulk = Field(j.at("bulk_modulus"), x);
        else {
            const double speed = Field(j.at("sound_speed"), x);
            Require(std::isfinite(speed) && speed > 0, "Sound speed must be finite and positive");
            m.Bulk = m.Density * speed * speed;
        }
        Require(std::isfinite(m.Bulk) && m.Bulk > 0, "Bulk modulus must be finite and positive");
        m.Speed = std::sqrt(m.Bulk / m.Density);
    } else {
        Require(j.at("type") == "solid", "Material type must be fluid or solid");
        m.Stiffness = Mat::Zero(6, 6);
        if (j.contains("stiffness")) {
            Require(j.at("stiffness").size() == 6, "Stiffness must be 6 by 6");
            for (int i = 0; i < 6; ++i) {
                Require(j.at("stiffness")[i].size() == 6, "Stiffness must be 6 by 6");
                for (int k = 0; k < 6; ++k) m.Stiffness(i, k) = Field(j.at("stiffness")[i][k], x);
            }
        } else {
            const double young = Field(j.at("young_modulus"), x), poisson = Field(j.at("poisson_ratio"), x);
            Require(young > 0 && poisson > -1 && poisson < .5, "Invalid isotropic elastic constants");
            const double mu = young / (2 * (1 + poisson)), lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson));
            m.Stiffness.topLeftCorner(3, 3).setConstant(lambda);
            m.Stiffness.topLeftCorner(3, 3).diagonal().array() += 2 * mu;
            m.Stiffness.bottomRightCorner(3, 3).diagonal().setConstant(mu);
        }
        Require(m.Stiffness.allFinite() && (m.Stiffness - m.Stiffness.transpose()).norm() < 1e-12 * m.Stiffness.norm(), "Stiffness must be finite and symmetric");
        Eigen::SelfAdjointEigenSolver<Mat> const eigen(m.Stiffness);
        Require(eigen.info() == Eigen::Success && eigen.eigenvalues().minCoeff() > 0, "Stiffness must be positive definite");
        m.Speed = std::sqrt(2 * eigen.eigenvalues().maxCoeff() / m.Density);
        if (j.contains("relaxation")) {
            const auto &mechanisms = j.at("relaxation");
            Require(mechanisms.is_array(), "relaxation must be an array of stress/strain time pairs");
            m.RelaxationCoupling = Mat::Zero(6, 6 * mechanisms.size());
            m.RelaxationRates.resize(6 * mechanisms.size());
            const double g = m.Stiffness.bottomRightCorner(3, 3).trace() / 3;
            const double k = m.Stiffness.topLeftCorner(3, 3).trace() / 3 - 4 * g / 3;
            for (int l = 0; l < int(mechanisms.size()); ++l) {
                const auto &branch = mechanisms[l];
                std::array<double, 4> strength, rate;
                for (int mode = 0; mode < 4; ++mode) {
                    const auto time = [&](const char *name) {
                        const auto &value = branch.at(name);
                        Require(!value.is_array() || value.size() == 4, "Relaxation times need four modes: bulk, yz, xz, xy");
                        return Field(value.is_array() ? value[mode] : value, x);
                    };
                    const double stress = time("stress_time"), strain = time("strain_time");
                    Require(std::isfinite(stress) && std::isfinite(strain) && stress > 0 && strain >= stress && std::isfinite(1 / stress), "Require finite strain_time >= stress_time > 0");
                    rate[mode] = 1 / stress;
                    strength[mode] = 1 - stress / strain;
                }
                Require(k >= 0 || strength[0] == 0, "Dilatational relaxation requires nonnegative mean bulk modulus");
                auto coupling = m.RelaxationCoupling.middleCols(6 * l, 6);
                coupling.col(0).head(3).setConstant(std::sqrt(std::max(0., k) * strength[0]));
                // Orthonormal normal-deviatoric directions, followed by the three engineering shears.
                coupling(0, 1) = std::sqrt(g * strength[1]);
                coupling(1, 1) = -coupling(0, 1);
                coupling(0, 2) = coupling(1, 2) = std::sqrt(g * strength[1] / 3);
                coupling(2, 2) = -2 * coupling(0, 2);
                for (int mode = 0; mode < 3; ++mode) coupling(3 + mode, 3 + mode) = std::sqrt(m.Stiffness(3 + mode, 3 + mode) * strength[1 + mode]);
                m.RelaxationRates.segment(6 * l, 6) << rate[0], rate[1], rate[1], rate[1], rate[2], rate[3];
            }
            // The equilibrium stiffness must remain positive after all parallel relaxation decrements.
            Inverse(m.Stiffness - m.RelaxationCoupling * m.RelaxationCoupling.transpose());
        }
    }
    m.Impedance = m.Density * m.Speed;
    Require(std::isfinite(m.Speed) && std::isfinite(m.Impedance) && m.Impedance > 0, "Material exceeds numerical range");
    return m;
}
Mat NormalMatrix(const V3 &n) {
    Mat b = Mat::Zero(6, 3);
    b(0, 0) = n[0];
    b(1, 1) = n[1];
    b(2, 2) = n[2];
    b(3, 1) = n[2];
    b(3, 2) = n[1];
    b(4, 0) = n[2];
    b(4, 2) = n[0];
    b(5, 0) = n[1];
    b(5, 1) = n[0];
    return b;
}
Vec Flux(const Vec &left, const Vec &right, const Material &ml, const Material &mr, const V3 &n, double penalty) {
    const V3 vl = left.head<3>(), vr = right.head<3>();
    const Mat b = NormalMatrix(n);
    const V3 tl = ml.Fluid() ? V3(left[3] * n) : V3(b.transpose() * left.segment(3, 6));
    const V3 tr = mr.Fluid() ? V3(right[3] * n) : V3(b.transpose() * right.segment(3, 6));
    const double tv = penalty * 2 / (1 / ml.Impedance + 1 / mr.Impedance), ts = penalty * 2 / (ml.Impedance + mr.Impedance);
    Vec result = Vec::Zero(left.size());
    if (ml.Fluid()) {
        result.head<3>() = (.5 * n.dot(tl + tr) + .5 * tv * n.dot(vr - vl)) * n;
        result[3] = .5 * n.dot(vl + vr) + .5 * ts * n.dot(tr - tl);
    } else if (mr.Fluid()) {
        result.head<3>() = (.5 * n.dot(tl + tr) + .5 * tv * n.dot(vr - vl)) * n;
        result.segment(3, 6) = b * (vl + .5 * n.dot(vr - vl) * n + .5 * ts * (tr - tl));
    } else {
        result.head<3>() = .5 * (tl + tr) + .5 * tv * b.transpose() * b * (vr - vl);
        result.segment(3, 6) = b * (.5 * (vl + vr) + .5 * ts * (tr - tl));
    }
    return result;
}
Model::Model(const Json &config, const Json &mesh, bool assembleReference) : GeometryDegree(mesh.at("geometry_degree")) {
    Degree = config.value("degree", 2);
    Penalty = config.value("penalty", 1.);
    Require(Degree >= 1 && Degree <= 6 && std::isfinite(Penalty) && Penalty >= 0, "Invalid degree or penalty");
    const std::string mass = config.value("mass", "weight_adjusted");
    Require(mass == "full" || mass == "weight_adjusted", "Unknown mass operator");
    WeightAdjusted = mass == "weight_adjusted";
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer): Degree must be loaded and validated before sizing.
    Nodes = (Degree + 1) * (Degree + 2) * (Degree + 3) / 6;
    Require(GeometryDegree >= 1 && GeometryDegree <= 4, "Geometry degree must be 1..4");
    const int quadratureOrder = config.value("quadrature_order", Degree + GeometryDegree + 1);
    Require(quadratureOrder >= Degree + GeometryDegree + 1 && quadratureOrder <= 20, "Invalid quadrature order");
    const auto quadrature = dg::reference::Integrate(quadratureOrder);
    const auto basis = dg::reference::Bernstein(quadrature.Points, Degree);
    V = basis.V;
    Weights = quadrature.Weights;
    ReferenceInverse = Inverse(V.transpose() * Weights.asDiagonal() * V);
    MassInterpolation = V * ReferenceInverse;
    const Mat xyz = Points(mesh.at("nodes")), referenceNodes = Points(mesh.at("reference_nodes"));
    const Mat geometryBasis = Basis(referenceNodes, GeometryDegree);
    Eigen::FullPivLU<Mat> const gl(geometryBasis);
    Require(gl.isInvertible(), "Singular geometry interpolation");
    GeometryInverse = gl.inverse();
    const Mat corners = ReferenceVertices();
    Require(referenceNodes.rows() == (GeometryDegree + 1) * (GeometryDegree + 2) * (GeometryDegree + 3) / 6 && (referenceNodes.topRows(4) - corners).norm() < 1e-12, "Geometry nodes must start with canonical vertices");
    std::map<std::array<int, 3>, std::vector<std::pair<int, std::array<int, 3>>>> topology;
    const std::array<std::array<int, 3>, 4> localFaces{{{0, 1, 2}, {0, 1, 3}, {1, 2, 3}, {0, 2, 3}}};
    for (const auto &item : mesh.at("elements")) {
        Element e;
        e.Tag = item.at("tag");
        const auto nodes = item.at("nodes").get<std::vector<int>>();
        Require(nodes.size() == size_t(referenceNodes.rows()), "Wrong element node count");
        e.Geometry.resize(nodes.size(), 3);
        for (int i = 0; i < e.Geometry.rows(); ++i) {
            Require(nodes[i] >= 0 && nodes[i] < xyz.rows(), "Bad node index");
            e.Geometry.row(i) = xyz.row(nodes[i]);
        }
        const auto geometry = Map(e, quadrature.Points);
        e.Points = geometry.X;
        const auto &material = config.at("materials").at(std::to_string(e.Tag));
        const auto first = EvaluateMaterial(material, e.Points.row(0));
        e.FieldCount = first.Fields();
        Viscoelastic |= first.RelaxationRates.size() > 0;
        e.Offset = Size;
        Size += Nodes * e.Fields();
        std::vector<Mat> h;
        Mat rates(V.rows(), first.RelaxationRates.size());
        std::array<Mat, 3> derivative{Mat(V.rows(), Nodes), Mat(V.rows(), Nodes), Mat(V.rows(), Nodes)};
        for (int q = 0; q < V.rows(); ++q) {
            const auto m = EvaluateMaterial(material, e.Points.row(q));
            const Mat inverseJ = geometry.J[q].inverse();
            const double jacobian = geometry.J[q].determinant();
            const Mat value = jacobian * Constitutive(m);
            h.push_back(value);
            rates.row(q) = jacobian * m.RelaxationRates.transpose();
            if (WeightAdjusted) e.QuadratureInverse.push_back(Inverse(value));
            for (int a = 0; a < 3; ++a) {
                derivative[a].row(q).setZero();
                for (int r = 0; r < 3; ++r) derivative[a].row(q) += jacobian * inverseJ(a, r) * basis.D[r].row(q);
            }
        }
        e.PhysicalMass = Mass(V, Weights, h);
        if (rates.cols()) {
            e.Dissipation = Mat::Zero(Nodes * e.Fields(), Nodes * e.Fields());
            for (int r = 0; r < rates.cols(); ++r)
                e.Dissipation.block((9 + r) * Nodes, (9 + r) * Nodes, Nodes, Nodes) = V.transpose() * Weights.cwiseProduct(rates.col(r)).asDiagonal() * V;
        }
        e.InverseMass = WeightAdjusted ? ApplyReference(Mass(V, Weights, e.QuadratureInverse), ReferenceInverse, e.Fields()) : Inverse(e.PhysicalMass);
        e.EnergyMass = WeightAdjusted ? Inverse(e.InverseMass) : e.PhysicalMass;
        for (int a = 0; a < 3; ++a) e.Weak[a] = derivative[a].transpose() * Weights.asDiagonal() * V;
        for (auto local : localFaces) {
            std::sort(local.begin(), local.end(), [&](int a, int b) { return nodes[a] < nodes[b]; });
            std::array<int, 3> const key{nodes[local[0]], nodes[local[1]], nodes[local[2]]};
            topology[key].emplace_back(int(Elements.size()), local);
        }
        Elements.push_back(std::move(e));
    }
    Require(!Elements.empty(), "Empty mesh");
    std::map<std::array<int, 3>, std::string> boundaryTags;
    for (const auto &face : mesh.value("boundaries", Json::array())) {
        auto key = face.at("nodes").get<std::array<int, 3>>();
        std::sort(key.begin(), key.end());
        Require(boundaryTags.emplace(key, std::to_string(int(face.at("tag")))).second, "Duplicate boundary face");
    }
    for (const auto &[key, sides] : topology) {
        Require(sides.size() <= 2, "Nonmanifold face");
        Face f;
        f.Left = sides[0].first;
        const auto prepare = [&](int side, Mat &trace, std::vector<Material> &materials) {
            const auto &[ei, ids] = sides[side];
            const auto &e = Elements[ei];
            Mat points = quadrature.Faces[0];
            for (int q = 0; q < points.rows(); ++q) {
                const double u = (points(q, 0) + 1) / 2, v = (points(q, 1) + 1) / 2;
                points.row(q) = (1 - u - v) * corners.row(ids[0]) + u * corners.row(ids[1]) + v * corners.row(ids[2]);
            }
            trace = Basis(points, Degree);
            // Bernstein functions containing the opposite barycentric coordinate vanish identically on the face.
            const int oppositeVertex = 6 - ids[0] - ids[1] - ids[2];
            std::vector<std::pair<std::array<int, 3>, int>> faceNodes;
            int node = 0;
            for (int i = 0; i <= Degree; ++i)
                for (int j = 0; j <= Degree - i; ++j)
                    for (int k = 0; k <= Degree - i - j; ++k, ++node) {
                        const std::array<int, 4> powers{Degree - i - j - k, i, j, k};
                        if (powers[oppositeVertex]) trace.col(node).setZero();
                        else faceNodes.push_back({{powers[ids[0]], powers[ids[1]], powers[ids[2]]}, node});
                    }
            std::sort(faceNodes.begin(), faceNodes.end());
            for (const auto &[powers, index] : faceNodes) f.TraceNodes[side].push_back(index);
            if (side)
                for (int j = 0; j < int(faceNodes.size()); ++j) trace.col(f.TraceNodes[1][j]) = f.L.col(f.TraceNodes[0][j]);
            const auto geometry = Map(e, points);
            if (side == 0) {
                f.Points = geometry.X;
                f.Normals.resize(points.rows(), 3);
                f.Weights.resize(points.rows());
            } else Require((geometry.X - f.Points).norm() < 1e-9 * std::max(1., f.Points.norm()), "Nonconforming curved interface");
            const V3 du = .5 * (corners.row(ids[1]) - corners.row(ids[0])), dv = .5 * (corners.row(ids[2]) - corners.row(ids[0]));
            const bool reverse = du.cross(dv).dot(corners.row(oppositeVertex).transpose() - corners.row(ids[0]).transpose()) > 0;
            for (int q = 0; q < points.rows(); ++q) {
                V3 normal = (geometry.J[q].transpose() * du).cross(geometry.J[q].transpose() * dv);
                if (reverse) normal = -normal;
                if (side == 0) {
                    f.Weights[q] = quadrature.FaceWeights[q] * normal.norm();
                    f.Normals.row(q) = normal.normalized();
                } else Require((normal.normalized() + f.Normals.row(q).transpose()).norm() < 1e-8, "Overlapping elements or inconsistent face orientation");
            }
            for (int q = 0; q < points.rows(); ++q) materials.push_back(EvaluateMaterial(config.at("materials").at(std::to_string(e.Tag)), geometry.X.row(q)));
        };
        prepare(0, f.L, f.LeftMaterial);
        if (sides.size() == 2) {
            Require(!boundaryTags.contains(key), "Interior face tagged as exterior boundary");
            f.Right = sides[1].first;
            prepare(1, f.R, f.RightMaterial);
        } else {
            const auto tag = boundaryTags.find(key);
            f.Boundary = tag == boundaryTags.end() ? config.value("default_boundary", std::string("absorbing")) : config.at("boundaries").at(tag->second).get<std::string>();
            BoundaryFlux(Vec::Zero(Elements[f.Left].Fields()), f.LeftMaterial.front(), f.Normals.row(0), f.Boundary, Penalty);
        }
        Faces.push_back(std::move(f));
    }
    for (const auto &[key, tag] : boundaryTags) Require(topology.contains(key), "Boundary does not belong to mesh");
    Assemble(assembleReference);
    Initial = Vec::Zero(Size);
    for (const auto &receiver : config.value("receivers", Json::array())) {
        const auto [index, weights] = Locate(Point(receiver.at("position")), receiver.value("region", 0));
        const auto &e = Elements[index];
        const std::string field = receiver.value("field", std::string("pressure"));
        const std::vector<std::string> fields = e.Fluid() ? std::vector<std::string>{"vx", "vy", "vz", "pressure"} : std::vector<std::string>{"vx", "vy", "vz", "sxx", "syy", "szz", "syz", "sxz", "sxy"};
        auto const it = std::find(fields.begin(), fields.end(), field);
        Require(it != fields.end(), "Receiver field incompatible with material");
        Receivers.push_back({index, int(it - fields.begin()), field == "pressure" ? -1. : 1., weights});
    }
    for (const auto &source : config.value("sources", Json::array())) {
        const auto [index, weights] = Locate(Point(source.at("position")), source.value("region", 0));
        const auto &e = Elements[index];
        Vec load = Vec::Zero(Nodes * e.Fields());
        if (source.at("type") == "volume_flow") {
            Require(e.Fluid(), "Volume source requires fluid");
            load.segment(3 * Nodes, Nodes) = -weights;
        } else {
            Require(source.at("type") == "force", "Unknown source type");
            const V3 direction = Point(source.at("direction"));
            for (int i = 0; i < 3; ++i) load.segment(i * Nodes, Nodes) = direction[i] * weights;
        }
        Source const s{index, e.InverseMass * load, source.value("amplitude", 1.), source.at("frequency"), source.value("time", .002)};
        Require(s.Load.allFinite() && std::isfinite(s.Amplitude) && std::isfinite(s.Frequency) && s.Frequency > 0 && std::isfinite(s.Time), "Invalid source parameters");
        Sources.push_back(s);
    }
}
std::pair<int, Vec> Model::Locate(const V3 &position, int region) const {
    Require(position.allFinite(), "Nonfinite point");
    for (int i = 0; i < int(Elements.size()); ++i) {
        const auto &e = Elements[i];
        if (region && e.Tag != region) continue;
        Mat r(1, 3);
        r.setConstant(-.5);
        for (int k = 0; k < 20; ++k) {
            const auto g = Map(e, r, false);
            const V3 error = g.X.row(0).transpose() - position;
            if (error.norm() < 1e-11) break;
            const Eigen::FullPivLU<Eigen::Matrix3d> factor(g.J[0].transpose());
            if (!factor.isInvertible()) break;
            const V3 step = factor.solve(error);
            if (!step.allFinite() || step.norm() > 1e6) break;
            r.row(0) -= step.transpose();
        }
        const auto g = Map(e, r, false);
        if ((g.X.row(0).transpose() - position).norm() < 1e-8 && r.minCoeff() >= -1 - 1e-9 && r.sum() <= -1 + 1e-9) return {i, Basis(r, Degree).row(0)};
    }
    throw std::invalid_argument("Source or receiver lies outside mesh");
}
void Model::Assemble(bool assembleReference) {
    std::vector<std::map<int, Mat>> blocks(Elements.size());
    Entries boundaryEntries;
    for (int i = 0; i < int(Elements.size()); ++i) {
        const auto &e = Elements[i];
        Mat volume = Mat::Zero(Nodes * e.Fields(), Nodes * e.Fields());
        for (int d = 0; d < 3; ++d) {
            const Mat a = Operator(e.Fields(), V3::Unit(d));
            for (int row = 0; row < e.Fields(); ++row)
                for (int col = 0; col < e.Fields(); ++col)
                    if (a(row, col) != 0) volume.block(row * Nodes, col * Nodes, Nodes, Nodes) -= a(row, col) * e.Weak[d];
        }
        if (e.Dissipation.size()) volume -= e.Dissipation;
        blocks[i][i] = std::move(volume);
    }
    for (auto &f : Faces) {
        const int leftRows = Elements[f.Left].Fields() * f.TraceNodes[0].size();
        const int rightRows = f.Right < 0 ? 0 : Elements[f.Right].Fields() * f.TraceNodes[1].size();
        f.Load.resize(leftRows + rightRows, leftRows + rightRows);
        const auto add = [&](bool reverse) {
            const int i = reverse ? f.Right : f.Left, j = reverse ? f.Left : f.Right;
            const auto &a = Elements[i];
            const int nf = a.Fields(), ng = j >= 0 ? Elements[j].Fields() : nf;
            const Mat &v = reverse ? f.R : f.L, &w = reverse ? f.L : f.R;
            std::vector<Mat> own(nf, Mat(v.rows(), nf)), other(ng, Mat(v.rows(), nf));
            for (int q = 0; q < v.rows(); ++q) {
                const V3 normal = reverse ? V3(-f.Normals.row(q)) : V3(f.Normals.row(q));
                const auto &ma = reverse ? f.RightMaterial[q] : f.LeftMaterial[q];
                for (int c = 0; c < nf; ++c) own[c].row(q) = f.Weights[q] * (j < 0 ? BoundaryFlux(Vec::Unit(nf, c), ma, normal, f.Boundary, Penalty) : Flux(Vec::Unit(nf, c), Vec::Zero(ng), ma, reverse ? f.LeftMaterial[q] : f.RightMaterial[q], normal, Penalty));
                if (j >= 0)
                    for (int c = 0; c < ng; ++c) other[c].row(q) = f.Weights[q] * Flux(Vec::Zero(nf), Vec::Unit(ng, c), ma, reverse ? f.LeftMaterial[q] : f.RightMaterial[q], normal, Penalty);
            }
            const auto &vn = f.TraceNodes[reverse], &wn = f.TraceNodes[!reverse];
            const int start = reverse ? leftRows : 0, otherStart = reverse ? 0 : leftRows;
            const auto accumulate = [&](Mat &block, const std::vector<Mat> &flux, const Mat &trial, const std::vector<int> &nodes, int columnStart) {
                for (int row = 0; row < nf; ++row)
                    for (int col = 0; col < int(flux.size()); ++col) {
                        const Mat value = v.transpose() * flux[col].col(row).asDiagonal() * trial;
                        block.block(row * Nodes, col * Nodes, Nodes, Nodes) += value;
                        f.Load.block(start + row * vn.size(), columnStart + col * nodes.size(), vn.size(), nodes.size()) = value(vn, nodes);
                    }
            };
            accumulate(blocks[i][i], own, v, vn, start);
            if (j < 0) {
                for (int q = 0; q < v.rows(); ++q) {
                    const Mat physical = .5 * f.Weights[q] * Operator(nf, f.Normals.row(q));
                    for (int col = 0; col < nf; ++col) own[col].row(q) -= physical.col(col).transpose();
                }
                for (int row = 0; row < nf; ++row)
                    for (int col = 0; col < nf; ++col) {
                        const Mat power = v.transpose() * own[col].col(row).asDiagonal() * v;
                        AppendBlock(boundaryEntries, power, a.Offset + row * Nodes, a.Offset + col * Nodes);
                    }
            }
            if (j >= 0) {
                auto [it, inserted] = blocks[i].try_emplace(j, Mat::Zero(nf * Nodes, ng * Nodes));
                accumulate(it->second, other, w, wn, otherStart);
            }
        };
        add(false);
        if (f.Right >= 0) add(true);
    }
    if (assembleReference) {
        Entries entries;
        for (int i = 0; i < int(Elements.size()); ++i)
            for (const auto &[j, load] : blocks[i]) AppendBlock(entries, Elements[i].InverseMass * load, Elements[i].Offset, Elements[j].Offset);
        System = MakeSparse(Size, Size, entries);
        RowStarts.assign(System.outerIndexPtr(), System.outerIndexPtr() + Size + 1);
    }
    BoundaryPower = MakeSparse(Size, Size, boundaryEntries);
    SetTimestep(blocks);
}
void Model::SetTimestep(std::vector<std::map<int, Mat>> &weakBlocks) {
    std::vector<Mat> inverse;
    for (const auto &e : Elements) {
        Eigen::LLT<Mat> const factor(e.EnergyMass);
        if (factor.info() != Eigen::Success) throw std::runtime_error("Invalid CFL energy mass");
        inverse.emplace_back(factor.matrixU().solve(Mat::Identity(e.EnergyMass.rows(), e.EnergyMass.cols())));
    }
    // M = U^T U; U^-T L U^-1 is similar to the physical evolution operator M^-1 L.
    for (int i = 0; i < int(Elements.size()); ++i)
        for (auto &[j, block] : weakBlocks[i]) block = inverse[i].transpose() * block * inverse[j];
    Vec real = Vec::Zero(Elements.size()), imaginary = real;
    for (int i = 0; i < int(Elements.size()); ++i)
        for (const auto &[j, block] : weakBlocks[i]) {
            if (j < i) continue;
            const Mat transpose = weakBlocks[j].at(i).transpose();
            const double s = SpectralNorm(.5 * (block + transpose)), k = SpectralNorm(.5 * (block - transpose));
            real[i] += s;
            imaginary[i] += k;
            if (j != i) {
                real[j] += s;
                imaginary[j] += k;
            }
        }
    // Chan et al. (2016), section 3.1, with block-norm bounds for the symmetric and skew parts.
    CflRealBound = real.maxCoeff();
    CflImaginaryBound = imaginary.maxCoeff();
    DtLimit = RkRadius / std::hypot(CflRealBound, CflImaginaryBound);
    if (!std::isfinite(DtLimit) || DtLimit <= 0) throw std::runtime_error("Invalid CFL timestep");
}
Vec Model::ApplyMatrixFree(const Vec &state) const {
    Require(state.size() == Size, "Wrong state size");
    std::vector<Mat> load;
    for (const auto &e : Elements) {
        const Eigen::Map<const Mat> q(state.data() + e.Offset, Nodes, e.Fields());
        Mat local = Mat::Zero(Nodes, e.Fields());
        for (int d = 0; d < 3; ++d) local.noalias() -= e.Weak[d] * q * Operator(e.Fields(), V3::Unit(d)).transpose();
        if (e.Dissipation.size()) {
            Eigen::Map<Vec> flat(local.data(), local.size());
            flat.noalias() -= e.Dissipation * state.segment(e.Offset, local.size());
        }
        load.push_back(std::move(local));
    }
    for (const auto &f : Faces) {
        const auto &left = Elements[f.Left];
        const Mat l = f.L * Eigen::Map<const Mat>(state.data() + left.Offset, Nodes, left.Fields());
        Mat r;
        if (f.Right >= 0) {
            const auto &right = Elements[f.Right];
            r = f.R * Eigen::Map<const Mat>(state.data() + right.Offset, Nodes, right.Fields());
        }
        Mat flux(l.rows(), l.cols()), reverse(r.rows(), r.cols());
        for (int q = 0; q < l.rows(); ++q) {
            const V3 n = f.Normals.row(q);
            flux.row(q) = f.Weights[q] * (f.Right < 0 ? BoundaryFlux(l.row(q), f.LeftMaterial[q], n, f.Boundary, Penalty) : Flux(l.row(q), r.row(q), f.LeftMaterial[q], f.RightMaterial[q], n, Penalty));
            if (f.Right >= 0) reverse.row(q) = f.Weights[q] * Flux(r.row(q), l.row(q), f.RightMaterial[q], f.LeftMaterial[q], -n, Penalty);
        }
        load[f.Left].noalias() += f.L.transpose() * flux;
        if (f.Right >= 0) load[f.Right].noalias() += f.R.transpose() * reverse;
    }
    Vec result(Size);
    for (int i = 0; i < int(Elements.size()); ++i) {
        const auto &e = Elements[i];
        if (WeightAdjusted) {
            Mat q = MassInterpolation * load[i];
            for (int j = 0; j < q.rows(); ++j) q.row(j) = Weights[j] * (e.QuadratureInverse[j] * q.row(j).transpose()).transpose();
            Eigen::Map<Mat>(result.data() + e.Offset, Nodes, e.Fields()).noalias() = MassInterpolation.transpose() * q;
        } else result.segment(e.Offset, Nodes * e.Fields()).noalias() = e.InverseMass * Eigen::Map<const Vec>(load[i].data(), load[i].size());
    }
    return result;
}
Vec Model::Rhs(const Vec &state, double time) const {
    Vec result = Apply(state);
    AddForcing(result, time);
    return result;
}
void Model::AddForcing(Vec &result, double time) const {
    if (PrescribedLoad) {
        const Vec load = PrescribedLoad(time);
        Require(load.size() == Size && load.allFinite(), "Invalid prescribed boundary load");
        result += load;
    } else
        for (const auto &f : Faces)
            if (f.Right < 0 && f.Boundary == "prescribed") {
                Require(bool(BoundaryState), "Prescribed boundary requires an incident field");
                const auto &e = Elements[f.Left];
                Mat values(f.L.rows(), e.Fields());
                for (int q = 0; q < values.rows(); ++q) {
                    const Vec value = BoundaryState(f.Points.row(q), e.Tag, time);
                    Require(value.size() == e.Fields() && value.allFinite(), "Invalid prescribed boundary state");
                    values.row(q) = f.Weights[q] * Flux(Vec::Zero(e.Fields()), 2 * value, f.LeftMaterial[q], f.LeftMaterial[q], f.Normals.row(q), Penalty);
                }
                const Mat load = f.L.transpose() * values;
                result.segment(e.Offset, Nodes * e.Fields()).noalias() += e.InverseMass * Eigen::Map<const Vec>(load.data(), load.size());
            }
    for (const auto &s : Sources) result.segment(Elements[s.ElementIndex].Offset, s.Load.size()) += (s.Amplitude * s.Drive(time)) * s.Load;
}
double Model::Energy(const Vec &state, bool physical) const {
    Require(state.size() == Size, "Wrong state size");
    double energy = 0;
    for (const auto &e : Elements) {
        auto const q = state.segment(e.Offset, Nodes * e.Fields());
        energy += .5 * q.dot((physical ? e.PhysicalMass : e.EnergyMass) * q);
    }
    return energy;
}
double Model::Power(const Vec &state, const Vec &rhs) const {
    Require(state.size() == Size && rhs.size() == Size, "Wrong state size");
    double power = 0;
    for (const auto &e : Elements) power += state.segment(e.Offset, Nodes * e.Fields()).dot(e.EnergyMass * rhs.segment(e.Offset, Nodes * e.Fields()));
    return power;
}
Eigen::Matrix<double, 6, 1> Model::Diagnostics(const Vec &state, double time) const {
    Vec massState(Size);
    double physical = 0, dissipation = 0;
    for (const auto &e : Elements) {
        auto const q = state.segment(e.Offset, Nodes * e.Fields());
        massState.segment(e.Offset, q.size()).noalias() = e.EnergyMass * q;
        physical += .5 * q.dot(e.PhysicalMass * q);
        if (e.Dissipation.size()) dissipation += q.dot(e.Dissipation * q);
    }
    double source = 0;
    for (const auto &s : Sources) source += (s.Amplitude * s.Drive(time)) * massState.segment(Elements[s.ElementIndex].Offset, s.Load.size()).dot(s.Load);
    Vec forcing = Vec::Zero(Size);
    AddForcing(forcing, time);
    const double boundary = state.dot(BoundaryPower * state);
    return {.5 * state.dot(massState), physical, source, boundary + massState.dot(forcing) - source, boundary - massState.dot(Apply(state)) - dissipation, dissipation};
}
Eigen::Vector3d Model::Powers(const Vec &state, double time) const {
    return Diagnostics(state, time).segment<3>(2);
}
Vec Model::Project(const std::function<Vec(const V3 &, int)> &field) const {
    Vec state(Size);
    for (const auto &e : Elements) {
        Mat values(V.rows(), e.Fields());
        for (int q = 0; q < V.rows(); ++q) {
            const Vec value = field(e.Points.row(q), e.Tag);
            Require(value.size() == e.Fields() && value.allFinite(), "Invalid projection field");
            values.row(q) = value;
        }
        Eigen::Map<Mat>(state.data() + e.Offset, Nodes, e.Fields()) = ReferenceInverse * V.transpose() * Weights.asDiagonal() * values;
    }
    return state;
}
Trace Model::RunReference(double seconds, double sampleRate, double dt) const {
    Require(std::isfinite(dt) && dt >= 0 && dt <= DtLimit, "Reference timestep exceeds CFL limit");
    return Run(seconds, sampleRate, RunOptions{true, dt > 0 ? dt : DtLimit});
}
} // namespace elastic
