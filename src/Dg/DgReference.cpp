#include "DgReference.h"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <sstream>

namespace dg::reference {
static_assert(std::endian::native == std::endian::little);
Json Load(const fs::path &p) { return Json::parse(std::ifstream(p)); }
void Save(const fs::path &p, const Json &j) { Require(bool(std::ofstream(p) << j.dump(2) << '\n'), "Cannot write: " + p.string()); }
std::string Hash(const fs::path &p) {
    auto bytes = Read<unsigned char>(p);
    Require(bytes.size() <= UINT32_MAX, "Hash input too large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes.data(), CC_LONG(bytes.size()), digest);
    std::ostringstream out;
    for (auto const c : digest) out << std::hex << std::setw(2) << std::setfill('0') << int(c);
    return out.str();
}
void NewDirectory(const fs::path &p) {
    Require(!fs::exists(p), "Use a new output directory: " + p.string());
    fs::create_directories(p);
}
Mat ReadMatrix(const fs::path &p, int rows, int cols) {
    auto data = Read<double>(p, size_t(rows) * cols);
    return Eigen::Map<const Mat>(data.data(), rows, cols);
}
Mat ReadNodes(const fs::path &p, int count) {
    Mat result(count, 3);
    std::ifstream in(p);
    for (int i = 0; i < count; ++i)
        for (int d = 0; d < 3; ++d) Require(bool(in >> result(i, d)), "Invalid node table");
    std::string extra;
    Require(!(in >> extra) && result.allFinite(), "Invalid node table length");
    return result;
}
namespace {
constexpr std::array<double, 5> Rka{0, -.41789047449985195, -1.192151694642677, -1.6977846924715279, -1.5141834442571558};
constexpr std::array<double, 5> Rkb{.14965902199922912, .37921031299962726, .8229550293869817, .6994504559491221, .15305724796815198};
Mat Inverse(const Mat &a) {
    Eigen::FullPivLU<Mat> const lu(a);
    Require(lu.isInvertible(), "Singular matrix");
    return lu.inverse();
}
Mat SpdInverse(const Mat &a) {
    Eigen::LLT<Mat> const llt(a);
    Require(llt.info() == Eigen::Success, "Nonpositive mass matrix");
    return llt.solve(Mat::Identity(a.rows(), a.cols()));
}
Mat Symmetric(const Mat &a) { return .5 * (a + a.transpose()); }
Vec Spectrum(const Mat &a) {
    Eigen::SelfAdjointEigenSolver<Mat> const eigen(a);
    Require(eigen.info() == Eigen::Success, "Mass spectrum failed");
    return eigen.eigenvalues();
}
Mat Random(int rows, int cols, uint32_t seed) {
    std::mt19937 generator(seed);
    Mat q(rows, cols);
    for (Eigen::Index i = 0; i < q.size(); ++i) q.data()[i] = 2. * double(generator()) / double(std::mt19937::max()) - 1.;
    return q;
}
Basis Nodal(const Mat &points, int degree, const Mat &inverse) {
    Basis b = Bernstein(points, degree);
    b.V = (b.V * inverse).eval();
    for (auto &d : b.D) d = (d * inverse).eval();
    return b;
}
std::array<Mat, 3> Geometry(const Basis &b, const Mat &xyz) {
    return {b.D[0] * xyz, b.D[1] * xyz, b.D[2] * xyz};
}
Eigen::Matrix3d Jacobian(const std::array<Mat, 3> &geometry, int row) {
    Eigen::Matrix3d j;
    for (int a = 0; a < 3; ++a) j.row(a) = geometry[a].row(row);
    return j;
}
Eigen::Vector4d Exact(const Eigen::Vector3d &, double, double, double, double);
State PhysicalState(const fs::path &file, int e, int n, double z) {
    Mat packed = ReadMatrix(file, e * 4, n), q(e * n, 4);
    for (int k = 0; k < e; ++k) q.middleRows(k * n, n) = packed.middleRows(k * 4, 4).transpose();
    q.rightCols(3) *= z;
    Require(q.allFinite(), "Nonfinite state");
    return q;
}
void WriteState(const fs::path &file, const State &q, int e, int n, double z) {
    Mat packed(e * 4, n);
    for (int k = 0; k < e; ++k) {
        packed.middleRows(k * 4, 4) = q.middleRows(k * n, n).transpose();
        packed.middleRows(k * 4 + 1, 3) /= z;
    }
    Write<double>(file, {packed.data(), size_t(packed.size())});
}
} // namespace
Basis Bernstein(const Mat &points, int degree) {
    int const n = (degree + 1) * (degree + 2) * (degree + 3) / 6;
    Basis out{Mat(points.rows(), n), {Mat::Zero(points.rows(), n), Mat::Zero(points.rows(), n), Mat::Zero(points.rows(), n)}};
    auto const factorial = [](int k) { double v=1; for(int i=2;i<=k;++i)v*=i; return v; };
    int col = 0;
    for (int i = 0; i <= degree; ++i)
        for (int j = 0; j <= degree - i; ++j)
            for (int k = 0; k <= degree - i - j; ++k, ++col) {
                std::array<int, 4> powers{degree - i - j - k, i, j, k};
                double factor = factorial(degree);
                for (int const power : powers) factor /= factorial(power);
                for (int q = 0; q < points.rows(); ++q) {
                    std::array<double, 4> bary{-(points.row(q).sum() + 1) / 2, (points(q, 0) + 1) / 2, (points(q, 1) + 1) / 2, (points(q, 2) + 1) / 2};
                    auto const product = [&](int lower) { double value=factor; for(int a=0;a<4;++a) value*=std::pow(bary[a],powers[a]-(a==lower)); return value; };
                    out.V(q, col) = product(-1);
                    for (int a = 0; a < 3; ++a) {
                        if (powers[a + 1]) out.D[a](q, col) += .5 * powers[a + 1] * product(a + 1);
                        if (powers[0]) out.D[a](q, col) -= .5 * powers[0] * product(0);
                    }
                }
            }
    return out;
}
Quadrature Integrate(int order) {
    Require(order >= 2 && order <= 64, "Invalid quadrature order");
    Mat jacobi = Mat::Zero(order, order);
    for (int i = 1; i < order; ++i) jacobi(i, i - 1) = jacobi(i - 1, i) = i / std::sqrt(4. * i * i - 1);
    Eigen::SelfAdjointEigenSolver<Mat> const eig(jacobi);
    Require(eig.info() == Eigen::Success, "Gauss quadrature failed");
    Vec g = (eig.eigenvalues().array() + 1) / 2, w = eig.eigenvectors().row(0).array().square();
    Quadrature out;
    int const count = order * order * order;
    out.Points.resize(count, 3);
    out.Weights.resize(count);
    int q = 0;
    for (int i = 0; i < order; ++i)
        for (int j = 0; j < order; ++j)
            for (int k = 0; k < order; ++k, ++q) {
                double a = g[i], b = g[j], c = g[k];
                out.Points.row(q) << 2 * a - 1, 2 * (1 - a) * b - 1, 2 * (1 - a) * (1 - b) * c - 1;
                out.Weights[q] = 8 * w[i] * w[j] * w[k] * (1 - a) * (1 - a) * (1 - b);
            }
    int const nf = 6 * order * order;
    for (auto &f : out.Faces) f.resize(nf, 3);
    out.FaceWeights.resize(nf);
    // Permutation symmetry gives adjacent elements the same face measure.
    std::array<int, 3> permutation{0, 1, 2};
    q = 0;
    do {
        for (int i = 0; i < order; ++i)
            for (int j = 0; j < order; ++j, ++q) {
                std::array<double, 3> bary{1 - g[i] - (1 - g[i]) * g[j], g[i], (1 - g[i]) * g[j]};
                double u = bary[permutation[1]], v = bary[permutation[2]];
                out.Faces[0].row(q) << 2 * u - 1, 2 * v - 1, -1;
                out.Faces[1].row(q) << 2 * u - 1, -1, 2 * v - 1;
                out.Faces[2].row(q) << 2 * u - 1, 2 * v - 1, 1 - 2 * u - 2 * v;
                out.Faces[3].row(q) << -1, 2 * u - 1, 2 * v - 1;
                out.FaceWeights[q] = 4 * w[i] * w[j] * (1 - g[i]) / 6;
            }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return out;
}
Mesh::Mesh(const fs::path &directory, const fs::path &nodes) : Metadata(Load(directory / "mesh.json")), Directory(directory), E(Metadata.at("elements")), N(Metadata.at("nodes_per_element")), Nfp(Metadata.at("nodes_per_face")), F(4 * Nfp), Degree(Metadata.at("degree")), Receivers(Metadata.at("receivers")), C(Metadata.at("sound_speed")), Z(double(Metadata.at("density")) * C) {
    Require(E > 0 && Degree >= 1 && Degree <= 12 && N == (Degree + 1) * (Degree + 2) * (Degree + 3) / 6 && Nfp == (Degree + 1) * (Degree + 2) / 2 && Receivers > 0, "Invalid mesh dimensions");
    Require(Metadata.at("scalar_bytes") == 8, "Mesh geometry must be FP64");

    Require(std::isfinite(C) && std::isfinite(Z) && C > 0 && Z > 0, "Invalid acoustic material");
    Nodes = ReadNodes(nodes, N);
    XYZ = ReadMatrix(directory / "xyz.bin", E * N, 3);
    const std::string initial_condition = Metadata.value("initial_condition", "captured");
    if (initial_condition == "cylinder_mode") {
        const double radius = Metadata.at("geometry").at("radius");
        const double height = Metadata.at("geometry").at("height");
        Require(std::isfinite(radius) && std::isfinite(height) && radius > 0 && height > 0, "Invalid cylinder dimensions");
        Initial.resize(E * N, 4);
        for (int i = 0; i < E * N; ++i) Initial.row(i) = Exact(XYZ.row(i), 0, radius, height, C);
    } else {
        Require(initial_condition == "captured", "Unknown initial condition");
        Initial = PhysicalState(directory / "initial_q.bin", E, N, Z);
    }
    std::vector<int32_t> face_nodes;
    for (int f = 0; f < 4; ++f) {
        const size_t first = face_nodes.size();
        for (int i = 0; i < N; ++i) {
            const std::array<double, 4> distance{Nodes(i, 2) + 1, Nodes(i, 1) + 1, Nodes.row(i).sum() + 1, Nodes(i, 0) + 1};
            if (std::abs(distance[f]) < 1e-12) face_nodes.push_back(i);
        }
        Require(face_nodes.size() - first == size_t(Nfp), "Incorrect reference face node count");
    }
    Vm.resize(size_t(E) * F);
    for (int e = 0; e < E; ++e)
        for (int i = 0; i < F; ++i) Vm[e * F + i] = e * N + face_nodes[i];
    Vp = Read<int32_t>(directory / "vmapP.bin", size_t(E) * F);
    Boundary.resize(size_t(E) * F);
    // A rigid exterior face maps every trace node back to itself.
    for (int face = 0; face < E * 4; ++face) {
        const int first = face * Nfp;
        const bool boundary = Vp[first] == Vm[first];
        for (int j = 0; j < Nfp; ++j) {
            const int i = first + j;
            Require(Vp[i] >= 0 && Vp[i] < E * N, "Invalid face map");
            Require((Vp[i] == Vm[i]) == boundary, "Mixed interior and boundary face nodes");
            Boundary[i] = boundary;
        }
    }
    ReceiverElements = Metadata.at("receiver_elements").get<std::vector<int32_t>>();
    Require(ReceiverElements.size() == size_t(Receivers), "Incorrect receiver element count");
    for (int const e : ReceiverElements) Require(e >= 0 && e < E, "Invalid receiver element");
    ReceiverWeights = ReadMatrix(directory / "receiver_interpolation.bin", Receivers, N);
    const auto &positions = Metadata.at("receiver_xyz");
    Require(positions.is_array() && positions.size() == size_t(Receivers), "Incorrect receiver position count");
    ReceiverXYZ.resize(Receivers, 3);
    for (int r = 0; r < Receivers; ++r) {
        Require(positions[r].is_array() && positions[r].size() == 3, "Invalid receiver position");
        for (int a = 0; a < 3; ++a) ReceiverXYZ(r, a) = positions[r][a].get<double>();
    }
    Require(XYZ.allFinite() && Initial.allFinite() && ReceiverWeights.allFinite() && ReceiverXYZ.allFinite(), "Nonfinite mesh input");
}
Reference::Reference(const fs::path &mesh, const fs::path &nodes, bool wadg) : Source(mesh, nodes), InverseBasis(Inverse(Bernstein(Source.Nodes, Source.Degree).V)) {
    auto quad = Integrate(2 * Source.Degree + 4);
    auto b = Nodal(quad.Points, Source.Degree, InverseBasis);
    V = b.V;
    InverseMass = SpdInverse(V.transpose() * quad.Weights.asDiagonal() * V);
    T = V * InverseMass;
    W.resize(Source.E, V.rows());
    Elements.resize(Source.E);
    std::array<Basis, 4> fb;
    for (int f = 0; f < 4; ++f) {
        fb[f] = Nodal(quad.Faces[f], Source.Degree, InverseBasis);
        FaceBasis[f].resize(fb[f].V.rows(), Source.Nfp);
        for (int j = 0; j < Source.Nfp; ++j) FaceBasis[f].col(j) = fb[f].V.col(Source.Vm[f * Source.Nfp + j] % Source.N);
    }
    constexpr double normal[4][3] = {{0, 0, -1}, {0, -1, 0}, {1, 1, 1}, {-1, 0, 0}};
    double minj = INFINITY, moment_error = 0, factor_error = 0, project_error = 0, min_spectrum = INFINITY, max_spectrum = 0;
    for (int e = 0; e < Source.E; ++e) {
        auto &el = Elements[e];
        int n = Source.N, fcount = Source.F;
        Mat const xyz = Source.XYZ.middleRows(e * n, n);
        auto const geometry = Geometry(b, xyz);
        el.Jacobian.resize(V.rows());
        std::array<Mat, 3> derivative{Mat::Zero(V.rows(), n), Mat::Zero(V.rows(), n), Mat::Zero(V.rows(), n)};
        for (int q = 0; q < V.rows(); ++q) {
            Eigen::Matrix3d const j = Jacobian(geometry, q);
            double const det = j.determinant();
            Require(std::isfinite(det) && det > 0, "Nonpositive volume Jacobian");
            el.Jacobian[q] = det;
            MinimumJacobian = std::min(MinimumJacobian, det);
            Eigen::Matrix3d cof = det * j.inverse().transpose();
            for (int d = 0; d < 3; ++d)
                for (int a = 0; a < 3; ++a) derivative[d].row(q) += cof(a, d) * b.D[a].row(q);
        }
        el.Physical = V.transpose() * (quad.Weights.array() * el.Jacobian.array()).matrix().asDiagonal() * V;
        Mat const physical_inverse = SpdInverse(el.Physical);
        el.Moments = el.Physical.colwise().sum();
        el.WeakG.resize(6 * n, n);
        for (int d = 0; d < 3; ++d) {
            Mat k = V.transpose() * quad.Weights.asDiagonal() * derivative[d];
            el.WeakG.middleRows(d * n, n) = k;
            el.WeakG.middleRows((d + 3) * n, n) = k.transpose();
        }
        el.WeakL = Mat::Zero(10 * n, fcount);
        for (int f = 0; f < 4; ++f) {
            auto const geom = Geometry(fb[f], xyz);
            int const nq = fb[f].V.rows();
            el.Normals[f].resize(nq, 3);
            el.Measures[f].resize(nq);
            Mat terms(nq, 10);
            for (int q = 0; q < nq; ++q) {
                Eigen::Matrix3d const j = Jacobian(geom, q);
                double const det = j.determinant();
                Require(std::isfinite(det) && det > 0, "Nonpositive face Jacobian");
                Eigen::Vector3d const nj = det * j.inverse() * Eigen::Map<const Eigen::Vector3d>(normal[f]);
                double const sj = nj.norm();
                Require(sj > 0 && std::isfinite(sj), "Invalid face normal");
                Eigen::Vector3d nn = nj / sj;
                el.Normals[f].row(q) = nn;
                el.Measures[f][q] = quad.FaceWeights[q] * sj;
                terms.row(q) << 1, nn[0], nn[1], nn[2], nn[0] * nn[0], nn[0] * nn[1], nn[0] * nn[2], nn[1] * nn[1], nn[1] * nn[2], nn[2] * nn[2];
            }
            for (int t = 0; t < 10; ++t) {
                Mat load = FaceBasis[f].transpose() * (el.Measures[f].array() * terms.col(t).array()).matrix().asDiagonal() * FaceBasis[f];
                for (int i = 0; i < Source.Nfp; ++i) el.WeakL.block(t * n + Source.Vm[f * Source.Nfp + i] % n, f * Source.Nfp, 1, Source.Nfp) = load.row(i);
            }
        }
        // Project the Jacobian only for mass inversion; spatial loads retain the original geometry.
        Vec const coefficients = InverseMass * V.transpose() * (quad.Weights.array() * el.Jacobian.array()).matrix();
        Vec projected = V * coefficients;
        Require(projected.minCoeff() > 0, "Nonpositive projected Jacobian");
        minj = std::min(minj, projected.minCoeff());
        project_error = std::max(project_error, ((projected - el.Jacobian).array() / el.Jacobian.array()).abs().maxCoeff());
        W.row(e) = (quad.Weights.array() / projected.array()).matrix();
        Mat const a = Symmetric(T.transpose() * W.row(e).asDiagonal() * T);
        Mat load = Random(n, 4, 753 + e), factored = InverseMass * V.transpose() * W.row(e).asDiagonal() * V * InverseMass * load;
        factor_error = std::max(factor_error, (a * load - factored).norm() / factored.norm());
        moment_error = std::max(moment_error, (a * el.Moments - Vec::Ones(n)).cwiseAbs().maxCoeff());
        Mat chol = el.Physical.llt().matrixL();
        Vec const spectrum = Spectrum(chol.transpose() * a * chol);
        min_spectrum = std::min(min_spectrum, spectrum.minCoeff());
        max_spectrum = std::max(max_spectrum, spectrum.maxCoeff());
        el.A = wadg ? a : physical_inverse;
        el.H = wadg ? SpdInverse(a) : el.Physical;
        el.G.resize(6 * n, n);
        el.L.resize(10 * n, fcount);
        for (int d = 0; d < 6; ++d) el.G.middleRows(d * n, n) = el.A * el.WeakG.middleRows(d * n, n);
        for (int t = 0; t < 10; ++t) el.L.middleRows(t * n, n) = el.A * el.WeakL.middleRows(t * n, n);
    }
    Require(factor_error < 1e-11 && moment_error < 1e-10, "WADG mass identity failed");
    MassChecks = {{"minimum_mass_jacobian", minj}, {"maximum_relative_jacobian_change", project_error}, {"factored_vs_dense_inverse_relative_l2", factor_error}, {"physical_constant_moment_max_error", moment_error}, {"mass_relative_spectrum_min", min_spectrum}, {"mass_relative_spectrum_max", max_spectrum}};
}
namespace {
template<class T> Matrix<T> Traces(const Mesh &mesh, const Matrix<T> &q, int e) {
    Matrix<T> traces(mesh.F, 7);
    for (int i = 0; i < mesh.F; ++i) {
        int const id = e * mesh.F + i;
        Eigen::Matrix<T, 1, 4> qm = q.row(mesh.Vm[id]), qp = q.row(mesh.Vp[id]);
        if (mesh.Boundary[id]) qp.template rightCols<3>() = -qm.template rightCols<3>();
        traces.row(i).template leftCols<4>() = qm - qp;
        traces.row(i).template rightCols<3>() = qm.template rightCols<3>() + qp.template rightCols<3>();
    }
    return traces;
}
template<class T> Matrix<T> Loads(const Matrix<T> &g, const Matrix<T> &l, const Matrix<T> &q, const Matrix<T> &traces, T c) {
    int const n = q.rows();
    Matrix<T> d = g * q, s = l * traces, out(n, 4);
    out.col(0) = c * (d.block(3 * n, 1, n, 1) + d.block(4 * n, 2, n, 1) + d.block(5 * n, 3, n, 1) - T(.5) * (s.block(0, 0, n, 1) + s.block(n, 4, n, 1) + s.block(2 * n, 5, n, 1) + s.block(3 * n, 6, n, 1)));
    constexpr int nn[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};
    for (int a = 0; a < 3; ++a) out.col(a + 1) = c * (-d.block(a * n, 0, n, 1) + T(.5) * s.block((a + 1) * n, 0, n, 1) - T(.5) * (s.block(nn[a][0] * n, 1, n, 1) + s.block(nn[a][1] * n, 2, n, 1) + s.block(nn[a][2] * n, 3, n, 1)));
    return out;
}
} // namespace
State Reference::Rhs(const State &q) const {
    State out(q.rows(), 4);
    for (int e = 0; e < Source.E; ++e) out.middleRows(e * Source.N, Source.N) = Loads<double>(Elements[e].G, Elements[e].L, q.middleRows(e * Source.N, Source.N), Traces(Source, q, e), Source.C);
    return out;
}
double Reference::Energy(const State &q) const {
    double sum = 0;
    for (int e = 0; e < Source.E; ++e) {
        Mat local = q.middleRows(e * Source.N, Source.N);
        sum += (local.array() * (Elements[e].H * local).array()).sum();
    }
    Require(std::isfinite(sum), "Nonfinite energy");
    return .5 * sum;
}
double Reference::Mean(const State &q) const {
    double sum = 0, volume = 0;
    for (int e = 0; e < Source.E; ++e) {
        sum += Elements[e].Moments.dot(q.col(0).segment(e * Source.N, Source.N));
        volume += Elements[e].Moments.sum();
    }
    Require(std::isfinite(sum), "Nonfinite mean");
    return sum / volume;
}
double Reference::Dissipation(const State &q) const {
    double sum = 0;
    for (int e = 0; e < Source.E; ++e) {
        Mat jumps = Traces(Source, q, e).leftCols(4);
        for (int f = 0; f < 4; ++f) {
            Mat values = FaceBasis[f] * jumps.middleRows(f * Source.Nfp, Source.Nfp);
            Vec vn = (values.rightCols(3).array() * Elements[e].Normals[f].array()).rowwise().sum();
            sum += (Elements[e].Measures[f].array() * (values.col(0).array().square() + vn.array().square())).sum();
        }
    }
    return -.25 * Source.C * sum;
}
Json Reference::Check() const {
    double load_error = 0;
    for (const auto &el : Elements) {
        for (int d = 0; d < 6; ++d) load_error = std::max(load_error, (el.H * el.G.middleRows(d * Source.N, Source.N) - el.WeakG.middleRows(d * Source.N, Source.N)).norm() / el.WeakG.middleRows(d * Source.N, Source.N).norm());
        for (int t = 0; t < 10; ++t) load_error = std::max(load_error, (el.H * el.L.middleRows(t * Source.N, Source.N) - el.WeakL.middleRows(t * Source.N, Source.N)).norm() / std::max(el.WeakL.middleRows(t * Source.N, Source.N).norm(), 1e-30));
    }
    Require(load_error < 1e-11, "Inverse mass changed the spatial loads");
    Json report{{"fixed_spatial_load_relative_l2", load_error}};
    for (int i = 0; i < 4; ++i) {
        State q = i ? Random(Source.E * Source.N, 4, 750 + i) : Source.Initial;
        State rhs = Rhs(q);
        double rate = 0;
        for (int e = 0; e < Source.E; ++e) rate += (q.middleRows(e * Source.N, Source.N).array() * (Elements[e].H * rhs.middleRows(e * Source.N, Source.N)).array()).sum();
        double diss = Dissipation(q), error = std::abs(rate - diss) / std::max({std::abs(diss), Source.C * Energy(q), 1e-30});
        Require(error < 1e-10, "Discrete energy identity failed");
        report[std::to_string(i)] = {{"energy_derivative", rate}, {"upwind_dissipation", diss}, {"scaled_balance_error", error}};
    }
    State constant = State::Zero(Source.E * Source.N, 4);
    constant.col(0).setOnes();
    double const error = Rhs(constant).cwiseAbs().maxCoeff();
    Require(error < 1e-7, "Constant pressure is not stationary");
    report["constant_pressure_max_rhs"] = error;
    report["minimum_quadrature_jacobian"] = MinimumJacobian;
    report["mass"] = MassChecks;
    return report;
}
Json Reference::Run(const fs::path &output, int steps, double dt) const {
    Require(steps > 0 && dt > 0 && std::isfinite(dt), "Invalid run schedule");
    NewDirectory(output);
    State q = Source.Initial, res = State::Zero(q.rows(), 4);
    Mat receivers(Source.Receivers, steps);
    std::vector<double> times(steps);
    double e0 = Energy(q), m0 = Mean(q), max_growth = 0, max_mean = 0, previous = e0;
    Json history = Json::array({{0, e0, m0}});
    auto const start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        times[step] = step * dt;
        for (int r = 0; r < Source.Receivers; ++r) receivers(r, step) = Source.ReceiverWeights.row(r).dot(q.col(0).segment(Source.ReceiverElements[r] * Source.N, Source.N));
        for (int stage = 0; stage < 5; ++stage) {
            res = Rka[stage] * res + dt * Rhs(q);
            q += Rkb[stage] * res;
        }
        if ((step + 1) % 1024 == 0 || step + 1 == steps) {
            double energy = Energy(q), mean = Mean(q);
            Require(energy <= e0 * (1 + 1e-8), "Reference energy grew");
            max_growth = std::max(max_growth, (energy - previous) / e0);
            previous = energy;
            max_mean = std::max(max_mean, std::abs(mean - m0) / std::max(std::abs(m0), 1e-30));
            history.push_back({(step + 1) * dt, energy, mean});
        }
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    Require(q.allFinite() && receivers.allFinite() && max_growth <= 1e-8 && max_mean <= 1e-8, "Reference trajectory gate failed");
    WriteState(output / "final_q.bin", q, Source.E, Source.N, Source.Z);
    Write<double>(output / "receivers.bin", {receivers.data(), size_t(receivers.size())});
    Write(output / "times.bin", times);
    Save(output / "energy.json", history);
    Json report = {{"steps", steps}, {"dt", dt}, {"precision", "float64"}, {"output_scalar_bytes", 8}, {"propagation_with_energy_checks_seconds", elapsed}, {"final_to_initial_energy", Energy(q) / e0}, {"maximum_sampled_energy_increase_relative_to_initial", max_growth}, {"physical_mean_relative_change", max_mean}};
    Save(output / "result.json", report);
    return report;
}
void Reference::Export(const fs::path &output, double offset, int order) const {
    Require(std::isfinite(offset), "Nonfinite pressure offset");
    fs::create_directories(output);
    fs::remove(output / "operator.json");
    Mat t = T, w = W;
    std::vector<Mat> masses;
    Json reduction;
    if (order) {
        auto quad = Integrate(order);
        Mat const v = Nodal(quad.Points, Source.Degree, InverseBasis).V;
        t = v * InverseMass;
        w.resize(Source.E, v.rows());
        double error = 0, moment_error = 0, minj = INFINITY;
        for (int e = 0; e < Source.E; ++e) {
            Vec j = v * InverseMass * Elements[e].Moments;
            Require(j.minCoeff() > 0, "Nonpositive reduced Jacobian");
            minj = std::min(minj, j.minCoeff());
            w.row(e) = (quad.Weights.array() / j.array()).matrix();
            Mat const a = Symmetric(t.transpose() * w.row(e).asDiagonal() * t);
            Mat h = SpdInverse(a), chol = Elements[e].H.llt().matrixL();
            Vec spectrum = Spectrum(chol.transpose() * a * chol);
            error = std::max(error, (spectrum.array() - 1).abs().maxCoeff());
            moment_error = std::max(moment_error, (a * Elements[e].Moments - Vec::Ones(Source.N)).cwiseAbs().maxCoeff());
            masses.push_back(h);
        }
        Require(error < 1e-6 && moment_error < 1e-10, "Reduced mass quadrature failed");
        reduction = {{"spectral_relative_inverse_mass_error", error}, {"physical_constant_moment_max_error", moment_error}, {"minimum_mass_jacobian", minj}, {"accepted", true}};
    } else
        for (const auto &el : Elements) masses.push_back(el.H);
    int ecount = Source.E, n = Source.N, f = Source.F;
    Mat g(ecount * 6 * n, n), l(ecount * 10 * n, f), h(ecount * n, n), moments(ecount, n);
    for (int e = 0; e < ecount; ++e) {
        g.middleRows(e * 6 * n, 6 * n) = Elements[e].WeakG;
        l.middleRows(e * 10 * n, 10 * n) = Elements[e].WeakL;
        h.middleRows(e * n, n) = masses[e];
        moments.row(e) = Elements[e].Moments;
    }
    State initial = Source.Initial;
    initial.col(0).array() -= offset;
    Matrix<float> random = Random(ecount * n, 4, 20260905).cast<float>(), init = initial.cast<float>(), constant = Matrix<float>::Zero(ecount * n, 4);
    constant.col(0).setOnes();
    Json meta = {{"elements", ecount}, {"nodes", n}, {"face_nodes", f}, {"quadrature_points", t.rows()}, {"receivers", Source.Receivers}, {"sound_speed", Source.C}, {"impedance", Source.Z}, {"pressure_offset", offset}, {"rka", Rka}, {"rkb", Rkb}, {"propagation_precision", "float32"}, {"diagnostics_precision", "float64"}, {"layout", "element,node,field; four balanced fields; row-major matrices"}};
    auto write = [&]<class T>(const std::string &name, const Matrix<T> &data, const std::vector<int> &shape) {
        Write<T>(output / (name + ".bin"), {data.data(), size_t(data.size())});
        meta["files"][name] = {{"shape", shape}, {"dtype", std::is_same_v<T, float> ? "float32" : std::is_same_v<T, double> ? "float64" :
                                                                                                                              "int32"},
                               {"sha256", Hash(output / (name + ".bin"))}};
    };
    write("G", Matrix<float>(g.cast<float>()), {ecount, 6 * n, n});
    write("L", Matrix<float>(l.cast<float>()), {ecount, 10 * n, f});
    write("T", Matrix<float>(t.cast<float>()), {int(t.rows()), n});
    write("W", Matrix<float>(w.cast<float>()), {ecount, int(w.cols())});
    write("initial", init, {ecount, n, 4});
    write("modified_mass", h, {ecount, n, n});
    write("physical_mass_weights", moments, {ecount, n});
    auto const indices = [&](const std::string &name, const std::vector<int32_t> &v, int rows, int cols, const std::vector<int> &shape) {
        Matrix<int32_t> const data = Eigen::Map<const Matrix<int32_t>>(v.data(), rows, cols);
        write(name, data, shape);
    };
    indices("vmapM", Source.Vm, ecount, f, {ecount, f});
    indices("vmapP", Source.Vp, ecount, f, {ecount, f});
    indices("boundary", Source.Boundary, ecount, f, {ecount, f});
    indices("receiver_elements", Source.ReceiverElements, Source.Receivers, 1, {Source.Receivers});
    write("receiver_interp", Matrix<float>(Source.ReceiverWeights.cast<float>()), {Source.Receivers, n});
    auto const factored = [&]<class T>(const Matrix<T> &q) {
        Matrix<T> out(q.rows(), 4), tt = t.cast<T>();
        for (int e = 0; e < ecount; ++e) {
            Matrix<T> local = q.middleRows(e * n, n);
            // Factor out the local pressure anchor; dense reference stepping does not use this cancellation.
            local.col(0).array() -= q(e * n, 0);
            Matrix<T> const load = Loads<T>(g.middleRows(e * 6 * n, 6 * n).cast<T>(), l.middleRows(e * 10 * n, 10 * n).cast<T>(), local, Traces(Source, q, e), T(Source.C));
            out.middleRows(e * n, n) = tt.transpose() * w.row(e).cast<T>().asDiagonal() * (tt * load);
        }
        return out;
    };
    Json checks;
    for (const auto &name : {"random", "initial", "constant"}) {
        Matrix<float> const q = std::string(name) == "random" ? random : std::string(name) == "initial" ? init :
                                                                                                          constant;
        State const truth = Rhs(q.cast<double>());
        double const scale = std::max(truth.norm(), Source.C * q.cast<double>().norm());
        Matrix<float> const r32 = factored(q);
        Mat const r64 = factored(Mat(q.cast<double>()));
        double error32 = (r32.cast<double>() - truth).norm() / scale, error64 = (r64 - truth).norm() / scale;
        Require(error32 < 2e-5 && error64 < (order ? 2e-5 : 1e-10), "Factored RHS check failed: " + std::string(name));
        if (std::string(name) == "constant") Require(r32.isZero(0) && r64.isZero(0), "Constant pressure RHS must be exactly zero");
        checks[name] = {{"fp32_rhs_scaled_l2", error32}, {"fp64_rhs_scaled_l2", error64}};
        write(std::string(name) + "_state", q, {ecount, n, 4});
        write(std::string(name) + "_rhs", truth, {ecount, n, 4});
    }
    Mat truth(ecount * n, 4), actual64(ecount * n, 4), actual32(ecount * n, 4);
    Matrix<float> tf = t.cast<float>();
    for (int e = 0; e < ecount; ++e) {
        Mat const load = random.middleRows(e * n, n).cast<double>();
        truth.middleRows(e * n, n) = SpdInverse(Elements[e].H) * load;
        actual64.middleRows(e * n, n) = t.transpose() * w.row(e).asDiagonal() * (t * load);
        actual32.middleRows(e * n, n) = (tf.transpose() * w.row(e).cast<float>().asDiagonal() * (tf * Matrix<float>(random.middleRows(e * n, n)))).cast<double>();
    }
    double error64 = (actual64 - truth).norm() / truth.norm(), error32 = (actual32 - truth).norm() / truth.norm();
    Require(error32 < 2e-5 && error64 < (order ? 1e-6 : 1e-10), "Factored mass check failed");
    checks["mass"] = {{"fp32_relative_l2", error32}, {"fp64_relative_l2", error64}};
    write("mass_load", random, {ecount, n, 4});
    write("mass_result", truth, {ecount, n, 4});
    if (order) {
        meta["mass_quadrature_order"] = order;
        meta["mass_quadrature_check"] = reduction;
    }
    Save(output / "operator.json", meta);
    Save(output / "operator-check.json", checks);
}
namespace {
constexpr double Alpha = 3.8317059702075123156;
struct Mode {
    double Phi;
    Eigen::Vector3d Gradient;
    double K;
};
Mode Cylinder(const Eigen::Vector3d &xyz, double radius, double height) {
    double r = std::hypot(xyz[0], xyz[1]), a = Alpha / radius, b = std::numbers::pi / height, radial = ::j0(a * r), axial = std::cos(b * xyz[2]), ratio = r == 0 ? a / 2 : ::j1(a * r) / r;
    return {radial * axial, {-a * ratio * xyz[0] * axial, -a * ratio * xyz[1] * axial, -b * radial * std::sin(b * xyz[2])}, std::hypot(a, b)};
}
Eigen::Vector4d Exact(const Eigen::Vector3d &x, double time, double radius = 1, double height = 1, double c = 343) {
    auto const m = Cylinder(x, radius, height);
    Eigen::Vector4d q;
    q[0] = 2 + m.Phi * std::cos(c * m.K * time);
    q.tail<3>() = -m.Gradient / m.K * std::sin(c * m.K * time);
    return q;
}
} // namespace
namespace {
std::vector<double> SampleTimes(const fs::path &record) {
    if (fs::exists(record / "times.bin")) return Read<double>(record / "times.bin");
    const auto result = Load(record / "result.json");
    const int steps = result.at("steps");
    const double dt = result.at("dt"), start = result.at("sample_start");
    Require(steps > 0 && dt > 0 && std::isfinite(dt) && std::isfinite(start), "Invalid receiver clocks");
    std::vector<double> times(steps);
    for (int i = 0; i < steps; ++i) times[i] = start + i * dt;
    return times;
}
} // namespace
Json Score(const fs::path &mesh, const fs::path &record, const fs::path &nodes, int order) {
    Mesh m(mesh, nodes);
    Json result = Load(record / "result.json"), geometry = m.Metadata.at("geometry");
    double radius = geometry.at("radius"), height = geometry.at("height"), end = double(result.at("steps")) * double(result.at("dt"));
    if (!order) order = 2 * m.Degree + 5;
    auto quad = Integrate(order);
    auto const b = Nodal(quad.Points, m.Degree, Inverse(Bernstein(m.Nodes, m.Degree).V));
    std::array<State, 2> states{m.Initial, PhysicalState(record / "final_q.bin", m.E, m.N, m.Z)};
    double volume = 0, phi_norm = 0, velocity_norm = 0, minj = INFINITY, k = 0;
    Eigen::Matrix<double, 2, 5> sums = Eigen::Matrix<double, 2, 5>::Zero();
    for (int e = 0; e < m.E; ++e) {
        Mat xyz = m.XYZ.middleRows(e * m.N, m.N), physical = b.V * xyz;
        auto const derivatives = Geometry(b, xyz);
        std::array<Mat, 2> numerical{b.V * states[0].middleRows(e * m.N, m.N), b.V * states[1].middleRows(e * m.N, m.N)};
        for (int q = 0; q < physical.rows(); ++q) {
            double const det = Jacobian(derivatives, q).determinant();
            Require(std::isfinite(det) && det > 0, "Invalid scoring Jacobian");
            minj = std::min(minj, det);
            double const measure = quad.Weights[q] * det;
            auto const mode = Cylinder(physical.row(q), radius, height);
            k = mode.K;
            Eigen::Vector3d const velocity = -mode.Gradient / k;
            volume += measure;
            phi_norm += measure * mode.Phi * mode.Phi;
            velocity_norm += measure * velocity.squaredNorm();
            for (int i = 0; i < 2; ++i) {
                double phase = m.C * k * (i ? end : 0), pt = mode.Phi * std::cos(phase), dp = numerical[i](q, 0) - 2 - pt;
                Eigen::Vector3d vt = velocity * std::sin(phase), vn = numerical[i].row(q).tail(3), dv = vn - vt;
                sums(i, 0) += measure * (dp * dp + dv.squaredNorm());
                sums(i, 1) += measure * (pt * pt + vt.squaredNorm());
                sums(i, 2) += measure * dp * dp;
                sums(i, 3) += measure * (numerical[i](q, 0) - 2) * mode.Phi;
                sums(i, 4) += measure * vn.dot(velocity);
            }
        }
    }
    auto time = SampleTimes(record);
    Require(time.size() == size_t(int(result.at("steps"))) && !time.empty(), "Incorrect receiver clock length");
    for (size_t i = 0; i < time.size(); ++i) Require(std::isfinite(time[i]) && time[i] == i * double(result.at("dt")), "Incorrect receiver clock");
    Mat receivers = ReadMatrix(record / "receivers.bin", m.Receivers, time.size());
    Require(receivers.allFinite(), "Nonfinite receiver record");
    std::vector<double> errors;
    for (int r = 0; r < m.Receivers; ++r) {
        auto const mode = Cylinder(m.ReceiverXYZ.row(r), radius, height);
        double error = 0, norm = 0;
        for (int i = 0; i < int(time.size()); ++i) {
            double truth = mode.Phi * std::cos(m.C * k * time[i]), d = receivers(r, i) - 2 - truth;
            error += d * d;
            norm += truth * truth;
        }
        errors.push_back(std::sqrt(error / norm));
    }
    Require(sums.allFinite() && std::isfinite(volume + phi_norm + velocity_norm), "Nonfinite analytic comparison");
    double a = sums(1, 3) / phi_norm, v = sums(1, 4) / velocity_norm, phase_error = std::remainder(std::atan2(v, a) - m.C * k * end, 2 * std::numbers::pi);
    return {{"geometry", geometry}, {"time", end}, {"scoring_points_per_axis", order}, {"frequency_hz", m.C * k / (2 * std::numbers::pi)}, {"cycles", m.C * k * end / (2 * std::numbers::pi)}, {"initial_relative_acoustic_l2", std::sqrt(sums(0, 0) / sums(0, 1))}, {"terminal_relative_acoustic_l2", std::sqrt(sums(1, 0) / sums(1, 1))}, {"terminal_pressure_error_over_mode_norm", std::sqrt(sums(1, 2) / phi_norm)}, {"receiver_relative_l2", errors}, {"terminal_modal_phase_error_radians", phase_error}, {"terminal_modal_amplitude", std::hypot(a, v)}, {"volume_relative_error", std::abs(volume - std::numbers::pi * radius * radius * height) / (std::numbers::pi * radius * radius * height)}, {"minimum_scoring_jacobian", minj}};
}
Json Compare(const fs::path &reference, const fs::path &candidate, const fs::path &tables, int stride) {
    auto meta = Load(tables / "operator.json");
    int e = meta.at("elements"), n = meta.at("nodes");
    double offset = meta.at("pressure_offset"), z = meta.at("impedance");
    Mat mass = ReadMatrix(tables / "modified_mass.bin", e * n, n);
    State ref = PhysicalState(reference / "final_q.bin", e, n, z), got = PhysicalState(candidate / "final_q.bin", e, n, z);
    ref.col(0).array() -= offset;
    got.col(0).array() -= offset;
    auto const norm = [&](const Mat &q) {double sum=0;for(int i=0;i<e;++i)sum+=(q.middleRows(i*n,n).array()*(mass.middleRows(i*n,n)*q.middleRows(i*n,n)).array()).sum();return sum; };
    auto rt = SampleTimes(reference), ct = SampleTimes(candidate);
    Require(stride > 0 && !rt.empty() && (ct.size() + stride - 1) / stride == rt.size(), "Receiver clocks differ");
    for (size_t i = 0; i < rt.size(); ++i) Require(rt[i] == ct[i * stride], "Receiver clocks differ");
    auto rr = Read<double>(reference / "receivers.bin"), cr = Read<double>(candidate / "receivers.bin");
    Require(rr.size() % rt.size() == 0 && cr.size() == rr.size() / rt.size() * ct.size(), "Receiver sizes differ");
    double err = 0, den = 0;
    for (size_t r = 0; r < rr.size() / rt.size(); ++r)
        for (size_t t = 0; t < rt.size(); ++t) {
            double x = rr[r * rt.size() + t] - offset, y = cr[r * ct.size() + t * stride] - offset;
            Require(std::isfinite(x) && std::isfinite(y), "Nonfinite receiver");
            err += (y - x) * (y - x);
            den += x * x;
        }
    double state_error = std::sqrt(norm(got - ref) / norm(ref)), receiver_error = std::sqrt(err / den);
    Require(std::isfinite(state_error) && std::isfinite(receiver_error), "Invalid precision norm");
    return {{"modified_mass_state_relative_l2", state_error}, {"receiver_fluctuation_relative_l2", receiver_error}};
}
void CheckAnalytic(const fs::path &output) {
    NewDirectory(output);
    Require(std::abs(::j1(Alpha)) < 1e-15, "Incorrect Bessel root");
    std::array<Eigen::Vector3d, 3> const points{{{.2, .3, .4}, {-.3, .1, .2}, {.1, -.1, .6}}};
    for (double const radius : {.5, 1., 2.}) {
        for (const auto &x : points) {
            auto const m = Cylinder(x, radius, 1);
            double lap = 0, h = 1e-4;
            for (int a = 0; a < 3; ++a) {
                Eigen::Vector3d const dx = Eigen::Vector3d::Unit(a) * h;
                lap += (Cylinder(x + dx, radius, 1).Phi - 2 * m.Phi + Cylinder(x - dx, radius, 1).Phi) / (h * h);
            }
            Require(std::abs(lap + m.K * m.K * m.Phi) / std::abs(m.K * m.K * m.Phi) < 1e-6, "Helmholtz check failed");
        }
        for (int i = 0; i < 19; ++i) {
            double const theta = 2 * std::numbers::pi * i / 18;
            Eigen::Vector3d normal{std::cos(theta), std::sin(theta), 0}, x{radius * normal[0], radius * normal[1], .37};
            Require(std::abs(Cylinder(x, radius, 1).Gradient.dot(normal)) < 1e-13, "Rigid side wall failed");
        }
        for (double const z : {0., 1.}) Require(std::abs(Cylinder({.2 * radius, .1 * radius, z}, radius, 1).Gradient[2]) < 1e-13, "Rigid cap failed");
    }
    auto axis = Exact({0, 0, .3}, .00031);
    Require(axis.allFinite() && axis.segment<2>(1).isZero(0), "Axis limit failed");
    for (const auto &x : points) {
        double ht = 1e-8, hx = 1e-5, t = .00031;
        Eigen::Vector4d derivative = (Exact(x, t + ht) - Exact(x, t - ht)) / (2 * ht), residual = derivative;
        for (int a = 0; a < 3; ++a) {
            Eigen::Vector3d const dx = Eigen::Vector3d::Unit(a) * hx;
            Eigen::Vector4d spatial = (Exact(x + dx, t) - Exact(x - dx, t)) / (2 * hx);
            residual[0] += 343 * spatial[a + 1];
            residual[a + 1] += 343 * spatial[0];
        }
        Require(residual.norm() / derivative.norm() < 1e-7, "Acoustic PDE check failed");
    }
    auto tables = output / "operator", reference = output / "reference", candidate = output / "candidate";
    for (const auto &p : {tables, reference, candidate}) fs::create_directories(p);
    Save(tables / "operator.json", {{"elements", 1}, {"nodes", 2}, {"pressure_offset", 1000.}, {"impedance", 400.}});
    Write(tables / "modified_mass.bin", std::vector<double>{1, 0, 0, 1});
    for (int i = 0; i < 2; ++i) {
        auto const path = i ? candidate : reference;
        double const gain = i ? 1.01 : 1.;
        Write(path / "final_q.bin", std::vector<double>{1000 + gain, 1000 - gain, gain / 400, -gain / 400, 0, 0, 0, 0});
        Write(path / "receivers.bin", std::vector<double>{1000 + gain, 1000, 1000 - gain, 1000});
        Write(path / "times.bin", std::vector<double>{0, 1, 2, 3});
    }
    auto comparison = Compare(reference, candidate, tables);
    for (const auto &value : comparison) Require(std::abs(double(value) - .01) < 1e-12, "Stationary pressure hid precision error");
    Write(candidate / "times.bin", std::vector<double>{.01, 1.01, 2.01, 3.01});
    bool rejected = false;
    try {
        Compare(reference, candidate, tables);
    } catch (const std::exception &e) { rejected = std::string(e.what()).find("clocks") != std::string::npos; }
    Require(rejected, "Mismatched clocks were accepted");
    fs::remove(candidate / "times.bin");
    Save(candidate / "result.json", {{"steps", 4}, {"dt", 1.}, {"sample_start", 0.}});
    Require(Compare(reference, candidate, tables) == comparison, "Derived clocks changed comparison");
    for (const auto schedule : {std::array<double, 2>{1., .01}, std::array<double, 2>{2., 0.}}) {
        Save(candidate / "result.json", {{"steps", 4}, {"dt", schedule[0]}, {"sample_start", schedule[1]}});
        rejected = false;
        try {
            Compare(reference, candidate, tables);
        } catch (const std::exception &e) { rejected = std::string(e.what()).find("clocks") != std::string::npos; }
        Require(rejected, "Mismatched derived clocks were accepted");
    }
    Save(output / "checks.json", {{"passed", true}, {"precision", comparison}});
}
} // namespace dg::reference
