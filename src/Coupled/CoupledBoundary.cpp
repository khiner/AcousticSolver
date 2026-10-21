#include "CoupledBoundary.h"
#include <Eigen/Cholesky>
#include <Eigen/SparseLU>
#include <chrono>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "AirboxIndexed.h"
extern "C" {
#include "matgen.h"
}
#undef real

namespace coupled {
namespace {
using Vector = Eigen::VectorXd;
using Triplet = Eigen::Triplet<double>;
Sparse Convert(const CSRmatrix *m) {
    return Eigen::Map<const Sparse>(m->nrow, m->ncol, m->rowStart[m->nrow], m->rowStart, m->colIndex, m->values).pruned(0.0);
}
Matrix Store(Tables &t, Sparse m) {
    m.prune(0.0);
    m.makeCompressed();
    return t.Store(m.rows(), m.outerIndexPtr(), m.innerIndexPtr(), m.valuePtr());
}
Sparse PlateOperator(int nx, int ny, int bc, double nu, Sparse *laplacian = nullptr) {
    CSRmatrix *b, *d, *d2, *d3, *dxy, *dx, *dy;
    double *mass;
    plateMatGen(nx, ny, bc, nu, &b, &d, &d2, &d3, &dxy, &mass, nullptr, nullptr, &dx, &dy);
    Sparse const result = Convert(b);
    if (laplacian) *laplacian = Convert(d);
    for (auto *m : {b, d, d2, d3, dxy, dx, dy}) CSR_free(m);
    free(mass);
    return result;
}
} // namespace

BoundaryOperators BuildBoundaryOperators(int nx, int ny, double h, double nu, const std::string &boundary) {
    const int bc = boundary == "clamped" ? 1 : boundary == "simply_supported" ? 2 :
        boundary == "free"                                                    ? 3 :
                                                                                0;
    if (!bc || nx < 4 || ny < 4 || !(h > 0)) throw std::invalid_argument("Invalid rectangular plate boundary/grid");
    const int n = (nx + 1) * (ny + 1);
    BoundaryOperators result;
    Vector active(n), inverseMass(n);
    for (int x = 0; x <= nx; ++x)
        for (int y = 0; y <= ny; ++y) {
            const bool edgeX = x == 0 || x == nx, edgeY = y == 0 || y == ny;
            active[x * (ny + 1) + y] = !(edgeX || edgeY);
            inverseMass[x * (ny + 1) + y] = bc == 3 ? (edgeX ? 2 : 1) * (edgeY ? 2 : 1) : active[x * (ny + 1) + y];
        }
    result.Bending = PlateOperator(nx, ny, bc, nu, &result.Laplacian) / std::pow(h, 4);
    result.Laplacian /= h * h;
    if (bc != 3) {
        result.Bending = active.asDiagonal() * result.Bending * active.asDiagonal();
        result.Laplacian = active.asDiagonal() * result.Laplacian * active.asDiagonal();
    }
    // Traction-free in-plane edges admit the Airy gauge Phi = normal derivative Phi = 0.
    result.Airy = active.asDiagonal() * PlateOperator(nx, ny, 1, 0) * active.asDiagonal();
    for (int i = 0; i < n; ++i)
        if (!active[i]) result.Airy.coeffRef(i, i) = 1;
    result.Airy /= std::pow(h, 4);
    result.InverseMass = inverseMass;
    return result;
}

Sparse ReadMatrix(const Tables &t, Matrix m, int columns) {
    std::vector<Triplet> entries;
    for (uint32_t i = 0; i < m.Rows; ++i)
        for (uint32_t j = t.Indices[m.Row + i]; j < t.Indices[m.Row + i + 1]; ++j)
            entries.emplace_back(i, t.Indices[m.Column + j], t.Values[m.Value + j]);
    Sparse result(m.Rows, columns);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}

void PrepareBoundary(Tables &t, Plate &p, const nlohmann::json &item) {
    const int nx = p.Size / p.Stride + 1, ny = p.Stride + 1, n = (nx + 1) * (ny + 1);
    if (n > 4096) throw std::invalid_argument("Panel exceeds 4096 grid points including boundary nodes");
    const std::string boundary = item.at("boundary");
    const double nu = item.value("poisson_ratio", .3), rhoH = item.value("density", 7800.) * item.value("thickness", .001);
    const double young = item.value("young_modulus", 2e11), thickness = item.value("thickness", .001);
    const double tension = item.value("tension", 0.), dt = 1 / t.SampleRate;
    const double sigma1 = t.Lossless ? 0 : item.value("sigma1", .001);
    const double a0 = t.Lossless ? 0 : 6 * std::numbers::ln10 * dt / (item.value("t60", 4.) * rhoH);
    const double rigidity = young * thickness * thickness / (12 * (1 - nu * nu) * item.value("density", 7800.));
    auto ops = BuildBoundaryOperators(nx, ny, p.H, nu, boundary);
    Vector const active = (ops.InverseMass.array() != 0).cast<double>();
    Sparse identity(n, n);
    identity.setIdentity();
    const Sparse damping = (2 * sigma1 * dt / (tension > 0 ? 1 : rhoH)) * ops.Laplacian;
    p.B = Store(t, (active.asDiagonal() * (2 * identity) - dt * dt * rigidity * ops.Bending + dt * dt * tension / rhoH * ops.Laplacian + damping) / (1 + a0));
    p.C = Store(t, (active.asDiagonal() * ((1 - a0) * identity) + damping) / (1 + a0));
    p.K = Store(t, ops.Airy);
    p.Size = n;
    p.Stride = ny + 1;
    p.Adjoint = true;
    p.Nonlinear = item.value("nonlinear", true);
    p.MassInverse = t.Append(ops.InverseMass.data(), n);
    p.Alpha = dt * dt / ((1 + a0) * rhoH * p.H * p.H);
    p.FScale = .5 * dt * dt / ((1 + a0) * rhoH);
    p.WScale = young * thickness;
    std::vector<double> derivatives(54 * size_t(n)), mask(9 * size_t(n));
    const double h2 = 1 / (p.H * p.H);
    for (int x = 1; x < nx; ++x)
        for (int y = 1; y < ny; ++y) {
            const int row = x * (ny + 1) + y;
            const std::array<std::array<double, 9>, 6> stencils{{{0, 0, 0, 1, -2, 1, 0, 0, 0}, {0, 1, 0, 0, -2, 0, 0, 1, 0}, {1, -1, 0, -1, 1, 0, 0, 0, 0}, {0, 1, -1, 0, -1, 1, 0, 0, 0}, {0, 0, 0, 1, -1, 0, -1, 1, 0}, {0, 0, 0, 0, 1, -1, 0, -1, 1}}};
            for (int d = 0; d < 6; ++d)
                for (int j = 0; j < 9; ++j) derivatives[9 * (d * n + row) + j] = h2 * stencils[d][j];
            std::fill_n(mask.begin() + 9 * row, 9, 1.);
        }
    p.Derivatives = t.Append(derivatives.data(), derivatives.size());
    p.Mask = t.Append(mask.data(), mask.size());
    if (p.Nonlinear) {
        Eigen::LLT<Eigen::MatrixXd> const factor{Eigen::MatrixXd(ops.Airy)};
        if (factor.info() != Eigen::Success) throw std::runtime_error("Airy boundary operator is not positive definite");
        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inverse = factor.solve(Eigen::MatrixXd::Identity(n, n));
        p.Inverse = t.Append(inverse.data(), inverse.size());
    }
    const auto center = item.at("center").get<std::array<double, 3>>();
    const int slice = t.Nx * t.Ny;
    std::vector<Triplet> interpolation;
    p.AirMask.assign(slice, 0);
    const double x0 = center[0] - nx * p.H / 2, y0 = center[1] - ny * p.H / 2;
    for (uint32_t x = 0; x < t.Nx; ++x)
        for (uint32_t y = 0; y < t.Ny; ++y) {
            const double px = (x - .5 * (t.Nx - 1)) * t.Q, py = (y - .5 * (t.Ny - 1)) * t.Q;
            const double gx = (px - x0) / p.H, gy = (py - y0) / p.H;
            if (gx < 0 || gx > nx || gy < 0 || gy > ny) continue;
            const int ix = std::min(int(gx), nx - 1), iy = std::min(int(gy), ny - 1), row = x * t.Ny + y;
            const double ax = gx - ix, ay = gy - iy;
            p.AirMask[row] = 1;
            for (int dx = 0; dx < 2; ++dx)
                for (int dy = 0; dy < 2; ++dy)
                    interpolation.emplace_back(row, (ix + dx) * (ny + 1) + iy + dy, (dx ? ax : 1 - ax) * (dy ? ay : 1 - ay));
        }
    Sparse j(slice, n);
    j.setFromTriplets(interpolation.begin(), interpolation.end());
    // The mass-weighted transpose makes plate and air interface work equal and opposite.
    const Sparse i = (t.Q * t.Q / (p.H * p.H)) * ops.InverseMass.asDiagonal() * Sparse(j.transpose());
    p.Bf = dt * t.AirDensity / (2 * (1 + a0) * rhoH);
    const double coupling = t.Q * t.SampleRate * t.Gamma * t.Gamma * p.Bf;
    p.IJ = Store(t, coupling * i * j);
    p.I = Store(t, p.Bf * i);
    p.J = Store(t, j);
}

Sparse NonlinearOperator(const Tables &t, const Plate &p, const Vector &w) {
    std::vector<Triplet> entries;
    const int n = p.Size, stride = p.Stride;
    for (int row = 0; row < n; ++row) {
        double d[6]{};
        for (int k = 0; k < 6; ++k)
            for (int j = 0; j < 9; ++j) {
                const int col = row + (j / 3 - 1) * stride + j % 3 - 1;
                if (col >= 0 && col < n) d[k] += t.Values[p.Derivatives + 9 * (k * n + row) + j] * w[col];
            }
        const double a[]{-.5 * d[2], d[0] + .5 * (d[2] - d[3]), .5 * d[3], d[1] + .5 * (d[2] - d[4]), -2 * (d[0] + d[1]) - .5 * (d[2] - d[3] - d[4] + d[5]), d[1] + .5 * (d[5] - d[3]), .5 * d[4], d[0] + .5 * (d[5] - d[4]), -.5 * d[5]};
        for (int j = 0; j < 9; ++j) {
            const int col = row + (j / 3 - 1) * stride + j % 3 - 1;
            const double value = a[j] * p.InvH2 * t.Values[p.Mask + 9 * row + j];
            if (value && col >= 0 && col < n) entries.emplace_back(row, col, value);
        }
    }
    Sparse result(n, n);
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
}

Trace RunBoundaryReference(const Tables &t, AirboxIndexed &air) {
    struct State {
        Sparse B, C, K, IJ, I, J;
        Vector W, U1, U2, Phi, Phi1, Phi0, Transfer, Difference, PreviousDifference, Sum;
    };
    const int slice = t.Nx * t.Ny;
    std::vector<State> states;
    for (const auto &p : t.Plates) {
        const int n = p.Size;
        states.push_back({ReadMatrix(t, p.B, n), ReadMatrix(t, p.C, n), p.Nonlinear ? ReadMatrix(t, p.K, n) : Sparse(n, n), ReadMatrix(t, p.IJ, n), ReadMatrix(t, p.I, slice), ReadMatrix(t, p.J, n), Vector::Zero(n), Vector::Zero(n), Vector::Zero(n), Vector::Zero(n), Vector::Zero(n), Vector::Zero(n), Vector::Zero(n), Vector::Zero(slice), Vector::Zero(slice), Vector::Zero(slice)});
    }
    Trace trace;
    trace.Samples.reserve(size_t(t.Steps) * t.Receivers.size());
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t step = 0; step < t.Steps; ++step) {
        air.runTimestep(step);
        for (size_t index = 0; index < states.size(); ++index) {
            auto &s = states[index];
            const auto &p = t.Plates[index];
            Vector aa = s.Transfer - s.C * s.U2 + s.B * s.U1 + s.IJ * s.U2;
            Sparse l, f;
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solve;
            if (p.Nonlinear) {
                l = NonlinearOperator(t, p, s.U1);
                f = p.Adjoint ? Sparse(Eigen::Map<const Vector>(t.Values.data() + p.MassInverse, p.Size).asDiagonal() * Sparse(l.transpose())) : l;
                f *= p.FScale;
                aa += f * s.Phi0;
                solve.compute(s.K + Sparse(p.WScale * l * f));
                if (solve.info() != Eigen::Success) throw std::runtime_error("FP64 plate factorization failed");
            }
            for (const auto &strike : t.Strikes)
                if (strike.Plate == index && step >= strike.Start && step - strike.Start < strike.Length) {
                    const double mass = p.Adjoint ? t.Values[p.MassInverse + strike.Index] : 1;
                    aa[strike.Index] += mass * p.Alpha * .5 * strike.Amplitude * (1 - std::cos(2 * std::numbers::pi * (step - strike.Start) / strike.Length));
                }
            if (!p.Nonlinear) s.W = aa;
            for (uint32_t iteration = 0; iteration < p.Iterations; ++iteration) {
                Vector const right = aa - s.IJ * s.W;
                if (p.Nonlinear) {
                    s.Phi = solve.solve(-s.K * s.Phi1 - p.WScale * l * right);
                    if (solve.info() != Eigen::Success) throw std::runtime_error("FP64 plate solve failed");
                    s.W = right + f * s.Phi;
                } else s.W = right;
            }
            s.Transfer = s.W - s.U2;
        }
        for (size_t index = 0; index < states.size(); ++index) {
            auto &s = states[index];
            const auto &p = t.Plates[index];
            const Vector mask = Eigen::Map<const Vector>(p.AirMask.data(), slice);
            const Vector star = -(mask.array() * s.PreviousDifference.array()).matrix() + t.Q * t.SampleRate * (s.J * s.Transfer);
            s.Difference += t.Gamma * t.Gamma * ((mask.array() * s.PreviousDifference.array()).matrix() - star);
            double *down = air.getU1() + p.LowerSlice * slice, *up = down + slice;
            for (int i = 0; i < slice; ++i)
                if (mask[i] > .5) {
                    down[i] = .5 * (s.Sum[i] + s.Difference[i]);
                    up[i] = .5 * (s.Sum[i] - s.Difference[i]);
                }
            air.runPartialUpdate(p.LowerSlice - 1, 4);
            const Vector nextDown = Eigen::Map<const Vector>(air.getU() + p.LowerSlice * slice, slice);
            const Vector nextUp = Eigen::Map<const Vector>(air.getU() + (p.LowerSlice + 1) * slice, slice);
            const Vector next = nextDown - nextUp + t.Lambda * t.Lambda * (star - s.PreviousDifference);
            s.Sum = nextDown + nextUp;
            s.Transfer = s.I * (next + 2 * t.Gamma * t.Gamma * s.Difference - s.PreviousDifference);
            s.PreviousDifference = s.Difference;
            s.Difference = next;
        }
        for (const auto &r : t.Receivers) {
            const double sample = r.Plate < 0 ? t.PressureScale * (air.getU()[r.Index] - air.getU2()[r.Index]) : states[r.Plate].W[r.Index];
            if (!std::isfinite(sample)) throw std::runtime_error("FP64 boundary reference became nonfinite");
            trace.Samples.push_back(sample);
        }
        air.swapBuffers(step);
        for (auto &s : states) {
            s.U2 = s.U1;
            s.U1 = s.W;
            s.Phi0 = s.Phi1;
            s.Phi1 = s.Phi;
        }
    }
    trace.Seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    trace.Air.assign(air.getU1(), air.getU1() + air.getStateSize());
    for (auto &s : states) trace.Displacements.insert(trace.Displacements.end(), s.U1.data(), s.U1.data() + s.U1.size());
    return trace;
}
} // namespace coupled
