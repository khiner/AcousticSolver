#include "Cqm.h"

#include "Parallel.h"
#include "RadiationMesh.h"

#include <Eigen/Geometry>

#define ACCELERATE_NEW_LAPACK // the modern interface, as everywhere else this links Accelerate
#include <Accelerate/Accelerate.h>

#include <bit>
#include <cassert>
#include <cmath>
#include <numbers>
#include <stdexcept>

using Eigen::Vector3d;

namespace {
// BDF2 characteristic polynomial gamma(z) = (1 - z) + (1 - z)^2 / 2.
Complex Gamma(Complex z) {
    const Complex one_minus = 1. - z;
    return one_minus + 0.5 * one_minus * one_minus;
}

// 8-point Gauss-Legendre on [0, 1], for the analytic-in-r self integral's edge parameter.
constexpr double GaussU[8]{0.01985507175123188, 0.10166676129318664, 0.2372337950418355, 0.4082826787521751, 0.5917173212478249, 0.7627662049581645, 0.8983332387068134, 0.9801449282487681};
constexpr double GaussW[8]{0.05061426814518813, 0.11119051722668724, 0.15685332293894364, 0.18134189168918097, 0.18134189168918097, 0.15685332293894364, 0.11119051722668724, 0.05061426814518813};

struct QuadPoint {
    Vector3d Y;
    double W; // includes the area factor
};

// Source-side quadrature: base 1-point (centroid) or 3-point (edge midpoints, degree 2),
// refined by 4-way subdivision while the sub-element is large relative to its distance from
// the target. The paper's fixed 1/3-point policy spikes on those near-singular pairs.
void AppendQuad(std::vector<QuadPoint> &out, const Vector3d &x, const Vector3d &a, const Vector3d &b, const Vector3d &c, double area, const QuadPolicy &policy, int depth) {
    const Vector3d centroid = (a + b + c) / 3.;
    const double max_edge = std::max({(b - a).norm(), (c - b).norm(), (a - c).norm()});
    if (depth < policy.MaxDepth && max_edge > policy.AdaptiveRatio * (centroid - x).norm()) {
        const Vector3d ab = 0.5 * (a + b), bc = 0.5 * (b + c), ca = 0.5 * (c + a);
        const double quarter = 0.25 * area;
        AppendQuad(out, x, a, ab, ca, quarter, policy, depth + 1);
        AppendQuad(out, x, b, bc, ab, quarter, policy, depth + 1);
        AppendQuad(out, x, c, ca, bc, quarter, policy, depth + 1);
        AppendQuad(out, x, ab, bc, ca, quarter, policy, depth + 1);
        return;
    }
    if (policy.BaseOrder >= 3 && depth == 0) {
        out.push_back({0.5 * (a + b), area / 3.});
        out.push_back({0.5 * (b + c), area / 3.});
        out.push_back({0.5 * (c + a), area / 3.});
    } else {
        out.push_back({centroid, area});
    }
}

// Assigns rather than returns, so `out` keeps its capacity across a worker's pairs.
void TrimInto(const std::vector<double> &w, int begin, double trim_eps, LagSeries &out) {
    double max_abs = 0.;
    for (double const v : w) max_abs = std::max(max_abs, std::abs(v));
    const double cut = trim_eps * max_abs;
    int lo = 0, hi = int(w.size());
    while (lo < hi && std::abs(w[lo]) < cut) ++lo;
    while (hi > lo && std::abs(w[hi - 1]) < cut) --hi;
    out.Begin = begin + lo;
    out.W.assign(w.begin() + lo, w.begin() + hi);
}

void TrimLagsInto(const CqmContour &contour, SplitSamples &samples, int begin, int end, double trim_eps, LagSeries &out) {
    TrimInto(contour.LagsFromSamples(samples, begin, end), begin, trim_eps, out);
}
} // namespace

CqmContour::CqmContour(double tau, double c, int n_contour)
    : Tau(tau), C(c), N(n_contour), Lambda(std::pow(1e-15, 1. / (2. * N))) {
    assert(N > 0 && (N & (N - 1)) == 0);
    S.resize(N);
    SRe.resize(N);
    SIm.resize(N);
    LambdaPow.resize(N);
    for (int l = 0; l < N; ++l) {
        const double angle = 2. * std::numbers::pi * l / N;
        S[l] = Gamma(Lambda * Complex{std::cos(angle), std::sin(angle)}) / Tau;
        SRe[l] = S[l].real();
        SIm[l] = S[l].imag();
        LambdaPow[l] = std::pow(Lambda, -double(l));
    }
    Log2N = std::countr_zero(unsigned(N));
    Fft = vDSP_create_fftsetupD(vDSP_Length(Log2N), kFFTRadix2);
    if (!Fft) throw std::runtime_error("vDSP FFT setup failed");
}

CqmContour::~CqmContour() {
    if (Fft) vDSP_destroy_fftsetupD(static_cast<FFTSetupD>(Fft));
}

std::vector<double> CqmContour::LagsFromSamples(SplitSamples &samples, int begin_lag, int end_lag) const {
    assert(int(samples.Re.size()) == N && begin_lag >= 0 && end_lag <= N);
    // vDSP's forward transform is sum_l x_l e^(-2 pi i l j / N), which is the sum the
    // weights are defined by.
    DSPDoubleSplitComplex const split{samples.Re.data(), samples.Im.data()};
    vDSP_fft_zipD(static_cast<FFTSetupD>(Fft), &split, 1, vDSP_Length(Log2N), kFFTDirection_Forward);
    std::vector<double> w(end_lag - begin_lag);
    for (int j = begin_lag; j < end_lag; ++j) w[j - begin_lag] = LambdaPow[j] * samples.Re[j] / N;
    return w;
}

std::vector<double> CqmContour::LagsFromSamples(const std::vector<Complex> &kernel_samples, int begin_lag, int end_lag) const {
    SplitSamples split{N};
    for (int l = 0; l < N; ++l) {
        split.Re[l] = kernel_samples[l].real();
        split.Im[l] = kernel_samples[l].imag();
    }
    return LagsFromSamples(split, begin_lag, end_lag);
}

void CqmKernelTable::Build(const CqmContour &contour, double theta_max) {
    // A node every 1/256 of a timestep of delay. The kernels vary on a scale of one lag, so
    // linear interpolation at that spacing sits near 1e-5 relative, far under the precision
    // the weights end up stored in, and it costs table bytes rather than per-pair work.
    // RadiationTest's self-test measures the deviation it actually leaves.
    constexpr double NodesPerLag = 256.;
    ThetaMax = std::max(0., theta_max);
    Step = 1. / NodesPerLag;
    // Every covered pair's window ends by ceil(theta) + 64 (the retardation window
    // ComputePairSeries takes below), so rows reaching that far drop nothing.
    LagCount = std::min(contour.N, int(std::ceil(ThetaMax)) + 65);
    const int nodes = int(std::ceil(ThetaMax * NodesPerLag)) + 2;
    Psi.assign(size_t(nodes) * LagCount, 0.f);
    Chi.assign(size_t(nodes) * LagCount, 0.f);
    ParallelFor(nodes, 1, [&](size_t k) {
        const double theta = double(k) * Step;
        SplitSamples psi{contour.N}, chi{contour.N};
        for (int l = 0; l < contour.N; ++l) {
            const Complex gamma = contour.S[l] * contour.Tau;
            const Complex e = std::exp(-gamma * theta), ge = gamma * e;
            psi.Re[l] = e.real();
            psi.Im[l] = e.imag();
            chi.Re[l] = ge.real();
            chi.Im[l] = ge.imag();
        }
        const auto p = contour.LagsFromSamples(psi, 0, LagCount);
        const auto c = contour.LagsFromSamples(chi, 0, LagCount);
        for (int j = 0; j < LagCount; ++j) {
            Psi[k * LagCount + j] = float(p[j]);
            Chi[k * LagCount + j] = float(c[j]);
        }
    });
}

void ComputePairSeries(const CqmContour &contour, const Vector3d &x, const RadiationMesh &mesh, int elem, bool is_self, const QuadPolicy &policy, PairSeries &out) {
    // A reused series must start empty: the self branch below fills only the single layer.
    out.D.Begin = out.V.Begin = 0;
    out.D.W.clear();
    out.V.W.clear();

    const Vector3d a = mesh.Vertices[mesh.Tris[elem][0]], b = mesh.Vertices[mesh.Tris[elem][1]],
                   c = mesh.Vertices[mesh.Tris[elem][2]];
    const Vector3d normal = mesh.Normals[elem];
    const double inv_c = 1. / contour.C;
    const int n = contour.N;

    if (is_self) {
        // Single layer over the element from its own centroid: split at the centroid into
        // three sub-triangles and integrate radially — the in-plane polar Jacobian cancels
        // the 1/r, leaving a closed form in r per angular node:
        //   integral = 1/(4 pi) * sum_sub 2 S_sub * int_0^1 (c/s) (1 - e^(-s r(u)/c)) / r(u)^2 du.
        // The double layer vanishes on a flat element.
        SplitSamples v_samples{n};
        const Vector3d verts[3]{a, b, c};
        double two_s[3], r_edge[3][8];
        for (int s = 0; s < 3; ++s) {
            const Vector3d va = verts[s], vb = verts[(s + 1) % 3];
            two_s[s] = (va - x).cross(vb - x).norm();
            for (int g = 0; g < 8; ++g) r_edge[s][g] = (va + GaussU[g] * (vb - va) - x).norm();
        }
        for (int l = 0; l < n; ++l) {
            const Complex s = contour.S[l];
            Complex acc = 0.;
            for (int sub = 0; sub < 3; ++sub) {
                Complex edge = 0.;
                for (int g = 0; g < 8; ++g) {
                    const double r = r_edge[sub][g];
                    edge += GaussW[g] * (1. - std::exp(-s * r * inv_c)) / (r * r);
                }
                acc += two_s[sub] * edge;
            }
            const Complex v = acc * (contour.C / s) / (4. * std::numbers::pi);
            v_samples.Re[l] = v.real();
            v_samples.Im[l] = v.imag();
        }
        TrimLagsInto(contour, v_samples, 0, std::min(n, 80), policy.TrimEps, out.V);
        return;
    }

    // Scratch reused across the millions of pairs one worker builds: allocating a pair's few
    // hundred bytes per pair costs as much as computing its weights.
    static thread_local std::vector<QuadPoint> quad;
    quad.clear();
    AppendQuad(quad, x, a, b, c, mesh.Areas[elem], policy, 0);

    // Geometry per quadrature point, shared across contour frequencies.
    struct Geom {
        double W, R, InvR, Dot; // Dot = (n . (y - x)) / r
    };
    static thread_local std::vector<Geom> geom;
    geom.assign(quad.size(), {});
    double r_min = 1e30, r_max = 0.;
    for (size_t q = 0; q < quad.size(); ++q) {
        const Vector3d diff = quad[q].Y - x;
        const double r = diff.norm();
        geom[q] = {quad[q].W, r, 1. / r, normal.dot(diff) / r};
        r_min = std::min(r_min, r);
        r_max = std::max(r_max, r);
    }
    // Lag range from retardation: weights concentrate at j ~ r / (c tau) with a BDF2 tail.
    const double inv_ctau = 1. / (contour.C * contour.Tau);
    const int j_lo = std::max(0, int(r_min * inv_ctau) - 24);
    const int j_hi = std::min(n, int(std::ceil(r_max * inv_ctau)) + 64);

    // The delay kernels, where they reach: each quadrature point's contribution is two
    // interpolated rows scaled by its own geometry, the same sum the transform below makes.
    if (const auto &kernels = contour.Kernels; policy.UseKernels && kernels.Covers(r_max * inv_ctau, j_hi)) {
        const int span = j_hi - j_lo;
        static thread_local std::vector<double> d, v;
        d.assign(span, 0.);
        v.assign(span, 0.);
        for (const auto &gq : geom) {
            const double node = gq.R * inv_ctau / kernels.Step;
            const int k = int(node);
            const double t = node - k, u = 1. - t;
            const float *psi0 = kernels.PsiRow(k), *psi1 = kernels.PsiRow(k + 1);
            const float *chi0 = kernels.ChiRow(k), *chi1 = kernels.ChiRow(k + 1);
            const double coeff = gq.W * gq.InvR / (4. * std::numbers::pi);
            const double v_scale = coeff, d_chi = -gq.Dot * coeff * inv_ctau, d_psi = -gq.Dot * coeff * gq.InvR;
            for (int j = j_lo; j < j_hi; ++j) {
                const double psi = u * psi0[j] + t * psi1[j], chi = u * chi0[j] + t * chi1[j];
                v[j - j_lo] += v_scale * psi;
                d[j - j_lo] += d_chi * chi + d_psi * psi;
            }
        }
        TrimInto(d, j_lo, policy.TrimEps, out.D);
        TrimInto(v, j_lo, policy.TrimEps, out.V);
        return;
    }

    // Sample the kernels at every contour node, vectorized over nodes rather than over
    // quadrature points: the exponential and the sine/cosine are the build's whole cost, so
    // each quadrature point pays three vectorized calls instead of N scalar ones.
    SplitSamples v_samples{n}, d_samples{n};
    std::vector<double> arg(n), mag(n), cosv(n), sinv(n);
    for (const auto &gq : geom) {
        const double a = -gq.R * inv_c, coeff = gq.W * gq.InvR / (4. * std::numbers::pi);
        for (int l = 0; l < n; ++l) arg[l] = contour.SRe[l] * a;
        vvexp(mag.data(), arg.data(), &n);
        for (int l = 0; l < n; ++l) arg[l] = contour.SIm[l] * a;
        vvsincos(sinv.data(), cosv.data(), arg.data(), &n);
        for (int l = 0; l < n; ++l) {
            const double gre = coeff * mag[l] * cosv[l], gim = coeff * mag[l] * sinv[l];
            v_samples.Re[l] += gre;
            v_samples.Im[l] += gim;
            // -(s / c + 1 / r) * green * (n . (y - x)) / r
            const double fre = (contour.SRe[l] * inv_c + gq.InvR) * gq.Dot, fim = contour.SIm[l] * inv_c * gq.Dot;
            d_samples.Re[l] -= fre * gre - fim * gim;
            d_samples.Im[l] -= fre * gim + fim * gre;
        }
    }

    TrimLagsInto(contour, d_samples, j_lo, j_hi, policy.TrimEps, out.D);
    TrimLagsInto(contour, v_samples, j_lo, j_hi, policy.TrimEps, out.V);
}
