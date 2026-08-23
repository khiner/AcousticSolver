// Validation harness for the SonicRadiation prototype, mirroring the paper's monopole test
// (Sec. 5.1): a unit monopole inside a closed icosphere whose analytic Neumann data drives
// the solver, judged by SNR against the free-space field on nested bounding boxes
// (2.4x/3.0x/3.6x/4.2x the object's) plus the surface Dirichlet trace.
//
// V2 is gated at six configurations, three varying the source and three the body, ordered by
// what they ask of the surface trace: the paper's centred monopole holds a constant trace
// that the element-constant basis represents exactly, off centre it varies, a dipole changes
// sign, and the ellipsoid, box and L add curvature that varies, then edges and corners, then
// a concavity. See VALIDATION for what each costs.
//
// Ladder:
//   --src-offset displaces the monopole from the body's centre, in body radii
//   --dipole     swaps the monopole for a centred dipole, whose surface trace changes sign
//   --ellipsoid  squashes the body from a sphere to a triaxial ellipsoid
//   --box        replaces it with a box of the same extent: flat faces, edges and corners
//   --lshape     removes a quadrant of that box, adding a reflex edge and a concavity
//                (all sweep flags: the gate runs every configuration it has thresholds for)
//   --gate       every rung below at the settings its thresholds were recorded at,
//                checked rather than printed, exiting nonzero on any miss (~3s). The
//                scene path is not on this ladder — script/ValidateRadiation runs both.
//   --self-test  CQM weight identities (BDF2 stencil, pure delay, DC limits)
//   --v1         pure TDBEM (dense pairs, no grid): isolates CQM + boundary-equation
//                accuracy; --tau-div sweeps the time step to expose O(tau^2) convergence,
//                --full swaps in the dense left-matrix solve to bound the paper's
//                row-diagonal approximation
//   --v2         the full hybrid at --res, reporting grid-sampled and retarded-potential
//                (Eq. 5) SNR per box, the surface SNR, and interior extinction;
//                --check-sets verifies the set-difference tables mid-run
//   --stability  long 30^3 run watching for coupled-scheme growth
//   --move S     dynamic geometry: the same test with the body translating at S m/s,
//                rebuilding the tables every --epoch steps; --rebase adds the far-field
//                re-basing across near-set changes (measured a wash, see RebasePlan)
//   --mesh F     retargets a scene mesh to a grid of --cell metres and reports the element
//                count, closedness, and edge-length spread the solver would see

#include "Parallel.h"
#include "RadiationGpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <string>

using Eigen::Vector3d;
using std::numbers::pi;

namespace {
constexpr double SoundC = 343., Freq = 1000., Domain = 0.7;
constexpr double ObjRadius = Domain / 6. / 2.; // mesh spans 1/6 of the domain
constexpr double RampT = 5. / Freq; // smoothstep onset over 5 periods

double Env(double t) {
    if (t >= RampT) return 1.;
    const double s = t / RampT;
    return s * s * (3. - 2. * s);
}

// Both analytic sources below are harmonic about X0 and differ only in their Laplace-domain
// amplitude, so the time signal and the ramped Neumann trace are written once over that.
template<typename Src> double Truth(const Src &src, const Vector3d &x, double t) {
    return (src.Amp(x) * std::exp(Complex{0., src.Omega * t})).real();
}
template<typename Src> double Neumann(const Src &src, const Vector3d &x, const Vector3d &n, double t) {
    return Env(t) * (src.DnAmp(x, n) * std::exp(Complex{0., src.Omega * t})).real();
}

// p_hat(x) = e^(-i k r) / (4 pi r) about X0.
//
// The source sits inside the body, so the exterior field is this free-space one and the
// solver's job is to reproduce it from the Neumann trace alone. Where X0 sits decides how
// much of the solver the rung exercises: at the centre of a sphere every element carries
// identical Neumann data, the most symmetric case the element coupling can be asked for.
struct Monopole {
    double K = 2. * pi * Freq / SoundC, Omega = 2. * pi * Freq;
    Vector3d X0{Vector3d::Zero()};

    Complex Amp(const Vector3d &x) const {
        const double r = (x - X0).norm();
        return std::exp(Complex{0., -K * r}) / (4. * pi * r);
    }
    Complex DnAmp(const Vector3d &x, const Vector3d &n) const {
        const Vector3d d = x - X0;
        const double r = d.norm();
        return Amp(x) * Complex{-1. / r, -K} * (n.dot(d) / r);
    }
};

// A point dipole about X0 along D: the monopole differentiated once along D, so its exterior
// field is the same closed form.
//
// What it asks for that no monopole does is a surface trace that changes *sign* across the
// body, which exposes the double-layer series, the window trim and the block scales all at
// once — each of them is relative to what a window sums to, and here that cancels.
struct Dipole {
    double K = 2. * pi * Freq / SoundC, Omega = 2. * pi * Freq;
    Vector3d X0{Vector3d::Zero()};
    Vector3d D{Vector3d{1., 1., 1.}.normalized()};

    // G, and its first and second radial derivatives: G' = G (-1/r - ik), G'' = G ((-1/r - ik)^2 + 1/r^2).
    Complex G(double r) const { return std::exp(Complex{0., -K * r}) / (4. * pi * r); }

    Complex Amp(const Vector3d &x) const {
        const Vector3d r_vec = x - X0;
        const double r = r_vec.norm();
        return G(r) * Complex{-1. / r, -K} * (D.dot(r_vec) / r);
    }
    // d^T (grad grad G) n, which is the dipole's own normal derivative.
    Complex DnAmp(const Vector3d &x, const Vector3d &n) const {
        const Vector3d r_vec = x - X0;
        const double r = r_vec.norm();
        const Complex g = G(r), radial{-1. / r, -K};
        const Complex g1 = g * radial, g2 = g * (radial * radial + 1. / (r * r));
        const double a = D.dot(r_vec) / r, b = n.dot(r_vec) / r, c = D.dot(n);
        return g2 * a * b + (g1 / r) * (c - a * b);
    }
};

struct Snr {
    double Sig{0.}, Err{0.};
    void Add(double truth, double approx) {
        Sig += truth * truth;
        Err += (approx - truth) * (approx - truth);
    }
    double Db() const { return 10. * std::log10(Sig / std::max(Err, 1e-300)); }
};

// The gate. Every threshold is this implementation's own measurement, recorded in
// VALIDATION.md's ladder table. The ladder is deterministic to every digit it prints, so the
// margins are for a future toolchain's rounding rather than for drift: what this catches is
// silent wrong answers, which arrive as tens of percent.
int Failures = 0;

void Fail(const char *what, double got, const char *relation, double want) {
    std::printf("[gate] FAIL %s: %.4g, want %s %.4g\n", what, got, relation, want);
    ++Failures;
}
void AtLeast(const char *what, double got, double least) {
    if (got < least) Fail(what, got, ">=", least);
}
void AtMost(const char *what, double got, double most) {
    if (got > most) Fail(what, got, "<=", most);
}
void Near(const char *what, double got, double want, double tol) {
    if (std::abs(got - want) > tol) Fail(what, got - want, "|error| <=", tol);
}

// Semi-axis ratios of the non-spherical rung, against the sphere's radius. Triaxial enough
// that element areas, aspect ratios and near-set sizes all vary over the body, without
// thinning it so far that the interior source or the near sets get crowded.
constexpr double EllipsoidAxes[3]{1., 0.7, 0.5};

// The bodies the rungs run on. Sphere is the paper's; the rest hold its extent and change
// what the surface is made of.
enum class Body { Sphere,
                  Ellipsoid,
                  Box,
                  LShape };

Vector3d BodyAxes(Body body) {
    if (body == Body::Sphere) return Vector3d::Ones();
    return Vector3d{EllipsoidAxes[0], EllipsoidAxes[1], EllipsoidAxes[2]};
}

// Where inside the body the source goes. The L keeps one arm of an extruded L, so its
// bounding-box centre sits on the reflex edge and the source moves into the arm instead.
Vector3d BodyCentre(Body body, const Vector3d &half) {
    return body == Body::LShape ? Vector3d{-0.5 * half.x(), 0., 0.} : Vector3d::Zero();
}

// Total solid angle a closed mesh subtends at `p`: 4pi from strictly inside, 0 from outside
// (Van Oosterom & Strackee). Every analytic rung rests on the source being inside the body,
// and this degenerates on a wrongly wound or unclosed mesh too, so it checks both at once.
double SolidAngleAt(const RadiationMesh &mesh, const Vector3d &p) {
    double total = 0.;
    for (const auto &tri : mesh.Tris) {
        const Vector3d a = mesh.Vertices[tri[0]] - p, b = mesh.Vertices[tri[1]] - p, c = mesh.Vertices[tri[2]] - p;
        const double la = a.norm(), lb = b.norm(), lc = c.norm();
        total += 2. * std::atan2(a.dot(b.cross(c)), la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb);
    }
    return std::abs(total);
}

// A closed axis-aligned solid, from a lattice of cells each either in the body or not: every
// face where a solid cell meets an empty one becomes two triangles wound outward, and lattice
// points are shared, so the result is edge-manifold. `notch` removes one quadrant across the
// full depth, turning the box into an extruded L with a reflex edge down its middle.
//
// Generated at the resolution it runs at rather than remeshed towards it: isotropic remeshing
// moves vertices tangentially, which would round off the edges and corners these bodies exist
// to provide.
RadiationMesh SolidMesh(const Vector3d &half, double target, bool notch) {
    const int n[3]{std::max(2, int(std::lround(2. * half.x() / target))), std::max(2, int(std::lround(2. * half.y() / target))), std::max(2, int(std::lround(2. * half.z() / target)))};
    const auto solid = [&](int i, int j, int k) {
        if (i < 0 || j < 0 || k < 0 || i >= n[0] || j >= n[1] || k >= n[2]) return false;
        return !(notch && i >= n[0] / 2 && j >= n[1] / 2);
    };
    RadiationMesh mesh;
    std::vector<int> index(size_t(n[0] + 1) * (n[1] + 1) * (n[2] + 1), -1);
    const auto vert = [&](int i, int j, int k) {
        int &id = index[(size_t(k) * (n[1] + 1) + j) * (n[0] + 1) + i];
        if (id < 0) {
            id = int(mesh.Vertices.size());
            mesh.Vertices.emplace_back(-half.x() + 2. * half.x() * i / n[0], -half.y() + 2. * half.y() * j / n[1], -half.z() + 2. * half.z() * k / n[2]);
        }
        return id;
    };
    const auto quad = [&](int a, int b, int c, int d) {
        mesh.Tris.emplace_back(a, b, c);
        mesh.Tris.emplace_back(a, c, d);
    };
    for (int k = 0; k < n[2]; ++k) {
        for (int j = 0; j < n[1]; ++j) {
            for (int i = 0; i < n[0]; ++i) {
                if (!solid(i, j, k)) continue;
                if (!solid(i + 1, j, k)) quad(vert(i + 1, j, k), vert(i + 1, j + 1, k), vert(i + 1, j + 1, k + 1), vert(i + 1, j, k + 1));
                if (!solid(i - 1, j, k)) quad(vert(i, j, k), vert(i, j, k + 1), vert(i, j + 1, k + 1), vert(i, j + 1, k));
                if (!solid(i, j + 1, k)) quad(vert(i, j + 1, k), vert(i, j + 1, k + 1), vert(i + 1, j + 1, k + 1), vert(i + 1, j + 1, k));
                if (!solid(i, j - 1, k)) quad(vert(i, j, k), vert(i + 1, j, k), vert(i + 1, j, k + 1), vert(i, j, k + 1));
                if (!solid(i, j, k + 1)) quad(vert(i, j, k + 1), vert(i + 1, j, k + 1), vert(i + 1, j + 1, k + 1), vert(i, j + 1, k + 1));
                if (!solid(i, j, k - 1)) quad(vert(i, j, k), vert(i, j + 1, k), vert(i + 1, j + 1, k), vert(i + 1, j, k));
            }
        }
    }
    mesh.Finalize();
    return mesh;
}

// Face-grid sample points on the paper's four nested boxes, `per_face` x `per_face` each.
std::vector<Vector3d> BoxPoints(int per_face) {
    static constexpr double Factors[4]{2.4, 3.0, 3.6, 4.2};
    std::vector<Vector3d> points;
    for (const double f : Factors) {
        const double half = f * ObjRadius;
        for (int face = 0; face < 6; ++face) {
            const int axis = face / 2;
            const double sign = face % 2 ? 1. : -1.;
            for (int a = 0; a < per_face; ++a) {
                for (int b = 0; b < per_face; ++b) {
                    Vector3d x;
                    x[axis] = sign * half;
                    x[(axis + 1) % 3] = half * (2. * (a + 0.5) / per_face - 1.);
                    x[(axis + 2) % 3] = half * (2. * (b + 0.5) / per_face - 1.);
                    points.push_back(x);
                }
            }
        }
    }
    return points;
}

int PickSubdiv(double h) {
    for (int subdiv = 1; subdiv <= 5; ++subdiv) {
        if (RadiationMesh::Icosphere(Vector3d::Zero(), ObjRadius, subdiv).MaxEdge < 1.05 * h) return subdiv;
    }
    return 5;
}

// Reports what RadiationMesh::Remesh does to a scene mesh at a given cell size: the
// solver needs a closed manifold whose elements fit inside a cell, and the element count
// it lands on sets the size of every table built over it.
void MeshReport(const std::string &path, double cell) {
    const double t0 = Now();
    auto mesh = RadiationMesh::FromObj(path, Vector3d::Zero());
    std::printf("[mesh] %s: %d elements, closed %d, max edge %.2f cells\n", path.c_str(), mesh.NumElems(), int(mesh.IsClosed()), mesh.MaxEdge / cell);
    const double max_edge = 1.05 * cell;
    mesh.Remesh(RadiationMesh::RemeshTarget * max_edge, max_edge);
    std::vector<double> lengths;
    for (const auto &tri : mesh.Tris) {
        for (int i = 0; i < 3; ++i) {
            const int a = tri[i], b = tri[(i + 1) % 3];
            if (a < b) lengths.push_back((mesh.Vertices[a] - mesh.Vertices[b]).norm() / cell);
        }
    }
    std::ranges::sort(lengths);
    const auto pct = [&](double p) { return lengths[size_t(p * (lengths.size() - 1))]; };
    std::printf("[mesh] remeshed: %d elements, closed %d, edges/cell min %.2f p50 %.2f p99 %.2f max %.2f, over "
                "ceiling %zu | %.1fs\n",
                mesh.NumElems(), int(mesh.IsClosed()), lengths.front(), pct(0.5), pct(0.99), lengths.back(), size_t(std::ranges::count_if(lengths, [&](double l) { return l > 1.05; })), Now() - t0);
}

void SelfTest(bool gate) {
    const double tau = 1e-4;
    const CqmContour contour(tau, SoundC);
    const std::vector<Complex> ones(contour.N, 1.), s_samples(contour.S.begin(), contour.S.end());
    std::vector<Complex> delay(contour.N);
    for (int l = 0; l < contour.N; ++l) delay[l] = std::exp(-contour.S[l] * (10. * tau));

    const auto w_one = contour.LagsFromSamples(ones, 0, 6);
    const auto w_s = contour.LagsFromSamples(s_samples, 0, 6);
    const auto w_delay = contour.LagsFromSamples(delay, 0, 40);
    double delay_sum = 0., delay_max = 0.;
    int delay_peak = -1;
    for (int j = 0; j < 40; ++j) {
        delay_sum += w_delay[j];
        if (std::abs(w_delay[j]) > delay_max) {
            delay_max = std::abs(w_delay[j]);
            delay_peak = j;
        }
    }
    std::printf("[self-test] identity w0=%.6f (want 1), w1=%.1e | bdf2*tau (%.3f, %.3f, %.3f) (want 1.5, -2, 0.5) | "
                "delay sum=%.6f peak j=%d (want 1, 10)\n",
                w_one[0], w_one[1], w_s[0] * tau, w_s[1] * tau, w_s[2] * tau, delay_sum, delay_peak);
    if (gate) {
        Near("self-test identity w0", w_one[0], 1., 1e-12);
        Near("self-test identity w1", w_one[1], 0., 1e-12);
        for (const auto &[j, want] : {std::pair{0, 1.5}, {1, -2.}, {2, 0.5}}) Near("self-test bdf2 stencil", w_s[j] * tau, want, 1e-9);
        Near("self-test delay DC gain", delay_sum, 1., 1e-6);
        // The ideal peak is at 10 lags; BDF2's approximation of a pure delay lands one late.
        Near("self-test delay peak lag", delay_peak, 10., 1.);
    }

    // DC limit of a pair series vs the static kernels, well-separated elements.
    const auto mesh = RadiationMesh::Icosphere(Vector3d::Zero(), ObjRadius, 2);
    const Vector3d x = mesh.Centroids[0] + Vector3d{0.05, 0.02, -0.03};
    // Untrimmed: this identity is about the weights themselves, and the default trim drops
    // the tail that carries the last digits of their sum.
    QuadPolicy exact;
    exact.TrimEps = 0.;
    PairSeries pair;
    ComputePairSeries(contour, x, mesh, 7, false, exact, pair);
    double v_sum = 0., d_sum = 0.;
    for (const double w : pair.V.W) v_sum += w;
    for (const double w : pair.D.W) d_sum += w;
    const Vector3d y = mesh.Centroids[7], diff = y - x;
    const double r = diff.norm();
    const double v_ref = mesh.Areas[7] / (4. * pi * r);
    const double d_ref = -mesh.Areas[7] / (4. * pi * r * r) * mesh.Normals[7].dot(diff) / r;
    const double v_rel = std::abs(v_sum - v_ref) / std::abs(v_ref), d_rel = std::abs(d_sum - d_ref) / std::abs(d_ref);
    std::printf("[self-test] pair DC: V %.6e vs static %.6e (rel %.1e) | D %.6e vs static %.6e (rel %.1e)\n", v_sum, v_ref, v_rel, d_sum, d_ref, d_rel);
    if (gate) {
        AtMost("self-test pair DC single-layer rel", v_rel, 1e-12);
        AtMost("self-test pair DC double-layer rel", d_rel, 1e-12);
    }

    // The delay kernels against the per-pair transform they stand in for. Same contour and
    // same window, so the whole difference is the interpolation in theta — the one
    // approximation the table introduces, measured here rather than argued.
    CqmContour tabled(tau, SoundC);
    double theta_max = 0.;
    std::vector<Vector3d> targets(6);
    for (int k = 0; k < 6; ++k) targets[k] = Vector3d{1.4, -0.7, 0.5}.normalized() * (1.2 + 1.4 * k) * ObjRadius;
    for (const auto &t : targets)
        for (int e = 0; e < mesh.NumElems(); ++e) theta_max = std::max(theta_max, ((t - mesh.Centroids[e]).norm() + ObjRadius) / (SoundC * tau));
    tabled.BuildKernels(theta_max);

    double worst = 0.;
    int pairs = 0;
    for (const auto &t : targets) {
        for (int e = 0; e < mesh.NumElems(); ++e) {
            PairSeries ref, got;
            ComputePairSeries(contour, t, mesh, e, false, exact, ref);
            ComputePairSeries(tabled, t, mesh, e, false, exact, got);
            for (const auto &[a, b] : {std::pair{&ref.D, &got.D}, std::pair{&ref.V, &got.V}}) {
                double peak = 0., dev = 0.;
                for (const double w : a->W) peak = std::max(peak, std::abs(w));
                if (a->Begin != b->Begin || a->W.size() != b->W.size()) {
                    dev = peak; // a window mismatch is a failure, not a deviation to average
                } else {
                    for (size_t j = 0; j < a->W.size(); ++j) dev = std::max(dev, std::abs(a->W[j] - b->W[j]));
                }
                worst = std::max(worst, dev / std::max(peak, 1e-300));
            }
            ++pairs;
        }
    }
    std::printf("[self-test] delay kernels vs transform: max dev %.1e of peak over %d pairs, theta to %.1f, table %.1f MB\n", worst, pairs, theta_max, double(tabled.Kernels.Psi.size() + tabled.Kernels.Chi.size()) * sizeof(float) / 1048576.);
    // The interpolation in theta is the one approximation the delay tables introduce, so
    // this is the number that moves if their resolution or window handling changes.
    if (gate) AtMost("self-test delay kernel deviation", worst, 1.5e-5);
}

// The quadrature policy every rung starts from. `quad3` raises the base order, the sweep
// that prices the default against a costlier contour.
QuadPolicy MakePolicy(bool kernels, bool quad3 = false) {
    QuadPolicy policy;
    policy.UseKernels = kernels;
    if (quad3) {
        policy.BaseOrder = 3;
        policy.AdaptiveRatio = 0.3;
    }
    return policy;
}

void RunV1(int subdiv, double tau_div, bool full, bool quad3, double total_time, bool kernels, bool gate) {
    const double t0 = Now();
    const double h30 = Domain / 30.;
    const double tau = h30 / (SoundC * std::numbers::sqrt3) / tau_div;
    auto mesh = RadiationMesh::Icosphere(Vector3d::Zero(), ObjRadius, subdiv);
    const int m = mesh.NumElems();
    const Monopole src;

    const QuadPolicy policy = MakePolicy(kernels, quad3);
    Tdbem bem;
    std::vector<Vector3d> centroids = mesh.Centroids, normals = mesh.Normals;
    bem.Init(std::move(mesh), tau, SoundC, {}, policy);
    if (full) bem.PrepareFullSolve();

    const auto points = BoxPoints(4);
    const auto ptable = bem.BuildPointTable(points);
    bem.EnsureHistLen(std::max(bem.ElemElem.MaxLagEnd(), ptable.MaxLagEnd()), 0);

    const double t_settle = RampT + 0.002, t_meas = 2. / Freq;
    const int steps = int(std::ceil((total_time > 0. ? total_time : t_settle + t_meas) / tau));
    Snr surface, radiated;
    for (int n = 0; n <= steps; ++n) {
        const double t = n * tau;
        for (int e = 0; e < m; ++e) bem.GAt(n, e) = Neumann(src, centroids[e], normals[e], t);
        if (full) bem.SolveDirichletFull(n);
        else bem.SolveDirichletDiag(n, nullptr);
        if (total_time > 0. && n % 250 == 0) {
            double peak = 0.;
            for (int e = 0; e < m; ++e) peak = std::max(peak, std::abs(bem.PhiAt(n, e)));
            std::printf("[v1 long] t=%.1fms max|phi|=%.3e\n", t * 1e3, peak);
        }
        if (t < t_settle || total_time > 0.) continue;
        for (int e = 0; e < m; ++e) surface.Add(Truth(src, centroids[e], t), bem.PhiAt(n, e));
        std::vector<double> vals(points.size());
        ParallelFor(points.size(), 8, [&](size_t p) { vals[p] = bem.EvaluatePoint(ptable, int(p), n); });
        for (size_t p = 0; p < points.size(); ++p) radiated.Add(Truth(src, points[p], t), vals[p]);
    }
    std::printf("[v1%s%s] subdiv=%d M=%d tau=%.2fus steps=%d | surface SNR %.1f dB | radiated SNR %.1f dB | %.1fs\n", full ? " full" : "", quad3 ? " quad3" : "", subdiv, m, tau * 1e6, steps, surface.Db(), radiated.Db(), Now() - t0);
    // The dense left-matrix solve buys surface accuracy and gives back a little radiated,
    // which is the measurement behind keeping the row-diagonal one.
    if (gate) {
        AtLeast(full ? "v1 full surface SNR" : "v1 surface SNR", surface.Db(), full ? 41.2 : 24.8);
        AtLeast(full ? "v1 full radiated SNR" : "v1 radiated SNR", radiated.Db(), full ? 23.1 : 24.4);
    }
}

// `src_offset` displaces the monopole from the body's centre, as a fraction of its radius,
// along a direction off every axis so it lines up with neither the mesh's symmetries nor the
// sampling boxes' faces. `dipole` swaps the source for a centred dipole along that same
// direction, whose surface trace changes sign rather than only varying in amplitude.
struct V2Config {
    int Res{30};
    bool CheckSets{false};
    double TotalTime{0.}; // nonzero runs the long-run stability probe instead of the rung
    bool Quad3{false};
    double Cfl{1.};
    int R1{2}, R2{3};
    double Damp{0.};
    bool Filter{true};
    double Trim{QuadPolicy{}.TrimEps};
    double SrcOffset{0.};
    bool Dipole{false};
    Body B{Body::Sphere};
    bool Kernels{true};
    bool Gate{false};
};

void RunV2(const V2Config &in) {
    const auto [res, check_sets, total_time, quad3, cfl, r1, r2, damp, filter, trim, src_offset, dipole, body, kernels, gate] = in;
    const double t0 = Now();
    const double h = Domain / res;
    const int subdiv = PickSubdiv(h);
    const Vector3d axes = BodyAxes(body);
    const bool solid_body = body == Body::Box || body == Body::LShape;
    RadiationMesh mesh = solid_body ? SolidMesh(axes * ObjRadius, RadiationMesh::RemeshTarget * 1.05 * h, body == Body::LShape) : RadiationMesh::Icosphere(Vector3d::Zero(), ObjRadius, subdiv);
    if (body == Body::Ellipsoid) {
        // Squashing an icosphere only shrinks edges, so the subdivision picked against the
        // grid still clears the ceiling, and the body stays exactly an ellipsoid rather than
        // being remeshed towards one. The analytic truth does not care what shape it is: a
        // source inside any closed surface has the free-space field outside it.
        for (auto &v : mesh.Vertices) v = v.cwiseProduct(axes);
        mesh.Finalize();
    }
    const int m = mesh.NumElems();
    // Both analytic sources, selected per call: one branch in a test harness is cheaper than
    // templating the rung on the source. The offset is scaled by the body's axes, so it sits
    // the same fraction of the way out whatever the shape.
    const Vector3d centre = BodyCentre(body, axes * ObjRadius);
    const Monopole mono{.X0 = centre + axes.cwiseProduct(Vector3d{1., 1., 1.}.normalized()) * (src_offset * ObjRadius)};
    const Dipole dip{.X0 = centre};
    const auto truth = [&](const Vector3d &x, double t) { return dipole ? Truth(dip, x, t) : Truth(mono, x, t); };
    const auto &centroids = mesh.Centroids;
    const auto &normals = mesh.Normals;
    const auto neumann = [&](int e, double t) {
        return dipole ? Neumann(dip, centroids[e], normals[e], t) : Neumann(mono, centroids[e], normals[e], t);
    };

    QuadPolicy policy = MakePolicy(kernels, quad3);
    policy.TrimEps = trim;
    RadiationGrid grid = RadiationGrid::Cubic(res, Domain);
    grid.C = SoundC;
    grid.CflFraction = cfl;
    grid.R1 = r1;
    grid.R2 = r2;
    grid.Damping = damp;
    grid.FilterFeedback = filter;
    grid.Finalize();

    // Sample points: the paper's four nested boxes, then interior probes for the
    // extinction diagnostic (the exterior representation should cancel inside the body).
    std::vector<Vector3d> sample_points = BoxPoints(4);
    const size_t n_box = sample_points.size(), per_box = n_box / 4;
    for (double const frac : {0., 0.25, 0.5}) {
        for (int axis = 0; axis < 3; ++axis) {
            Vector3d x = Vector3d::Zero();
            // Strictly inside whatever the body's shape: the L's arm is the narrow case, its
            // reflex edge sitting half a bounding box from the source.
            x[axis] = frac * ObjRadius * axes[axis] * (body == Body::LShape ? 0.6 : 1.);
            sample_points.emplace_back(centre + x);
        }
    }

    Tdbem bem;
    RadiationTables tables;
    BuildRadiation(grid, mesh, policy, sample_points, bem, tables);
    const auto ptable = bem.BuildPointTable(sample_points);
    bem.EnsureHistLen(ptable.MaxLagEnd(), 0);
    SeedRestState(bem, grid.Tau, neumann);
    const double build_s = Now() - t0;

    RadiationGpu gpu;
    gpu.Init(grid, bem, tables, PackRadiation(bem, tables));
    gpu.SetListeners(int(sample_points.size()), ptable);
    std::printf("[v2 build] tables %.0f MB | band rows %zu entries %zu weights %zu | corr cells %zu refs %zu | "
                "elem entries %zu | hist %d | mean lag window %.1f (trim %.0e)\n",
                gpu.Bytes() / 1048576., tables.CellElem.Begin.size() - 1, tables.CellElem.Entries.size(), tables.CellElem.Weights.size(), tables.CorrCell.size(), tables.CorrRefs.size(), bem.ElemElem.Entries.size(), bem.HistLen, double(tables.CellElem.Weights.size()) / double(2 * tables.CellElem.Entries.size()), trim);
    if (check_sets) {
        const double mismatch = tables.CheckSetAlgebra(grid, bem, 12, 0);
        std::printf("[v2] set-algebra max rel mismatch %.2e (want ~1e-12)\n", mismatch);
        if (gate) AtMost("v2 set-algebra mismatch", mismatch, 1e-12);
    }

    const double t_settle = RampT + 0.006, t_meas = 2. / Freq;
    const double t_end = total_time > 0. ? total_time : t_settle + t_meas;
    const int steps = int(std::ceil(t_end / grid.Tau));
    const int batch = 256;

    Snr surface, grid_all, eq5_all, grid_vs_eq5, grid_box[4], eq5_box[4];
    double interior_rms = 0., exterior_rms = 0., max_abs = 0.;
    std::vector<float> g_rows(size_t(batch) * m);

    for (int n = 0; n < steps;) {
        const int take = std::min(batch, steps - n);
        // Each step appends its successor's Neumann samples, so the row uploaded with
        // step q is the one for q + 1.
        for (int b = 0; b < take; ++b) {
            const double t_next = (n + b + 2) * grid.Tau;
            for (int e = 0; e < m; ++e) g_rows[size_t(b) * m + e] = float(neumann(e, t_next));
        }
        gpu.RunSteps(take, g_rows.data());
        const float *gsamp = gpu.GridSamples(), *esamp = gpu.PointSamples();
        const float *phi = gpu.Phi();

        for (int b = 0; b < take; ++b) {
            const int step_index = n + b + 1;
            const double t = step_index * grid.Tau;
            if (total_time > 0.) {
                if (step_index % std::max(1, steps / 10)) continue;
                const float *pf = gpu.FarField();
                double peak = 0.;
                for (size_t c = 0; c < gpu.NumCells(); ++c) peak = std::max(peak, std::abs(double(pf[c])));
                max_abs = std::max(max_abs, peak);
                std::printf("[stability] t=%.1fms max|p|=%.3e\n", t * 1e3, peak);
                continue;
            }
            if (t < t_settle) continue;
            const int slot = step_index & (gpu.HistLength() - 1);
            for (int e = 0; e < m; ++e) surface.Add(truth(centroids[e], t), phi[size_t(e) * gpu.HistLength() + slot]);
            for (size_t p = 0; p < sample_points.size(); ++p) {
                const double gv = gsamp[size_t(b) * sample_points.size() + p];
                const double ev = esamp[size_t(b) * sample_points.size() + p];
                if (p >= n_box) {
                    interior_rms += gv * gv;
                    continue;
                }
                const double want = truth(sample_points[p], t);
                grid_all.Add(want, gv);
                eq5_all.Add(want, ev);
                grid_vs_eq5.Add(ev, gv);
                grid_box[p / per_box].Add(want, gv);
                eq5_box[p / per_box].Add(want, ev);
                if (p < per_box) exterior_rms += gv * gv;
            }
        }
        n += take;
    }
    if (total_time > 0.) {
        std::printf("[stability] res=%d steps=%d max|p| %.3e | %.1fs\n", res, steps, max_abs, Now() - t0);
        // The coupled scheme's failure is growth, and its other failure is a field that
        // never arrives, so this brackets signal level rather than bounding it from above.
        if (gate) {
            AtLeast("stability max|p|", max_abs, 0.4);
            AtMost("stability max|p|", max_abs, 0.8);
        }
        return;
    }
    char offset_tag[32] = "";
    static constexpr const char *ShapeTags[]{"", " ellipsoid", " box", " lshape"};
    const char *shape_tag = ShapeTags[int(body)];
    if (dipole) std::snprintf(offset_tag, sizeof offset_tag, " dipole");
    else if (src_offset != 0.) std::snprintf(offset_tag, sizeof offset_tag, " src=%.2fR", src_offset);
    std::printf("[v2%s%s%s r1=%d r2=%d] res=%d subdiv=%d M=%d tau=%.2fus steps=%d build=%.1fs | surface SNR %.1f dB | "
                "grid SNR %.1f dB | eq5 SNR %.1f dB | grid-vs-eq5 %.1f dB | extinction %.1f dB | %.1fs\n",
                quad3 ? " quad3" : "", shape_tag, offset_tag, r1, r2, res, subdiv, m, grid.Tau * 1e6, steps, build_s, surface.Db(), grid_all.Db(), eq5_all.Db(), grid_vs_eq5.Db(), 10. * std::log10(interior_rms / std::max(exterior_rms, 1e-300)), Now() - t0);
    for (int b = 0; b < 4; ++b)
        std::printf("[v2]   box %.1fx: grid SNR %.1f dB | eq5 SNR %.1f dB\n", 2.4 + 0.6 * b, grid_box[b].Db(), eq5_box[b].Db());
    if (gate) {
        // A threshold belongs to the configuration it was recorded at, so each rung indexes
        // its own: sphere rungs first, ordered by what the source asks of the surface trace,
        // then the bodies in Body order.
        const int cfg = body != Body::Sphere ? int(body) + 2 : dipole ? 2 :
            src_offset != 0.                                          ? 1 :
                                                                        0;
        static constexpr double SurfaceFloor[6]{24.2, 21.5, 16.3, 18.1, 17.5, 15.6};
        static constexpr double GridAllFloor[6]{35.2, 29.2, 20.2, 33.6, 36.9, 32.5};
        static constexpr double Eq5AllFloor[6]{24.9, 23.6, 18.3, 23.5, 24.2, 23.5};
        static constexpr double CrossFloor[6]{24.6, 24.5, 23.5, 23.8, 23.9, 23.8};
        AtLeast("v2 surface SNR", surface.Db(), SurfaceFloor[cfg]);
        AtLeast("v2 grid SNR", grid_all.Db(), GridAllFloor[cfg]);
        AtLeast("v2 eq5 SNR", eq5_all.Db(), Eq5AllFloor[cfg]);
        AtLeast("v2 grid-vs-eq5 SNR", grid_vs_eq5.Db(), CrossFloor[cfg]);
        // The exterior representation assumes a closed surface, and the solid bodies are
        // built here rather than loaded.
        AtLeast("v2 body closed", mesh.IsClosed() ? 1. : 0., 1.);
        // The analytic truth is only truth while the source is inside the body, and the
        // extinction probes only diagnose anything while they are too. Neither is obvious
        // once the body stops being a sphere at the origin, and neither fails loudly.
        double inside = SolidAngleAt(mesh, mono.X0);
        for (size_t p = n_box; p < sample_points.size(); ++p) inside = std::min(inside, SolidAngleAt(mesh, sample_points[p]));
        AtLeast("v2 source and probes inside the body", inside, 12.); // 4pi inside, 2pi on the surface, 0 outside
        // The exterior representation cancels inside the body, and its failing to is how a
        // wrong sign or a lost double-layer term shows up while the boxes still look fine.
        // Each body cancels differently at this resolution, so each carries its own ceiling
        // rather than widening the one the sphere rungs are held to.
        static constexpr double ExtinctionCeiling[6]{-20., -21.0, -29.4, -16.9, -23.5, -21.7};
        AtMost("v2 extinction", 10. * std::log10(interior_rms / std::max(exterior_rms, 1e-300)), ExtinctionCeiling[cfg]);
        static constexpr double GridFloor[6][4]{{37.9, 32.5, 39.6, 33.3}, {29.8, 28.0, 30.6, 28.4}, {20.5, 19.6, 20.9, 19.6}, {36.3, 31.1, 36.4, 31.8}, {39.1, 34.4, 40.6, 35.5}, {33.4, 31.1, 34.2, 31.6}};
        static constexpr double Eq5Floor[6][4]{{28.3, 25.4, 23.2, 21.4}, {26.1, 24.0, 22.2, 20.6}, {19.5, 18.4, 17.4, 16.4}, {26.4, 23.9, 22.0, 20.4}, {27.2, 24.6, 22.6, 20.9}, {26.1, 23.9, 22.0, 20.4}};
        for (int b = 0; b < 4; ++b) {
            AtLeast("v2 box grid SNR", grid_box[b].Db(), GridFloor[cfg][b]);
            AtLeast("v2 box eq5 SNR", eq5_box[b].Db(), Eq5Floor[cfg][b]);
        }
    }
}

// Dynamic-geometry gate: the monopole test with the body translating, rebuilding the
// geometry tables at each epoch's pose and optionally re-basing the far-field state onto the
// new near sets (see RebasePlan). The source travels with the body, so the analytic field is
// the same monopole about the moving centre and the SNR should hold near the static value.
// A step in max|p| at an epoch boundary is the failure this watches for.
void RunMoving(int res, double speed, int epoch_steps, double total_time, int r1, int r2, bool filter, bool rebase, bool kernels, bool gate) {
    const double t0 = Now();
    const double h = Domain / res;
    const int subdiv = PickSubdiv(h);
    const Monopole src;
    const QuadPolicy policy = MakePolicy(kernels);

    RadiationGrid grid = RadiationGrid::Cubic(res, Domain);
    grid.C = SoundC;
    grid.R1 = r1;
    grid.R2 = r2;
    grid.FilterFeedback = filter;
    grid.Finalize();

    const auto points = BoxPoints(2);
    Vector3d centre = Vector3d::Zero();
    auto mesh = RadiationMesh::Icosphere(centre, ObjRadius, subdiv);
    const int m = mesh.NumElems();

    auto bem = std::make_unique<Tdbem>();
    auto tables = std::make_unique<RadiationTables>();
    BuildRadiation(grid, mesh, policy, points, *bem, *tables);
    auto ptable = bem->BuildPointTable(points);
    // The rings carry across every epoch, so they have to cover the lags of the last pose
    // rather than the first: the body recedes from the fixed sample points as it travels.
    // RunRadiationScene sizes for the same reason, over its animation track; here the travel
    // is the run's.
    bem->EnsureHistLen(ptable.MaxLagEnd() + Tdbem::RecedingLags(speed * total_time, SoundC, grid.Tau), 0);
    SeedRestState(*bem, grid.Tau, [&](int e, double t) { return Neumann(src, mesh.Centroids[e] - centre, mesh.Normals[e], t); });

    RadiationGpu gpu;
    gpu.Init(grid, *bem, *tables, PackRadiation(*bem, *tables));
    gpu.SetListeners(int(points.size()), ptable);

    const double t_settle = RampT + 0.006;
    const int steps = int(std::ceil(total_time / grid.Tau));
    Snr grid_all, eq5_all;
    double max_abs = 0.;
    size_t rebase_cells = 0, rebase_refs = 0;
    int epochs = 0;
    std::vector<float> g_rows(size_t(epoch_steps) * m);
    int step = 0;
    while (step < steps) {
        const int take = std::min(epoch_steps, steps - step);
        for (int b = 0; b < take; ++b) {
            const double t_next = (step + b + 2) * grid.Tau;
            for (int e = 0; e < m; ++e)
                g_rows[size_t(b) * m + e] = float(Neumann(src, mesh.Centroids[e] - centre, mesh.Normals[e], t_next));
        }
        gpu.RunSteps(take, g_rows.data());
        const float *gs = gpu.GridSamples(), *es = gpu.PointSamples();
        for (int b = 0; b < take; ++b) {
            const double t = (step + b + 1) * grid.Tau;
            if (t < t_settle) continue;
            for (size_t p = 0; p < points.size(); ++p) {
                const double truth = Truth(src, points[p] - centre, t);
                grid_all.Add(truth, gs[size_t(b) * points.size() + p]);
                eq5_all.Add(truth, es[size_t(b) * points.size() + p]);
                max_abs = std::max(max_abs, std::abs(double(gs[size_t(b) * points.size() + p])));
            }
        }
        step += take;
        if (step >= steps) break;

        // Geometry epoch: the body advances one epoch's worth of travel.
        const Vector3d next_centre = centre + Vector3d{speed * take * grid.Tau, 0., 0.};
        auto next_mesh = RadiationMesh::Icosphere(next_centre, ObjRadius, subdiv);
        auto next_bem = std::make_unique<Tdbem>();
        auto next_tables = std::make_unique<RadiationTables>();
        BuildRadiation(grid, next_mesh, policy, points, *next_bem, *next_tables);
        // The rings carry across an epoch, so their length has to.
        next_bem->EnsureHistLen(bem->HistLen - 1, 0);
        RebasePlan plan;
        if (rebase) plan = PlanRebase(grid, *bem->Contour, mesh, tables->ElemCell, next_tables->ElemCell, policy);
        rebase_cells += plan.Cells.size();
        rebase_refs += plan.Refs.size();
        ++epochs;
        gpu.ApplyEpoch(plan, *next_bem, *next_tables, PackRadiation(*next_bem, *next_tables));
        ptable = next_bem->BuildPointTable(points);
        gpu.SetListeners(int(points.size()), ptable);
        mesh = std::move(next_mesh);
        bem = std::move(next_bem);
        tables = std::move(next_tables);
        centre = next_centre;
    }
    std::printf("[move%s] res=%d M=%d speed=%.2fm/s (Mach %.1e) epoch=%d steps steps=%d epochs=%d | rebased cells/epoch "
                "%.0f refs/epoch %.0f | grid SNR %.1f dB | eq5 SNR %.1f dB | max|p| %.3e | %.1fs\n",
                rebase ? " rebase" : "", res, m, speed, speed / SoundC, epoch_steps, steps, epochs, double(rebase_cells) / std::max(1, epochs), double(rebase_refs) / std::max(1, epochs), grid_all.Db(), eq5_all.Db(), max_abs, Now() - t0);
    // Travel per epoch is what governs this, so the thresholds belong to the cadence the gate
    // runs: 2 m/s at 16 steps an epoch, 0.054 cells of travel between table builds.
    if (gate) {
        AtLeast("move grid SNR", grid_all.Db(), 34.8);
        AtLeast("move eq5 SNR", eq5_all.Db(), 25.2);
        AtLeast("move max|p|", max_abs, 0.4);
        AtMost("move max|p|", max_abs, 0.8);
    }
}

} // namespace

int main(int argc, char **argv) {
    bool self_test = false, v1 = false, v2 = false, full = false, check_sets = false, stability = false, quad3 = false,
         filter = true;
    int res = 30, subdiv = 2, r1 = 2, r2 = 3;
    double tau_div = 1., cfl = 1., v1_time = 0., damp = 0., stab_time = 0.1, trim = QuadPolicy{}.TrimEps, cell = 0.005;
    std::string mesh_file;
    double move_speed = 0., move_time = 0.03;
    int epoch_steps = 64;
    bool moving = false, rebase = false, kernels = true, gate = false;
    double src_offset = 0.;
    bool dipole = false;
    Body body = Body::Sphere;
    bool any = false;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--gate") gate = any = true;
        else if (arg == "--self-test") self_test = any = true;
        else if (arg == "--v1") v1 = any = true;
        else if (arg == "--v2") v2 = any = true;
        else if (arg == "--full") full = true;
        else if (arg == "--check-sets") check_sets = true;
        else if (arg == "--stability") stability = any = true;
        else if (arg == "--quad3") quad3 = true;
        else if (arg == "--res" && a + 1 < argc) res = std::atoi(argv[++a]);
        else if (arg == "--subdiv" && a + 1 < argc) subdiv = std::atoi(argv[++a]);
        else if (arg == "--tau-div" && a + 1 < argc) tau_div = std::atof(argv[++a]);
        else if (arg == "--cfl" && a + 1 < argc) cfl = std::atof(argv[++a]);
        else if (arg == "--r1" && a + 1 < argc) r1 = std::atoi(argv[++a]);
        else if (arg == "--r2" && a + 1 < argc) r2 = std::atoi(argv[++a]);
        else if (arg == "--v1-time" && a + 1 < argc) v1_time = std::atof(argv[++a]);
        else if (arg == "--damp" && a + 1 < argc) damp = std::atof(argv[++a]);
        else if (arg == "--no-filter") filter = false;
        else if (arg == "--no-kernels") kernels = false;
        else if (arg == "--trim" && a + 1 < argc) trim = std::atof(argv[++a]);
        else if (arg == "--src-offset" && a + 1 < argc) src_offset = std::atof(argv[++a]);
        else if (arg == "--dipole") dipole = true;
        else if (arg == "--ellipsoid") body = Body::Ellipsoid;
        else if (arg == "--box") body = Body::Box;
        else if (arg == "--lshape") body = Body::LShape;
        else if (arg == "--mesh" && a + 1 < argc) mesh_file = argv[++a], any = true;
        else if (arg == "--cell" && a + 1 < argc) cell = std::atof(argv[++a]);
        else if (arg == "--move" && a + 1 < argc) move_speed = std::atof(argv[++a]), moving = any = true;
        else if (arg == "--epoch" && a + 1 < argc) epoch_steps = std::atoi(argv[++a]);
        else if (arg == "--move-time" && a + 1 < argc) move_time = std::atof(argv[++a]);
        else if (arg == "--rebase") rebase = true;
        else if (arg == "--time" && a + 1 < argc) stab_time = std::atof(argv[++a]);
        else {
            std::printf("unknown arg %s\n", arg.c_str());
            return 1;
        }
    }
    if (!any) self_test = v1 = v2 = true; // default ladder

    if (!mesh_file.empty()) MeshReport(mesh_file, cell);
    if (self_test) SelfTest(false);
    if (v1) RunV1(subdiv, tau_div, full, quad3, v1_time, kernels, false);
    const V2Config swept{.Res = res, .CheckSets = check_sets, .Quad3 = quad3, .Cfl = cfl, .R1 = r1, .R2 = r2, .Damp = damp, .Filter = filter, .Trim = trim, .SrcOffset = src_offset, .Dipole = dipole, .B = body, .Kernels = kernels};
    if (v2) RunV2(swept);
    if (stability) RunV2({.Res = res, .TotalTime = stab_time, .Quad3 = quad3, .Cfl = cfl, .R1 = r1, .R2 = r2, .Damp = damp, .Filter = filter, .Trim = trim, .SrcOffset = src_offset, .Dipole = dipole, .B = body, .Kernels = kernels});
    if (moving) RunMoving(res, move_speed, epoch_steps, move_time, r1, r2, filter, rebase, kernels, false);
    // The gate ignores every sweep flag above: a threshold belongs to the configuration it
    // was recorded at, and the sweeps exist to leave that configuration.
    if (gate) {
        SelfTest(true);
        // Only the tabulated path. The per-pair transform it stands in for is compared
        // against directly in the self-test, pair by pair, so running the rungs a second
        // time under --no-kernels only re-covers the flag that selects between them.
        RunV1(2, 1., false, false, 0., true, true);
        RunV2({.CheckSets = true, .Gate = true});
        // The same rung with the source off the body's centre, then with a dipole: a
        // surface trace that varies, then one that changes sign.
        RunV2({.SrcOffset = 0.5, .Gate = true});
        RunV2({.Dipole = true, .Gate = true});
        // And the paper's source on a body that is not a sphere, which is the rung that asks
        // for curvature that varies and elements that are not all alike.
        RunV2({.B = Body::Ellipsoid, .Gate = true});
        // And two with edges and corners, the second concave, where the surface trace
        // converges slowest of anything on the ladder.
        RunV2({.B = Body::Box, .Gate = true});
        RunV2({.B = Body::LShape, .Gate = true});
        RunV1(2, 1., true, false, 0., true, true); // the dense left-matrix reference
        RunV2({.TotalTime = 0.1, .Gate = true}); // stability
        RunMoving(30, 2., 16, 0.03, 2, 3, true, false, true, true); // geometry epochs
        std::printf("[gate] %s (%d failures)\n", Failures ? "FAIL" : "PASS", Failures);
    }
    return Failures ? 1 : 0;
}
