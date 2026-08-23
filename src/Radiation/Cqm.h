#pragma once

// Convolution quadrature (CQM) weights for the time-domain BEM in the SonicRadiation
// solver [Jin et al. 2025, arXiv:2508.08775]. BDF2-based CQM after [Lubich 1994;
// Banjai & Sauter 2008]: the convolution weights of a Laplace-domain kernel K(s) are the
// power-series coefficients of K(gamma(zeta)/tau), computed by a scaled inverse DFT over a
// contour |zeta| = Lambda with Lambda^N = sqrt(eps), leaving a ~3e-8 relative error floor.
//
// Two kernels per (target point, source element) pair, both integrated over the source
// element with the element's outward (into-fluid) normal:
//   single layer  V: G(s) = e^(-s r / c) / (4 pi r)
//   double layer  D: dG/dn_y = -(s / c + 1 / r) G(s) (n . (y - x)) / r
// The exterior representation is p(x) = D[phi] - V[g] with g = dp/dn, and the exterior
// trace jump is +phi/2, so the surface Dirichlet solve reads phi = 2 * (principal value).
// The double-layer self term vanishes for flat elements ((y - x) . n = 0), making the
// collocation diagonal exactly |elem| / 2 — the basis of the paper's row-diagonal solve.

#include <Eigen/Core>

#include <complex>
#include <vector>

struct RadiationMesh;

using Complex = std::complex<double>;

struct CqmContour;

// The universal BDF2 delay kernels every non-self pair's weights are made of.
//
// Both Laplace symbols carry the geometry only through the retarded delay r / c, which with
// s = gamma(zeta) / tau is the dimensionless theta = r / (c tau). The weight operator is
// linear in the contour samples with real per-pair coefficients, so exactly
//   V_j = sum_q coeff_q Psi_j(theta_q)
//   D_j = -sum_q dot_q coeff_q (Chi_j(theta_q) / (c tau) + Psi_j(theta_q) / r_q)
// where Psi_j(theta) = [zeta^j] e^(-gamma(zeta) theta) and Chi_j(theta) = [zeta^j]
// gamma(zeta) e^(-gamma(zeta) theta) depend on nothing but theta. Tabulating those two over
// theta once turns each pair's contour sampling and transform into two interpolated row
// reads. Rows run to the end of the widest window any covered pair reaches, so the only new
// error is the interpolation in theta, orders below the precision the weights are kept in.
struct CqmKernelTable {
    double ThetaMax{0.}; // largest delay the table covers, in timesteps
    double Step{0.}; // theta between nodes
    int LagCount{0}; // lags per node, from lag 0
    std::vector<float> Psi, Chi; // node-major, LagCount each

    // Samples both kernels over [0, theta_max]: a few thousand transforms once, against the
    // twelve million a scene's tables would otherwise need.
    void Build(const CqmContour &, double theta_max);

    // Whether a pair reaching `theta` and lag `lag_end` can be built from the table.
    bool Covers(double theta, int lag_end) const { return LagCount >= lag_end && theta >= 0. && theta <= ThetaMax; }
    const float *PsiRow(int node) const { return Psi.data() + size_t(node) * LagCount; }
    const float *ChiRow(int node) const { return Chi.data() + size_t(node) * LagCount; }
};

// Kernel samples at the contour nodes, split into real and imaginary arrays: the layout
// both vDSP's FFT and its vectorized transcendentals want.
struct SplitSamples {
    std::vector<double> Re, Im;

    explicit SplitSamples(int n = 0) : Re(n, 0.), Im(n, 0.) {}
};

struct CqmContour {
    CqmContour(double tau, double c, int n_contour = 256); // n_contour must be a power of two
    ~CqmContour();
    CqmContour(const CqmContour &) = delete;
    CqmContour &operator=(const CqmContour &) = delete;

    double Tau, C;
    int N;
    double Lambda;
    std::vector<Complex> S; // N contour frequencies, Re(s) > 0
    std::vector<double> SRe, SIm; // the same, split for the vectorized sampling

    // Weights w_j for j in [begin_lag, end_lag) from kernel samples K_l at the contour
    // frequencies: w_j = Re(Lambda^(-j) / N * sum_l K_l e^(-2 pi i l j / N)). That sum over
    // l is the DFT of the samples, so one transform yields every lag. It runs in place, so
    // `samples` is consumed.
    std::vector<double> LagsFromSamples(SplitSamples &samples, int begin_lag, int end_lag) const;
    std::vector<double> LagsFromSamples(const std::vector<Complex> &kernel_samples, int begin_lag, int end_lag) const;

    // Delay kernels for pairs reaching at most `theta_max` timesteps of retardation.
    // ComputePairSeries falls back to the transform beyond them, so this is a speed
    // decision, not a correctness one.
    void BuildKernels(double theta_max) { Kernels.Build(*this, theta_max); }
    CqmKernelTable Kernels;

private:
    std::vector<double> LambdaPow; // Lambda^(-j)
    // vDSP's FFT setup (an FFTSetupD), held opaquely to keep Accelerate out of this header.
    // Read-only once created, so every worker shares this one.
    void *Fft{nullptr};
    int Log2N{0};
};

// One kernel's weights on a trimmed lag window [Begin, Begin + W.size()).
struct LagSeries {
    int Begin{0};
    std::vector<double> W;
};

struct PairSeries {
    LagSeries D, V;
};

struct QuadPolicy {
    int BaseOrder{1}; // 1-point (centroid) or 3-point (edge midpoints) base rule
    double AdaptiveRatio{0.5}; // subdivide while max edge > ratio * distance to target
    int MaxDepth{3};
    // Drop lags below TrimEps * max |w|. The BDF2 weights have a long, slowly decaying tail
    // and it is the tables' dominant cost, so this is the knob that sets their size: 3e-3
    // takes the mean window from 40 lags to 20. Being relative to each series' own peak, it
    // leaves an error floor at a roughly constant level whatever the geometry, source or
    // resolution, and what it perturbs is mode amplitude, not frequency or decay. 1e-2 is
    // past where that holds. See VALIDATION for the sweep behind 3e-3.
    double TrimEps{3e-3};
    // Build weights from the contour's delay kernels where they reach. Off runs every pair
    // through its own transform instead, which is what the kernels are measured against.
    bool UseKernels{true};
};

// CQM weight series for the potentials radiated by source element `elem` onto target
// point x. `is_self` selects the analytic single-layer self integral (x must then be the
// element centroid) and drops the double layer entirely.
//
// Fills `out` rather than returning it, so a caller walking millions of pairs reuses one
// series' capacity instead of paying two heap round trips per pair. `out` is reset on
// entry, so a reused one carries nothing over.
void ComputePairSeries(const CqmContour &, const Eigen::Vector3d &x, const RadiationMesh &, int elem, bool is_self, const QuadPolicy &, PairSeries &out);
