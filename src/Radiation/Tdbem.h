#pragma once

// Element-constant time-domain BEM stepped by BDF2 convolution quadrature (see Cqm.h).
// Owns the Dirichlet/Neumann history rings and the element-to-element weight table, and
// evaluates the retarded-potential convolutions everything else consumes: the diagonal
// Dirichlet solve (Eq. 13/17 of [Jin et al. 2025]), the dense reference solve of the full
// boundary equation (Eq. 4), and pressure evaluation at arbitrary points (Eq. 5).
//
// Weight tables are flat: each entry names its source element and two lag windows into a
// shared weight array (RadEntry), grouped by source (see PairTable). The GPU stepper
// consumes the same records, so a table built here is the bit-comparison reference for the
// ported kernels.

#include "Cqm.h"
#include "RadiationMesh.h"
#include "RadiationParams.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <memory>

// A table's weights in the shape the GPU reads them (RadStagedEntry). Filled by Finish
// instead of PairTable::Weights when nothing on the CPU will read the table — every table a
// geometry epoch builds after the first — because the float weights are gigabytes that would
// only be converted and thrown away.
struct StagedWeights {
    std::vector<RadStagedEntry> Entries; // the records the kernels index, without the source
    std::vector<signed char> Weights; // signed bytes, scaled by their window's RadStagedEntry exponent
    // Each entry's lag-0 double-layer weight, which leaves the table: the diagonal solve
    // moves that term to the left side.
    std::vector<float> D0;
};

// Weight table over target points, laid out for the way the solver reads it: entries are
// grouped by *source element*, with their weight windows contiguous in that order. The
// per-step convolution passes walk one source at a time — every entry of a source reads the
// same history window, so the pass stages that window once and streams the weights — and
// that pass is the solver's dominant bandwidth. Target-major consumers go through Order.
struct PairTable {
    std::vector<int> SrcBegin; // per source element, into Entries
    std::vector<int> Begin, Order; // per target, into Order; Order holds entry ids
    // Windows and offsets into whichever of the two weight arrays below is filled. Staged,
    // they are the aligned and padded windows rather than the trimmed ones.
    std::vector<RadEntry> Entries;
    // Narrowed to float against the contour's own ~3e-8 relative error floor: these tables
    // are both the largest allocation and the per-step bandwidth cost.
    std::vector<float> Weights;
    StagedWeights Staged; // filled instead of Weights when the builder was asked to stage

    bool IsStaged() const { return !Staged.Weights.empty(); }
    int MaxLagEnd() const; // largest lag any entry reaches, for history-ring sizing
    // Entry id for a (target, source) pair, -1 when absent. A target's entries appear in
    // increasing source order, so this is a binary search.
    int Find(int target, int src) const;
};

// A window's block scale: the power of two bringing its largest weight to the top of a
// signed byte. A power of two so the scale itself is exact.
struct BlockScale {
    short Exp{0};
    float Inv{1.f};

    explicit BlockScale(float max_abs);
    signed char operator()(float w) const;
};

// A window widened down to a quad boundary and up to a quad length, for the four-lag
// alignment RadStagedEntry describes. An empty window stays empty.
inline void WidenQuad(int begin, int len, int &lo, int &wide) {
    lo = begin & ~3;
    wide = len ? ((begin + len + 3) & ~3) - lo : 0;
}

// One entry's weights as the block-scaled signed bytes the convolution passes read. `w` is
// the trimmed window, double-layer lags then single-layer, at whatever width it was built
// in; `d_dst` and `v_dst` point at where lag `d_begin` and the first single-layer lag land
// inside the widened record.
//
// The lag-0 double-layer term leaves for `d0` before either scale is chosen: the diagonal
// solve moves it to the left side, and it is larger than the rest of its window by orders
// of magnitude, so a scale sized to it would quantize everything behind it to nothing. It
// keeps the precision it arrived with.
//
// Both packing paths route through here — the builder quantizing its rows in place, and
// PackRadiation repacking a float table — so the layout, the lift rule and the exponent
// choice have one definition. A pair carried across epochs is quantized by whichever path
// built it, and the two must agree bit for bit.
template<typename Weight>
void QuantizeWindows(const Weight *w, int d_begin, int d_len, int v_len, signed char *d_dst, signed char *v_dst, float *d0, signed char &d_exp, signed char &v_exp) {
    const bool lift = d_begin == 0 && d_len > 0;
    if (lift && d0) *d0 = float(w[0]);
    float d_max = 0.f, v_max = 0.f;
    for (int j = lift ? 1 : 0; j < d_len; ++j) d_max = std::max(d_max, std::abs(float(w[j])));
    for (int j = 0; j < v_len; ++j) v_max = std::max(v_max, std::abs(float(w[d_len + j])));
    const BlockScale d_scale{d_max}, v_scale{v_max};
    d_exp = (signed char)d_scale.Exp;
    v_exp = (signed char)v_scale.Exp;
    for (int j = lift ? 1 : 0; j < d_len; ++j) d_dst[j] = d_scale(float(w[j]));
    for (int j = 0; j < v_len; ++j) v_dst[j] = v_scale(float(w[d_len + j]));
}

// Builds a PairTable one target row at a time, in parallel: each row accumulates into its
// own weight blob, and Finish lays the blobs out in source order, releasing each as it is
// read so the peak stays near the packed size rather than twice it.
struct PairTableBuilder {
    // `staged` builds the GPU's byte layout in place of the float weights.
    explicit PairTableBuilder(int n_targets, bool staged = false) : Staged(staged), Rows(n_targets) {}

    // Appends one source element's series to row `target`. Call from the row's own thread.
    // Sources must arrive in ascending order — Finish counts on it. The second form takes a
    // pair already built at another pose, whose weights carry over unchanged.
    void Add(int target, int src, const PairSeries &);
    void Add(int target, int src, const RadEntry &windows, const float *weights);
    // Staged weights arrive quantized, with their lag-0 double-layer term lifted out and
    // zeroed, so carrying one over dequantizes and puts that term back. Both are exact: the
    // scale is a power of two, and requantizing a dequantized window lands on the same
    // exponent and the same bytes, so a pair carried through any number of epochs does not
    // drift.
    void Add(int target, int src, const RadEntry &windows, const signed char *weights, float d0, short d_exp, short v_exp);
    PairTable Finish(int n_sources);

private:
    bool Staged;
    // A row holds whichever width its table will end in, so a weight is rounded once, and no
    // earlier than this: a weight is summed over up to 64 quadrature points, the double layer
    // is a partly cancelling combination, and Trim thresholds a window against its own peak.
    struct Row {
        std::vector<RadEntry> Entries;
        std::vector<float> Weights; // float tables, which the CPU convolutions read
        std::vector<__fp16> Half; // staged tables, which only the GPU reads
    };
    std::vector<Row> Rows;

    // A row holds one width or the other, never both, and an entry's Off indexes whichever
    // it holds — so every append and every offset an Add hands out goes through these two.
    int RowSize(const Row &row) const { return int(Staged ? row.Half.size() : row.Weights.size()); }
    template<typename It> void Append(Row &row, It first, It last) {
        if (Staged) row.Half.insert(row.Half.end(), first, last);
        else row.Weights.insert(row.Weights.end(), first, last);
    }
};

struct Tdbem {
    RadiationMesh Mesh;
    std::unique_ptr<CqmContour> Contour;
    QuadPolicy Policy;
    int HistLen{0}; // power of two, set by EnsureHistLen
    PairTable ElemElem; // target = element centroid; sources = the element's near list

    // `near_lists[i]` are the source elements whose potentials element i evaluates
    // directly (always including itself); empty outer vector means dense (all pairs).
    // `theta_max` sizes the contour's delay kernels (Cqm.h) — the largest retardation, in
    // timesteps, that any table built on this contour will reach. A caller that will build
    // further tables over the same contour, as BuildRadiation does for the cell table,
    // passes their reach too; the default covers these pairs alone.
    //
    // `previous` is the same element set's table at an earlier pose. The kernels see only
    // distances between elements, element normals and areas, all of which rigid motion
    // leaves alone, so a pair in both near sets keeps the weights it had and only membership
    // turns over. Passing it therefore builds only the pairs the near sets gained, which is a
    // third off this table. Exact in exact arithmetic — recomputing transforms the mesh in
    // floating point first, so the two agree to ~5e-6 and neither is the more correct.
    // `staged` builds the element-element weights in the GPU's byte layout and skips the
    // float ones, which only the CPU convolutions read.
    void Init(RadiationMesh mesh, double tau, double c, const std::vector<std::vector<int>> &near_lists, const QuadPolicy &policy, double theta_max = 0., const PairTable *previous = nullptr, bool staged = false);
    // Grows the history rings to cover `max_lag_end` lags, preserving the history already
    // stored up to `current_step`. Every table whose entries feed a convolution must be
    // covered — including evaluation tables built after the solve started, whose targets
    // sit farther away and so reach longer lags.
    void EnsureHistLen(int max_lag_end, int current_step);
    // Extra lags a receding target needs on top of a table's reach: the body travelling
    // `distance` away from a fixed sample point pushes that point's lags out by the time the
    // sound takes to cross it, plus one for the rounding. Callers whose geometry moves size
    // their ring by this, and an epoch that outgrows it fails rather than wrapping.
    static int RecedingLags(double distance, double c, double tau) { return int(std::ceil(distance / (c * tau))) + 1; }

    int NumElems() const { return Mesh.NumElems(); }
    // Rings are element-major: one contiguous power-of-two lag window per element, so a
    // convolution walks memory in order (the layout the GPU gather wants too).
    double &PhiAt(int n, int e) { return PhiRing[size_t(e) * HistLen + (n & (HistLen - 1))]; }
    double PhiAt(int n, int e) const { return PhiRing[size_t(e) * HistLen + (n & (HistLen - 1))]; }
    double &GAt(int n, int e) { return GRing[size_t(e) * HistLen + (n & (HistLen - 1))]; }
    const std::vector<double> &Phi() const { return PhiRing; }
    const std::vector<double> &G() const { return GRing; }

    // sum_j (D_j phi_(n-j) - V_j g_(n-j)) for one table entry. skip_current_phi zeroes the
    // unknown phi_n (the lag-0 double-layer term the diagonal solve moves to the left side).
    double Convolve(const PairTable &, int entry, int n, bool skip_current_phi) const;

    // Every element's near-field convolution at step `n`, accumulated onto `seed` (null for
    // zero) and written to `out`. Both Dirichlet solves start here and both want the seed
    // accumulated first, so their rounding is unchanged by sharing this.
    void NearSums(int n, const double *seed, double *out) const;
    // phi_n = 2 * (near-field principal value + far), far per element or null (pure TDBEM).
    void SolveDirichletDiag(int n, const double *far);

    // Dense reference for validation: factor (I/2 - D0) once, then solve Eq. 4 exactly.
    void PrepareFullSolve();
    void SolveDirichletFull(int n);

    // Dense evaluation table from all elements onto arbitrary points.
    PairTable BuildPointTable(const std::vector<Eigen::Vector3d> &points) const;
    double EvaluatePoint(const PairTable &, int point, int n) const;

private:
    std::vector<double> PhiRing, GRing;
    Eigen::PartialPivLU<Eigen::MatrixXd> FullLu;
    bool FullReady{false};
};
