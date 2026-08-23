#include "Tdbem.h"

#include "Parallel.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>

// A staged record counts lags in quads and scales in a signed byte, which bounds both.
static_assert(RadMaxHistLen / 4 <= 255, "a staged window no longer fits a byte of quads");

using Eigen::Vector3d;

int PairTable::MaxLagEnd() const {
    int max_end = 0;
    for (const auto &e : Entries) max_end = std::max({max_end, e.DBegin + e.DLen, e.VBegin + e.VLen});
    return max_end;
}

int PairTable::Find(int target, int src) const {
    const auto *first = Order.data() + Begin[target], *last = Order.data() + Begin[target + 1];
    const auto *it = std::lower_bound(first, last, src, [&](int entry, int s) { return Entries[entry].Src < s; });
    return it != last && Entries[*it].Src == src ? *it : -1;
}

// 2^(ilogb(x) - 6) puts the largest weight in [64, 128), so the byte keeps its full seven
// bits of magnitude. A window whose largest rounds to 128 clamps, which costs that one
// weight a half step and nothing else.
BlockScale::BlockScale(float max_abs) {
    if (max_abs > 0.f) Exp = short(std::ilogb(max_abs) - 6);
    Inv = std::ldexp(1.f, -Exp);
}

signed char BlockScale::operator()(float w) const {
    return (signed char)std::clamp(std::lrint(double(w) * double(Inv)), -127L, 127L);
}

void PairTableBuilder::Add(int target, int src, const PairSeries &series) {
    auto &row = Rows[target];
    assert(row.Entries.empty() || row.Entries.back().Src < src); // Finish counts on it
    // The single-layer window follows the double-layer one immediately: the convolution
    // passes read both as one span, and RadEntry carries only the one offset.
    const RadEntry entry{src, RowSize(row), (unsigned short)series.D.Begin, (unsigned short)series.D.W.size(), (unsigned short)series.V.Begin, (unsigned short)series.V.W.size()};
    Append(row, series.D.W.begin(), series.D.W.end());
    Append(row, series.V.W.begin(), series.V.W.end());
    row.Entries.push_back(entry);
}

void PairTableBuilder::Add(int target, int src, const RadEntry &windows, const signed char *weights, float d0, short d_exp, short v_exp) {
    // Rows hold the unquantized window, so a carried pair is dequantized back into one and
    // Finish quantizes it again. Exact in both directions, so a pair does not drift across
    // epochs — see the header.
    const float d_scale = std::ldexp(1.f, d_exp), v_scale = std::ldexp(1.f, v_exp);
    const int span = windows.DLen + windows.VLen;
    static thread_local std::vector<float> widened; // millions of pairs carry over per epoch
    widened.resize(span);
    for (int j = 0; j < windows.DLen; ++j) widened[j] = float(weights[j]) * d_scale;
    for (int j = windows.DLen; j < span; ++j) widened[j] = float(weights[j]) * v_scale;
    // The lag-0 double-layer term Finish lifted out of the table and zeroed. It never went
    // through the quantizer, so it goes back at the precision it left with.
    if (windows.DBegin == 0 && windows.DLen > 0) widened[0] = d0;
    Add(target, src, windows, widened.data());
}

void PairTableBuilder::Add(int target, int src, const RadEntry &windows, const float *weights) {
    auto &row = Rows[target];
    assert(row.Entries.empty() || row.Entries.back().Src < src);
    RadEntry entry = windows;
    entry.Src = src;
    entry.Off = RowSize(row);
    Append(row, weights, weights + windows.DLen + windows.VLen);
    row.Entries.push_back(entry);
}

PairTable PairTableBuilder::Finish(int n_sources) {
    const size_t n_targets = Rows.size();
    // Where each target's entries land in the target-major listing, which is PairTable::Begin.
    std::vector<int> row_off(n_targets + 1, 0);
    for (size_t t = 0; t < n_targets; ++t) row_off[t + 1] = row_off[t] + int(Rows[t].Entries.size());
    const int n_entries = row_off[n_targets];

    // Order every entry by (source, target): the convolution passes stream the weights in
    // exactly that order, one source element per threadgroup. Rows are built in ascending
    // source order (see Add), so counting each source's entries and scattering the rows in
    // turn lands them there — a counting sort over millions of entries rather than a
    // comparison sort, and every pass of it parallel over disjoint target ranges.
    const size_t n_parts = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t per_part = std::max<size_t>(1, (n_targets + n_parts - 1) / n_parts);
    const auto part_end = [&](size_t p) { return std::min(n_targets, (p + 1) * per_part); };
    std::vector<int> hist(n_parts * n_sources, 0);
    ParallelFor(n_parts, 1, [&](size_t p) {
        int *counts = hist.data() + p * n_sources;
        for (size_t t = p * per_part; t < part_end(p); ++t)
            for (const auto &entry : Rows[t].Entries) ++counts[entry.Src];
    });

    PairTable table;
    table.SrcBegin.assign(n_sources + 1, 0);
    std::vector<int> cursor(n_parts * n_sources); // each part's first slot for each source
    for (int src = 0; src < n_sources; ++src) {
        int slot = table.SrcBegin[src];
        for (size_t p = 0; p < n_parts; ++p) {
            cursor[p * n_sources + src] = slot;
            slot += hist[p * n_sources + src];
        }
        table.SrcBegin[src + 1] = slot;
    }

    table.Entries.resize(n_entries);
    table.Order.assign(n_entries, 0);
    ParallelFor(n_parts, 1, [&](size_t p) {
        std::vector<int> slot(cursor.begin() + p * n_sources, cursor.begin() + (p + 1) * n_sources);
        for (size_t t = p * per_part; t < part_end(p); ++t) {
            const auto &row = Rows[t];
            for (size_t k = 0; k < row.Entries.size(); ++k) {
                const int dst = slot[row.Entries[k].Src]++;
                table.Entries[dst] = row.Entries[k];
                table.Order[row_off[t] + k] = dst;
            }
        }
    });

    // Weight spans follow the same order, so their offsets are one prefix sum. Scattering
    // them row by row rather than gathering them entry by entry lets each row go as soon as
    // it has been read: a gather holds every row until the last entry is placed, which is
    // the whole table's weights twice over at the peak.
    //
    // Staged, the windows widen to four-lag boundaries here and the weights are quantized on
    // the way out, so the float array the GPU would only have converted is never built.
    std::vector<int> weight_off(n_entries);
    int total = 0;
    for (int i = 0; i < n_entries; ++i) {
        RadEntry &entry = table.Entries[i];
        if (Staged) {
            int d_lo, d_len, v_lo, v_len;
            WidenQuad(entry.DBegin, entry.DLen, d_lo, d_len);
            WidenQuad(entry.VBegin, entry.VLen, v_lo, v_len);
            entry.DBegin = (unsigned short)d_lo;
            entry.DLen = (unsigned short)d_len;
            entry.VBegin = (unsigned short)v_lo;
            entry.VLen = (unsigned short)v_len;
        }
        weight_off[i] = total;
        entry.Off = total; // the write pass needs it, and so does the staged record below
        total += entry.DLen + entry.VLen;
    }
    if (Staged) {
        table.Staged.Weights.assign(total, 0);
        table.Staged.D0.assign(n_entries, 0.f);
        // Built before the weights so the write pass can leave each window's block scale in
        // it; everything else about the record is known from the widened windows already.
        table.Staged.Entries.resize(n_entries);
        ParallelFor(n_entries, 4096, [&](size_t i) {
            const RadEntry &e = table.Entries[i];
            table.Staged.Entries[i] = {e.Off, (unsigned char)(e.DBegin / 4), (unsigned char)(e.DLen / 4), (unsigned char)(e.VBegin / 4), (unsigned char)(e.VLen / 4), 0, 0};
        });
    } else {
        table.Weights.resize(total);
    }
    ParallelFor(n_targets, 1, [&](size_t t) {
        auto &row = Rows[t];
        for (size_t k = 0; k < row.Entries.size(); ++k) {
            const RadEntry &was = row.Entries[k]; // the trimmed windows, before any padding
            const int dst = table.Order[row_off[t] + k];
            const int off = weight_off[dst];
            const RadEntry &entry = table.Entries[dst];
            if (!Staged) {
                std::copy_n(row.Weights.data() + was.Off, was.DLen + was.VLen, table.Weights.data() + off);
                continue;
            }
            const __fp16 *w = row.Half.data() + was.Off;
            signed char *out = table.Staged.Weights.data() + off;
            RadStagedEntry &staged = table.Staged.Entries[dst];
            QuantizeWindows(w, was.DBegin, was.DLen, was.VLen, out + (was.DBegin - entry.DBegin), out + entry.DLen + (was.VBegin - entry.VBegin), &table.Staged.D0[dst], staged.DExp, staged.VExp);
        }
        row.Weights = {};
        row.Half = {};
        row.Entries = {};
    });
    Rows.clear();
    Rows.shrink_to_fit();

    table.Begin = std::move(row_off);
    return table;
}

void Tdbem::Init(RadiationMesh mesh, double tau, double c, const std::vector<std::vector<int>> &near_lists, const QuadPolicy &policy, double theta_max, const PairTable *previous, bool staged) {
    Mesh = std::move(mesh);
    Contour = std::make_unique<CqmContour>(tau, c);
    Policy = policy;
    const int m = NumElems();

    std::vector<std::vector<int>> lists;
    if (near_lists.empty()) {
        std::vector<int> all(m);
        for (int e = 0; e < m; ++e) all[e] = e;
        lists.assign(m, all);
    }
    const auto &near = near_lists.empty() ? lists : near_lists;

    if (theta_max <= 0.) {
        // The furthest quadrature point any of these pairs reaches: centroid distance plus
        // the source element's own extent, since the quadrature samples anywhere inside it.
        std::vector<double> extent(m, 0.);
        for (int e = 0; e < m; ++e)
            for (int v = 0; v < 3; ++v) extent[e] = std::max(extent[e], (Mesh.Vertices[Mesh.Tris[e][v]] - Mesh.Centroids[e]).norm());
        double max_r = 0.;
        for (int i = 0; i < m; ++i)
            for (const int src : near[i]) max_r = std::max(max_r, (Mesh.Centroids[i] - Mesh.Centroids[src]).norm() + extent[src]);
        theta_max = max_r / (c * tau);
    }
    Contour->BuildKernels(theta_max);

    PairTableBuilder builder{m, staged};
    const bool prior_staged = previous && previous->IsStaged();
    ParallelFor(m, 1, [&](size_t i) {
        PairSeries series; // reused down the row, so its windows allocate once rather than per pair
        // Both lists ascend in source, so carrying the retained pairs over is one merge.
        int kept = previous ? previous->Begin[i] : 0;
        const int kept_end = previous ? previous->Begin[i + 1] : 0;
        for (const int src : near[i]) {
            if (previous) {
                while (kept < kept_end && previous->Entries[previous->Order[kept]].Src < src) ++kept;
                if (kept < kept_end && previous->Entries[previous->Order[kept]].Src == src) {
                    const RadEntry &was = previous->Entries[previous->Order[kept]];
                    if (prior_staged) {
                        const RadStagedEntry &sw = previous->Staged.Entries[previous->Order[kept]];
                        builder.Add(int(i), src, was, previous->Staged.Weights.data() + was.Off, previous->Staged.D0[previous->Order[kept]], sw.DExp, sw.VExp);
                    } else builder.Add(int(i), src, was, previous->Weights.data() + was.Off);
                    ++kept;
                    continue;
                }
            }
            QuadPolicy p = Policy;
            if (src != int(i) && Mesh.Adjacent(int(i), src)) p.BaseOrder = 3;
            ComputePairSeries(*Contour, Mesh.Centroids[i], Mesh, src, src == int(i), p, series);
            builder.Add(int(i), src, series);
        }
    });
    ElemElem = builder.Finish(m);
}

void Tdbem::EnsureHistLen(int max_lag_end, int current_step) {
    const int want = int(std::bit_ceil(unsigned(std::max(64, max_lag_end + 1))));
    if (want <= HistLen) return;

    const int m = NumElems(), old_len = HistLen, old_mask = old_len - 1, mask = want - 1;
    std::vector<double> phi(size_t(want) * m, 0.), g(size_t(want) * m, 0.);
    for (int t = std::max(0, current_step - old_len + 1); old_len > 0 && t <= current_step; ++t) {
        for (int e = 0; e < m; ++e) {
            phi[size_t(e) * want + (t & mask)] = PhiRing[size_t(e) * old_len + (t & old_mask)];
            g[size_t(e) * want + (t & mask)] = GRing[size_t(e) * old_len + (t & old_mask)];
        }
    }
    HistLen = want;
    PhiRing = std::move(phi);
    GRing = std::move(g);
}

double Tdbem::Convolve(const PairTable &table, int entry, int n, bool skip_current_phi) const {
    const auto &e = table.Entries[entry];
    const int mask = HistLen - 1, d_begin = e.DBegin, v_begin = e.VBegin;
    const float *w = table.Weights.data();
    const double *phi = &PhiRing[size_t(e.Src) * HistLen];
    const double *g = &GRing[size_t(e.Src) * HistLen];
    double acc = 0.;
    for (int j = std::max(d_begin, skip_current_phi ? 1 : 0), end = d_begin + e.DLen; j < end; ++j) {
        const int t = n - j;
        if (t < 0) break;
        acc += w[e.Off + (j - d_begin)] * phi[t & mask];
    }
    for (int j = v_begin, end = v_begin + e.VLen; j < end; ++j) {
        const int t = n - j;
        if (t < 0) break;
        acc -= w[e.Off + e.DLen + (j - v_begin)] * g[t & mask];
    }
    return acc;
}

void Tdbem::NearSums(int n, const double *seed, double *out) const {
    ParallelFor(NumElems(), 16, [&](size_t i) {
        double acc = seed ? seed[i] : 0.;
        for (int k = ElemElem.Begin[i]; k < ElemElem.Begin[i + 1]; ++k) acc += Convolve(ElemElem, ElemElem.Order[k], n, true);
        out[i] = acc;
    });
}

void Tdbem::SolveDirichletDiag(int n, const double *far) {
    const int m = NumElems();
    std::vector<double> phi(m);
    NearSums(n, far, phi.data());
    for (int i = 0; i < m; ++i) PhiAt(n, i) = 2. * phi[i];
}

void Tdbem::PrepareFullSolve() {
    const int m = NumElems();
    assert(ElemElem.Begin[m] == m * m); // dense table required
    Eigen::MatrixXd a = 0.5 * Eigen::MatrixXd::Identity(m, m);
    for (int i = 0; i < m; ++i) {
        for (int k = ElemElem.Begin[i]; k < ElemElem.Begin[i + 1]; ++k) {
            const auto &e = ElemElem.Entries[ElemElem.Order[k]];
            if (e.DBegin == 0 && e.DLen > 0) a(i, e.Src) -= ElemElem.Weights[e.Off];
        }
    }
    FullLu.compute(a);
    FullReady = true;
}

void Tdbem::SolveDirichletFull(int n) {
    assert(FullReady);
    const int m = NumElems();
    Eigen::VectorXd rhs(m);
    NearSums(n, nullptr, rhs.data());
    const Eigen::VectorXd phi = FullLu.solve(rhs);
    for (int i = 0; i < m; ++i) PhiAt(n, i) = phi[i];
}

PairTable Tdbem::BuildPointTable(const std::vector<Vector3d> &points) const {
    const int m = NumElems();
    PairTableBuilder builder{int(points.size())};
    ParallelFor(points.size(), 1, [&](size_t p) {
        PairSeries series;
        for (int e = 0; e < m; ++e) {
            ComputePairSeries(*Contour, points[p], Mesh, e, false, Policy, series);
            builder.Add(int(p), e, series);
        }
    });
    return builder.Finish(m);
}

double Tdbem::EvaluatePoint(const PairTable &table, int point, int n) const {
    double acc = 0.;
    for (int k = table.Begin[point]; k < table.Begin[point + 1]; ++k) acc += Convolve(table, table.Order[k], n, false);
    return acc;
}
