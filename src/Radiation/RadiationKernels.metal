// GPU kernels for the SonicRadiation hybrid solver [Jin et al. 2025, arXiv:2508.08775].
//
// Compiled at runtime with RadiationParams.h prepended (see MetalContext::RadiationPipeline).
// A step is a fixed sequence of dispatches over the tables the CPU builder produced, with
// the weights narrowed to block-scaled signed bytes and the history rings to float:
//
//   RadConvolveCells / RadConvolveElems
//                 every weight-table entry's retarded-potential convolution at step n + 1,
//                 with the unknown current phi zeroed (the lag-0 double-layer term the
//                 diagonal solve moves to the left side)
//   RadVelocity   shell velocities of the split-field PML, from p(n)
//   RadPressure   p(n+1) everywhere: PML split pressures in the shell, the scalar
//                 second-order update in the interior
//   RadCorrect    Eq. 12 set-difference corrections onto p(n+1) at band cells
//   RadDirichlet  Eqs. 17-20 element update: interpolated far field (optionally filtered)
//                 plus direct near sums, written into the Phi ring
//   RadSample /   listener output, on the grid path and on the Eq. 5 retarded-potential
//   RadEvalPoint  path respectively
//
// The convolution passes are what make this fast. Every consumer needs the same per-entry
// convolutions at two adjacent times, so they run once per step over the whole table and the
// consumers gather from the results. The two times differ only by the lag-0 double-layer
// term, so one pass covers both: it writes this step's skip-current-phi value and, in the
// same read-modify-write, completes the previous step's now that phi(n) is known. Consumers
// touch one float2 per reference and nothing else.
//
// They are dispatched one threadgroup per *source element*, not per entry: every entry of a
// source reads the same history window, so the group stages that window in threadgroup
// memory once, and the table is laid out in that order (PairTable::SrcBegin), so the device
// reads reduce to one sequential stream over the weights.
//
// The Dirichlet kernel appends the *next* step's Neumann sample to each element's ring as
// its last act: nothing in the step reads that slot, so the next step's lag-0 single-layer
// term is in place without another pass.

#include <metal_stdlib>
using namespace metal;

// Threadgroup width of the reducing kernels, matching their shared arrays.
constant uint EvalThreads = 128;

// Lanes sharing one weight-table entry in the convolution passes (see PartialStaged). Eight
// is measured best on WineglassTap's two tables, and a whole SIMD group per entry costs
// 30 / 44%: once a lane covers four lags, a ~27-lag window no longer fills 32 of them and
// the reduction stops amortizing.
constant int QuadLanes = 8;

// Attributes the convolution passes' fixed cost; 0 is the shipping path. See RadiationParams.h.
constant int RadFloorProbe [[function_constant(RadFloorProbeFcIndex)]];

inline int Cid(int i, int j, int k, int nx, int ny) { return i + nx * (j + ny * k); }

inline bool InShell(int i, int j, int k, constant RadGridParams &g) {
    return min(min(min(i, g.Nx - 1 - i), min(j, g.Ny - 1 - j)), min(k, g.Nz - 1 - k)) < g.PmlWidth;
}

// sum_j (D_j phi_(n-j) - V_j g_(n-j)) for one table entry. `skip_current_phi` zeroes the
// unknown phi_n (the lag-0 double-layer term the diagonal solve moves to the left side).
inline float Convolve(device const RadEntry *entries, device const float *weights,
                      device const float *phi_ring, device const float *g_ring,
                      int hist_len, int entry, int n, bool skip_current_phi) {
    const RadEntry e = entries[entry];
    const int mask = hist_len - 1, d_begin = e.DBegin, v_begin = e.VBegin;
    device const float *phi = phi_ring + e.Src * hist_len;
    device const float *g = g_ring + e.Src * hist_len;
    float acc = 0.f;
    for (int j = max(d_begin, skip_current_phi ? 1 : 0), end = d_begin + e.DLen; j < end; ++j) {
        const int t = n - j;
        if (t < 0) break;
        acc += weights[e.Off + (j - d_begin)] * phi[t & mask];
    }
    for (int j = v_begin, end = v_begin + e.VLen; j < end; ++j) {
        const int t = n - j;
        if (t < 0) break;
        acc -= weights[e.Off + e.DLen + (j - v_begin)] * g[t & mask];
    }
    return acc;
}

// sum_j (D_j phi_(n-j) - V_j g_(n-j)) for one entry, over history already staged in
// threadgroup memory. `lanes` threads share the entry, striding its lag window four weights
// at a time: the windows are four-lag aligned and zero padded (RadStagedEntry), so a lane
// covers four lags with one vector load of the weights and one of the staged history.
//
// **These passes are bound by the weight stream**, so bytes are what to spend: pointing
// every entry at one shared run, with everything else untouched, is worth 2.9 and 2.5x on
// the two tables. Hence signed bytes with a power-of-two scale per window, whose unpacking
// hides entirely under the stalls the smaller stream removes. Eight bits is the floor the
// ladder holds at — six fails five rungs, and no number of scales recovers it — and the two
// windows must carry separate scales, since they differ by orders of magnitude and sharing
// one costs 5-19 dB. See VALIDATION.
//
// The unknown current phi needs no special case: its weight is zeroed out of the table and
// carried in `d0`. The start-of-run clamp still hoists out, by testing the whole window
// once — only there does a lag past step 0 have to be masked.
inline float PartialStaged(const RadStagedEntry e, device const char *weights, threadgroup const float *phi,
                           threadgroup const float *g, int n, uint lane, int lanes) {
    const int d_begin = 4 * int(e.DBeginQ), d_n = 4 * int(e.DLenQ), v_begin = 4 * int(e.VBeginQ), v_n = 4 * int(e.VLenQ);
    device const char4 *dw = (device const char4 *)(weights + e.Off);
    device const char4 *vw = (device const char4 *)(weights + e.Off + d_n);
    // Applied once to each window's sum rather than to every weight, which is what makes
    // the scale free: two multiplies an entry, not two per lag.
    const float d_scale = exp2(float(e.DExp)), v_scale = exp2(float(e.VExp));
    float d_acc = 0.f, v_acc = 0.f;
    if (n >= d_begin + d_n && n >= v_begin + v_n) { // the whole window is in range
        for (int q = int(lane); 4 * q < d_n; q += lanes)
            d_acc += dot(float4(dw[q]), *(threadgroup const float4 *)(phi + d_begin + 4 * q));
        for (int q = int(lane); 4 * q < v_n; q += lanes)
            v_acc += dot(float4(vw[q]), *(threadgroup const float4 *)(g + v_begin + 4 * q));
        return d_acc * d_scale - v_acc * v_scale;
    }
    // Near the start of a run, where part of the window predates step 0 and reads as zero.
    for (int j = 4 * int(lane); j < d_n; j += 4 * lanes) {
        for (int c = 0; c < 4; ++c) {
            if (d_begin + j + c <= n) d_acc += float(weights[e.Off + j + c]) * phi[d_begin + j + c];
        }
    }
    for (int j = 4 * int(lane); j < v_n; j += 4 * lanes) {
        for (int c = 0; c < 4; ++c) {
            if (v_begin + j + c <= n) v_acc += float(weights[e.Off + d_n + j + c]) * g[v_begin + j + c];
        }
    }
    return d_acc * d_scale - v_acc * v_scale;
}

// Sum of one entry's lane partials, left in every lane sharing it. The caller must reach
// this with the whole SIMD group converged.
inline float QuadSum(float v) {
    for (uint m = 1; m < uint(QuadLanes); m <<= 1) v += simd_shuffle_xor(v, m);
    return v;
}

// Stages source element `e`'s history windows in threadgroup memory, indexed by lag from
// step `n` rather than by ring slot, so the convolutions carry no ring arithmetic. The host
// sizes the buffer to the ring in use, since declaring it at RadMaxHistLen would cost
// occupancy for every scene whose rings are shorter. RadStageTail zeroed slots follow each
// window, because a padded lag window can reach three lags past the ring's end.
inline void StageHistory(threadgroup float *phi, threadgroup float *g, device const float *phi_ring,
                         device const float *g_ring, int e, int hist_len, int n, uint lid, uint tg_size) {
    const int mask = hist_len - 1;
    const bool from_ring = RadFloorProbe != 2; // probe 2 keeps the fill and the barrier, drops the reads
    for (int lag = int(lid); lag < hist_len + RadStageTail; lag += int(tg_size)) {
        phi[lag] = from_ring ? (lag < hist_len ? phi_ring[e * hist_len + ((n - lag) & mask)] : 0.f) : 1.f;
        g[lag] = from_ring ? (lag < hist_len ? g_ring[e * hist_len + ((n - lag) & mask)] : 0.f) : 1.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One entry per source is the floor the probe measures; 0 leaves the row whole.
inline int ProbeEntryEnd(int lo, int hi) { return RadFloorProbe == 0 ? hi : min(hi, lo + 1); }

// A sink the compiler cannot fold away, so dropping a kernel's real write does not take the
// work that fed it with it. The comparison never holds at run time.
inline bool ProbeSink(float t) { return t == 3.4028235e38f; }

// Cell-target table: x holds the skip-current-phi convolution at n + 1, y the completed
// convolution at n (the previous step's x plus its lag-0 double-layer term). `conv` is
// indexed by target-major position, which is what the three gathering passes want, so the
// writes here scatter; `xs` keeps the same step's x in this pass's own source-major order
// so completing the previous step's value reads sequentially.
kernel void RadConvolveCells(device const RadStagedEntry *entries [[buffer(0)]], device const char *weights [[buffer(1)]],
                             device const float *phi_ring [[buffer(2)]], device const float *g_ring [[buffer(3)]],
                             device const int *src_begin [[buffer(4)]], device const float *d0 [[buffer(5)]],
                             device const int *dst [[buffer(6)]], device float *xs [[buffer(7)]],
                             device float2 *conv [[buffer(8)]],
                             constant RadGridParams &g [[buffer(9)]], constant RadStepParams &s [[buffer(10)]],
                             threadgroup float *staged [[threadgroup(0)]],
                             uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                             uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float *phi = staged, *gg = staged + g.HistLen + RadStageTail;
    const int e = int(gid);
    if (RadFloorProbe == 5) return; // the dispatch and nothing else
    StageHistory(phi, gg, phi_ring, g_ring, e, g.HistLen, s.N1, lid, tg_size);
    const float phi_n = phi[s.N1 - s.N]; // one lag back from the step being solved
    const uint ql = lid & (QuadLanes - 1);
    const int quads = int(tg_size) / QuadLanes;
    const int lo = src_begin[e], hi = ProbeEntryEnd(lo, src_begin[e + 1]);
    // The loop counter is uniform across the threadgroup, so every SIMD group reaches the
    // shuffles converged even on the ragged last pass.
    for (int base = lo; base < hi; base += quads) {
        const int k = base + int(lid / QuadLanes);
        const float part = k < hi ? PartialStaged(entries[k], weights, phi, gg, s.N1, ql, QuadLanes) : 0.f;
        const float t = RadFloorProbe == 3 ? part : QuadSum(part);
        if (RadFloorProbe == 4) {
            if (ProbeSink(t)) conv[0] = float2(t, t);
        } else if (ql == 0 && k < hi) {
            conv[dst[k]] = float2(t, xs[k] + d0[k] * phi_n);
            xs[k] = t;
        }
    }
}

// Element-element table: the BEM's own near sums, which only ever need step n + 1.
kernel void RadConvolveElems(device const RadStagedEntry *entries [[buffer(0)]], device const char *weights [[buffer(1)]],
                             device const float *phi_ring [[buffer(2)]], device const float *g_ring [[buffer(3)]],
                             device const int *src_begin [[buffer(4)]], device float *conv [[buffer(5)]],
                             constant RadGridParams &g [[buffer(6)]], constant RadStepParams &s [[buffer(7)]],
                             threadgroup float *staged [[threadgroup(0)]],
                             uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                             uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float *phi = staged, *gg = staged + g.HistLen + RadStageTail;
    const int e = int(gid);
    if (RadFloorProbe == 5) return; // the dispatch and nothing else
    StageHistory(phi, gg, phi_ring, g_ring, e, g.HistLen, s.N1, lid, tg_size);
    const uint ql = lid & (QuadLanes - 1);
    const int quads = int(tg_size) / QuadLanes;
    const int lo = src_begin[e], hi = ProbeEntryEnd(lo, src_begin[e + 1]);
    for (int base = lo; base < hi; base += quads) {
        const int k = base + int(lid / QuadLanes);
        const float part = k < hi ? PartialStaged(entries[k], weights, phi, gg, s.N1, ql, QuadLanes) : 0.f;
        const float t = RadFloorProbe == 3 ? part : QuadSum(part);
        if (RadFloorProbe == 4) {
            if (ProbeSink(t)) conv[0] = t;
        } else if (ql == 0 && k < hi) conv[k] = t;
    }
}

// Split-field PML velocities in the shell, plus the one interior layer whose face is
// shared with a shell cell (the pressure pass reads it).
kernel void RadVelocity(device const float *p [[buffer(0)]],
                        device float *vx [[buffer(1)]], device float *vy [[buffer(2)]], device float *vz [[buffer(3)]],
                        device const float *pml_nv [[buffer(4)]], device const float *pml_dv [[buffer(5)]],
                        constant RadGridParams &g [[buffer(6)]],
                        uint3 tid [[thread_position_in_grid]]) {
    const int i = int(tid.x), j = int(tid.y), k = int(tid.z);
    if (i >= g.Nx || j >= g.Ny || k >= g.Nz) return;
    const bool c_shell = InShell(i, j, k, g);
    const int cid = Cid(i, j, k, g.Nx, g.Ny);
    const float p_c = p[cid];
    if (c_shell || (i + 1 < g.Nx && InShell(i + 1, j, k, g))) {
        const int dist = min(i + 1, g.Nx - 1 - i);
        const float p_r = i + 1 < g.Nx ? p[cid + 1] : 0.f;
        vx[cid] = (pml_nv[dist] * vx[cid] - g.InvRhoDtH * (p_r - p_c)) * pml_dv[dist];
    }
    if (c_shell || (j + 1 < g.Ny && InShell(i, j + 1, k, g))) {
        const int dist = min(j + 1, g.Ny - 1 - j);
        const float p_u = j + 1 < g.Ny ? p[cid + g.Nx] : 0.f;
        vy[cid] = (pml_nv[dist] * vy[cid] - g.InvRhoDtH * (p_u - p_c)) * pml_dv[dist];
    }
    if (c_shell || (k + 1 < g.Nz && InShell(i, j, k + 1, g))) {
        const int dist = min(k + 1, g.Nz - 1 - k);
        const float p_b = k + 1 < g.Nz ? p[cid + g.Nx * g.Ny] : 0.f;
        vz[cid] = (pml_nv[dist] * vz[cid] - g.InvRhoDtH * (p_b - p_c)) * pml_dv[dist];
    }
}

// p(n+1) over the whole grid: the split-field PML pressure in the shell, and the scalar
// second-order update (the first-order system with the velocities eliminated) inside.
kernel void RadPressure(device const float *p [[buffer(0)]], device const float *p_prev [[buffer(1)]],
                        device float *p_next [[buffer(2)]],
                        device const float *vx [[buffer(3)]], device const float *vy [[buffer(4)]], device const float *vz [[buffer(5)]],
                        device float *px [[buffer(6)]], device float *py [[buffer(7)]], device float *pz [[buffer(8)]],
                        device const float *pml_np [[buffer(9)]], device const float *pml_dp [[buffer(10)]],
                        constant RadGridParams &g [[buffer(11)]],
                        uint3 tid [[thread_position_in_grid]]) {
    const int i = int(tid.x), j = int(tid.y), k = int(tid.z);
    if (i >= g.Nx || j >= g.Ny || k >= g.Nz) return;
    const int cid = Cid(i, j, k, g.Nx, g.Ny);
    if (InShell(i, j, k, g)) {
        const float vx_l = i > 0 ? vx[cid - 1] : 0.f, vy_d = j > 0 ? vy[cid - g.Nx] : 0.f,
                    vz_f = k > 0 ? vz[cid - g.Nx * g.Ny] : 0.f;
        const int dx = min(i, g.Nx - 1 - i), dy = min(j, g.Ny - 1 - j), dz = min(k, g.Nz - 1 - k);
        const float x = (pml_np[dx] * px[cid] - g.RhoCcDtH * (vx[cid] - vx_l)) * pml_dp[dx];
        const float y = (pml_np[dy] * py[cid] - g.RhoCcDtH * (vy[cid] - vy_d)) * pml_dp[dy];
        const float z = (pml_np[dz] * pz[cid] - g.RhoCcDtH * (vz[cid] - vz_f)) * pml_dp[dz];
        px[cid] = x;
        py[cid] = y;
        pz[cid] = z;
        p_next[cid] = x + y + z;
    } else {
        const float lap = p[cid - 1] + p[cid + 1] + p[cid - g.Nx] + p[cid + g.Nx] + p[cid - g.Nx * g.Ny] +
            p[cid + g.Nx * g.Ny] - 6.f * p[cid];
        p_next[cid] = p[cid] + (p[cid] - p_prev[cid] + g.S2 * lap) * g.InvDamp;
    }
}

// Eq. 12: each band cell's Laplacian is corrected for the set difference between its far set
// and each neighbor's, by direct retarded-potential evaluation at the neighbor. One SIMD
// group per cell, not one thread: a cell carries a few hundred references, and a thread
// apiece left the pass with too few threads to cover its gather latency.
kernel void RadCorrect(device float *p_next [[buffer(0)]],
                       device const int *corr_cell [[buffer(1)]], device const int *corr_begin [[buffer(2)]],
                       device const RadRef *corr_refs [[buffer(3)]], device const float2 *conv [[buffer(4)]],
                       constant RadGridParams &g [[buffer(5)]], constant RadStepParams &s [[buffer(6)]],
                       uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                       uint simd_id [[simdgroup_index_in_threadgroup]], uint simds [[simdgroups_per_threadgroup]]) {
    const int c = int(gid * simds + simd_id);
    if (c >= g.NumCorrCells) return; // uniform over the SIMD group: every lane shares `c`
    float acc = 0.f;
    for (int r = corr_begin[c] + int(lane), end = corr_begin[c + 1]; r < end; r += 32) {
        const int packed = corr_refs[r].Packed;
        const float y = conv[packed & ~RadRefNegative].y; // completed at step n
        acc += (packed & RadRefNegative) ? -y : y;
    }
    const float total = simd_sum(acc);
    if (lane == 0) p_next[corr_cell[c]] += g.S2 * total * g.InvDamp;
}

// Threadgroup sum of `value`, left in every lane.
inline float GroupSum(threadgroup float *partial, uint lid, uint tg_size, float value) {
    partial[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return partial[0];
}

// Eqs. 17-20: the element Dirichlet update, one threadgroup per element. The far field
// arrives by trilinear interpolation of the corner cells' far-field state, re-based from
// each corner's far set onto the element's own (Eq. 20); near elements are summed
// directly. The row-diagonal solve then reads phi = 2 * (that principal value).
kernel void RadDirichlet(device const float *p [[buffer(0)]], device const float *p_prev [[buffer(1)]],
                         device const RadCorner *interp [[buffer(2)]], device const RadRef *interp_refs [[buffer(3)]],
                         device const float2 *conv [[buffer(4)]],
                         device const int *elem_begin [[buffer(5)]], device const int *elem_order [[buffer(6)]],
                         device const float *conv_elem [[buffer(7)]],
                         device float *phi_ring [[buffer(8)]], device float *g_ring [[buffer(9)]],
                         device const float *g_src [[buffer(10)]],
                         constant RadGridParams &g [[buffer(11)]], constant RadStepParams &s [[buffer(12)]],
                         uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                         uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float partial[128];
    const int i = int(gid);
    // Without the feedback filter the far term is the interpolation at n + 1 alone; with
    // it, the average of that and the same construction one step back.
    const float w = g.FilterFeedback ? 0.5f : 1.f;
    float acc = 0.f;
    for (int corner = 0; corner < 8; ++corner) {
        const RadCorner ic = interp[i * 8 + corner];
        if (lid == 0) {
            acc += w * ic.Gamma * p[ic.Cell];
            if (g.FilterFeedback) acc += w * ic.Gamma * p_prev[ic.Cell];
        }
        for (int r = ic.RefBegin + int(lid); r < ic.RefEnd; r += int(tg_size)) {
            const int packed = interp_refs[r].Packed;
            const float2 c = conv[packed & ~RadRefNegative];
            const float v = g.FilterFeedback ? c.x + c.y : c.x;
            acc += w * ic.Gamma * ((packed & RadRefNegative) ? -v : v);
        }
    }
    for (int k = elem_begin[i] + int(lid), end = elem_begin[i + 1]; k < end; k += int(tg_size)) acc += conv_elem[elem_order[k]];

    const float phi = GroupSum(partial, lid, tg_size, acc);
    if (lid != 0) return;
    const int mask = g.HistLen - 1;
    phi_ring[i * g.HistLen + (s.N1 & mask)] = 2.f * phi;
    // The next step's Neumann sample: no read this step touches that slot, so appending it
    // here saves a pass and leaves the lag-0 single-layer term in place for step N1 + 1.
    g_ring[i * g.HistLen + ((s.N1 + 1) & mask)] = g_src[s.GOffset * g.NumElems + i];
}

// Re-bases the far-field state across a geometry epoch (see RebasePlan): each cell whose
// near set changed takes the signed retarded potentials of the elements that crossed its
// boundary, at both time levels the second-order update reads.
kernel void RadRebase(device float *p [[buffer(0)]], device float *p_prev [[buffer(1)]],
                      device const int *cells [[buffer(2)]], device const int *begin [[buffer(3)]],
                      device const RadRef *refs [[buffer(4)]],
                      device const RadEntry *entries [[buffer(5)]], device const float *weights [[buffer(6)]],
                      device const float *phi_ring [[buffer(7)]], device const float *g_ring [[buffer(8)]],
                      constant RadGridParams &g [[buffer(9)]], constant RadStepParams &s [[buffer(10)]],
                      uint tid [[thread_position_in_grid]]) {
    const int c = int(tid);
    if (c >= s.Count) return;
    float now = 0.f, before = 0.f;
    for (int r = begin[c], end = begin[c + 1]; r < end; ++r) {
        const int packed = refs[r].Packed, entry = packed & ~RadRefNegative;
        const float sign = (packed & RadRefNegative) ? -1.f : 1.f;
        now += sign * Convolve(entries, weights, phi_ring, g_ring, g.HistLen, entry, s.N, false);
        before += sign * Convolve(entries, weights, phi_ring, g_ring, g.HistLen, entry, s.N - 1, false);
    }
    p[cells[c]] += now;
    p_prev[cells[c]] += before;
}

// Grid-path listener output: trilinear sample of far-field state plus the near elements'
// retarded potentials. This is the one consumer that needs a convolution at the step just
// solved, so it completes the lag-0 term itself.
kernel void RadSample(device const float *p [[buffer(0)]],
                      device const RadCorner *corners [[buffer(1)]], device const RadRef *refs [[buffer(2)]],
                      device const float2 *conv [[buffer(3)]], device const float *cell_d0 [[buffer(4)]],
                      device const int *cell_src [[buffer(5)]], device const float *phi_ring [[buffer(6)]],
                      device float *out [[buffer(7)]],
                      constant RadGridParams &g [[buffer(8)]], constant RadStepParams &s [[buffer(9)]],
                      uint gid [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
                      uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float partial[128];
    const int l = int(gid);
    const int mask = g.HistLen - 1;
    float acc = 0.f;
    for (int corner = 0; corner < 8; ++corner) {
        const RadCorner ic = corners[l * 8 + corner];
        if (lid == 0) acc += ic.Gamma * p[ic.Cell];
        for (int r = ic.RefBegin + int(lid); r < ic.RefEnd; r += int(tg_size)) {
            const int packed = refs[r].Packed, entry = packed & ~RadRefNegative;
            const float full = conv[entry].x + cell_d0[entry] * phi_ring[cell_src[entry] * g.HistLen + (s.N1 & mask)];
            acc += ic.Gamma * ((packed & RadRefNegative) ? -full : full);
        }
    }
    const float total = GroupSum(partial, lid, tg_size, acc);
    if (lid == 0) out[s.SampleIndex * g.NumListeners + l] = total;
}

// Eq. 5: pressure at a listener straight from the boundary histories, with no grid in the
// path. The table is dense over every element and its lag windows are long (a listener sits
// far from the body), so the pass is split across EvalGroups threadgroups per point and
// summed by RadEvalReduce — one group per point leaves all but one GPU core idle.
kernel void RadEvalPoint(device const int *begin [[buffer(0)]], device const int *order [[buffer(1)]],
                         device const RadEntry *entries [[buffer(2)]], device const float *weights [[buffer(3)]],
                         device const float *phi_ring [[buffer(4)]], device const float *g_ring [[buffer(5)]],
                         device float *partials [[buffer(6)]],
                         constant RadGridParams &g [[buffer(7)]], constant RadStepParams &s [[buffer(8)]],
                         uint2 gid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[EvalThreads];
    const int point = int(gid.y), chunk = int(gid.x);
    const int lo = begin[point], hi = begin[point + 1];
    const int span = (hi - lo + g.EvalGroups - 1) / g.EvalGroups;
    const int from = lo + chunk * span, to = min(hi, from + span);
    float acc = 0.f;
    for (int k = from + int(lid); k < to; k += EvalThreads)
        acc += Convolve(entries, weights, phi_ring, g_ring, g.HistLen, order[k], s.N1, false);
    const float total = GroupSum(partial, lid, EvalThreads, acc);
    if (lid == 0) partials[point * g.EvalGroups + chunk] = total;
}

// Sums one point's partials, in a fixed order so the output stays reproducible.
kernel void RadEvalReduce(device const float *partials [[buffer(0)]], device float *out [[buffer(1)]],
                          constant RadGridParams &g [[buffer(2)]], constant RadStepParams &s [[buffer(3)]],
                          uint tid [[thread_position_in_grid]]) {
    const int point = int(tid);
    if (point >= g.NumListeners) return;
    float acc = 0.f;
    for (int c = 0; c < g.EvalGroups; ++c) acc += partials[point * g.EvalGroups + c];
    out[s.SampleIndex * g.NumListeners + point] = acc;
}
