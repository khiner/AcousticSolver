// Metal port of WaveBlender's CUDA kernels (GPUSolver.cu, WaveBlender.cu, Shaders/*.cu).
//
// Compiled at runtime with KernelParams.h prepended (see MetalContext.cpp).
// Every floating-point expression matches the CUDA reference operation-for-operation
// (fast-math is off), so per-cell results are bit-exact against the reference given
// identical inputs. The step kernels share one persistent argument table (see
// KernelParams.h) and sequence into a batch in WaveBlender::RunFdtd.
//
// References:
//   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains.

#include <metal_stdlib>
using namespace metal;

using real = float;

constant bool ApplyFaces [[function_constant(ApplyFacesFcIndex)]];

constant int XDir = 0;
constant int YDir = 1;
constant int ZDir = 2;
constant float Pi = 3.14159265358979323846f;

inline int Cid(int i, int j, int k, int nx, int ny) { return (ny * nx) * k + nx * j + i; }

// ============================== Core FDTD (GPUSolver.cu) ==============================

// Beta from a cell's blending state (see the Cell* constants in KernelParams.h) at
// blending time `tb`, with the reference's cubic-smoothstep arithmetic.
inline real BetaOf(uchar state, real tb) {
    if (state == CellAir) return 0.f;
    if (state == CellSolid) return 1.f;
    if (state == CellRising) return 3.f * tb * tb - 2.f * tb * tb * tb;
    real t = 1.f - tb;
    return 3.f * t * t - 2.f * t * t * t;
}

// Beta at batch start, for the fresh-cell prologue kernels: exactly 0 or 1, as
// transitioning cells still hold the last batch's endpoint.
inline real Beta0Of(uchar state) {
    return (state == CellSolid || state == CellFalling) ? 1.f : 0.f;
}

// Index into the shell-packed PML split-field storage: the shell (any cell within
// PmlWidth of a domain face) as six disjoint dense slabs, z before y before x, each
// contiguous along x. Call only for shell cells.
inline int PsiIndex(int i, int j, int k, int nx, int ny, int nz) {
    const int w = PmlWidth;
    if (k < w) return (k * ny + j) * nx + i; // z-lo
    int off = nx * ny * w;
    if (k >= nz - w) return off + (((k - (nz - w)) * ny + j) * nx + i); // z-hi
    off += nx * ny * w;
    const int kz = k - w, nzi = nz - 2 * w;
    if (j < w) return off + ((kz * w + j) * nx + i); // y-lo
    off += nx * w * nzi;
    if (j >= ny - w) return off + ((kz * w + (j - (ny - w))) * nx + i); // y-hi
    off += nx * w * nzi;
    const int jy = j - w, nyi = ny - 2 * w;
    if (i < w) return off + ((kz * nyi + jy) * w + i); // x-lo
    off += w * nyi * nzi;
    return off + ((kz * nyi + jy) * w + (i - (nx - w))); // x-hi
}

// Pressure update kernel (Eq. 8a from [Xue et al. 2024])
kernel void StepPressure(device real *p [[buffer(0)]],
                         device real *px [[buffer(1)]], device real *py [[buffer(2)]], device real *pz [[buffer(3)]],
                         device const real *vx [[buffer(4)]], device const real *vy [[buffer(5)]], device const real *vz [[buffer(6)]],
                         constant real *pml_n [[buffer(9)]], constant real *pml_d [[buffer(10)]],
                         constant FdtdBatchParams &batch [[buffer(FdtdBatchParamsIndex)]],
                         uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y, k = tid.z;
    if (i >= batch.Nx || j >= batch.Ny || k >= batch.Nz) return;
    int cid = Cid(i, j, k, batch.Nx, batch.Ny);

    // Distance to the closest domain boundary
    int min_xdist = min(i, batch.Nx - 1 - i);
    int min_ydist = min(j, batch.Ny - 1 - j);
    int min_zdist = min(k, batch.Nz - 1 - k);

    // Individual components of divergence
    real vx_c = vx[cid];
    real vx_l = (i > 0) ? vx[Cid(i - 1, j, k, batch.Nx, batch.Ny)] : 0.f; // LEFT
    real vy_c = vy[cid];
    real vy_d = (j > 0) ? vy[Cid(i, j - 1, k, batch.Nx, batch.Ny)] : 0.f; // DOWN
    real vz_c = vz[cid];
    real vz_f = (k > 0) ? vz[Cid(i, j, k - 1, batch.Nx, batch.Ny)] : 0.f; // FRONT

    real divx = (vx_c - vx_l) * batch.InvDx;
    real divy = (vy_c - vy_d) * batch.InvDx;
    real divz = (vz_c - vz_f) * batch.InvDx;

    if (min_xdist < PmlWidth || min_ydist < PmlWidth || min_zdist < PmlWidth) { // Inside the PML: split pressure update
        int pidx = PsiIndex(i, j, k, batch.Nx, batch.Ny, batch.Nz);
        real px_c = px[pidx], py_c = py[pidx], pz_c = pz[pidx];

        px_c = (pml_n[min_xdist] * px_c - batch.RhoCcDt * divx) * pml_d[min_xdist];
        py_c = (pml_n[min_ydist] * py_c - batch.RhoCcDt * divy) * pml_d[min_ydist];
        pz_c = (pml_n[min_zdist] * pz_c - batch.RhoCcDt * divz) * pml_d[min_zdist];

        px[pidx] = px_c;
        py[pidx] = py_c;
        pz[pidx] = pz_c;
        p[cid] = px_c + py_c + pz_c;
    } else {
        p[cid] = p[cid] - batch.RhoCcDt * (divx + divy + divz);
    }
}

// Interpolated shader value for shader-data row `bid` at the step's fractional sample index.
inline real ShaderValue(int bid, device const real *shader_data, constant FdtdBatchParams &batch, constant FdtdStepParams &step) {
    real frac = (bid * batch.NShaderSamples) + step.SampleFrac; // Fractional index into shader_data
    int lo = int(frac), hi = lo + 1;
    return (hi - frac) * shader_data[lo] + (frac - lo) * shader_data[hi];
}

// Standalone boundary application (the Eq. 8b boundary-conditions term). Indexed by shader
// point rather than by face, so a dual-claimed face needs the host's two ordered ranges.
kernel void ApplyShader(device real *vx [[buffer(4)]], device real *vy [[buffer(5)]], device real *vz [[buffer(6)]],
                        device const uchar *state [[buffer(8)]],
                        device const real *shader_data [[buffer(13)]], device const int *shader_map [[buffer(14)]],
                        constant FdtdBatchParams &batch [[buffer(FdtdBatchParamsIndex)]],
                        constant FdtdStepParams &step [[buffer(FdtdStepParamsIndex)]],
                        constant ApplyShaderRange &range [[buffer(FdtdApplyRangeIndex)]],
                        uint tid [[thread_position_in_grid]]) {
    int global_bid = tid + range.Start;
    if (global_bid >= range.End) return;

    // Decode the shader map. is_force is encoded via the sign of shader_map[cid].
    bool is_force = shader_map[global_bid] < 0;
    int dir = (!is_force) ? shader_map[global_bid] % 3 : (-shader_map[global_bid]) % 3;
    int cid = (!is_force) ? shader_map[global_bid] / 3 : (-shader_map[global_bid]) / 3;

    int i = cid % batch.Nx;
    int j = (cid / batch.Nx) % batch.Ny;
    int k = (cid / batch.Nx) / batch.Ny;

    real val = ShaderValue(global_bid, shader_data, batch, step);

    if (dir == XDir) {
        real betax = max(BetaOf(state[cid], step.Tb), BetaOf(state[Cid(i + 1, j, k, batch.Nx, batch.Ny)], step.Tb));
        if (!is_force) vx[cid] += betax * (val - vx[cid]);
        else vx[cid] += (1.f - betax) * val * batch.InvRhoDt;
    } else if (dir == YDir) {
        real betay = max(BetaOf(state[cid], step.Tb), BetaOf(state[Cid(i, j + 1, k, batch.Nx, batch.Ny)], step.Tb));
        if (!is_force) vy[cid] += betay * (val - vy[cid]);
        else vy[cid] += (1.f - betay) * val * batch.InvRhoDt;
    } else if (dir == ZDir) {
        real betaz = max(BetaOf(state[cid], step.Tb), BetaOf(state[Cid(i, j, k + 1, batch.Nx, batch.Ny)], step.Tb));
        if (!is_force) vz[cid] += betaz * (val - vz[cid]);
        else vz[cid] += (1.f - betaz) * val * batch.InvRhoDt;
    }
}

// Layout in KernelParams.h.
struct ShaderFaces {
    device const uchar *Mask;
    device const int *Slot;
    device const uchar *Tiles;
    device const int *Bids;
};
inline ShaderFaces FacesOf(device const uchar *buffer, constant FdtdBatchParams &batch) {
    const int g = batch.Nx * batch.Ny * batch.Nz;
    const int n_tiles = batch.NTilesX * batch.NTilesY * batch.NTilesZ;
    return {
        buffer,
        (device const int *)(buffer + ShaderFacesSlotOffset(g)),
        buffer + ShaderFacesTilesOffset(g),
        (device const int *)(buffer + ShaderFacesBidsOffset(g, n_tiles)),
    };
}

inline bool TileFlagged(ShaderFaces faces, constant FdtdBatchParams &batch, uint3 tgid) {
    return faces.Tiles[FdtdTileIndex(tgid.x, tgid.y, tgid.z, batch.NTilesX, batch.NTilesY)] != 0;
}

inline int FaceBid(ShaderFaces faces, int cid, uchar mask, int plane) {
    return faces.Bids[faces.Slot[cid] + popcount(uint(mask) & ((1u << plane) - 1u))];
}

// The pending boundary application for the face at `cid` in direction `dir`. Same
// arithmetic, and the same blend-before-force order, as ApplyShader.
inline real ApplyPending(real v, real beta_face, int dir, int cid, uchar mask, ShaderFaces faces,
                         device const real *shader_data, constant FdtdBatchParams &batch, constant FdtdStepParams &step) {
    if (mask & (1u << dir)) { // Pending blend
        real val = ShaderValue(FaceBid(faces, cid, mask, dir), shader_data, batch, step);
        v += beta_face * (val - v);
    }
    if (mask & (1u << (dir + 3))) { // Pending force
        real val = ShaderValue(FaceBid(faces, cid, mask, dir + 3), shader_data, batch, step);
        v += (1.f - beta_face) * val * batch.InvRhoDt;
    }
    return v;
}

// One face's blended velocity update (Eq. 8b from [Xue et al. 2024]), for the face stored
// at cell (i, j, k). Shared by StepVelocity and the fused step so both compile identical
// arithmetic. Any pending boundary application already landed on the velocity read here.
inline real FaceVelocityX(int i, int j, int k, device const real *p, device const real *vx, device const uchar *state,
                          constant real *pml_n, constant real *pml_d,
                          constant FdtdBatchParams &batch, constant FdtdStepParams &step) {
    int cid = Cid(i, j, k, batch.Nx, batch.Ny);
    int min_xdist = min(i + 1, batch.Nx - i - 1);
    real p_c = p[cid];
    real p_r = (i < batch.Nx - 1) ? p[Cid(i + 1, j, k, batch.Nx, batch.Ny)] : 0.f; // RIGHT
    real beta = BetaOf(state[cid], step.Tb);
    real beta_r = (i < batch.Nx - 1) ? BetaOf(state[Cid(i + 1, j, k, batch.Nx, batch.Ny)], step.Tb) : 1.f;
    real gradx = (p_r - p_c) * batch.InvDx;
    real betax = max(beta, beta_r);
    // pml_n = 1. and pml_d = 1. / (1. + damping) outside the PML layer
    return (pml_n[min_xdist] * vx[cid] - (1.f - betax) * batch.InvRhoDt * gradx) * pml_d[min_xdist];
}
inline real FaceVelocityY(int i, int j, int k, device const real *p, device const real *vy, device const uchar *state,
                          constant real *pml_n, constant real *pml_d,
                          constant FdtdBatchParams &batch, constant FdtdStepParams &step) {
    int cid = Cid(i, j, k, batch.Nx, batch.Ny);
    int min_ydist = min(j + 1, batch.Ny - j - 1);
    real p_c = p[cid];
    real p_u = (j < batch.Ny - 1) ? p[Cid(i, j + 1, k, batch.Nx, batch.Ny)] : 0.f; // UP
    real beta = BetaOf(state[cid], step.Tb);
    real beta_u = (j < batch.Ny - 1) ? BetaOf(state[Cid(i, j + 1, k, batch.Nx, batch.Ny)], step.Tb) : 1.f;
    real grady = (p_u - p_c) * batch.InvDx;
    real betay = max(beta, beta_u);
    return (pml_n[min_ydist] * vy[cid] - (1.f - betay) * batch.InvRhoDt * grady) * pml_d[min_ydist];
}
inline real FaceVelocityZ(int i, int j, int k, device const real *p, device const real *vz, device const uchar *state,
                          constant real *pml_n, constant real *pml_d,
                          constant FdtdBatchParams &batch, constant FdtdStepParams &step) {
    int cid = Cid(i, j, k, batch.Nx, batch.Ny);
    int min_zdist = min(k + 1, batch.Nz - k - 1);
    real p_c = p[cid];
    real p_b = (k < batch.Nz - 1) ? p[Cid(i, j, k + 1, batch.Nx, batch.Ny)] : 0.f; // BACK
    real beta = BetaOf(state[cid], step.Tb);
    real beta_b = (k < batch.Nz - 1) ? BetaOf(state[Cid(i, j, k + 1, batch.Nx, batch.Ny)], step.Tb) : 1.f;
    real gradz = (p_b - p_c) * batch.InvDx;
    real betaz = max(beta, beta_b);
    return (pml_n[min_zdist] * vz[cid] - (1.f - betaz) * batch.InvRhoDt * gradz) * pml_d[min_zdist];
}

// Blended velocity update kernel: the pressure gradient term (Eq. 8b from [Xue et al. 2024]),
// also sampling the pressure field at listener cells. Runs as the batch tail, where nothing
// consumes the velocities afterwards, so no boundary application folds in.
kernel void StepVelocity(device const real *p [[buffer(0)]],
                         device real *vx [[buffer(4)]], device real *vy [[buffer(5)]], device real *vz [[buffer(6)]],
                         device const uchar *state [[buffer(8)]],
                         constant real *pml_n [[buffer(11)]], constant real *pml_d [[buffer(12)]],
                         constant int *listener_cids [[buffer(15)]], device real *listener_out [[buffer(16)]],
                         constant FdtdBatchParams &batch [[buffer(FdtdBatchParamsIndex)]],
                         constant FdtdStepParams &step [[buffer(FdtdStepParamsIndex)]],
                         uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y, k = tid.z;
    if (i >= batch.Nx || j >= batch.Ny || k >= batch.Nz) return;
    int cid = Cid(i, j, k, batch.Nx, batch.Ny);

    vx[cid] = FaceVelocityX(i, j, k, p, vx, state, pml_n, pml_d, batch, step);
    vy[cid] = FaceVelocityY(i, j, k, p, vy, state, pml_n, pml_d, batch, step);
    vz[cid] = FaceVelocityZ(i, j, k, p, vz, state, pml_n, pml_d, batch, step);

    // Listener sampling (p is unchanged by the velocity update)
    for (int l = 0; l < batch.NListeners; ++l) {
        if (cid == listener_cids[l]) listener_out[step.SampleIndex * batch.NListeners + l] = p[cid];
    }
}

// Fused step: the velocity update at step s followed by the pressure update at step s + 1,
// in one full-grid pass. Each thread computes all six face velocities its cell's divergence
// needs (the three lower faces recompute bit-identically in their owning threads), writes
// its own three, then updates its pressure. Step s + 1's boundary application lands on those
// writes, but the pressure update deliberately uses the untransformed velocities, matching
// the reference's [pressure, boundary application, velocity] step order.
//
// Reads p(s) and v from the in slots and writes p(s+1) and v(s) to the out slots (the host
// ping-pongs the bindings per step) — neighbor reads would otherwise race in-kernel writes.
// Listener slot step.SampleIndex samples p(s), the pre-update pressure.
kernel void StepVelocityPressure(device const real *p [[buffer(0)]],
                                 device real *px [[buffer(1)]], device real *py [[buffer(2)]], device real *pz [[buffer(3)]],
                                 device const real *vx [[buffer(4)]], device const real *vy [[buffer(5)]], device const real *vz [[buffer(6)]],
                                 device const uchar *shader_faces [[buffer(7)]], device const uchar *state [[buffer(8)]],
                                 constant real *pml_np [[buffer(9)]], constant real *pml_dp [[buffer(10)]],
                                 constant real *pml_nv [[buffer(11)]], constant real *pml_dv [[buffer(12)]],
                                 device const real *shader_data [[buffer(13)]],
                                 constant int *listener_cids [[buffer(15)]], device real *listener_out [[buffer(16)]],
                                 device real *p_out [[buffer(21)]],
                                 device real *vx_out [[buffer(22)]], device real *vy_out [[buffer(23)]], device real *vz_out [[buffer(24)]],
                                 constant FdtdBatchParams &batch [[buffer(FdtdBatchParamsIndex)]],
                                 constant FdtdStepParams &step [[buffer(FdtdStepParamsIndex)]],
                                 constant FdtdStepParams &next [[buffer(FdtdNextStepParamsIndex)]],
                                 uint3 tid [[thread_position_in_grid]],
                                 uint3 tgid [[threadgroup_position_in_grid]]) {
    int i = tid.x, j = tid.y, k = tid.z;
    if (i >= batch.Nx || j >= batch.Ny || k >= batch.Nz) return;
    int cid = Cid(i, j, k, batch.Nx, batch.Ny);

    // Face mask first: the velocity stores depend on it, so loading it here lets the
    // velocity arithmetic cover its latency. Unflagged tiles never touch the mask.
    const ShaderFaces faces = ApplyFaces ? FacesOf(shader_faces, batch) : ShaderFaces{nullptr, nullptr, nullptr, nullptr};
    const uchar mask = (ApplyFaces && TileFlagged(faces, batch, tgid)) ? faces.Mask[cid] : 0;

    // ----- Velocity update at step s: this cell's faces, plus the lower faces its
    // divergence needs -----
    real vx_c = FaceVelocityX(i, j, k, p, vx, state, pml_nv, pml_dv, batch, step);
    real vy_c = FaceVelocityY(i, j, k, p, vy, state, pml_nv, pml_dv, batch, step);
    real vz_c = FaceVelocityZ(i, j, k, p, vz, state, pml_nv, pml_dv, batch, step);
    real vx_l = (i > 0) ? FaceVelocityX(i - 1, j, k, p, vx, state, pml_nv, pml_dv, batch, step) : 0.f; // LEFT
    real vy_d = (j > 0) ? FaceVelocityY(i, j - 1, k, p, vy, state, pml_nv, pml_dv, batch, step) : 0.f; // DOWN
    real vz_f = (k > 0) ? FaceVelocityZ(i, j, k - 1, p, vz, state, pml_nv, pml_dv, batch, step) : 0.f; // FRONT

    real p_s = p[cid]; // p(s), for the listener sample

    // ----- Step s + 1's boundary application, onto the faces this thread writes -----
    if (ApplyFaces && mask) {
        const real beta_c = BetaOf(state[cid], next.Tb);
        const real beta_x = max(beta_c, (i < batch.Nx - 1) ? BetaOf(state[Cid(i + 1, j, k, batch.Nx, batch.Ny)], next.Tb) : 1.f);
        const real beta_y = max(beta_c, (j < batch.Ny - 1) ? BetaOf(state[Cid(i, j + 1, k, batch.Nx, batch.Ny)], next.Tb) : 1.f);
        const real beta_z = max(beta_c, (k < batch.Nz - 1) ? BetaOf(state[Cid(i, j, k + 1, batch.Nx, batch.Ny)], next.Tb) : 1.f);
        vx_out[cid] = ApplyPending(vx_c, beta_x, XDir, cid, mask, faces, shader_data, batch, next);
        vy_out[cid] = ApplyPending(vy_c, beta_y, YDir, cid, mask, faces, shader_data, batch, next);
        vz_out[cid] = ApplyPending(vz_c, beta_z, ZDir, cid, mask, faces, shader_data, batch, next);
    } else {
        vx_out[cid] = vx_c;
        vy_out[cid] = vy_c;
        vz_out[cid] = vz_c;
    }

    for (int l = 0; l < batch.NListeners; ++l) {
        if (cid == listener_cids[l]) listener_out[step.SampleIndex * batch.NListeners + l] = p_s;
    }

    // ----- Pressure update at step s + 1 (Eq. 8a), from the fresh face velocities -----
    int min_xdist = min(i, batch.Nx - 1 - i);
    int min_ydist = min(j, batch.Ny - 1 - j);
    int min_zdist = min(k, batch.Nz - 1 - k);

    real divx = (vx_c - vx_l) * batch.InvDx;
    real divy = (vy_c - vy_d) * batch.InvDx;
    real divz = (vz_c - vz_f) * batch.InvDx;

    if (min_xdist < PmlWidth || min_ydist < PmlWidth || min_zdist < PmlWidth) { // Inside the PML: split pressure update
        int pidx = PsiIndex(i, j, k, batch.Nx, batch.Ny, batch.Nz);
        real px_c = px[pidx], py_c = py[pidx], pz_c = pz[pidx];

        px_c = (pml_np[min_xdist] * px_c - batch.RhoCcDt * divx) * pml_dp[min_xdist];
        py_c = (pml_np[min_ydist] * py_c - batch.RhoCcDt * divy) * pml_dp[min_ydist];
        pz_c = (pml_np[min_zdist] * pz_c - batch.RhoCcDt * divz) * pml_dp[min_zdist];

        px[pidx] = px_c;
        py[pidx] = py_c;
        pz[pidx] = pz_c;
        p_out[cid] = px_c + py_c + pz_c;
    } else {
        p_out[cid] = p_s - batch.RhoCcDt * (divx + divy + divz);
    }
}

// ====================== Fresh cells + shader re-init (WaveBlender.cu) ======================

// Computes a_n, required to enforce Neumann boundary conditions for a fresh cell.
kernel void PrepareFreshCell(device real *ax [[buffer(0)]], device real *ay [[buffer(1)]], device real *az [[buffer(2)]],
                             device const real *shader_data [[buffer(3)]], device const int *shader_map [[buffer(4)]],
                             constant PrepareFreshCellParams &params [[buffer(5)]],
                             uint tid [[thread_position_in_grid]]) {
    int global_bid = tid;
    if (global_bid >= params.NShaderPoints) return;

    // Decode the shader map
    if (shader_map[global_bid] < 0) return; // Skip if is_force
    int dir = shader_map[global_bid] % 3;
    int cid = shader_map[global_bid] / 3;

    int base = (global_bid * params.NShaderSamples);
    real vb0 = shader_data[base];
    real vb0_dt = (1.f - params.Incr) * shader_data[base] + (params.Incr)*shader_data[base + 1];

    real a_0 = (vb0_dt - vb0) * params.InvDt;
    if (dir == XDir) ax[cid] = a_0;
    else if (dir == YDir) ay[cid] = a_0;
    else if (dir == ZDir) az[cid] = a_0;
}

// Eq. 13 from [Xue et al. 2024]
kernel void FreshCellPressure(device real *p [[buffer(0)]],
                              device const real *ax [[buffer(1)]], device const real *ay [[buffer(2)]], device const real *az [[buffer(3)]],
                              device const uchar *state [[buffer(4)]],
                              constant FreshCellPressureParams &params [[buffer(5)]],
                              uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x + PmlWidth;
    int j = tid.y + PmlWidth;
    int k = tid.z + PmlWidth;

    if (i >= params.Nx - 1 - PmlWidth || j >= params.Ny - 1 - PmlWidth || k >= params.Nz - 1 - PmlWidth) return;
    int cid = Cid(i, j, k, params.Nx, params.Ny);

    int neighbor_cids[6] = {
        Cid(i - 1, j, k, params.Nx, params.Ny), Cid(i + 1, j, k, params.Nx, params.Ny), // left, right
        Cid(i, j - 1, k, params.Nx, params.Ny), Cid(i, j + 1, k, params.Nx, params.Ny), // down, up
        Cid(i, j, k - 1, params.Nx, params.Ny), Cid(i, j, k + 1, params.Nx, params.Ny) // front, back
    };
    real p_fresh = 0.f;
    int n_air = 0;
    for (int n = 0; n < 6; n++) {
        if (Beta0Of(state[neighbor_cids[n]]) >= 1.f) continue; // Neighboring cell is solid

        int dir = n / 3;
        real sign = (n % 2 == 0) ? 1.f : -1.f;
        int vid = (n % 2 == 0) ? neighbor_cids[n] : cid;

        if (dir == XDir) p_fresh += p[neighbor_cids[n]] - sign * params.RhoDx * ax[vid];
        else if (dir == YDir) p_fresh += p[neighbor_cids[n]] - sign * params.RhoDx * ay[vid];
        else if (dir == ZDir) p_fresh += p[neighbor_cids[n]] - sign * params.RhoDx * az[vid];
        n_air += 1;
    }
    if (n_air > 0) p_fresh /= n_air;

    real beta = Beta0Of(state[cid]);
    p[cid] = (1.f - beta) * p[cid] + beta * p_fresh;
}

// Sets interior velocities to 0 (and clears the acceleration a_n buffers).
kernel void ClearSolid(device real *vx [[buffer(0)]], device real *vy [[buffer(1)]], device real *vz [[buffer(2)]],
                       device real *ax [[buffer(3)]], device real *ay [[buffer(4)]], device real *az [[buffer(5)]],
                       device const uchar *state [[buffer(6)]],
                       constant ClearSolidParams &params [[buffer(7)]],
                       uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x + PmlWidth;
    int j = tid.y + PmlWidth;
    int k = tid.z + PmlWidth;

    if (i >= params.Nx - 1 - PmlWidth || j >= params.Ny - 1 - PmlWidth || k >= params.Nz - 1 - PmlWidth) return;
    int cid = Cid(i, j, k, params.Nx, params.Ny);

    if (Beta0Of(state[cid]) >= 1.f) {
        if (Beta0Of(state[Cid(i + 1, j, k, params.Nx, params.Ny)]) >= 1.f) vx[cid] = 0.f;
        if (Beta0Of(state[Cid(i, j + 1, k, params.Nx, params.Ny)]) >= 1.f) vy[cid] = 0.f;
        if (Beta0Of(state[Cid(i, j, k + 1, params.Nx, params.Ny)]) >= 1.f) vz[cid] = 0.f;
    }
    ax[cid] = 0.f;
    ay[cid] = 0.f;
    az[cid] = 0.f;
}

// Section 6.2.2 from [Xue et al. 2024]
kernel void ShaderReInit(device const real *vx [[buffer(0)]], device const real *vy [[buffer(1)]], device const real *vz [[buffer(2)]],
                         device real *shader_data [[buffer(3)]], device const int *shader_map [[buffer(4)]],
                         constant ShaderReInitParams &params [[buffer(5)]],
                         uint tid [[thread_position_in_grid]]) {
    int global_bid = tid;
    if (global_bid >= params.NShaderPoints) return;

    // Decode the shader map
    if (shader_map[global_bid] < 0) return; // Skip if is_force
    int dir = shader_map[global_bid] % 3;
    int cid = shader_map[global_bid] / 3;

    int base = (global_bid * params.NShaderSamples);
    real vb0 = shader_data[base];
    real v_0 = 0.f;

    if (dir == XDir) v_0 = vx[cid];
    else if (dir == YDir) v_0 = vy[cid];
    else if (dir == ZDir) v_0 = vz[cid];

    for (int t = 0; t < params.NShaderSamples; t++) shader_data[base + t] -= (vb0 - v_0);
}

// ============================== Acoustic shaders (Shaders/*.cu) ==============================

// Monopole source (Eq. (D-5b), Blackstock pg. 358)
kernel void Monopole(device real *vb [[buffer(0)]],
                     device const real *b [[buffer(1)]], device const real *bn [[buffer(2)]],
                     constant MonopoleParams &params [[buffer(3)]],
                     uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, k = tid.y;
    if (i >= params.NPoints || k >= params.NSamples) return;

    real x_src = (params.StartStep >= params.NSamples - 1) ? -params.Speed + params.Speed * (params.StartStep + k) / (params.NSamples - 1) : 0.f;
    real y_src = 0.f, z_src = 0.f;

    real x = b[3 * i], y = b[3 * i + 1], z = b[3 * i + 2];
    real xn = bn[3 * i], yn = bn[3 * i + 1], zn = bn[3 * i + 2];

    real time = ((real)(params.StartStep + k) / params.Srate);

    real r = sqrt((x - x_src) * (x - x_src) + (y - y_src) * (y - y_src) + (z - z_src) * (z - z_src));
    real v = (cos(2.f * Pi * params.FreqHz * (time - r / params.C)) / (r * r * 2.f * Pi * params.FreqHz)
              - sin(2.f * Pi * params.FreqHz * (time - r / params.C)) / (r * params.C))
        * (xn + yn + zn);

    if (time - r / params.C < 0.f) v = 0.f;
    vb[(params.GlobalBid + i) * params.NSamples + k] = v;
}

// Section 5.1.1 from [Xue et al. 2024]
kernel void Speaker(device real *vb [[buffer(0)]],
                    device const real *bn [[buffer(1)]], device const real *audio [[buffer(2)]],
                    constant SpeakerParams &params [[buffer(3)]],
                    uint2 tid [[thread_position_in_grid]]) {
    const real Scale = 100.f; // Manually scale the input audio amplitude

    int i = tid.x, k = tid.y;
    if (i >= params.NPoints || k >= params.NSamples) return;

    real xn = bn[3 * i], yn = bn[3 * i + 1], zn = bn[3 * i + 2];

    real v = 0.f; // Manually specify the speaker direction
    if (params.Dir == 0 && xn <= -1.f) v = -audio[params.StartStep + k]; // LEFT
    else if (params.Dir == 1 && xn >= 1.f) v = audio[params.StartStep + k]; // RIGHT
    else if (params.Dir == 2 && yn <= -1.f) v = -audio[params.StartStep + k]; // DOWN
    else if (params.Dir == 3 && yn >= 1.f) v = audio[params.StartStep + k]; // UP
    else if (params.Dir == 4 && zn <= -1.f) v = -audio[params.StartStep + k]; // FRONT
    else if (params.Dir == 5 && zn >= 1.f) v = audio[params.StartStep + k]; // BACK

    vb[(params.GlobalBid + i) * params.NSamples + k] = Scale * v;
}

// Projects modal velocities to boundary faces, interpolating the transfer matrix over the batch.
kernel void ModeMatmul(device real *vb [[buffer(0)]],
                       device const real *mode_to_boundary1 [[buffer(1)]], device const real *mode_to_boundary2 [[buffer(2)]],
                       device const real *mode_vels [[buffer(3)]], device const real *accel_noise [[buffer(4)]],
                       constant ModeMatmulParams &params [[buffer(5)]],
                       uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, k = tid.y;
    if (i >= params.NPoints || k >= params.NSamples) return;

    real v = 0.f;
    real alpha = ((real)k) / (params.NSamples - 1.f);
    for (int j = 0; j < params.NModes; j++) {
        v += (1.f - alpha) * mode_to_boundary1[params.NPoints * j + i] * mode_vels[params.NModes * k + j];
        v += alpha * mode_to_boundary2[params.NPoints * j + i] * mode_vels[params.NModes * k + j];
    }
    vb[(params.GlobalBid + i) * params.NSamples + k] = v + accel_noise[params.NPoints * k + i];
}

// Constructs the (NPoints x NBubs) "bubble-to-boundary" transfer matrix (Eq. 11 from [Xue et al. 2024]).
kernel void BubToBoundary(device real *bub_to_boundary [[buffer(0)]],
                          device const real *b [[buffer(1)]], device const real *bn [[buffer(2)]],
                          device const real *bub_data [[buffer(3)]],
                          constant BubToBoundaryParams &params [[buffer(4)]],
                          uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y;
    if (i >= params.NPoints || j >= params.NBubs) return;

    real x = b[3 * i], y = b[3 * i + 1], z = b[3 * i + 2];
    real xn = bn[3 * i], yn = bn[3 * i + 1], zn = bn[3 * i + 2];

    real r = bub_data[4 * j];
    real xbub = bub_data[4 * j + 1], ybub = bub_data[4 * j + 2], zbub = bub_data[4 * j + 3];

    real dist = sqrt((x - xbub) * (x - xbub) + (y - ybub) * (y - ybub) + (z - zbub) * (z - zbub));
    real a_bub = ((x - xbub) * abs(xn) + (y - ybub) * abs(yn) + (z - zbub) * abs(zn)) / (dist * 4.f * Pi);

    // Special handling for the d < r case (avoid d = 0 blowing up, etc.)
    if (a_bub < 0.f) a_bub = 0.f;
    else if (dist < r) a_bub *= -3.f * dist * dist / (r * r * r * r) + 4.f * dist / (r * r * r);
    else a_bub *= 1.f / (dist * dist);

    bub_to_boundary[params.NPoints * j + i] = a_bub;
}

// Per-bubble boundary flux: alpha times the column sums of the transfer matrix (replaces
// cublasSgemv(OP_T, ..., ones, ...) in the CUDA reference).
kernel void BubFlux(device const real *bub_to_boundary [[buffer(0)]], device real *flux [[buffer(1)]],
                    constant BubFluxParams &params [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {
    int j = tid;
    if (j >= params.NBubs) return;

    real sum = 0.f;
    for (int i = 0; i < params.NPoints; i++) sum += bub_to_boundary[params.NPoints * j + i];
    flux[j] = params.Alpha * sum;
}

// Multiplies the "bubble-to-boundary" transfer matrix with bubble volume velocities to compute vb.
kernel void BubMatmul(device real *vb [[buffer(0)]],
                      device const real *bub_to_boundary [[buffer(1)]], device const real *bub_vels [[buffer(2)]],
                      device const real *flux [[buffer(3)]],
                      constant BubMatmulParams &params [[buffer(4)]],
                      uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, k = tid.y;
    if (i >= params.NPoints || k >= params.Stop - params.Start) return;

    real v = 0.f;
    for (int j = 0; j < params.NBubs; j++) {
        real f = (flux[j] > 1e-6f) ? flux[j] : 1.f;
        v += bub_to_boundary[params.NPoints * j + i] * bub_vels[params.NBubs * k + j] / f;
    }
    vb[(params.GlobalBid + i) * params.NSamples + (params.Start + k)] = v;
}

// Constructs the (NPoints x NImpulses) "impulse-to-boundary" (particle-to-grid) transfer matrix.
kernel void ImpulseToBoundary(device real *impulse_to_boundary [[buffer(0)]],
                              device const real *b [[buffer(1)]], device const real *bn [[buffer(2)]],
                              device const real *impulse_data [[buffer(3)]],
                              constant ImpulseToBoundaryParams &params [[buffer(4)]],
                              uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y;
    if (i >= params.NPoints || j >= params.NImpulses) return;

    real x = b[3 * i], y = b[3 * i + 1], z = b[3 * i + 2];
    real xn = bn[3 * i], yn = bn[3 * i + 1], zn = bn[3 * i + 2];

    real xpt = impulse_data[6 * j], ypt = impulse_data[6 * j + 1], zpt = impulse_data[6 * j + 2];
    real xpt_dir = impulse_data[6 * j + 3], ypt_dir = impulse_data[6 * j + 4], zpt_dir = impulse_data[6 * j + 5];

    real distx = abs(x - xpt), disty = abs(y - ypt), distz = abs(z - zpt);

    real weight = 0.f;
    if (distx < params.Dx && disty < params.Dx && distz < params.Dx) weight = (1.f - distx / params.Dx) * (1.f - disty / params.Dx) * (1.f - distz / params.Dx);

    if (xn != 0.f) weight *= xpt_dir;
    else if (yn != 0.f) weight *= ypt_dir;
    else if (zn != 0.f) weight *= zpt_dir;

    impulse_to_boundary[params.NPoints * j + i] = weight;
}

// Multiplies the "impulse-to-boundary" transfer matrix with impulse magnitudes to compute forces.
kernel void ImpulseMatmul(device real *force [[buffer(0)]],
                          device const real *impulse_to_boundary [[buffer(1)]], device const real *impulse_vels [[buffer(2)]],
                          constant ImpulseMatmulParams &params [[buffer(3)]],
                          uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, k = tid.y;
    if (i >= params.NPoints || k >= params.NSamples) return;

    real v = 0.f;
    for (int j = 0; j < params.NImpulses; j++) {
        v += impulse_to_boundary[params.NPoints * j + i] * impulse_vels[params.NImpulses * k + j];
    }
    force[(params.GlobalBid + i) * params.NSamples + k] = v;
}
