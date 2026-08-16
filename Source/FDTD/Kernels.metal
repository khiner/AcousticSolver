/** Metal port of WaveBlender's CUDA kernels (GPUSolver.cu, WaveBlender.cu, Shaders/*.cu).
 *
 * \file Kernels.metal
 *
 * Compiled at runtime with KernelParams.h prepended (see MetalContext.cpp).
 * Every floating-point expression matches the CUDA reference operation-for-operation
 * (fast-math is off), so per-cell results are bit-exact against the reference given
 * identical inputs. The step kernels share one persistent argument table (see
 * KernelParams.h) and sequence into a batch in WaveBlender::RunFdtd.
 *
 * References:
 *   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains.
 */

#include <metal_stdlib>
using namespace metal;

typedef float REAL;

constant int X_DIR = 0;
constant int Y_DIR = 1;
constant int Z_DIR = 2;
constant float PI_F = 3.14159265358979323846f;

inline int CID(int i, int j, int k, int Nx, int Ny) { return ((Ny * Nx) * k + Nx * j + i); }

// ============================== Core FDTD (GPUSolver.cu) ==============================

/** \brief Pressure update kernel (Eq. 8a from [Xue et al. 2024]) */
kernel void step_pressure(device REAL *d_p [[buffer(0)]],
                          device REAL *d_px [[buffer(1)]], device REAL *d_py [[buffer(2)]], device REAL *d_pz [[buffer(3)]],
                          device const REAL *d_vx [[buffer(4)]], device const REAL *d_vy [[buffer(5)]], device const REAL *d_vz [[buffer(6)]],
                          device REAL *d_beta [[buffer(7)]], device const uchar *d_cell [[buffer(8)]],
                          device const REAL *d_pmlN [[buffer(9)]], device const REAL *d_pmlD [[buffer(10)]],
                          constant FdtdBatchParams &B [[buffer(FDTD_BATCH_PARAMS_INDEX)]],
                          constant FdtdStepParams &S [[buffer(FDTD_STEP_PARAMS_INDEX)]],
                          uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y, k = tid.z;
    if (i >= B.Nx || j >= B.Ny || k >= B.Nz) return;
    int cid = CID(i, j, k, B.Nx, B.Ny);

    // Distance to closest domain boundary
    int min_xdist = min(i, B.Nx - 1 - i);
    int min_ydist = min(j, B.Ny - 1 - j);
    int min_zdist = min(k, B.Nz - 1 - k);

    // Individual components of divergence
    REAL vx = d_vx[cid];
    REAL vx_L = (i > 0) ? d_vx[CID(i - 1, j, k, B.Nx, B.Ny)] : 0.f; // LEFT
    REAL vy = d_vy[cid];
    REAL vy_D = (j > 0) ? d_vy[CID(i, j - 1, k, B.Nx, B.Ny)] : 0.f; // DOWN
    REAL vz = d_vz[cid];
    REAL vz_F = (k > 0) ? d_vz[CID(i, j, k - 1, B.Nx, B.Ny)] : 0.f; // FRONT

    REAL divx = (vx - vx_L) * B.inv_dx;
    REAL divy = (vy - vy_D) * B.inv_dx;
    REAL divz = (vz - vz_F) * B.inv_dx;

    if (min_xdist < PML_WIDTH || min_ydist < PML_WIDTH || min_zdist < PML_WIDTH) { // Inside PML: split pressure update
        REAL px = d_px[cid], py = d_py[cid], pz = d_pz[cid];

        px = (d_pmlN[min_xdist] * px - B.RHO_CC_dt * divx) * d_pmlD[min_xdist];
        py = (d_pmlN[min_ydist] * py - B.RHO_CC_dt * divy) * d_pmlD[min_ydist];
        pz = (d_pmlN[min_zdist] * pz - B.RHO_CC_dt * divz) * d_pmlD[min_zdist];

        d_px[cid] = px;
        d_py[cid] = py;
        d_pz[cid] = pz;
        d_p[cid] = px + py + pz;
    } else { // Otherwise, regular pressure update
        d_p[cid] = d_p[cid] - B.RHO_CC_dt * (divx + divy + divz);
    }

    // Update beta (cubic smoothstep function)
    if (d_cell[cid] > 0 && d_beta[cid] < 1.f) {
        d_beta[cid] = 3.f * S.tb * S.tb - 2.f * S.tb * S.tb * S.tb;
    } else if (d_cell[cid] == 0 && d_beta[cid] > 0.f) {
        REAL tb = 1.f - S.tb;
        d_beta[cid] = 3.f * tb * tb - 2.f * tb * tb * tb;
    }
}

/** \brief Blended velocity update kernel: boundary conditions term (Eq. 8b from [Xue et al. 2024]) */
kernel void apply_shader(device REAL *d_vx [[buffer(4)]], device REAL *d_vy [[buffer(5)]], device REAL *d_vz [[buffer(6)]],
                         device const REAL *d_beta [[buffer(7)]],
                         device const REAL *d_shaderData [[buffer(13)]], device const int *d_shaderMap [[buffer(14)]],
                         constant FdtdBatchParams &B [[buffer(FDTD_BATCH_PARAMS_INDEX)]],
                         constant FdtdStepParams &S [[buffer(FDTD_STEP_PARAMS_INDEX)]],
                         constant ApplyShaderRange &R [[buffer(FDTD_APPLY_RANGE_INDEX)]],
                         uint tid [[thread_position_in_grid]]) {
    int global_bid = tid + R.start;
    if (global_bid >= R.end) return;

    // Decode shader map
    bool isForce = d_shaderMap[global_bid] < 0; // isForce is encoded via the sign of d_shaderMap[cid]
    int dir = (!isForce) ? d_shaderMap[global_bid] % 3 : (-d_shaderMap[global_bid]) % 3; // direction
    int cid = (!isForce) ? d_shaderMap[global_bid] / 3 : (-d_shaderMap[global_bid]) / 3; // cell index

    int i = cid % B.Nx;
    int j = (cid / B.Nx) % B.Ny;
    int k = (cid / B.Nx) / B.Ny;

    // Decode shader values
    REAL frac = (global_bid * B.N_shader_samples) + S.ss; // fractional index in d_shaderData memory (to allow for interpolation)
    int floor_ = int(frac), ceil_ = floor_ + 1;
    REAL val = (ceil_ - frac) * d_shaderData[floor_] + (frac - floor_) * d_shaderData[ceil_]; // interpolate

    // Update velocity
    if (dir == X_DIR) {
        REAL betax = max(d_beta[cid], d_beta[CID(i + 1, j, k, B.Nx, B.Ny)]);
        if (!isForce) d_vx[cid] += betax * (val - d_vx[cid]);
        else d_vx[cid] += (1.f - betax) * val * B.inv_RHO_dt;
    } else if (dir == Y_DIR) {
        REAL betay = max(d_beta[cid], d_beta[CID(i, j + 1, k, B.Nx, B.Ny)]);
        if (!isForce) d_vy[cid] += betay * (val - d_vy[cid]);
        else d_vy[cid] += (1.f - betay) * val * B.inv_RHO_dt;
    } else if (dir == Z_DIR) {
        REAL betaz = max(d_beta[cid], d_beta[CID(i, j, k + 1, B.Nx, B.Ny)]);
        if (!isForce) d_vz[cid] += betaz * (val - d_vz[cid]);
        else d_vz[cid] += (1.f - betaz) * val * B.inv_RHO_dt;
    }
}

/** \brief One face's blended velocity update (Eq. 8b pressure-gradient term), for the
 *         face stored at cell (i, j, k). Shared by step_velocity and the fused step so
 *         both compile identical arithmetic. */
inline REAL face_velocity_x(int i, int j, int k, device const REAL *d_p, device const REAL *d_vx, device const REAL *d_beta,
                            device const REAL *d_pmlN, device const REAL *d_pmlD, constant FdtdBatchParams &B) {
    int cid = CID(i, j, k, B.Nx, B.Ny);
    int min_xdist = min(i + 1, B.Nx - i - 1);
    REAL p = d_p[cid];
    REAL p_R = (i < B.Nx - 1) ? d_p[CID(i + 1, j, k, B.Nx, B.Ny)] : 0.f; // RIGHT
    REAL beta = d_beta[cid];
    REAL beta_R = (i < B.Nx - 1) ? d_beta[CID(i + 1, j, k, B.Nx, B.Ny)] : 1.f;
    REAL gradx = (p_R - p) * B.inv_dx;
    REAL betax = max(beta, beta_R);
    // pmlN = 1. and pmlD = 1. / (1. + damping) if outside PML layer
    return (d_pmlN[min_xdist] * d_vx[cid] - (1.f - betax) * B.inv_RHO_dt * gradx) * d_pmlD[min_xdist];
}
inline REAL face_velocity_y(int i, int j, int k, device const REAL *d_p, device const REAL *d_vy, device const REAL *d_beta,
                            device const REAL *d_pmlN, device const REAL *d_pmlD, constant FdtdBatchParams &B) {
    int cid = CID(i, j, k, B.Nx, B.Ny);
    int min_ydist = min(j + 1, B.Ny - j - 1);
    REAL p = d_p[cid];
    REAL p_U = (j < B.Ny - 1) ? d_p[CID(i, j + 1, k, B.Nx, B.Ny)] : 0.f; // UP
    REAL beta = d_beta[cid];
    REAL beta_U = (j < B.Ny - 1) ? d_beta[CID(i, j + 1, k, B.Nx, B.Ny)] : 1.f;
    REAL grady = (p_U - p) * B.inv_dx;
    REAL betay = max(beta, beta_U);
    return (d_pmlN[min_ydist] * d_vy[cid] - (1.f - betay) * B.inv_RHO_dt * grady) * d_pmlD[min_ydist];
}
inline REAL face_velocity_z(int i, int j, int k, device const REAL *d_p, device const REAL *d_vz, device const REAL *d_beta,
                            device const REAL *d_pmlN, device const REAL *d_pmlD, constant FdtdBatchParams &B) {
    int cid = CID(i, j, k, B.Nx, B.Ny);
    int min_zdist = min(k + 1, B.Nz - k - 1);
    REAL p = d_p[cid];
    REAL p_B = (k < B.Nz - 1) ? d_p[CID(i, j, k + 1, B.Nx, B.Ny)] : 0.f; // BACK
    REAL beta = d_beta[cid];
    REAL beta_B = (k < B.Nz - 1) ? d_beta[CID(i, j, k + 1, B.Nx, B.Ny)] : 1.f;
    REAL gradz = (p_B - p) * B.inv_dx;
    REAL betaz = max(beta, beta_B);
    return (d_pmlN[min_zdist] * d_vz[cid] - (1.f - betaz) * B.inv_RHO_dt * gradz) * d_pmlD[min_zdist];
}

/** \brief Blended velocity update kernel: pressure gradient term (Eq. 8b from [Xue et al. 2024]).
 *         Also samples the pressure field at listener cells into the listener output buffer. */
kernel void step_velocity(device const REAL *d_p [[buffer(0)]],
                          device REAL *d_vx [[buffer(4)]], device REAL *d_vy [[buffer(5)]], device REAL *d_vz [[buffer(6)]],
                          device const REAL *d_beta [[buffer(7)]],
                          device const REAL *d_pmlN [[buffer(11)]], device const REAL *d_pmlD [[buffer(12)]],
                          device const int *d_listenerCids [[buffer(15)]], device REAL *d_listenerOut [[buffer(16)]],
                          constant FdtdBatchParams &B [[buffer(FDTD_BATCH_PARAMS_INDEX)]],
                          constant FdtdStepParams &S [[buffer(FDTD_STEP_PARAMS_INDEX)]],
                          uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y, k = tid.z;
    if (i >= B.Nx || j >= B.Ny || k >= B.Nz) return;
    int cid = CID(i, j, k, B.Nx, B.Ny);

    d_vx[cid] = face_velocity_x(i, j, k, d_p, d_vx, d_beta, d_pmlN, d_pmlD, B);
    d_vy[cid] = face_velocity_y(i, j, k, d_p, d_vy, d_beta, d_pmlN, d_pmlD, B);
    d_vz[cid] = face_velocity_z(i, j, k, d_p, d_vz, d_beta, d_pmlN, d_pmlD, B);

    // Listener sampling (p is unchanged by the velocity update)
    for (int l = 0; l < B.N_listeners; ++l) {
        if (cid == d_listenerCids[l]) d_listenerOut[S.s * B.N_listeners + l] = d_p[cid];
    }
}

/** \brief Fused step: velocity update at step s followed by the pressure update at step
 *         s + 1, in one full-grid pass. Each thread computes all six face velocities its
 *         cell's divergence needs (the three lower faces recompute bit-identically in
 *         their owning threads via the shared face_velocity_* functions), writes its own
 *         three, then updates its pressure from the fresh divergence.
 *
 *         Reads p(s) and v from the in slots and writes p(s+1) and v(s) to the out slots
 *         (the host ping-pongs the bindings per step) — neighbor reads would otherwise
 *         race in-kernel writes. Beta updates live in update_beta for the same reason.
 *         Listener slot S.s samples p(s), the pre-update pressure. */
kernel void step_velocity_pressure(device const REAL *d_p [[buffer(0)]],
                                   device REAL *d_px [[buffer(1)]], device REAL *d_py [[buffer(2)]], device REAL *d_pz [[buffer(3)]],
                                   device const REAL *d_vx [[buffer(4)]], device const REAL *d_vy [[buffer(5)]], device const REAL *d_vz [[buffer(6)]],
                                   device const REAL *d_beta [[buffer(7)]],
                                   device const REAL *d_pmlNp [[buffer(9)]], device const REAL *d_pmlDp [[buffer(10)]],
                                   device const REAL *d_pmlNv [[buffer(11)]], device const REAL *d_pmlDv [[buffer(12)]],
                                   device const int *d_listenerCids [[buffer(15)]], device REAL *d_listenerOut [[buffer(16)]],
                                   device REAL *d_p_out [[buffer(21)]],
                                   device REAL *d_vx_out [[buffer(22)]], device REAL *d_vy_out [[buffer(23)]], device REAL *d_vz_out [[buffer(24)]],
                                   constant FdtdBatchParams &B [[buffer(FDTD_BATCH_PARAMS_INDEX)]],
                                   constant FdtdStepParams &S [[buffer(FDTD_STEP_PARAMS_INDEX)]],
                                   uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y, k = tid.z;
    if (i >= B.Nx || j >= B.Ny || k >= B.Nz) return;
    int cid = CID(i, j, k, B.Nx, B.Ny);

    // ----- Velocity update at step s: this cell's faces, plus the lower faces its
    // divergence needs -----
    REAL vx = face_velocity_x(i, j, k, d_p, d_vx, d_beta, d_pmlNv, d_pmlDv, B);
    REAL vy = face_velocity_y(i, j, k, d_p, d_vy, d_beta, d_pmlNv, d_pmlDv, B);
    REAL vz = face_velocity_z(i, j, k, d_p, d_vz, d_beta, d_pmlNv, d_pmlDv, B);
    REAL vx_L = (i > 0) ? face_velocity_x(i - 1, j, k, d_p, d_vx, d_beta, d_pmlNv, d_pmlDv, B) : 0.f; // LEFT
    REAL vy_D = (j > 0) ? face_velocity_y(i, j - 1, k, d_p, d_vy, d_beta, d_pmlNv, d_pmlDv, B) : 0.f; // DOWN
    REAL vz_F = (k > 0) ? face_velocity_z(i, j, k - 1, d_p, d_vz, d_beta, d_pmlNv, d_pmlDv, B) : 0.f; // FRONT

    REAL p_s = d_p[cid]; // p(s), for the listener sample
    d_vx_out[cid] = vx;
    d_vy_out[cid] = vy;
    d_vz_out[cid] = vz;

    for (int l = 0; l < B.N_listeners; ++l) {
        if (cid == d_listenerCids[l]) d_listenerOut[S.s * B.N_listeners + l] = p_s;
    }

    // ----- Pressure update at step s + 1 (Eq. 8a), from the fresh face velocities -----
    int min_xdist = min(i, B.Nx - 1 - i);
    int min_ydist = min(j, B.Ny - 1 - j);
    int min_zdist = min(k, B.Nz - 1 - k);

    REAL divx = (vx - vx_L) * B.inv_dx;
    REAL divy = (vy - vy_D) * B.inv_dx;
    REAL divz = (vz - vz_F) * B.inv_dx;

    if (min_xdist < PML_WIDTH || min_ydist < PML_WIDTH || min_zdist < PML_WIDTH) { // Inside PML: split pressure update
        REAL px = d_px[cid], py = d_py[cid], pz = d_pz[cid];

        px = (d_pmlNp[min_xdist] * px - B.RHO_CC_dt * divx) * d_pmlDp[min_xdist];
        py = (d_pmlNp[min_ydist] * py - B.RHO_CC_dt * divy) * d_pmlDp[min_ydist];
        pz = (d_pmlNp[min_zdist] * pz - B.RHO_CC_dt * divz) * d_pmlDp[min_zdist];

        d_px[cid] = px;
        d_py[cid] = py;
        d_pz[cid] = pz;
        d_p_out[cid] = px + py + pz;
    } else { // Otherwise, regular pressure update
        d_p_out[cid] = p_s - B.RHO_CC_dt * (divx + divy + divz);
    }
}

/** \brief Beta update (cubic smoothstep), over only the cells transitioning this batch.
 *         Each batch ends at tb = 1, so beta is exactly 0 or 1 at batch start and the
 *         per-step update set is exactly the cells whose solidity differs between the
 *         batch's raster endpoints — precomputed on the host. Entries encode
 *         (cid << 1) | 1 for air-to-solid, (cid << 1) for solid-to-air. */
kernel void update_beta(device REAL *d_beta [[buffer(7)]],
                        device const int *d_transitions [[buffer(20)]],
                        constant FdtdBatchParams &B [[buffer(FDTD_BATCH_PARAMS_INDEX)]],
                        constant FdtdStepParams &S [[buffer(FDTD_STEP_PARAMS_INDEX)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(B.N_transitions)) return;

    int encoded = d_transitions[tid];
    int cid = encoded >> 1;
    if (encoded & 1) {
        d_beta[cid] = 3.f * S.tb * S.tb - 2.f * S.tb * S.tb * S.tb;
    } else {
        REAL tb = 1.f - S.tb;
        d_beta[cid] = 3.f * tb * tb - 2.f * tb * tb * tb;
    }
}

// ====================== Fresh cells + shader re-init (WaveBlender.cu) ======================

/** \brief Computes a_n, required to enforce Neumann boundary conditions for fresh cell */
kernel void prepare_fresh_cell(device REAL *d_ax [[buffer(0)]], device REAL *d_ay [[buffer(1)]], device REAL *d_az [[buffer(2)]],
                               device const REAL *d_shaderData [[buffer(3)]], device const int *d_shaderMap [[buffer(4)]],
                               constant PrepareFreshCellParams &P [[buffer(5)]],
                               uint tid [[thread_position_in_grid]]) {
    int global_bid = tid;
    if (global_bid >= P.N_shader_points) return;

    // Decode shader map
    if (d_shaderMap[global_bid] < 0) return; // Skip if isForce
    int dir = d_shaderMap[global_bid] % 3;
    int cid = d_shaderMap[global_bid] / 3;

    // Decode shader values
    int floor_ = (global_bid * P.N_shader_samples);
    REAL vb0 = d_shaderData[floor_];
    REAL vb0_dt = (1.f - P.incr) * d_shaderData[floor_] + (P.incr)*d_shaderData[floor_ + 1];

    REAL a_0 = (vb0_dt - vb0) * P.inv_dt;
    if (dir == X_DIR) d_ax[cid] = a_0;
    else if (dir == Y_DIR) d_ay[cid] = a_0;
    else if (dir == Z_DIR) d_az[cid] = a_0;
}

/** \brief Eq. 13 from [Xue et al. 2024] */
kernel void fresh_cell_pressure(device REAL *d_p [[buffer(0)]],
                                device const REAL *d_ax [[buffer(1)]], device const REAL *d_ay [[buffer(2)]], device const REAL *d_az [[buffer(3)]],
                                device const REAL *d_beta [[buffer(4)]],
                                constant FreshCellPressureParams &P [[buffer(5)]],
                                uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x + PML_WIDTH;
    int j = tid.y + PML_WIDTH;
    int k = tid.z + PML_WIDTH;

    if (i >= P.Nx - 1 - PML_WIDTH || j >= P.Ny - 1 - PML_WIDTH || k >= P.Nz - 1 - PML_WIDTH) return;
    int cid = CID(i, j, k, P.Nx, P.Ny);

    int neighbor_cids[6] = {
        CID(i - 1, j, k, P.Nx, P.Ny), CID(i + 1, j, k, P.Nx, P.Ny), // left, right
        CID(i, j - 1, k, P.Nx, P.Ny), CID(i, j + 1, k, P.Nx, P.Ny), // down, up
        CID(i, j, k - 1, P.Nx, P.Ny), CID(i, j, k + 1, P.Nx, P.Ny) // front, back
    };
    REAL p_fresh = 0.f;
    int N_air = 0;
    for (int n = 0; n < 6; n++) {
        if (d_beta[neighbor_cids[n]] >= 1.f) continue; // neighboring cell is solid

        int dir = n / 3;
        REAL sign = (n % 2 == 0) ? 1.f : -1.f;
        int vid = (n % 2 == 0) ? neighbor_cids[n] : cid;

        if (dir == X_DIR) p_fresh += d_p[neighbor_cids[n]] - sign * P.rho_dx * d_ax[vid];
        else if (dir == Y_DIR) p_fresh += d_p[neighbor_cids[n]] - sign * P.rho_dx * d_ay[vid];
        else if (dir == Z_DIR) p_fresh += d_p[neighbor_cids[n]] - sign * P.rho_dx * d_az[vid];
        N_air += 1;
    }
    if (N_air > 0) p_fresh /= N_air;

    d_p[cid] = (1.f - d_beta[cid]) * d_p[cid] + d_beta[cid] * p_fresh;
}

/** \brief Sets interior velocities to 0 (and clears acceleration a_n buffers) */
kernel void clear_solid(device REAL *d_vx [[buffer(0)]], device REAL *d_vy [[buffer(1)]], device REAL *d_vz [[buffer(2)]],
                        device REAL *d_ax [[buffer(3)]], device REAL *d_ay [[buffer(4)]], device REAL *d_az [[buffer(5)]],
                        device const REAL *d_beta [[buffer(6)]],
                        constant ClearSolidParams &P [[buffer(7)]],
                        uint3 tid [[thread_position_in_grid]]) {
    int i = tid.x + PML_WIDTH;
    int j = tid.y + PML_WIDTH;
    int k = tid.z + PML_WIDTH;

    if (i >= P.Nx - 1 - PML_WIDTH || j >= P.Ny - 1 - PML_WIDTH || k >= P.Nz - 1 - PML_WIDTH) return;
    int cid = CID(i, j, k, P.Nx, P.Ny);

    if (d_beta[cid] >= 1.f) {
        if (d_beta[CID(i + 1, j, k, P.Nx, P.Ny)] >= 1.f) d_vx[cid] = 0.f;
        if (d_beta[CID(i, j + 1, k, P.Nx, P.Ny)] >= 1.f) d_vy[cid] = 0.f;
        if (d_beta[CID(i, j, k + 1, P.Nx, P.Ny)] >= 1.f) d_vz[cid] = 0.f;
    }
    d_ax[cid] = 0.f;
    d_ay[cid] = 0.f;
    d_az[cid] = 0.f;
}

/** \brief Section 6.2.2 from [Xue et al. 2024] */
kernel void shader_reinit(device const REAL *d_vx [[buffer(0)]], device const REAL *d_vy [[buffer(1)]], device const REAL *d_vz [[buffer(2)]],
                          device REAL *d_shaderData [[buffer(3)]], device const int *d_shaderMap [[buffer(4)]],
                          constant ShaderReInitParams &P [[buffer(5)]],
                          uint tid [[thread_position_in_grid]]) {
    int global_bid = tid;
    if (global_bid >= P.N_shader_points) return;

    // Decode shader map
    if (d_shaderMap[global_bid] < 0) return; // Skip if isForce
    int dir = d_shaderMap[global_bid] % 3;
    int cid = d_shaderMap[global_bid] / 3;

    // Decode shader values
    int floor_ = (global_bid * P.N_shader_samples);
    REAL vb0 = d_shaderData[floor_];
    REAL v_0 = 0.f;

    if (dir == X_DIR) v_0 = d_vx[cid];
    else if (dir == Y_DIR) v_0 = d_vy[cid];
    else if (dir == Z_DIR) v_0 = d_vz[cid];

    for (int t = 0; t < P.N_shader_samples; t++) d_shaderData[floor_ + t] -= (vb0 - v_0);
}

// ============================== Acoustic shaders (Shaders/*.cu) ==============================

/** \brief Monopole source (Eq. (D-5b), Blackstock pg. 358) */
kernel void monopole(device REAL *d_vb [[buffer(0)]],
                     device const REAL *d_B [[buffer(1)]], device const REAL *d_BN [[buffer(2)]],
                     constant MonopoleParams &P [[buffer(3)]],
                     uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, k = tid.y;
    if (i >= P.N_points || k >= P.N_samples) return;

    REAL x_src = (P.start_step >= P.N_samples - 1) ? -P.speed + P.speed * (P.start_step + k) / (P.N_samples - 1) : 0.f;
    REAL y_src = 0.f, z_src = 0.f;

    REAL x = d_B[3 * i], y = d_B[3 * i + 1], z = d_B[3 * i + 2];
    REAL xn = d_BN[3 * i], yn = d_BN[3 * i + 1], zn = d_BN[3 * i + 2];

    REAL time = ((REAL)(P.start_step + k) / P.srate);

    REAL r = sqrt((x - x_src) * (x - x_src) + (y - y_src) * (y - y_src) + (z - z_src) * (z - z_src));
    REAL vb = (cos(2.f * PI_F * P.freqHz * (time - r / P.C)) / (r * r * 2.f * PI_F * P.freqHz)
               - sin(2.f * PI_F * P.freqHz * (time - r / P.C)) / (r * P.C))
        * (xn + yn + zn);

    if (time - r / P.C < 0.f) vb = 0.f;
    d_vb[(P.global_bid + i) * P.N_samples + k] = vb;
}

/** \brief Section 5.1.1 from [Xue et al. 2024] */
kernel void speaker(device REAL *d_vb [[buffer(0)]],
                    device const REAL *d_BN [[buffer(1)]], device const REAL *d_audio [[buffer(2)]],
                    constant SpeakerParams &P [[buffer(3)]],
                    uint2 tid [[thread_position_in_grid]]) {
    const REAL SCALE = 100.f; // Manually scale input audio amplitude

    int i = tid.x, k = tid.y;
    if (i >= P.N_points || k >= P.N_samples) return;

    REAL xn = d_BN[3 * i], yn = d_BN[3 * i + 1], zn = d_BN[3 * i + 2];

    REAL vb = 0.f; // Manually specify speaker direction
    if (P.dir == 0 && xn <= -1.f) vb = -d_audio[P.start_step + k]; // LEFT
    else if (P.dir == 1 && xn >= 1.f) vb = d_audio[P.start_step + k]; // RIGHT
    else if (P.dir == 2 && yn <= -1.f) vb = -d_audio[P.start_step + k]; // DOWN
    else if (P.dir == 3 && yn >= 1.f) vb = d_audio[P.start_step + k]; // UP
    else if (P.dir == 4 && zn <= -1.f) vb = -d_audio[P.start_step + k]; // FRONT
    else if (P.dir == 5 && zn >= 1.f) vb = d_audio[P.start_step + k]; // BACK

    d_vb[(P.global_bid + i) * P.N_samples + k] = SCALE * vb;
}

/** \brief Projects modal velocities to boundary faces (interpolating the transfer matrix over the batch) */
kernel void mode_matmul(device REAL *d_vb [[buffer(0)]],
                        device const REAL *d_modeToBoundary1 [[buffer(1)]], device const REAL *d_modeToBoundary2 [[buffer(2)]],
                        device const REAL *d_modeVels [[buffer(3)]], device const REAL *d_accelNoise [[buffer(4)]],
                        constant ModeMatmulParams &P [[buffer(5)]],
                        uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, k = tid.y;
    if (i >= P.N_points || k >= P.N_samples) return;

    REAL vb = 0.f;
    REAL alpha = ((REAL)k) / (P.N_samples - 1.f);
    for (int j = 0; j < P.N_modes; j++) {
        vb += (1.f - alpha) * d_modeToBoundary1[P.N_points * j + i] * d_modeVels[P.N_modes * k + j];
        vb += alpha * d_modeToBoundary2[P.N_points * j + i] * d_modeVels[P.N_modes * k + j];
    }
    d_vb[(P.global_bid + i) * P.N_samples + k] = vb + d_accelNoise[P.N_points * k + i];
}

/** \brief Constructs (N_points x N_bubs) "bubble-to-boundary" transfer matrix (Eq. 11 from [Xue et al. 2024]) */
kernel void bub_to_boundary(device REAL *d_bubToBoundary [[buffer(0)]],
                            device const REAL *d_B [[buffer(1)]], device const REAL *d_BN [[buffer(2)]],
                            device const REAL *d_bubData [[buffer(3)]],
                            constant BubToBoundaryParams &P [[buffer(4)]],
                            uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y;
    if (i >= P.N_points || j >= P.N_bubs) return;

    REAL x = d_B[3 * i], y = d_B[3 * i + 1], z = d_B[3 * i + 2];
    REAL xn = d_BN[3 * i], yn = d_BN[3 * i + 1], zn = d_BN[3 * i + 2];

    REAL r = d_bubData[4 * j];
    REAL xbub = d_bubData[4 * j + 1], ybub = d_bubData[4 * j + 2], zbub = d_bubData[4 * j + 3];

    REAL dist = sqrt((x - xbub) * (x - xbub) + (y - ybub) * (y - ybub) + (z - zbub) * (z - zbub));
    REAL A_bub = ((x - xbub) * abs(xn) + (y - ybub) * abs(yn) + (z - zbub) * abs(zn)) / (dist * 4.f * PI_F);

    // Special handling for d < r case (avoid d = 0 blowing up, etc.)
    if (A_bub < 0.f) A_bub = 0.f;
    else if (dist < r) A_bub *= -3.f * dist * dist / (r * r * r * r) + 4.f * dist / (r * r * r);
    else A_bub *= 1.f / (dist * dist);

    d_bubToBoundary[P.N_points * j + i] = A_bub;
}

/** \brief Per-bubble boundary flux: alpha * column sums of the transfer matrix
 *         (replaces cublasSgemv(OP_T, ..., ones, ...) in the CUDA reference) */
kernel void bub_flux(device const REAL *d_bubToBoundary [[buffer(0)]], device REAL *d_flux [[buffer(1)]],
                     constant BubFluxParams &P [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    int j = tid;
    if (j >= P.N_bubs) return;

    REAL sum = 0.f;
    for (int i = 0; i < P.N_points; i++) sum += d_bubToBoundary[P.N_points * j + i];
    d_flux[j] = P.alpha * sum;
}

/** \brief Multiplies "bubble-to-boundary" transfer matrix with bubble volume velocities to compute vb */
kernel void bub_matmul(device REAL *d_vb [[buffer(0)]],
                       device const REAL *d_bubToBoundary [[buffer(1)]], device const REAL *d_bubVels [[buffer(2)]],
                       device const REAL *d_flux [[buffer(3)]],
                       constant BubMatmulParams &P [[buffer(4)]],
                       uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, k = tid.y;
    if (i >= P.N_points || k >= P.stop - P.start) return;

    REAL vb = 0.f;
    for (int j = 0; j < P.N_bubs; j++) {
        REAL flux = (d_flux[j] > 1e-6f) ? d_flux[j] : 1.f;
        vb += d_bubToBoundary[P.N_points * j + i] * d_bubVels[P.N_bubs * k + j] / flux;
    }
    d_vb[(P.global_bid + i) * P.N_samples + (P.start + k)] = vb;
}

/** \brief Constructs (N_points x N_impulses) "impulse-to-boundary" (particle-to-grid) transfer matrix */
kernel void impulse_to_boundary(device REAL *d_impulseToBoundary [[buffer(0)]],
                                device const REAL *d_B [[buffer(1)]], device const REAL *d_BN [[buffer(2)]],
                                device const REAL *d_impulseData [[buffer(3)]],
                                constant ImpulseToBoundaryParams &P [[buffer(4)]],
                                uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, j = tid.y;
    if (i >= P.N_points || j >= P.N_impulses) return;

    REAL x = d_B[3 * i], y = d_B[3 * i + 1], z = d_B[3 * i + 2];
    REAL xn = d_BN[3 * i], yn = d_BN[3 * i + 1], zn = d_BN[3 * i + 2];

    REAL xpt = d_impulseData[6 * j], ypt = d_impulseData[6 * j + 1], zpt = d_impulseData[6 * j + 2];
    REAL xpt_dir = d_impulseData[6 * j + 3], ypt_dir = d_impulseData[6 * j + 4], zpt_dir = d_impulseData[6 * j + 5];

    REAL distx = abs(x - xpt), disty = abs(y - ypt), distz = abs(z - zpt);

    REAL weight = 0.f;
    if (distx < P.dx && disty < P.dx && distz < P.dx) weight = (1.f - distx / P.dx) * (1.f - disty / P.dx) * (1.f - distz / P.dx);

    if (xn != 0.f) weight *= xpt_dir;
    else if (yn != 0.f) weight *= ypt_dir;
    else if (zn != 0.f) weight *= zpt_dir;

    d_impulseToBoundary[P.N_points * j + i] = weight;
}

/** \brief Multiplies "impulse-to-boundary" transfer matrix with impulse magnitudes to compute forces */
kernel void impulse_matmul(device REAL *d_force [[buffer(0)]],
                           device const REAL *d_impulseToBoundary [[buffer(1)]], device const REAL *d_impulseVels [[buffer(2)]],
                           constant ImpulseMatmulParams &P [[buffer(3)]],
                           uint2 tid [[thread_position_in_grid]]) {
    int i = tid.x, k = tid.y;
    if (i >= P.N_points || k >= P.N_samples) return;

    REAL vb = 0.f;
    for (int j = 0; j < P.N_impulses; j++) {
        vb += d_impulseToBoundary[P.N_points * j + i] * d_impulseVels[P.N_impulses * k + j];
    }
    d_force[(P.global_bid + i) * P.N_samples + k] = vb;
}
