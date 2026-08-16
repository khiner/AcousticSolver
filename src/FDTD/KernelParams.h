/** Metal port of WaveBlender.
 *
 * \file KernelParams.h
 * \brief Parameter blocks shared between host (C++) and device (MSL).
 *
 * This header is compiled both as C++ and, prepended to Kernels.metal, as MSL —
 * keep it to plain structs of 4-byte int/float members so the layouts agree.
 */

#ifndef KERNEL_PARAMS_H // include guard rather than pragma once: this header is textually prepended to Kernels.metal
#define KERNEL_PARAMS_H

#define PML_WIDTH 8

// ----- Core FDTD step kernels -----
// The batch's step kernels share one argument table:
//   0:P(in) 1:Px 2:Py 3:Pz 4:Vx(in) 5:Vy(in) 6:Vz(in) 7:ShaderFaces 8:CellState
//   9:PmlNp 10:PmlDp 11:PmlNv 12:PmlDv 13:ShaderData 14:ShaderMap 15:ListenerCids
//   16:ListenerOut 17:FdtdBatchParams (setBytes, once per batch)
//   18:FdtdStepParams (setBytes, once per step) 19:ApplyShaderRange (setBytes, per
//   standalone apply_shader dispatch) 21:P(out) 22:Vx(out) 23:Vy(out) 24:Vz(out)
// The in/out field slots ping-pong per fused step, everything else binds once per batch.
#define FDTD_BATCH_PARAMS_INDEX 17
#define FDTD_STEP_PARAMS_INDEX 18
#define FDTD_APPLY_RANGE_INDEX 19

// Per-cell blending state for the batch. Beta derives in-kernel via beta_of: AIR -> 0,
// SOLID -> 1, RISING/FALLING -> the cubic smoothstep of the step's blending time. Each
// batch ends at tb = 1, so states re-converge to AIR/SOLID by batch end.
#define CELL_AIR 0
#define CELL_SOLID 1
#define CELL_RISING 2
#define CELL_FALLING 3

// The boundary application (the reference's apply_shader kernel) has two paths, both
// bit-exact. Runtime probe batches lock whichever is faster per scene, via the
// FOLD_APPLY function constant specialized at pipeline creation:
//   - Standalone apply_shader dispatches per step. Velocity-blend points and force
//     points dispatch separately (blend first) via ApplyShaderRange, so a face claimed
//     by both updates in a defined order.
//   - Folded into the velocity updates, which transform their velocity reads on the fly
//     through the ShaderFaces lookup below, applying blend before force to match that
//     order.
#define FOLD_APPLY_FC_INDEX 0

// Threadgroup tile dims of the full-grid step kernels (host dispatch and tile flags)
#define FDTD_TGX 32
#define FDTD_TGY 4
#define FDTD_TGZ 4

// ShaderFaces buffer layout, shared by the host writer and both velocity kernels:
// [mask: G bytes][pad to 4][6 row-id int planes][tile flags], G = Nx * Ny * Nz.
// A cell's mask bit `dir` marks a pending blend on that face and bit `dir + 3` a pending
// force. The row-id planes (blend x/y/z then force x/y/z) are read only where the mask
// hits, and one flag byte per threadgroup tile marks tiles whose threads can read a
// nonzero mask (a face cell in the tile or its lower halo) — unflagged tiles skip all
// mask reads.
#define SHADER_FACES_BIDS_OFFSET(G) (((G) + 3) & ~3)
#define SHADER_FACES_TILES_OFFSET(G) (SHADER_FACES_BIDS_OFFSET(G) + 6 * (G) * 4)
#define FDTD_TILE_INDEX(tx, ty, tz, ntx, nty) (((tz) * (nty) + (ty)) * (ntx) + (tx))

struct FdtdBatchParams {
    float RHO_CC_dt;
    float inv_dx;
    float inv_RHO_dt;
    int Nx;
    int Ny;
    int Nz;
    int N_shader_samples;
    int N_listeners;
};
// Timing of the step whose boundary application is pending (one step behind the
// dispatch that consumes it: the fused step [V(q), P(q+1)] gets step q's params).
struct FdtdStepParams {
    float tb; // normalized blending time (0, 1]
    float ss; // fractional shader sample index (for interpolation)
    int s; // index within the batch (listener output slot)
};
// Shader-point range for one standalone apply_shader dispatch.
struct ApplyShaderRange {
    int start;
    int end;
};

// ----- Fresh cells + shader re-init -----
struct PrepareFreshCellParams {
    int N_shader_samples;
    int N_shader_points;
    float inv_dt;
    float incr;
};
struct FreshCellPressureParams {
    int Nx;
    int Ny;
    int Nz;
    float rho_dx;
};
struct ClearSolidParams {
    int Nx;
    int Ny;
    int Nz;
};
struct ShaderReInitParams {
    int N_shader_samples;
    int N_shader_points;
};

// ----- Acoustic shaders -----
struct MonopoleParams {
    int global_bid;
    float freqHz;
    float speed;
    float C;
    int srate;
    int N_points;
    int N_samples;
    int start_step;
};
struct SpeakerParams {
    int global_bid;
    int dir;
    int N_points;
    int N_samples;
    int start_step;
};
struct ModeMatmulParams {
    int global_bid;
    int N_points;
    int N_modes;
    int N_samples;
};

struct BubToBoundaryParams {
    int N_points;
    int N_bubs;
};
struct BubFluxParams {
    int N_points;
    int N_bubs;
    float alpha;
};
struct BubMatmulParams {
    int global_bid;
    int N_points;
    int N_bubs;
    int N_samples;
    int start;
    int stop;
};

struct ImpulseToBoundaryParams {
    int N_points;
    int N_impulses;
    float dx;
};
struct ImpulseMatmulParams {
    int global_bid;
    int N_points;
    int N_impulses;
    int N_samples;
};

#endif // #ifndef KERNEL_PARAMS_H
