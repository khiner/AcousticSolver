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
//   0:P(in) 1:Px 2:Py 3:Pz 4:Vx(in) 5:Vy(in) 6:Vz(in) 7:Beta 8:Cell 9:PmlNp 10:PmlDp
//   11:PmlNv 12:PmlDv 13:ShaderData 14:ShaderMap 15:ListenerCids 16:ListenerOut
//   17:FdtdBatchParams (setBytes, once per batch) 18:FdtdStepParams (setBytes, once per step)
//   19:ApplyShaderRange (setBytes, per apply_shader dispatch) 20:BetaTransitions
//   21:P(out) 22:Vx(out) 23:Vy(out) 24:Vz(out)
// The in/out field slots ping-pong per fused step, everything else binds once per batch.
#define FDTD_BATCH_PARAMS_INDEX 17
#define FDTD_STEP_PARAMS_INDEX 18
#define FDTD_APPLY_RANGE_INDEX 19

struct FdtdBatchParams {
    float RHO_CC_dt;
    float inv_dx;
    float inv_RHO_dt;
    int Nx;
    int Ny;
    int Nz;
    int N_shader_samples;
    int N_listeners;
    int N_transitions; // cells changing solidity this batch (see update_beta)
};
struct FdtdStepParams {
    float tb; // normalized blending time (0, 1]
    float ss; // fractional shader sample index (for interpolation)
    int s; // FDTD sample index within the batch (listener output slot)
};
// Shader-point range for one apply_shader dispatch. Velocity-blend points and force
// points dispatch separately (blend first): a face claimed by both — possible between a
// regular shader and a point source — then updates in a defined order instead of racing.
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
