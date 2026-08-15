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

struct StepPressureParams {
    float tb;
    float RHO_CC_dt;
    float inv_dx;
    int Nx;
    int Ny;
    int Nz;
};
struct ApplyShaderParams {
    int N_shader_samples;
    int N_shader_points;
    float inv_RHO_dt;
    int Nx;
    int Ny;
    int Nz;
    float ss;
};
struct StepVelocityParams {
    float inv_RHO_dt;
    float inv_dx;
    int Nx;
    int Ny;
    int Nz;
};
struct SampleListenerParams {
    int cid;
    int s;
};

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
