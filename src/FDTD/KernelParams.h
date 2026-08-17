// Metal port of WaveBlender. Parameter blocks shared between host (C++) and device (MSL).
//
// This header is compiled both as C++ and, prepended to Kernels.metal, as MSL — keep it to
// plain structs of 4-byte int/float members so the layouts agree, and to fundamental types
// (it lands ahead of <metal_stdlib>, so nothing from the standard library exists yet).

#ifndef KERNEL_PARAMS_H // Include guard rather than pragma once: this header is textually prepended to Kernels.metal
#define KERNEL_PARAMS_H

// Shared integer constants are enumerators rather than `constexpr` variables: MSL requires
// every program-scope variable to live in the constant address space, while enumerators are
// integer constant expressions in both languages and so can also index buffer attributes.
enum : int { PmlWidth = 8 };

// ----- Core FDTD step kernels -----
// The batch's step kernels share one argument table:
//   0:P(in) 1:PsiX 2:PsiY 3:PsiZ (shell-packed) 4:Vx(in) 5:Vy(in) 6:Vz(in) 7:ShaderFaces 8:CellState
//   9:PmlNp 10:PmlDp 11:PmlNv 12:PmlDv 13:ShaderData 14:ShaderMap 15:ListenerCids
//   16:ListenerOut 17:FdtdBatchParams (setBytes, once per batch)
//   18:FdtdStepParams (setBytes, once per step) 19:ApplyShaderRange (setBytes, per
//   standalone ApplyShader dispatch) 21:P(out) 22:Vx(out) 23:Vy(out) 24:Vz(out)
// The in/out field slots ping-pong per fused step, everything else binds once per batch.
enum : int {
    FdtdBatchParamsIndex = 17,
    FdtdStepParamsIndex = 18,
    FdtdApplyRangeIndex = 19,
};

// Per-cell blending state for the batch. Beta derives in-kernel via BetaOf: Air -> 0,
// Solid -> 1, Rising/Falling -> the cubic smoothstep of the step's blending time. Each
// batch ends at tb = 1, so states re-converge to Air/Solid by batch end.
enum : int {
    CellAir = 0,
    CellSolid = 1,
    CellRising = 2,
    CellFalling = 3,
};

// The boundary application (the reference's apply_shader kernel) has two paths, both
// bit-exact. Runtime probe batches lock whichever is faster per scene, via the FoldApply
// function constant specialized at pipeline creation:
//   - Standalone ApplyShader dispatches per step. Velocity-blend points and force points
//     dispatch separately (blend first) via ApplyShaderRange, so a face claimed by both
//     updates in a defined order.
//   - Folded into the velocity updates, which transform their velocity reads on the fly
//     through the ShaderFaces lookup below, applying blend before force to match that
//     order.
enum : int { FoldApplyFcIndex = 0 };

// Threadgroup tile dims of the full-grid step kernels. The host picks per scene and
// passes the tile-grid extents in FdtdBatchParams for the kernel-side flag lookup.
enum : int {
    FdtdTgXSmall = 32,
    FdtdTgYSmall = 4,
    FdtdTgZSmall = 4,
    FdtdTgXWide = 128,
    FdtdTgYWide = 4,
    FdtdTgZWide = 1,
};

// ShaderFaces buffer layout, shared by the host writer and both velocity kernels:
// [mask: G bytes][pad to 4][6 row-id int planes][tile flags], G = Nx * Ny * Nz.
// A cell's mask bit `dir` marks a pending blend on that face and bit `dir + 3` a pending
// force. The row-id planes (blend x/y/z then force x/y/z) are read only where the mask
// hits, and one flag byte per threadgroup tile marks tiles whose threads can read a
// nonzero mask (a face cell in the tile or its lower halo) — unflagged tiles skip all
// mask reads.
constexpr long ShaderFacesBidsOffset(long g) { return (g + 3) & ~3L; }
constexpr long ShaderFacesTilesOffset(long g) { return ShaderFacesBidsOffset(g) + 6 * g * 4; }
constexpr int FdtdTileIndex(int tx, int ty, int tz, int ntx, int nty) { return (tz * nty + ty) * ntx + tx; }

struct FdtdBatchParams {
    float RhoCcDt;
    float InvDx;
    float InvRhoDt;
    int Nx;
    int Ny;
    int Nz;
    int NShaderSamples;
    int NListeners;
    int NTilesX; // threadgroup counts of the full-grid dispatch, for the tile-flag lookup
    int NTilesY;
};
// Timing of the step whose boundary application is pending (one step behind the
// dispatch that consumes it: the fused step [V(q), P(q+1)] gets step q's params).
struct FdtdStepParams {
    float Tb; // Normalized blending time (0, 1]
    float SampleFrac; // Fractional shader sample index (for interpolation)
    int SampleIndex; // Index within the batch (listener output slot)
};
// Shader-point range for one standalone ApplyShader dispatch.
struct ApplyShaderRange {
    int Start;
    int End;
};

// ----- Fresh cells + shader re-init -----
struct PrepareFreshCellParams {
    int NShaderSamples;
    int NShaderPoints;
    float InvDt;
    float Incr;
};
struct FreshCellPressureParams {
    int Nx;
    int Ny;
    int Nz;
    float RhoDx;
};
struct ClearSolidParams {
    int Nx;
    int Ny;
    int Nz;
};
struct ShaderReInitParams {
    int NShaderSamples;
    int NShaderPoints;
};

// ----- Acoustic shaders -----
struct MonopoleParams {
    int GlobalBid;
    float FreqHz;
    float Speed;
    float C;
    int Srate;
    int NPoints;
    int NSamples;
    int StartStep;
};
struct SpeakerParams {
    int GlobalBid;
    int Dir;
    int NPoints;
    int NSamples;
    int StartStep;
};
struct ModeMatmulParams {
    int GlobalBid;
    int NPoints;
    int NModes;
    int NSamples;
};

struct BubToBoundaryParams {
    int NPoints;
    int NBubs;
};
struct BubFluxParams {
    int NPoints;
    int NBubs;
    float Alpha;
};
struct BubMatmulParams {
    int GlobalBid;
    int NPoints;
    int NBubs;
    int NSamples;
    int Start;
    int Stop;
};

struct ImpulseToBoundaryParams {
    int NPoints;
    int NImpulses;
    float Dx;
};
struct ImpulseMatmulParams {
    int GlobalBid;
    int NPoints;
    int NImpulses;
    int NSamples;
};

#endif // #ifndef KERNEL_PARAMS_H
