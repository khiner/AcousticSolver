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
//   18:FdtdStepParams of the step being computed (setBytes, once per step)
//   19:ApplyShaderRange (setBytes, per standalone ApplyShader dispatch)
//   20:FdtdStepParams of the *next* step, whose boundary application the fused kernel
//   folds into its velocity writes  21:P(out) 22:Vx(out) 23:Vy(out) 24:Vz(out)
// The in/out field slots ping-pong per fused step, everything else binds once per batch.
enum : int {
    FdtdBatchParamsIndex = 17,
    FdtdStepParamsIndex = 18,
    FdtdApplyRangeIndex = 19,
    FdtdNextStepParamsIndex = 20,
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

// The boundary application (the reference's apply_shader kernel) lands on the velocities
// as they are written: the step computing V(q) applies step q+1's to the three faces it
// writes, so blend-before-force ordering on a dual-claimed face is automatic. Only step
// 0, on inherited velocities, dispatches ApplyShader. ApplyFaces compiles the fold out.
enum : int { ApplyFacesFcIndex = 0 };

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

// ShaderFaces buffer layout, shared by the host writer and the fused step kernel:
// [mask: G bytes][pad to 4][slot: G ints][tile flags: NTiles bytes][pad to 4][bids ints],
// G = Nx * Ny * Nz. A cell's mask bit `dir` marks a pending blend on that face and bit
// `dir + 3` a pending force. (cell, plane) pairs are unique, so row ids pack contiguously:
// plane `pl`'s is bids[slot[cell] + popcount(mask & ((1 << pl) - 1))]. Only the mask is
// read densely, and only in tiles their flag byte marks as holding a face cell.
constexpr long ShaderFacesSlotOffset(long g) { return (g + 3) & ~3L; }
constexpr long ShaderFacesTilesOffset(long g) { return ShaderFacesSlotOffset(g) + g * 4; }
constexpr long ShaderFacesBidsOffset(long g, long n_tiles) { return (ShaderFacesTilesOffset(g) + n_tiles + 3) & ~3L; }
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
    int NTilesZ;
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
