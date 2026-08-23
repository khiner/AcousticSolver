// SonicRadiation solver: parameter blocks and table records shared between host (C++)
// and device (MSL).
//
// This header is compiled both as C++ and, prepended to RadiationKernels.metal, as MSL —
// keep it to plain structs of 4-byte int/float members so the layouts agree, and to
// fundamental types (it lands ahead of <metal_stdlib>, so nothing from the standard
// library exists yet).

#ifndef RADIATION_PARAMS_H // Include guard rather than pragma once: textually prepended to RadiationKernels.metal
#define RADIATION_PARAMS_H

// Longest history ring the convolution passes can stage in threadgroup memory. Rings are
// sized to the largest lag any table reaches — usually the listener table, which reaches
// much further than either convolution table — so the passes stage far more than they read.
// Staging each pass's own reach instead would decouple the two, for about 1%.
enum : int { RadMaxHistLen = 512 };

// Zeroed lag slots the convolution passes stage past the history ring, so a zero-padded
// window (see RadStagedEntry) can be read four lags at a time right up to the ring's end.
enum : int { RadStageTail = 4 };

// Selects a cut-down build of the two convolution passes, for attributing their fixed cost.
// Set by ACOUSTIC_RAD_FLOOR_PROBE and compiled in, so the shipping path (0) carries no
// branch. Every level above 0 computes the wrong answer on purpose. See VALIDATION for the
// measurements and how to read them.
//
//   0  off, the shipping path
//   1  the floor itself: one entry per source, everything else intact
//   2  as 1, with the staged history filled from a constant instead of read from the ring,
//      which leaves the fill and the barrier in place and takes out the ring traffic
//   3  as 1, without the QuadSum reduction
//   4  as 1, without the write block (a sink the compiler cannot fold keeps the work)
//   5  nothing but the dispatch: every threadgroup returns immediately
enum : int { RadFloorProbeFcIndex = 0 };

// Sign bit of a RadRef's packed entry index (spelled without a shift: MSL's base language
// leaves shifting into the sign bit undefined).
enum : int { RadRefNegative = -2147483647 - 1 };

// One (target, source element) CQM weight series, packed into a table's shared weight
// array: the double-layer window covers lags [DBegin, DBegin + DLen) starting at Off, and
// the single-layer window [VBegin, VBegin + VLen) follows it immediately at Off + DLen.
// Lags and window lengths are bounded by the history ring, so 16 bits carry them.
struct RadEntry {
    int Src;
    int Off;
    unsigned short DBegin, DLen, VBegin, VLen;
};

// The same series, repacked for the two hot tables the per-step convolution passes stream.
// Those passes are bound by this stream and stage their source element's history
// themselves, so `Src` is gone and every field is as narrow as it goes (see VALIDATION).
//
// Weights are signed bytes scaled by their window's binary exponent, `DExp` or `VExp` — one
// scale per window, never one shared between them, since the double- and single-layer
// series differ by orders of magnitude and sharing costs 5-19 dB. The lag-0 double-layer
// weight is zeroed here and carried separately, so the pass needs no special case for the
// unknown current phi.
//
// Windows are widened down to a four-lag boundary and up to a four-lag length with zero
// weights, so the pass reads four weights and four staged lags at a time. The window fields
// therefore count *quads* of lags, which is what lets a byte cover the ring's whole range.
struct RadStagedEntry {
    int Off;
    unsigned char DBeginQ, DLenQ, VBeginQ, VLenQ;
    signed char DExp, VExp;
};

// A signed reference to a table entry, for the Laplacian set-difference corrections
// (Eq. 12) and the interpolation re-basing corrections (Eq. 20). The sign is only ever
// +1 or -1, so it rides in the index's top bit rather than doubling the record.
struct RadRef {
    int Packed;
};

// Trilinear interpolation corner of one element's far-field feedback (Eqs. 19-20).
struct RadCorner {
    int Cell;
    float Gamma;
    int RefBegin, RefEnd;
};

// Constant for the whole run.
struct RadGridParams {
    int Nx, Ny, Nz;
    int PmlWidth;
    int NumElems;
    int HistLen; // power of two
    int NumCorrCells;
    int NumListeners;
    float S2; // (c tau / h)^2
    float InvDamp; // 1 / (1 + damping)
    float InvRhoDtH; // tau / (rho h)
    float RhoCcDtH; // rho c^2 tau / h
    int FilterFeedback;
    int EvalGroups; // threadgroups the Eq. 5 listener pass splits each point's entries over
};

// Per-step state: `N1` is the step being computed, `N` the last completed one.
struct RadStepParams {
    int N, N1;
    int GOffset; // row of the batch's Neumann samples holding g(N1)
    int SampleIndex; // listener output slot for this step
    int Count; // work items, for the passes that are not sized by RadGridParams
};

#endif // #ifndef RADIATION_PARAMS_H
