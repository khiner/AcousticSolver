#pragma once

#ifndef __METAL_VERSION__
#include "PlateReal.h"
#include <cstdint>
using PlateUint = std::uint32_t;
#else
using PlateUint = uint;
#endif

// Each transform dot product uses one four-lane subgroup of a 32-lane SIMD.
enum : PlateUint { PlateTransformLanes = 4 };

struct PlateParams {
    PlateUint Mx, My, Nx, Ny;
    PlateUint Modes, AiryModes, Grid, Nonlinear;
    PlateUint Receivers, Sample, RecordStates;
    PlateUint NormType, HasForces;
    PlateReal Dt, Kappa2, Lambda, Epsilon, Scale, Dt2, StepKappa2, HalfInvDt, InvDt2, InvScale;
};

enum PlateFftStage : PlateUint {
    FftDerivativeX,
    FftDerivativeY,
    FftProjectX,
    FftAiryY,
    FftPhiX,
    FftPhiY,
    FftForceX,
    FftForceY
};

struct PlateFftParams {
    PlateParams Plate;
    PlateUint N, Length, Bits, Chirp, Phase, Kernel, Waves;
    PlateReal AnalysisNorm, SynthesisNorm, Sqrt2, InvLength;
};
