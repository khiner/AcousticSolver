// Prepended to ImmersedKernels.metal at runtime. Use only plain 4-byte members;
// neither language's standard library is available.

#ifndef IMMERSED_PARAMS_H
#define IMMERSED_PARAMS_H

struct ImmersedGridParams {
    int Nx, Ny, Nz;
    int PmlWidth;
    int NumSources;
    int NumReceivers;
    int NumSteps;
    int NumSourceCells;
    int NumPatches;
    int NumVelocityActive;
    int NumPressureActive;
    int NumVelocityHistoryCells;
    int NumPressureHistoryCells;
    int NumVelocityCorrectionCells;
    int NumPressureCorrectionCells;
    int HasSolid;
    float InvRhoDtH;
    float RhoCcDtH;
    float SourceScale;
    float Courant;
};

struct ImmersedStepParams {
    int Step;
};

struct ImmersedIoCell {
    int Ixyz;
    int Begin;
    int End;
};

// Source tables use Index as a source number. Receiver tables use it as a
// pressure-cell index. In both cases Weight is one product-form delta weight.
struct ImmersedWeightedIndex {
    int Index;
    float Weight;
};

struct ImmersedFilter {
    float B0, B1, B2;
    float A1, A2;
};

struct ImmersedFilterState {
    float X1, X2;
    float Y1, Y2;
    float Previous;
};

struct ImmersedSolveParams {
    int Count;
};

#endif // IMMERSED_PARAMS_H
