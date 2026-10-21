#pragma once
#ifndef __METAL_VERSION__
#include "String/StringReal.h"
#include <cstdint>
using uint = uint32_t;
#endif
using Real = StringReal;
struct CoupledMatrix {
    uint Rows{}, Row{}, Column{}, Value{};
};
struct PanelParams {
    uint N{}, Stride{}, StateOffset{}, Iterations{}, Segments{}, Nonlinear{}, MaxIterations{}, IterationsPerDispatch{}, Adjoint{}, MassInverse{};
    uint Derivatives{}, Mask{}, Inverse{}, PreconditionSize{}, PreconditionIndices{}, ForceCount{}, ForceOffset{}, ForceIndices{};
    uint Channels{}, Channel{}, Pickup{};
    Real WScale{}, FScale{}, InvH2{}, Tol2{}, PreconditionBound{};
    CoupledMatrix B, C, K, IJ, I, J;
};
struct PanelDispatch {
    uint Phase{}, Step{}, Solve{};
};
struct AirParams {
    uint N{}, Ny{}, Slice{}, Lower{}, Step{}, Channels{}, Receivers{}, Reflecting{};
    Real Gamma2{}, Lambda2{}, QRate{}, Rate{};
};
struct InterfaceParams {
    PanelParams Panel;
    AirParams Air;
};
