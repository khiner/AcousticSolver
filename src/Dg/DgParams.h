#pragma once

#define DG_NODE_SIMDS 32

#ifndef __METAL_VERSION__
#include <cstdint>
using DgUint = uint32_t;
#else
using DgUint = uint;
#endif

struct DgParams {
    DgUint Elements, Nodes, FaceNodes, QuadraturePoints;
    DgUint Receivers, Sample, Steps, Update;
    float SoundSpeed, Dt, RkA, RkB;
};

struct DgFace {
    float Nx, Ny, Nz;
    DgUint Offset;
};
