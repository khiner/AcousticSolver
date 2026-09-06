#pragma once

#define DG_NODE_SIMDS 8
#define DG_MASS_SIMDS 4

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
