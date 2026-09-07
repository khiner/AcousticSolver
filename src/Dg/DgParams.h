#pragma once

#define DG_NODE_SIMDS 32

#ifndef __METAL_VERSION__
#include <cstdint>
using DgUint = uint32_t;
#else
using DgUint = uint;
#endif

// Degree 4 has 35 volume nodes and 15 face nodes; use both halves of each SIMD group.
constexpr DgUint DgNodeLanes(DgUint nodes) { return nodes <= 35 ? 16 : 32; }

struct DgParams {
    DgUint Elements, Nodes, FaceNodes, QuadraturePoints;
    DgUint Receivers, Sample, Steps, Update;
    float SoundSpeed, Dt, RkA, RkB;
};

struct DgFace {
    float Nx, Ny, Nz;
    DgUint Offset;
};
