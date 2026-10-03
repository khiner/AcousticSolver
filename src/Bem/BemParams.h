#pragma once
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using BemFloat4 = float4;
using BemUInt = uint;
#else
#include <cstdint>
struct alignas(16) BemFloat4 {
    float x, y, z, w;
};
using BemUInt = uint32_t;
#endif
struct BemFace {
    BemFloat4 A, B, C, Low; // Anchor high/low and two local edge vectors.
};
struct BemParams {
    BemUInt FaceCount, PointCount, HeightCount;
    float Wavenumber;
};
// Offsets count complex entries, including the intermediate and output buffers.
struct BemBlock {
    BemUInt Row, Col, Rows, Cols;
    BemUInt Offset, Rank, Temporary, Output;
};
struct BemBlockParams {
    BemUInt Row, Col, Rows, Cols;
    BemUInt Boundary, Transpose, BlockCount, OutputCount;
    float Wavenumber;
};
struct BemCoarseParams {
    BemUInt FineCount, CoarseCount, Transpose, Pad;
};
struct BemCoarseGroup {
    BemUInt Begin, Count;
    float Weight, Pad;
};
#ifndef __METAL_VERSION__
static_assert(sizeof(BemFace) == 64 && sizeof(BemParams) == 16);
static_assert(sizeof(BemBlock) == 32 && sizeof(BemBlockParams) == 36);
static_assert(sizeof(BemCoarseParams) == 16 && sizeof(BemCoarseGroup) == 16);
#endif
struct BemTexStencil {
    BemUInt I0, I1, I2, I3;
    BemFloat4 Weights;
};
struct BemTexLink {
    BemUInt Height;
    float Weight;
};
struct BemTextureParams {
    BemUInt Width, Height, Count, Pad;
};
#ifndef __METAL_VERSION__
static_assert(sizeof(BemTexStencil) == 32 && sizeof(BemTexLink) == 8 && sizeof(BemTextureParams) == 16);
#endif
