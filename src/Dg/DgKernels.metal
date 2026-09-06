#include <metal_stdlib>
using namespace metal;

constant uint DgNodes [[function_constant(0)]];
constant uint DgQuadrature [[function_constant(1)]];
constant uint DgFaceNodes [[function_constant(2)]];

kernel void DgWeakLoad(device const float4 *q [[buffer(0)]], device const float *g [[buffer(1)]], device const float *l [[buffer(2)]], device const uint *vm [[buffer(3)]], device const uint *vp [[buffer(4)]], device const uint *boundary [[buffer(5)]], device const int4 *faceRows [[buffer(6)]], device float4 *load [[buffer(7)]], constant DgParams &p [[buffer(8)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint id = group * DG_NODE_SIMDS + sg;
    if (id >= p.Elements * DgNodes) return;
    const uint e = id / DgNodes, i = id % DgNodes, n = DgNodes, f = DgFaceNodes;
    const uint tiles = (n + 7) / 8, k = f / 4;
    const float anchor = q[e * n].x;
    float4 value = 0;
    for (uint j = lane; j < n; j += 32) {
        const float4 state = q[e * n + j];
        const float pressure = state.x - anchor;
        const uint gb = ((e * tiles + i / 8) * tiles + j / 8) * 3 * 64 + i % 8 * 8 + j % 8;
        const uint gt = ((e * tiles + j / 8) * tiles + i / 8) * 3 * 64 + j % 8 * 8 + i % 8;
        value.x += g[gt] * state.y + g[gt + 64] * state.z + g[gt + 128] * state.w;
        value.y -= g[gb] * pressure;
        value.z -= g[gb + 64] * pressure;
        value.w -= g[gb + 128] * pressure;
    }
    for (uint side = 0; side < 4; ++side) {
        const int row = faceRows[i][side];
        if (row < 0) continue;
        const uint triangle = k * (k + 1) / 2, lb = (e * 4 + side) * 10 * triangle;
        for (uint j = lane; j < k; j += 32) {
            const uint face = e * f + side * k + j;
            const float4 minus = q[vm[face]];
            float4 plus = q[vp[face]];
            if (boundary[face]) plus.yzw = -minus.yzw;
            const float4 diff = minus - plus;
            const float3 sum = minus.yzw + plus.yzw;
            float a[10];
            for (uint term = 0; term < 10; ++term) a[term] = l[lb + term * triangle + max(uint(row), j) * (max(uint(row), j) + 1) / 2 + min(uint(row), j)];
            value.x -= .5f * (a[0] * diff.x + a[1] * sum.x + a[2] * sum.y + a[3] * sum.z);
            value.y += .5f * (a[1] * diff.x - a[4] * diff.y - a[5] * diff.z - a[6] * diff.w);
            value.z += .5f * (a[2] * diff.x - a[5] * diff.y - a[7] * diff.z - a[8] * diff.w);
            value.w += .5f * (a[3] * diff.x - a[6] * diff.y - a[8] * diff.z - a[9] * diff.w);
        }
    }
    value = simd_sum(value);
    if (lane == 0) load[(e / 2 * ((n + 7) / 8 * 8) + i) * 2 + e % 2] = p.SoundSpeed * value;
}

// Eight matrix columns hold four fields from each of two elements.
kernel void DgMassInterpolate(device const float *load [[buffer(0)]], device const float *t [[buffer(1)]], device const float *w [[buffer(2)]], device float4 *weighted [[buffer(3)]], constant DgParams &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint np = (DgNodes + 7) / 8 * 8, qp = (DgQuadrature + 7) / 8 * 8;
    const uint row = (group.x * DG_MASS_SIMDS + sg) * 8, pair = group.y;
    if (row >= qp) return;
    simdgroup_float8x8 a, b, value(0.0f);
    for (uint j = 0; j < np; j += 8) {
        simdgroup_load(a, t + j * qp + row, qp, ulong2(0), true);
        simdgroup_load(b, load + (pair * np + j) * 8, 8);
        simdgroup_multiply_accumulate(value, a, b, value);
    }
    threadgroup float scratch[DG_MASS_SIMDS * 64];
    simdgroup_store(value, scratch + sg * 64, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16) {
        const uint q = row + lane / 2, e = pair * 2 + lane % 2;
        const float weight = q < DgQuadrature && e < p.Elements ? w[e * DgQuadrature + q] : 0.0f;
        weighted[(pair * qp + q) * 2 + lane % 2] = weight * ((threadgroup float4 *)(scratch + sg * 64))[lane];
    }
}

// SIMD groups split the quadrature sum; their partial tiles reduce in fixed order.
kernel void DgMassProject(device const float *weighted [[buffer(0)]], device const float *t [[buffer(1)]], device float4 *rhs [[buffer(2)]], device float4 *q [[buffer(3)]], device float4 *residual [[buffer(4)]], constant DgParams &p [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint qp = (DgQuadrature + 7) / 8 * 8;
    const uint row = group.x * 8, pair = group.y;
    if (row >= DgNodes) return;
    simdgroup_float8x8 a, b, value(0.0f);
    for (uint j = sg * 8; j < qp; j += 8 * DG_MASS_SIMDS) {
        simdgroup_load(a, t + row * qp + j, qp);
        simdgroup_load(b, weighted + (pair * qp + j) * 8, 8);
        simdgroup_multiply_accumulate(value, a, b, value);
    }
    threadgroup float scratch[DG_MASS_SIMDS * 64];
    simdgroup_store(value, scratch + sg * 64, 8);
    // Every SIMD group contributes a partial tile before any thread reads the sum.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < 16) {
        const uint i = row + lane / 2, e = pair * 2 + lane % 2, node = e * DgNodes + i;
        if (i < DgNodes && e < p.Elements) {
            float4 result = 0;
            for (uint part = 0; part < DG_MASS_SIMDS; ++part) result += ((threadgroup float4 *)(scratch + part * 64))[lane];
            rhs[node] = result;
            if (p.Update) {
                const float4 r = p.RkA * residual[node] + p.Dt * result;
                residual[node] = r;
                q[node] += p.RkB * r;
            }
        }
    }
}

kernel void DgReceivers(device const float4 *q [[buffer(0)]], device const uint *elements [[buffer(1)]], device const float *weights [[buffer(2)]], device float *record [[buffer(3)]], constant DgParams &p [[buffer(4)]], uint receiver [[thread_position_in_grid]]) {
    if (receiver >= p.Receivers) return;
    float pressure = 0;
    for (uint i = 0; i < DgNodes; ++i)
        pressure += weights[receiver * DgNodes + i] * q[elements[receiver] * DgNodes + i].x;
    record[receiver * p.Steps + p.Sample] = pressure;
}
