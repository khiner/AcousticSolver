#include <metal_stdlib>
using namespace metal;

constant uint DgNodes [[function_constant(0)]];
constant uint DgFaceNodes [[function_constant(2)]];

kernel void DgWeakLoad(device const float4 *q [[buffer(0)]], device const float *g [[buffer(1)]], device const float *l [[buffer(2)]], device const uint *vm [[buffer(3)]], device const uint *vp [[buffer(4)]], device const uint *boundary [[buffer(5)]], device const int4 *faceRows [[buffer(6)]], device float4 *load [[buffer(7)]], device const uint *elements [[buffer(8)]], device const float *weights [[buffer(9)]], device float *record [[buffer(10)]], device const DgFace *info [[buffer(11)]], constant DgParams &p [[buffer(12)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint id = group * DG_NODE_SIMDS + sg;
    if (id >= p.Elements * DgNodes) return;
    const uint e = id / DgNodes, i = id % DgNodes, n = DgNodes, f = DgFaceNodes;
    // LSERK4 has RkA == 0 only at its first stage; sample before changing the state.
    if (p.Update && p.RkA == 0 && i == 0 && lane == 0) {
        for (uint receiver = 0; receiver < p.Receivers; ++receiver) {
            if (elements[receiver] != e) continue;
            float pressure = 0;
            for (uint j = 0; j < n; ++j) pressure += weights[receiver * n + j] * q[e * n + j].x;
            record[receiver * p.Steps + p.Sample] = pressure;
        }
    }
    const uint k = f / 4;
    const float anchor = q[e * n].x;
    float4 value = 0;
    for (uint j = lane; j < n; j += 32) {
        const float4 state = q[e * n + j];
        const float pressure = state.x - anchor;
        const uint gb = (e * 3 * n + i) * n + j;
        const uint gt = (e * 3 * n + j) * n + i;
        value.x += g[gt] * state.y + g[gt + n * n] * state.z + g[gt + 2 * n * n] * state.w;
        value.y -= g[gb] * pressure;
        value.z -= g[gb + n * n] * pressure;
        value.w -= g[gb + 2 * n * n] * pressure;
    }
    for (uint side = 0; side < 4; ++side) {
        const int row = faceRows[i][side];
        if (row < 0) continue;
        const DgFace data = info[e * 4 + side];
        const uint triangle = k * (k + 1) / 2, lb = data.Offset & 0x7fffffffu;
        const bool planar = (data.Offset & 0x80000000u) != 0;
        const float3 normal(data.Nx, data.Ny, data.Nz);
        for (uint j = lane; j < k; j += 32) {
            const uint face = e * f + side * k + j;
            const float4 minus = q[vm[face]];
            float4 plus = q[vp[face]];
            if (boundary[face]) plus.yzw = -minus.yzw;
            const float4 diff = minus - plus;
            const float3 sum = minus.yzw + plus.yzw;
            if (planar) {
                const float weight = .5f * l[lb + max(uint(row), j) * (max(uint(row), j) + 1) / 2 + min(uint(row), j)];
                value.x -= weight * (diff.x + dot(normal, sum));
                value.yzw += weight * normal * (diff.x - dot(normal, diff.yzw));
                continue;
            }
            float a[10];
            for (uint term = 0; term < 10; ++term) a[term] = l[lb + term * triangle + max(uint(row), j) * (max(uint(row), j) + 1) / 2 + min(uint(row), j)];
            value.x -= .5f * (a[0] * diff.x + a[1] * sum.x + a[2] * sum.y + a[3] * sum.z);
            value.y += .5f * (a[1] * diff.x - a[4] * diff.y - a[5] * diff.z - a[6] * diff.w);
            value.z += .5f * (a[2] * diff.x - a[5] * diff.y - a[7] * diff.z - a[8] * diff.w);
            value.w += .5f * (a[3] * diff.x - a[6] * diff.y - a[8] * diff.z - a[9] * diff.w);
        }
    }
    value = simd_sum(value);
    if (lane == 0) load[id] = p.SoundSpeed * value;
}

kernel void DgMassProject(device const float4 *load [[buffer(0)]], device const float *mass [[buffer(1)]], device float4 *rhs [[buffer(2)]], device float4 *q [[buffer(3)]], device float4 *residual [[buffer(4)]], constant DgParams &p [[buffer(5)]], uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint id = group * DG_NODE_SIMDS + sg;
    if (id >= p.Elements * DgNodes) return;
    const uint e = id / DgNodes;
    float4 result = 0;
    for (uint j = lane; j < DgNodes; j += 32) result += mass[id * DgNodes + j] * load[e * DgNodes + j];
    result = simd_sum(result);
    if (lane == 0) {
        if (p.Update) {
            const float4 r = p.RkA * residual[id] + p.Dt * result;
            residual[id] = r;
            q[id] += p.RkB * r;
        } else rhs[id] = result;
    }
}
