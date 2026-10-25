#include <metal_stdlib>
using namespace metal;

float2 add(float2 a, float2 b) {
    float s = a.x + b.x, v = s - a.x;
    float t = ((b.x - v) + (a.x - (s - v))) + a.y + b.y;
    float h = s + t;
    return float2(h, t - (h - s));
}
float2 mul(float2 a, float2 b) {
    float p = a.x * b.x, e = fma(a.x, b.x, -p) + a.x * b.y + a.y * b.x;
    float h = p + e;
    return float2(h, e - (h - p));
}
float2 madd(float2 sum, float2 a, float2 b) {
    float product = a.x * b.x;
    float error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x;
    float high = sum.x + product, v = high - sum.x;
    float low = ((product - v) + (sum.x - (high - v))) + sum.y + error;
    float result = high + low;
    return float2(result, low - (result - high));
}
float2 sumLanes(float2 value, uint lane) {
    for (uint delta = 16; delta; delta /= 2) {
        float2 other = simd_shuffle_down(value, delta);
        if (lane + delta < 32) value = add(value, other);
    }
    return value;
}
float2 dotRow(device const uint *starts, device const uint *columns, device const float2 *values, device const float2 *x, uint row, uint lane) {
    if (starts[row] == starts[row + 1]) return float2(0);
    float2 sum = 0;
    for (uint j = starts[row] * NODES + lane; j < starts[row + 1] * NODES; j += 32) sum = madd(sum, values[j], x[columns[j / NODES] + j % NODES]);
    return sumLanes(sum, lane);
}
float2 sourceRow(device const uint *starts, device const uint *columns, device const float2 *values, device const float2 *drive, uint row) {
    float2 sum = 0;
    for (uint j = starts[row]; j < starts[row + 1]; ++j) sum = madd(sum, values[j], drive[columns[j]]);
    return sum;
}
struct StepParams {
    uint drive, stage;
    float2 a, b, dt;
};
struct RecordParams {
    uint count, index;
};
kernel void sample(device const uint *starts [[buffer(0)]], device const uint *columns [[buffer(1)]], device const float2 *values [[buffer(2)]], device const float2 *q [[buffer(3)]], device float2 *out [[buffer(4)]], constant RecordParams &p [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    uint row = id / 32, lane = id % 32;
    if (row >= p.count) return;
    float2 value = dotRow(starts, columns, values, q, row, lane);
    if (!lane) out[p.index * p.count + row] = value;
}
kernel void diagnostic(device const float2 *operatorRhs [[buffer(0)]], device const uint *ps [[buffer(3)]], device const uint *pc [[buffer(4)]], device const float2 *pv [[buffer(5)]], device const uint *bs [[buffer(6)]], device const uint *bc [[buffer(7)]], device const float2 *bv [[buffer(8)]], device const uint *ss [[buffer(9)]], device const uint *sc [[buffer(10)]], device const float2 *sv [[buffer(11)]], device const float2 *q [[buffer(12)]], device const float2 *drive [[buffer(13)]], device float2 *partial [[buffer(14)]], constant RecordParams &p [[buffer(15)]], uint id [[thread_position_in_grid]]) {
    uint row = id / 32, lane = id % 32;
    if (row >= p.count) return;
    float2 aq = operatorRhs[row], pq = dotRow(ps, pc, pv, q, row, lane);
    float2 bq = dotRow(bs, bc, bv, q, row, lane);
    if (!lane) {
        float2 sq = sourceRow(ss, sc, sv, drive + p.index, row);
        const float2 x = q[row];
        partial[row * 5] = mul(float2(.5f, 0), mul(x, x));
        partial[row * 5 + 1] = mul(float2(.5f, 0), mul(x, pq));
        partial[row * 5 + 2] = mul(x, sq);
        partial[row * 5 + 3] = mul(x, bq);
        partial[row * 5 + 4] = mul(x, add(bq, -aq));
    }
}
kernel void reduceDiagnostic(device const float2 *partial [[buffer(0)]], device float2 *out [[buffer(1)]], constant RecordParams &p [[buffer(2)]], uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup float2 sums[8];
    uint lane = tid % 32;
    for (uint col = 0; col < 5; ++col) {
        float2 sum = 0;
        for (uint row = group * 1024 + tid; row < min((group + 1) * 1024, p.count); row += 256) sum = add(sum, partial[row * 5 + col]);
        sum = sumLanes(sum, lane);
        if (!lane) sums[tid / 32] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32) {
            sum = sumLanes(tid < 8 ? sums[tid] : float2(0), lane);
            if (!tid) out[group * 5 + col] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void finishDiagnostic(device const float2 *partial [[buffer(0)]], device float2 *out [[buffer(1)]], constant RecordParams &p [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    uint lane = id % 32, col = id / 32;
    if (col >= 5) return;
    float2 sum = 0;
    for (uint row = lane; row < p.count; row += 32) sum = add(sum, partial[row * 5 + col]);
    sum = sumLanes(sum, lane);
    if (!lane) out[p.index * 5 + col] = sum;
}

struct ElementData {
    uint offset, fields, mass;
};
uint stressIndex(uint a, uint b) { return a == b ? 3 + a : 9 - a - b; }
struct FaceData {
    uint inputs, values, rows, outputs;
};
kernel void faceFlux(device const FaceData *faces [[buffer(0)]], device const uint *inputs [[buffer(1)]], device const float2 *values [[buffer(2)]], device const float2 *physical [[buffer(3)]], device float2 *flux [[buffer(4)]], uint fi [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 q[18 * FACE_NODES];
    FaceData f = faces[fi];
    for (uint col = tid; col < f.rows; col += 128) q[col] = physical[inputs[f.inputs + col]];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < f.outputs; row += 128) {
        float2 sum = 0;
        for (uint col = 0; col < f.rows; ++col) sum = madd(sum, values[f.values + col * f.outputs + row], q[col]);
        flux[f.inputs + row] = sum;
        if (f.outputs < f.rows) flux[f.inputs + f.outputs + row] = -sum;
    }
}
constant bool Update [[function_constant(0)]];
kernel void massStep(device const ElementData *elements [[buffer(0)]], device const float2 *basis [[buffer(1)]], device const float2 *coefficients [[buffer(2)]], device const float2 *weak [[buffer(3)]], device const uint *starts [[buffer(4)]], device const uint *columns [[buffer(5)]], device const float2 *values [[buffer(6)]], device const float2 *q [[buffer(7)]], device float2 *next [[buffer(8)]], device float2 *residual [[buffer(9)]], device const uint *ss [[buffer(10)]], device const uint *sc [[buffer(11)]], device const float2 *sv [[buffer(12)]], device const float2 *drive [[buffer(13)]], constant StepParams &p [[buffer(14)]], device float2 *rhs [[buffer(15)]], device const uint *faceMap [[buffer(16)]], device const float2 *flux [[buffer(17)]], device const uint *is [[buffer(18)]], device const uint *ic [[buffer(19)]], device const float2 *iv [[buffer(20)]], device const float2 *physical [[buffer(21)]], device float2 *nextPhysical [[buffer(22)]], uint tid [[thread_index_in_threadgroup]], uint ei [[threadgroup_position_in_grid]]) {
    threadgroup float2 input[9 * NODES], local[9 * NODES], quad[9 * (MASS_TILE + 1)];
    ElementData e = elements[ei];
    for (uint row = tid; row < e.fields * NODES; row += 128) input[row] = physical[e.offset + row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < e.fields * NODES; row += 128) {
        uint field = row / NODES, node = row % NODES;
        float2 sum = 0;
        for (uint j = 0; j < NODES; ++j) {
            if (field < 3) {
                if (e.fields == 4) sum = madd(sum, -weak[((ei * 3 + field) * NODES + node) * NODES + j], input[3 * NODES + j]);
                else
                    for (uint d = 0; d < 3; ++d) sum = madd(sum, -weak[((ei * 3 + d) * NODES + node) * NODES + j], input[stressIndex(field, d) * NODES + j]);
            } else if (e.fields == 4) {
                for (uint d = 0; d < 3; ++d) sum = madd(sum, -weak[((ei * 3 + d) * NODES + node) * NODES + j], input[d * NODES + j]);
            } else {
                const uint2 axes[6] = {uint2(0, 0), uint2(1, 1), uint2(2, 2), uint2(1, 2), uint2(0, 2), uint2(0, 1)};
                uint2 a = axes[field - 3];
                sum = madd(sum, -weak[((ei * 3 + a.x) * NODES + node) * NODES + j], input[a.y * NODES + j]);
                if (a.x != a.y) sum = madd(sum, -weak[((ei * 3 + a.y) * NODES + node) * NODES + j], input[a.x * NODES + j]);
            }
        }
        for (uint side = 0; side < 4; ++side) {
            uint faceRow = faceMap[4 * (e.offset + row) + side];
            if (faceRow != UINT_MAX) sum = add(sum, flux[faceRow]);
        }
        local[row] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < e.fields * NODES; row += 128) {
        input[row] = local[row];
        local[row] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if WEIGHT_ADJUSTED
    for (uint chunk = 0; chunk < VOL_Q; chunk += MASS_TILE) {
        uint qp = chunk + tid;
        if (tid < MASS_TILE && qp < VOL_Q) {
            float2 point[9];
            for (uint field = 0; field < e.fields; ++field) {
                point[field] = 0;
                for (uint j = 0; j < NODES; ++j) point[field] = madd(point[field], basis[qp * NODES + j], input[field * NODES + j]);
            }
            uint base = e.mass + qp * (e.fields == 4 ? 2 : 37);
            for (uint field = 0; field < 3; ++field) quad[field * (MASS_TILE + 1) + tid] = mul(coefficients[base], point[field]);
            if (e.fields == 4) quad[3 * (MASS_TILE + 1) + tid] = mul(coefficients[base + 1], point[3]);
            else
                for (uint field = 0; field < 6; ++field) {
                    float2 value = 0;
                    for (uint j = 0; j < 6; ++j) value = madd(value, coefficients[base + 1 + field * 6 + j], point[3 + j]);
                    quad[(3 + field) * (MASS_TILE + 1) + tid] = value;
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tid; row < e.fields * NODES; row += 128) {
            uint field = row / NODES, node = row % NODES;
            float2 sum = local[row];
            for (uint j = 0; j < min(uint(MASS_TILE), uint(VOL_Q) - chunk); ++j) sum = madd(sum, basis[(chunk + j) * NODES + node], quad[field * (MASS_TILE + 1) + j]);
            local[row] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#else
    for (uint row = tid; row < e.fields * NODES; row += 128) local[row] = input[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
    for (uint row = tid; row < e.fields * NODES; row += 128) {
        uint global = e.offset + row;
        float2 sum = 0;
        for (uint j = starts[global] * NODES; j < starts[global + 1] * NODES; ++j) sum = madd(sum, values[j], local[columns[j / NODES] - e.offset + j % NODES]);
        rhs[global] = sum;
        if (Update) {
            sum = add(sum, sourceRow(ss, sc, sv, drive + p.drive, global));
            float2 r = add(p.stage ? mul(p.a, residual[global]) : float2(0), mul(p.dt, sum));
            residual[global] = r;
            input[row] = next[global] = madd(q[global], p.b, r);
        }
    }
    if (Update) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tid; row < e.fields * NODES; row += 128) {
            uint global = e.offset + row;
            float2 sum = 0;
            for (uint j = is[global] * NODES; j < is[global + 1] * NODES; ++j) sum = madd(sum, iv[j], input[ic[j / NODES] - e.offset + j % NODES]);
            nextPhysical[global] = sum;
        }
    }
}
