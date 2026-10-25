#include <metal_stdlib>
using namespace metal;

struct Params { uint Rows, Columns, Sparse, Add; };
float2 add(float2 a, float2 b) {
    float s = a.x + b.x, v = s - a.x;
    float e = (a.x - (s - v)) + (b.x - v) + a.y + b.y;
    float r = s + e;
    return float2(r, e - (r - s));
}
float2 mul(float2 a, float2 b) {
    float p = a.x * b.x;
    float e = fma(a.x, b.x, -p) + a.x * b.y + a.y * b.x + a.y * b.y;
    float r = p + e;
    return float2(r, e - (r - p));
}
float4 add_complex(float4 a, float4 b) { return float4(add(a.xy, b.xy), add(a.zw, b.zw)); }
float4 mul_complex(float4 a, float4 b) {
    return float4(add(mul(a.xy, b.xy), -mul(a.zw, b.zw)), add(mul(a.xy, b.zw), mul(a.zw, b.xy)));
}
kernel void Apply(const device float4 *values [[buffer(0)]], const device uint *offsets [[buffer(1)]],
                  const device uint *indices [[buffer(2)]], const device float4 *x [[buffer(3)]],
                  device float4 *y [[buffer(4)]], constant Params &p [[buffer(5)]],
                  uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.Rows) return;
    uint begin = p.Sparse ? offsets[row] : row * p.Columns;
    uint end = p.Sparse ? offsets[row + 1] : begin + p.Columns;
    float4 sum = 0;
    for (uint j = begin + lane; j < end; j += 32) {
        sum = add_complex(sum, mul_complex(values[j], x[p.Sparse ? indices[j] : j - begin]));
    }
    for (uint offset = 16; offset; offset >>= 1) {
        float4 other = simd_shuffle_down(sum, offset);
        if (lane < offset) sum = add_complex(sum, other);
    }
    if (lane == 0) y[row] = p.Add ? add_complex(sum, y[row]) : sum;
}
