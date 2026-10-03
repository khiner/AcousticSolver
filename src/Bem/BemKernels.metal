// Six-point triangle rule and exterior double-layer convention of Acoustic Reliefs.
constant float3 BemQuadrature[6] = {
    float3(.223381589678011, .445948490915965, .108103018168070),
    float3(.223381589678011, .445948490915965, .445948490915965),
    float3(.223381589678011, .108103018168070, .445948490915965),
    float3(.109951743655322, .091576213509771, .816847572980459),
    float3(.109951743655322, .091576213509771, .091576213509771),
    float3(.109951743655322, .816847572980459, .091576213509771)
};
float2 bem_mul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
float2 bem_exp(float phase) { return float2(cos(phase), -sin(phase)); }
float3 bem_center_low(BemFace f) { return f.Low.xyz + (f.B.xyz + f.C.xyz) / 3.f; }
// Rationalizing |point-q|-|point| preserves the small local phase at distant
// listeners. The large exp(-ik|point|) phase is evaluated once in host FP64.
float2 bem_phase(float k, float3 point, float3 q, float distance, float2 base, bool listener) {
    float local = listener ? (dot(q, q) - 2.f * dot(point, q)) / (distance + length(point)) : distance;
    return bem_mul(base, bem_exp(k * local));
}
float2 bem_integral(float k, float3 point, BemFace f, float2 base = float2(1, 0), bool listener = false, float3 pointLow = float3(0)) {
    float3 ab = f.B.xyz, ac = f.C.xyz, normal = cross(ab, ac);
    float2 sum = 0;
    for (uint q = 0; q < 6; ++q) {
        float3 w = BemQuadrature[q];
        float3 local = w.y * ab + w.z * ac;
        float3 qpoint = f.A.xyz + (f.Low.xyz + local);
        float3 r = (point - f.A.xyz) + (pointLow - f.Low.xyz) - local;
        float d = length(r);
        sum += bem_mul(bem_phase(k, point, qpoint, d, base, listener), float2(1.f, k * d)) * (w.x * .5f * dot(r, normal) / (4.f * M_PI_F * d * d * d));
    }
    return sum;
}
void bem_add(thread float2 &hi, thread float2 &lo, float2 value);
float2 bem_derivative(float k, float3 point, BemFace f, float pointDy, int corner, float2 base = float2(1, 0), bool listener = false, float3 pointLow = float3(0)) {
    float3 ab = f.B.xyz, ac = f.C.xyz, normal = cross(ab, ac), ey = float3(0, 1, 0), dn = 0;
    if (corner == 0) dn = cross(-ey, ac) + cross(ab, -ey);
    if (corner == 1) dn = cross(ey, ac);
    if (corner == 2) dn = cross(ab, ey);
    float2 hi = 0, lo = 0;
    for (uint q = 0; q < 6; ++q) {
        float3 w = BemQuadrature[q];
        float sourceDy = corner == 0 ? 1.f - w.y - w.z : corner == 1 ? w.y :
            corner == 2                                              ? w.z :
                                                                       0.f;
        float3 local = w.y * ab + w.z * ac;
        float3 qpoint = f.A.xyz + (f.Low.xyz + local);
        float3 r = (point - f.A.xyz) + (pointLow - f.Low.xyz) - local, dr = (pointDy - sourceDy) * ey;
        float d = length(r), d2 = d * d;
        float2 e = bem_phase(k, point, qpoint, d, base, listener) / (4.f * M_PI_F);
        float2 radial = bem_mul(e, float2(1.f, k * d)) / (d2 * d);
        float2 derivative = bem_mul(e, float2(k * k * d2 - 3.f, -3.f * k * d)) / (d2 * d2 * d);
        bem_add(hi, lo, .5f * w.x * (radial * (dot(dr, normal) + dot(r, dn)) + derivative * (dot(r, normal) * dot(r, dr))));
    }
    return hi + lo;
}
// Two-component vectors preserve Krylov corrections and cancellation in the
// adjoint. Operator coefficients remain FP32. Explicit FMA recovers product errors.
void bem_add(thread float2 &hi, thread float2 &lo, float2 value) {
    float2 sum = hi + value, back = sum - hi;
    lo += (hi - (sum - back)) + (value - back);
    hi = sum;
}
float4 bem_pair(float2 hi, float2 lo) {
    float2 sum = hi + lo, back = sum - hi;
    return float4(sum, (hi - (sum - back)) + (lo - back));
}
void bem_dot_add(thread float2 &hi, thread float2 &lo, float2 a, float4 b) {
    float2 first = a.x * b.xy, second = float2(-a.y * b.y, a.y * b.x);
    float2 error = fma(float2(a.x), b.xy, -first) + fma(float2(-a.y, a.y), b.yx, -second);
    bem_add(hi, lo, first);
    bem_add(hi, lo, second);
    bem_add(hi, lo, error + bem_mul(a, b.zw));
}
// One threadgroup per triangle corner contracts the adjoint over boundary and
// listener targets. A fixed reduction followed by a vertex gather avoids atomics.
kernel void bem_gradient_faces(device const BemFace *faces [[buffer(0)]], device const float4 *points [[buffer(1)]], device const float2 *surface [[buffer(2)]], device const float2 *sensitivity [[buffer(3)]], device const float2 *adjoint [[buffer(4)]], device const float2 *incidentDy [[buffer(5)]], device float2 *contributions [[buffer(6)]], device const float2 *phases [[buffer(7)]], device const uint *corners [[buffer(8)]], constant BemParams &p [[buffer(9)]], uint lane [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    uint active = corners[group.x];
    uint t = active / 3;
    int corner = active % 3;
    BemFace f = faces[t];
    float2 hi = 0, lo = 0;
    for (uint j = lane; j < max(p.FaceCount, p.PointCount); j += 128) {
        if (j < p.PointCount) bem_add(hi, lo, bem_mul(bem_mul(sensitivity[j], surface[t]), bem_derivative(p.Wavenumber, points[j].xyz, f, 0.f, corner, phases[j], true)));
        if (j < p.FaceCount && j != t) {
            bem_add(hi, lo, -bem_mul(bem_mul(adjoint[t], surface[j]), bem_derivative(p.Wavenumber, f.A.xyz, faces[j], 1.f / 3.f, -1, float2(1, 0), false, bem_center_low(f))));
            bem_add(hi, lo, -bem_mul(bem_mul(adjoint[j], surface[t]), bem_derivative(p.Wavenumber, faces[j].A.xyz, f, 0.f, corner, float2(1, 0), false, bem_center_low(faces[j]))));
        }
    }
    if (lane == 0) bem_add(hi, lo, -bem_mul(adjoint[t], incidentDy[t]));
    threadgroup float4 partial[128];
    partial[lane] = bem_pair(hi, lo);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride > 0; stride /= 2) {
        if (lane < stride) {
            float2 high = partial[lane].xy, low = partial[lane].zw;
            bem_add(high, low, partial[lane + stride].xy);
            bem_add(high, low, partial[lane + stride].zw);
            partial[lane] = bem_pair(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) contributions[active] = partial[0].xz;
}
kernel void bem_gradient_gather(device const uint *offsets [[buffer(0)]], device const uint *corners [[buffer(1)]], device const float2 *contributions [[buffer(2)]], device float *gradient [[buffer(3)]], constant BemParams &p [[buffer(4)]], uint h [[thread_position_in_grid]]) {
    if (h >= p.HeightCount) return;
    float2 hi = 0, lo = 0;
    for (uint i = offsets[h]; i < offsets[h + 1]; ++i) {
        float2 value = contributions[corners[i]];
        bem_add(hi, lo, float2(value.x, 0));
        bem_add(hi, lo, float2(value.y, 0));
    }
    gradient[h] = hi.x + lo.x;
}

kernel void bem_assemble_block(device const BemFace *faces [[buffer(0)]], device const float4 *points [[buffer(1)]], device const uint *rows [[buffer(2)]], device const uint *cols [[buffer(3)]], device float2 *matrix [[buffer(4)]], device const float2 *phases [[buffer(5)]], constant BemBlockParams &p [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.Rows * p.Cols) return;
    uint row = rows[p.Row + id % p.Rows], col = cols[p.Col + id / p.Rows];
    float3 point = p.Boundary ? faces[row].A.xyz : points[row].xyz;
    matrix[id] = p.Boundary && row == col ? float2(-.5f, 0) : bem_integral(p.Wavenumber, point, faces[col], p.Boundary ? float2(1, 0) : phases[row], !p.Boundary, p.Boundary ? bem_center_low(faces[row]) : float3(0));
}
// A = U V uses a plain transpose in the adjoint; never conjugate these factors.
kernel void bem_block_reduce(device const BemBlock *blocks [[buffer(0)]], device const float2 *values [[buffer(1)]], device const float4 *input [[buffer(2)]], device float4 *temporary [[buffer(3)]], constant BemBlockParams &p [[buffer(4)]], uint lane [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    BemBlock b = blocks[group.x];
    if (b.Rank == 0xffffffffu) return;
    for (uint k = lane; k < b.Rank; k += 128) {
        float2 hi = 0, lo = 0;
        uint count = p.Transpose ? b.Rows : b.Cols;
        for (uint j = 0; j < count; ++j) {
            uint index = p.Transpose ? b.Offset + j + k * b.Rows : b.Offset + b.Rows * b.Rank + k + j * b.Rank;
            bem_dot_add(hi, lo, values[index], input[(p.Transpose ? b.Row : b.Col) + j]);
        }
        temporary[b.Temporary + k] = bem_pair(hi, lo);
    }
}
kernel void bem_block_product(device const BemBlock *blocks [[buffer(0)]], device const float2 *values [[buffer(1)]], device const float4 *input [[buffer(2)]], device const float4 *temporary [[buffer(3)]], device float4 *partial [[buffer(4)]], constant BemBlockParams &p [[buffer(5)]], uint lane [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    BemBlock b = blocks[group.x];
    uint count = p.Transpose ? b.Cols : b.Rows;
    for (uint i = lane; i < count; i += 128) {
        float2 hi = 0, lo = 0;
        if (b.Rank == 0xffffffffu) {
            uint inner = p.Transpose ? b.Rows : b.Cols;
            for (uint j = 0; j < inner; ++j) {
                uint index = b.Offset + (p.Transpose ? j + i * b.Rows : i + j * b.Rows);
                bem_dot_add(hi, lo, values[index], input[(p.Transpose ? b.Row : b.Col) + j]);
            }
        } else {
            for (uint k = 0; k < b.Rank; ++k) {
                uint index = p.Transpose ? b.Offset + b.Rows * b.Rank + k + i * b.Rank : b.Offset + i + k * b.Rows;
                bem_dot_add(hi, lo, values[index], temporary[b.Temporary + k]);
            }
        }
        partial[b.Output + i] = bem_pair(hi, lo);
    }
}
kernel void bem_block_gather(device const uint *offsets [[buffer(0)]], device const uint *indices [[buffer(1)]], device const float4 *partial [[buffer(2)]], device float4 *output [[buffer(3)]], constant BemBlockParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.OutputCount) return;
    float2 hi = 0, lo = 0;
    for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
        float4 value = partial[indices[j]];
        bem_add(hi, lo, value.xy);
        bem_add(hi, lo, value.zw);
    }
    output[i] = bem_pair(hi, lo);
}

kernel void bem_texture_sample(device const BemTexStencil *stencils [[buffer(0)]], device const float *pixels [[buffer(1)]], device float *values [[buffer(2)]], constant BemTextureParams &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.Count) return;
    BemTexStencil s = stencils[i];
    values[i] = s.Weights.x * pixels[s.I0] + s.Weights.y * pixels[s.I1] + s.Weights.z * pixels[s.I2] + s.Weights.w * pixels[s.I3];
}
kernel void bem_texture_transpose(device const uint *offsets [[buffer(0)]], device const BemTexLink *links [[buffer(1)]], device const float *values [[buffer(2)]], device float *gradient [[buffer(3)]], constant BemTextureParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.Width * p.Height) return;
    float sum = 0;
    for (uint j = offsets[i]; j < offsets[i + 1]; ++j) sum += links[j].Weight * values[links[j].Height];
    gradient[i] = sum;
}

// B = J + (P-J*A*P) * (P^T*A*P)^-1 * P^T. The adjoint uses B^T.
kernel void bem_coarse_restrict(device const BemCoarseGroup *groups [[buffer(0)]], device const float2 *correction [[buffer(1)]], device const float4 *input [[buffer(2)]], device float4 *output [[buffer(3)]], constant BemCoarseParams &p [[buffer(4)]], uint lane [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    uint c = group.x;
    float2 hi = 0, lo = 0;
    if (p.Transpose) {
        for (uint i = lane; i < p.FineCount; i += 128) bem_dot_add(hi, lo, correction[i + c * p.FineCount], input[i]);
    } else {
        BemCoarseGroup g = groups[c];
        for (uint i = g.Begin + lane; i < g.Begin + g.Count; i += 128) bem_dot_add(hi, lo, float2(g.Weight, 0), input[i]);
    }
    threadgroup float4 partial[128];
    partial[lane] = bem_pair(hi, lo);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride > 0; stride /= 2) {
        if (lane < stride) {
            float2 high = partial[lane].xy, low = partial[lane].zw;
            bem_add(high, low, partial[lane + stride].xy);
            bem_add(high, low, partial[lane + stride].zw);
            partial[lane] = bem_pair(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) output[c] = partial[0];
}
kernel void bem_coarse_solve(device const float2 *inverse [[buffer(0)]], device const float4 *input [[buffer(1)]], device float4 *output [[buffer(2)]], constant BemCoarseParams &p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.CoarseCount) return;
    float2 hi = 0, lo = 0;
    for (uint j = 0; j < p.CoarseCount; ++j) bem_dot_add(hi, lo, inverse[p.Transpose ? j + i * p.CoarseCount : i + j * p.CoarseCount], input[j]);
    output[i] = bem_pair(hi, lo);
}
kernel void bem_coarse_add(device const BemCoarseGroup *groups [[buffer(0)]], device const uint *indices [[buffer(1)]], device const float2 *correction [[buffer(2)]], device const float4 *coarse [[buffer(3)]], device float4 *output [[buffer(4)]], constant BemCoarseParams &p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.FineCount) return;
    float2 hi = output[i].xy, lo = output[i].zw;
    if (p.Transpose) {
        uint c = indices[i];
        bem_dot_add(hi, lo, float2(groups[c].Weight, 0), coarse[c]);
    } else {
        for (uint c = 0; c < p.CoarseCount; ++c) bem_dot_add(hi, lo, correction[i + c * p.FineCount], coarse[c]);
    }
    output[i] = bem_pair(hi, lo);
}
