#include <metal_stdlib>
using namespace metal;
using Real = PlateReal;
struct Real4 {
    Real x, y, z, w;
};
struct Real2 {
    Real x, y;
};
inline Real4 operator+(Real4 a, Real4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline thread Real4 &operator+=(thread Real4 &a, Real4 b) {
    a = a + b;
    return a;
}
inline Real PlateShuffle(Real value, uint offset) {
    Real out;
    for (uint i = 0; i < Real::Limbs; ++i) out.Word[i] = simd_shuffle_down(value.Word[i], offset);
    out.Exponent = simd_shuffle_down(value.Exponent, offset);
    out.Sign = simd_shuffle_down(value.Sign, offset);
    return out;
}
inline Real4 PlateShuffle(Real4 v, uint offset) {
    return {PlateShuffle(v.x, offset), PlateShuffle(v.y, offset), PlateShuffle(v.z, offset), PlateShuffle(v.w, offset)};
}

template<typename T> inline T PlateReduce(T value, uint lane) {
    for (uint offset = PlateTransformLanes / 2; offset; offset /= 2) {
        T other = PlateShuffle(value, offset);
        if (lane + offset < PlateTransformLanes) value += other;
    }
    return value;
}

// All transforms are orthonormal. The two fields of a nonlinear evaluation are
// q^n and (q^n+q^(n-1))/2; only the latter's potential is needed by the regulator.
kernel void PlateDerivativeX(device const Real4 *state [[buffer(0)]], device const Real4 *transform [[buffer(1)]], device Real4 *out [[buffer(2)]], constant PlateParams &p [[buffer(3)]], uint threadID [[thread_position_in_grid]]) {
    uint id = threadID / PlateTransformLanes, lane = threadID % PlateTransformLanes;
    if (id >= 2 * p.Nx * p.My) return;
    uint y = id % p.My, x = (id / p.My) % p.Nx, midpoint = id / (p.My * p.Nx);
    Real4 v{};
    for (uint i = lane; i < p.Mx; i += PlateTransformLanes) {
        Real4 s = state[i * p.My + y];
        Real q = s.x - (midpoint ? .5f * s.y : Real(0.f));
        Real4 t = transform[i * p.Nx + x];
        v.x += q * t.x;
        v.y += q * t.y;
        v.z += q * t.z;
    }
    v = PlateReduce(v, lane);
    if (!lane) out[id] = v;
}

kernel void PlateAiryY(device const Real *in [[buffer(0)]], device const Real *cy [[buffer(1)]], device const Real2 *coeff [[buffer(2)]], device const uint *active [[buffer(3)]], device Real *out [[buffer(4)]], constant PlateParams &p [[buffer(5)]], uint threadID [[thread_position_in_grid]]) {
    uint id = threadID / PlateTransformLanes, lane = threadID % PlateTransformLanes;
    if (id >= 2 * p.AiryModes) return;
    uint a = id % p.AiryModes, i = a / (p.My + 1), j = a % (p.My + 1), midpoint = id / p.AiryModes;
    Real sum = {};
    if (active[a]) {
        for (uint y = lane; y < p.Ny; y += PlateTransformLanes) sum += cy[j * p.Ny + y] * in[(midpoint * (p.Mx + 1) + i) * p.Ny + y];
        sum = PlateReduce(sum, lane);
        sum *= coeff[a].y;
    }
    if (!lane) out[id] = sum;
}

inline Real4 PlateSum(Real4 value, threadgroup Real4 *scratch, uint tid) {
    const uint lane = tid & 31;
    for (uint offset = 16; offset; offset /= 2) {
        Real4 other = PlateShuffle(value, offset);
        if (lane + offset < 32) value += other;
    }
    if (!lane) scratch[tid / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Real4 sum{};
    for (uint i = 0; i < 8; ++i) sum += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sum;
}

kernel void PlateSummary(device const Real *airy [[buffer(0)]], device const Real2 *coeff [[buffer(1)]], device const Real4 *state [[buffer(2)]], device Real *scalar [[buffer(3)]], constant PlateParams &p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint scratchWords[8 * sizeof(Real4) / sizeof(uint)];
    threadgroup Real4 *scratch = (threadgroup Real4 *)scratchWords;
    Real4 s{};
    if (p.Nonlinear)
        for (uint a = tid; a < p.AiryModes; a += 256) {
            Real k2 = coeff[a].x;
            Real x = k2 * airy[a], y = k2 * airy[p.AiryModes + a];
            s.x += .25f * x * x;
            s.y += .25f * y * y;
        }
    for (uint i = tid; i < p.Modes; i += 256) {
        Real delta = state[i].y;
        s.z += p.NormType == 1 ? PlateAbs(delta) : delta * delta;
    }
    s = PlateSum(s, scratch, tid);
    if (!tid) {
        scalar[2] = s.x;
        scalar[3] = s.y;
        scalar[4] = s.z;
        scalar[9] = scalar[0];
        scalar[8] = p.Nonlinear ? scalar[9] - PlateSqrt(2.f * s.y + p.Epsilon) : Real(0.f);
    }
}

kernel void PlatePhiX(device const Real *airy [[buffer(0)]], device const Real4 *transform [[buffer(1)]], device Real4 *out [[buffer(2)]], constant PlateParams &p [[buffer(3)]], uint threadID [[thread_position_in_grid]]) {
    uint id = threadID / PlateTransformLanes, lane = threadID % PlateTransformLanes;
    if (id >= p.Nx * (p.My + 1)) return;
    uint x = id / (p.My + 1), j = id % (p.My + 1);
    Real4 v{};
    for (uint i = lane; i <= p.Mx; i += PlateTransformLanes) {
        Real a = airy[i * (p.My + 1) + j];
        Real4 t = transform[i * p.Nx + x];
        v.x += a * t.x;
        v.y += a * t.y;
        v.z += a * t.z;
    }
    v = PlateReduce(v, lane);
    if (!lane) out[id] = v;
}

inline Real PlateGradient(uint id, device const Real *in, device const Real *sy, device const uint *active, constant PlateParams &p, uint lane) {
    if (!active[id] || !p.Nonlinear) return 0.f;
    uint i = id / p.My, j = id % p.My;
    Real value = 0;
    for (uint y = lane; y < p.Ny; y += PlateTransformLanes) value -= sy[j * p.Ny + y] * in[i * p.Ny + y];
    return PlateReduce(value, lane) * p.InvScale;
}

kernel void PlateForceY(device const Real *in [[buffer(0)]], device const Real *sy [[buffer(1)]], device const uint *active [[buffer(2)]], device Real *gradient [[buffer(3)]], constant PlateParams &p [[buffer(4)]], uint threadID [[thread_position_in_grid]]) {
    uint id = threadID / PlateTransformLanes, lane = threadID % PlateTransformLanes;
    if (id < p.Modes) {
        Real g = PlateGradient(id, in, sy, active, p, lane);
        if (!lane) gradient[id] = g;
    }
}

// One group owns the rank-one update. At the paper's 127 and 1027 active modes,
// combining these small reductions and updates avoids two dispatch barriers.
// Modal work is distributed over 256 lanes, with fixed-order SIMD reductions.
kernel void PlateStep(device const Real *gradient [[buffer(0)]], device const uint *active [[buffer(1)]], device Real4 *state [[buffer(2)]], device const Real4 *coeff [[buffer(3)]], device Real *scalar [[buffer(4)]], device const Real *forces [[buffer(5)]], device Real4 *coupling [[buffer(6)]], device const Real *pickups [[buffer(7)]], device Real *samples [[buffer(8)]], device Real *records [[buffer(9)]], device Real *states [[buffer(10)]], constant PlateParams &p [[buffer(11)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint scratchWords[8 * sizeof(Real4) / sizeof(uint)];
    threadgroup Real4 *scratch = (threadgroup Real4 *)scratchWords;
    Real4 sum{};
    Real psi = scalar[9], beta = .25f * p.StepKappa2;
    Real scale = 1.f / PlateSqrt(2.f * scalar[2] + p.Epsilon);
    // Velocity is delta/Dt: scaling the norm and regularizer together preserves
    // either upstream controller while avoiding a per-mode velocity division.
    Real regularizer = (p.NormType == 1 ? p.Dt : p.Dt2) * p.Epsilon;
    Real control = scalar[4] > 0.f ? p.Lambda * p.Dt * scalar[8] / (scalar[4] + regularizer) : Real(0.f);
    for (uint id = tid; id < p.Modes; id += 256) {
        Real4 s = state[id], c = coeff[id];
        Real g = gradient[id] * scale;
        if (p.Nonlinear && active[id]) g -= control * (p.NormType == 1 ? PlateSign(s.y) : s.y);
        Real rhs = 2.f * s.y - c.x * s.x + p.Dt2 * (p.HasForces ? forces[p.Sample * p.Modes + id] : Real(0.f));
        rhs -= p.StepKappa2 * g * psi;
        // 1/D = 1-loss, with the small loss retained accurately in coeff.z.
        Real4 v = active[id] ? Real4{g, rhs - rhs * c.z, g - g * c.z, 0.f} : Real4{};
        coupling[id] = v;
        sum.x += v.x * v.y;
        sum.y += v.x * v.z;
    }
    sum = PlateSum(sum, scratch, tid);
    Real correction = beta * sum.x / (1.f + beta * sum.y);
    if (!tid) scalar[0] = scalar[0] + .5f * sum.x / (1.f + beta * sum.y);
    sum = {};
    for (uint id = tid; id < p.Modes; id += 256) {
        Real4 s = state[id], c = coupling[id];
        Real z = c.y - c.z * correction, delta = z - s.y;
        Real q = s.x, velocity = z * p.HalfInvDt;
        sum += Real4{.5f * (s.y * s.y * p.InvDt2 + coeff[id].w * q * (q - s.y)), velocity * (p.HasForces ? forces[p.Sample * p.Modes + id] : Real(0.f)), 2.f * coeff[id].y * velocity * velocity, 0.f};
        state[id] = Real4{s.x + delta, delta, 0.f, 0.f};
        if (p.RecordStates) states[p.Sample * p.Modes + id] = state[id].x;
        if (p.Receivers) sum.w += state[id].x * pickups[id];
    }
    sum = PlateSum(sum, scratch, tid);
    if (!tid) {
        uint offset = p.Sample * 6;
        records[offset] = sum.x + (p.Nonlinear ? .5f * p.Kappa2 * psi * psi : Real(0.f));
        records[offset + 1] = sum.y;
        records[offset + 2] = sum.z;
        records[offset + 3] = psi;
        records[offset + 4] = scalar[3];
        records[offset + 5] = scalar[8];
        if (p.Receivers) samples[p.Sample * p.Receivers] = sum.w;
    }
    for (uint receiver = 1; receiver < p.Receivers; ++receiver) {
        sum = {};
        for (uint id = tid; id < p.Modes; id += 256)
            sum.x += state[id].x * pickups[receiver * p.Modes + id];
        sum = PlateSum(sum, scratch, tid);
        if (!tid) samples[p.Sample * p.Receivers + receiver] = sum.x;
    }
}

// Both stages synthesize a grid row, form its nonlinear bracket, then project
// it while it remains in threadgroup memory. Specialization preserves each
// stage's mode count, midpoint layout and arithmetic order.
template<bool Derivative>
inline void PlateProjectRow(device const Real4 *in, device const Real4 *transform,
    device Real4 *derivative, device const Real *projection, device Real *out,
    constant PlateParams &p, threadgroup Real *row, uint group, uint tid) {
    uint y = group % p.Ny, midpoint = Derivative ? group / p.Ny : 0;
    uint lane = tid % PlateTransformLanes, rank = tid / PlateTransformLanes;
    uint modes = p.My + (Derivative ? 0 : 1);
    for (uint x = rank; x < p.Nx; x += 256 / PlateTransformLanes) {
        Real4 v{};
        for (uint j = lane; j < modes; j += PlateTransformLanes) {
            Real4 a = in[(midpoint * p.Nx + x) * modes + j], t = transform[j * p.Ny + y];
            v.x += a.x * t.x;
            v.y += a.y * t.y;
            v.z += a.z * t.z;
        }
        v = PlateReduce(v, lane);
        if (!lane) {
            v.x *= p.Scale;
            v.y *= p.Scale;
            v.z *= p.Scale;
            if (Derivative) {
                row[x] = 2.f * (v.x * v.y - v.z * v.z);
                if (!midpoint) derivative[x * p.Ny + y] = v;
            } else {
                Real4 u = derivative[x * p.Ny + y];
                row[x] = u.x * v.y + u.y * v.x - 2.f * u.z * v.z;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint projectedModes = p.Mx + (Derivative ? 1 : 0);
    for (uint i = rank; i < projectedModes; i += 256 / PlateTransformLanes) {
        Real sum{};
        for (uint x = lane; x < p.Nx; x += PlateTransformLanes)
            sum += projection[i * p.Nx + x] * row[x];
        sum = PlateReduce(sum, lane);
        if (!lane) out[(midpoint * projectedModes + i) * p.Ny + y] = sum;
    }
}

kernel void PlateDerivativeProject(device const Real4 *in [[buffer(0)]], device const Real4 *transform [[buffer(1)]], device Real4 *derivative [[buffer(2)]], device const Real *cx [[buffer(3)]], device Real *out [[buffer(4)]], constant PlateParams &p [[buffer(5)]], threadgroup Real *row [[threadgroup(0)]], uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    PlateProjectRow<true>(in, transform, derivative, cx, out, p, row, group, tid);
}

kernel void PlatePhiForce(device const Real4 *in [[buffer(0)]], device const Real4 *transform [[buffer(1)]], device Real4 *derivative [[buffer(2)]], device const Real *sx [[buffer(3)]], device Real *out [[buffer(4)]], constant PlateParams &p [[buffer(5)]], threadgroup Real *row [[threadgroup(0)]], uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    PlateProjectRow<false>(in, transform, derivative, sx, out, p, row, group, tid);
}
