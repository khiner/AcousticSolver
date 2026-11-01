#include <metal_stdlib>
using namespace metal;
using Real = FloatPair;

struct Params {
    uint modes, hidden, frames, offset, steps;
    Real dt, lambda, epsilon, scale;
};

inline Real sum(Real value, threadgroup Real *scratch, uint tid) {
    for (uint stride = 16; stride; stride >>= 1)
        value = value + Real(simd_shuffle_down(value.Hi, stride), simd_shuffle_down(value.Lo, stride));
    if (tid % 32 == 0) scratch[tid / 32] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        value = tid < 8 ? scratch[tid] : Real(0);
        for (uint stride = 4; stride; stride >>= 1)
            value = value + Real(simd_shuffle_down(value.Hi, stride), simd_shuffle_down(value.Lo, stride));
        if (tid == 0) scratch[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Real result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

inline void transpose_product(device const Real *weights, threadgroup const Real *x, threadgroup Real *out, uint m, uint h, uint tid) {
    uint lane = tid % 8;
    for (uint base = 0; base < m; base += 32) {
        uint i = base + tid / 8;
        Real value;
        if (i < m)
            for (uint j = lane; j < h; j += 8) value = value + weights[j * m + i] * x[j];
        for (uint stride = 4; stride; stride >>= 1)
            value = value + Real(simd_shuffle_xor(value.Hi, stride), simd_shuffle_xor(value.Lo, stride));
        if (lane == 0 && i < m) out[i] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void modal(device const Real *weights [[buffer(0)]], device const Real *physical [[buffer(1)]], device const Real *excitation [[buffer(2)]], device Real *output [[buffer(3)]], constant Params &a [[buffer(4)]], uint tid [[thread_index_in_threadgroup]], uint body [[threadgroup_position_in_grid]]) {
    threadgroup float2 storage[858];
    threadgroup Real *q = reinterpret_cast<threadgroup Real *>(storage);
    threadgroup Real *p = q + 75, *qHalf = p + 75, *g = qHalf + 75, *rhs = g + 75, *av = rhs + 75, *bv = av + 75;
    threadgroup Real *activation = bv + 75, *reduction = activation + 76;
    threadgroup Real &psi = reduction[256];
    uint m = a.modes, h = a.hidden, dim = 2 * m + 1;
    device const Real *state = output + (body * a.frames + a.offset) * dim;
    device const Real *phys = physical + body * (3 * m + 1);
    Real omega2, damping, shape;
    Real mu = phys[3 * m];
    if (tid < m) {
        q[tid] = state[tid];
        p[tid] = state[m + tid];
        omega2 = phys[tid];
        damping = phys[m + tid];
        shape = phys[2 * m + tid];
    }
    if (tid == 0) psi = state[2 * m];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n = a.offset; n < a.offset + a.steps; ++n) {
        if (tid < m) qHalf[tid] = q[tid] + Real(.5f) * a.dt * p[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        Real potentialHalf, potentialQ;
        for (uint j = tid; j < h; j += 256) {
            Real z, zq;
            for (uint i = 0; i < m; ++i) {
                Real w = weights[j * m + i];
                z = z + w * qHalf[i];
                zq = zq + w * q[i];
            }
            Real r = FloatPairSqrt(Real(1) + z * z), rq = FloatPairSqrt(Real(1) + zq * zq);
            Real v = (r - Real(1)) * (r - Real(1)), vq = (rq - Real(1)) * (rq - Real(1));
            Real derivative = Real(2) * z * (Real(1) - Real(1) / r);
            activation[j] = a.scale * derivative;
            potentialHalf = potentialHalf + a.scale * v;
            potentialQ = potentialQ + a.scale * vq;
        }
        Real psiHalf = FloatPairSqrt(Real(2) * sum(potentialHalf, reduction, tid) + a.epsilon);
        Real psiQ = FloatPairSqrt(Real(2) * sum(potentialQ, reduction, tid) + a.epsilon);
        Real norm = sum(tid < m ? FloatPairSign(p[tid]) * p[tid] : Real(0), reduction, tid);
        transpose_product(weights, activation, g, m, h, tid);
        if (tid < m) {
            Real force = g[tid];
            g[tid] = force / psiHalf - a.lambda * (psi - psiQ) * FloatPairSign(p[tid]) / (norm + a.epsilon);
            av[tid] = Real(.5f) * a.dt * mu * g[tid];
            bv[tid] = av[tid] / (Real(1) + a.dt * damping);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        Real ap = sum(tid < m ? av[tid] * p[tid] : Real(0), reduction, tid);
        if (tid < m) rhs[tid] = (Real(1) - a.dt * damping) * p[tid] - av[tid] * ap +
            a.dt * (-omega2 * qHalf[tid] - mu * mu * g[tid] * psi + shape * excitation[body * a.frames + n]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        Real br = sum(tid < m ? bv[tid] * rhs[tid] : Real(0), reduction, tid);
        Real ab = sum(tid < m ? av[tid] * bv[tid] : Real(0), reduction, tid);
        Real nextP, pHalf;
        if (tid < m) {
            nextP = rhs[tid] / (Real(1) + a.dt * damping) - bv[tid] * br / (Real(1) + ab);
            pHalf = Real(.5f) * (p[tid] + nextP);
        }
        Real increment = sum(tid < m ? g[tid] * pHalf : Real(0), reduction, tid);
        if (tid == 0) {
            psi = psi + a.dt * increment;
            output[(body * a.frames + n + 1) * dim + 2 * m] = psi;
        }
        if (tid < m) {
            p[tid] = nextP;
            q[tid] = qHalf[tid] + Real(.5f) * a.dt * nextP;
            output[(body * a.frames + n + 1) * dim + tid] = q[tid];
            output[(body * a.frames + n + 1) * dim + m + tid] = p[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
