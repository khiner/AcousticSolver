#include <metal_stdlib>
using namespace metal;

inline Real mv(CoupledMatrix m, uint row, device const uint *ix, device const Real *v, device const Real *x, device const Real *previous = nullptr) {
    Real sum;
    for (uint j = ix[m.Row + row]; j < ix[m.Row + row + 1]; ++j)
        sum = sum + v[m.Value + j] * (previous ? x[ix[m.Column + j]] - previous[ix[m.Column + j]] : x[ix[m.Column + j]]);
    return sum;
}
inline Real band(device const Real *matrix, device const Real *x, uint row, uint n, int stride, bool transpose = false) {
    Real sum;
    for (int j = 0; j < 9; ++j) {
        int col = int(row) + (j / 3 - 1) * stride + j % 3 - 1;
        if (col >= 0 && col < int(n)) sum = sum + matrix[transpose ? 9 * col + 8 - j : 9 * row + j] * x[col];
    }
    return sum;
}
inline Real stressForce(constant PanelParams &p, device const Real *v, device const Real *l, device const Real *phi, uint row) {
    const Real scale = p.Adjoint ? p.FScale * v[p.MassInverse + row] : p.FScale;
    return scale * band(l, phi, row, p.N, p.Stride, p.Adjoint);
}
inline Real dot(device const Real *x, device const Real *y, uint n, uint tid, threadgroup float2 *sums) {
    Real sum;
    for (uint i = tid; i < n; i += 256) sum = sum + x[i] * y[i];
    for (uint stride = 16; stride; stride >>= 1)
        sum = sum + Real(simd_shuffle_down(sum.Hi, stride), simd_shuffle_down(sum.Lo, stride));
    if (tid % 32 == 0) sums[tid / 32] = float2(sum.Hi, sum.Lo);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        sum = tid < 8 ? Real(sums[tid].x, sums[tid].y) : Real(0);
        for (uint stride = 4; stride; stride >>= 1)
            sum = sum + Real(simd_shuffle_down(sum.Hi, stride), simd_shuffle_down(sum.Lo, stride));
        if (tid == 0) sums[0] = float2(sum.Hi, sum.Lo);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    Real result(sums[0].x, sums[0].y);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
inline void apply(constant PanelParams &p, device const uint *ix, device const Real *v, device const Real *l, device const Real *x, device Real *y, device Real *tmp, uint tid) {
    for (uint i = tid; i < p.N; i += 256) tmp[i] = stressForce(p, v, l, x, i);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < p.N; i += 256) y[i] = mv(p.K, i, ix, v, x) + p.WScale * band(l, tmp, i, p.N, p.Stride);
    threadgroup_barrier(mem_flags::mem_device);
}
inline bool converged(constant PanelParams &p, Real residualNorm) {
    // Include one minimum-normal allowance per squared residual entry for underflow.
    return p.PreconditionBound.Hi > 0 && (p.PreconditionBound * (residualNorm + Real(float(p.N) * 0x1p-126f))).Hi < p.Tol2.Hi;
}
template<bool Compact>
inline void preconditionRows(constant PanelParams &p, device const uint *ix, device const Real *v, device const Real *r, device Real *z, uint tid) {
    const uint lane = tid % 8, n = p.PreconditionSize;
    device const uint *order = ix + p.PreconditionIndices;
    for (uint i = n + tid; i < p.N; i += 256) z[order[i]] = v[p.Inverse + n * n + i - n] * r[order[i]];
    for (uint i = tid / 8; i < n; i += 32) {
        Real a, b, c, d;
        uint j = lane;
        for (; j + 24 < n; j += 32) {
            a = a + v[p.Inverse + i * n + j] * r[(Compact ? order[j] : j)];
            b = b + v[p.Inverse + i * n + j + 8] * r[(Compact ? order[j + 8] : j + 8)];
            c = c + v[p.Inverse + i * n + j + 16] * r[(Compact ? order[j + 16] : j + 16)];
            d = d + v[p.Inverse + i * n + j + 24] * r[(Compact ? order[j + 24] : j + 24)];
        }
        for (; j < n; j += 8) a = a + v[p.Inverse + i * n + j] * r[(Compact ? order[j] : j)];
        Real sum = (a + b) + (c + d);
        for (uint stride = 4; stride; stride >>= 1)
            sum = sum + Real(simd_shuffle_down(sum.Hi, stride), simd_shuffle_down(sum.Lo, stride));
        if (lane == 0) z[Compact ? order[i] : i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_device);
}
inline void precondition(constant PanelParams &p, device const uint *ix, device const Real *v, device const Real *r, device Real *z, uint tid) {
    if (p.PreconditionSize == p.N) preconditionRows<false>(p, ix, v, r, z, tid);
    else preconditionRows<true>(p, ix, v, r, z, tid);
}

kernel void CoupledPanel(device const uint *ix [[buffer(0)]], device const Real *v [[buffer(1)]], device Real *state [[buffer(2)]], device const Real *forces [[buffer(3)]], device Real *output [[buffer(4)]], device uint *status [[buffer(5)]], constant PanelParams *panels [[buffer(6)]], constant PanelDispatch &dispatch [[buffer(7)]], uint panel [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constant PanelParams &p = panels[panel];
    // Each group advances its own coupling iterations and PCG segments before the shared air update.
    if (dispatch.Phase == 1 && dispatch.Solve >= p.Iterations * p.Segments) return;
    const uint segment = dispatch.Solve % p.Segments;
    state += p.StateOffset;
    status += panel;
    threadgroup float2 sums[8];
    const uint n = p.N;
    device Real *w = state, *u1 = state + n, *u2 = state + 2 * n;
    device Real *phi = state + 3 * n, *phi1 = state + 4 * n, *phi0 = state + 5 * n;
    device Real *transfer = state + 6 * n, *aa = state + 7 * n, *bb = state + 8 * n, *right = state + 9 * n;
    device Real *r = state + 10 * n, *direction = state + 11 * n, *omega = state + 12 * n;
    device Real *z = state + 13 * n, *tmp = state + 14 * n, *l = state + 15 * n, *rho = state + 24 * n;
    if (dispatch.Phase == 0) {
        if (p.Nonlinear) {
            for (uint i = tid; i < n; i += 256) {
                Real d[6];
                for (uint j = 0; j < 6; ++j) d[j] = band(v + p.Derivatives + 9 * n * j, u1, i, n, p.Stride);
                Real a[9] = {
                    Real(-.5f) * d[2], d[0] + Real(.5f) * (d[2] - d[3]), Real(.5f) * d[3],
                    d[1] + Real(.5f) * (d[2] - d[4]), Real(-2) * (d[0] + d[1]) - Real(.5f) * (d[2] - d[3] - d[4] + d[5]),
                    d[1] + Real(.5f) * (d[5] - d[3]), Real(.5f) * d[4], d[0] + Real(.5f) * (d[5] - d[4]), Real(-.5f) * d[5]
                };
                for (uint j = 0; j < 9; ++j) l[9 * i + j] = a[j] * p.InvH2 * v[p.Mask + 9 * i + j];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < n; i += 256) {
            Real value = transfer[i] - mv(p.C, i, ix, v, u2) + mv(p.B, i, ix, v, u1);
            if (p.Nonlinear) value = value + stressForce(p, v, l, phi0, i);
            value = value + mv(p.IJ, i, ix, v, u2);
            for (uint f = 0; f < p.ForceCount; ++f)
                if (ix[p.ForceIndices + f] == i) value = value + forces[p.ForceOffset + dispatch.Step * p.ForceCount + f];
            aa[i] = value;
            if (p.Nonlinear) bb[i] = -mv(p.K, i, ix, v, phi1);
            else w[i] = value;
        }
    } else if (dispatch.Phase == 1) {
        Real value;
        if (segment == 0) {
            for (uint i = tid; i < n; i += 256) right[i] = aa[i] - mv(p.IJ, i, ix, v, w);
            threadgroup_barrier(mem_flags::mem_device);
            if (!p.Nonlinear) {
                for (uint i = tid; i < n; i += 256) w[i] = right[i];
                return;
            }
            apply(p, ix, v, l, phi, omega, tmp, tid);
            for (uint i = tid; i < n; i += 256) r[i] = bb[i] - p.WScale * band(l, right, i, n, p.Stride) - omega[i];
            threadgroup_barrier(mem_flags::mem_device);
            Real norm = dot(r, r, n, tid, sums);
            if (!isfinite(norm.Hi)) {
                if (tid == 0) status[0] = 1;
                return;
            }
            if (norm.Hi >= p.Tol2.Hi && !converged(p, norm)) {
                precondition(p, ix, v, r, direction, tid);
                value = dot(r, direction, n, tid, sums);
            }
        } else value = rho[0];
        for (uint iteration = 0; iteration < p.IterationsPerDispatch && p.IterationsPerDispatch * segment + iteration <= p.MaxIterations; ++iteration) {
            if (!(value.Hi > p.Tol2.Hi)) break;
            apply(p, ix, v, l, direction, omega, tmp, tid);
            Real denominator = dot(direction, omega, n, tid, sums);
            if (!(denominator.Hi > 0) || !isfinite(denominator.Hi)) {
                if (tid == 0) status[0] = 1;
                return;
            }
            Real alpha = value / denominator;
            for (uint i = tid; i < n; i += 256) {
                phi[i] = phi[i] + alpha * direction[i];
                r[i] = r[i] - alpha * omega[i];
            }
            threadgroup_barrier(mem_flags::mem_device);
            if (converged(p, dot(r, r, n, tid, sums))) {
                value = Real(0);
                break;
            }
            precondition(p, ix, v, r, z, tid);
            Real next = dot(r, z, n, tid, sums);
            Real beta = next / value;
            for (uint i = tid; i < n; i += 256) direction[i] = z[i] + beta * direction[i];
            threadgroup_barrier(mem_flags::mem_device);
            value = next;
        }
        if (tid == 0) rho[0] = value;
        if ((segment + 1) * p.IterationsPerDispatch > p.MaxIterations) {
            if (tid == 0 && (!isfinite(value.Hi) || value.Hi > p.Tol2.Hi)) status[0] = 2;
            for (uint i = tid; i < n; i += 256) w[i] = right[i] + stressForce(p, v, l, phi, i);
        }
    } else {
        if (tid == 0) output[dispatch.Step * p.Channels + p.Channel] = w[p.Pickup];
        for (uint i = tid; i < n; i += 256) {
            u2[i] = u1[i];
            u1[i] = w[i];
            phi0[i] = phi1[i];
            phi1[i] = phi[i];
        }
    }
}

kernel void CoupledAir(device Real *u [[buffer(0)]], device const Real *u1 [[buffer(1)]], device const Real *u2 [[buffer(2)]], device const Real *coeffs [[buffer(3)]], device const uchar *indices [[buffer(4)]], constant AirParams &p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.N) return;
    device const Real *c = coeffs + 16 * indices[i];
    int offsets[7] = {0, int(p.Ny), -int(p.Ny), 1, -1, int(p.Slice), -int(p.Slice)};
    Real a, b;
    for (int j = 0; j < 7; ++j) {
        int col = int(i) + offsets[j];
        if (col >= 0 && col < int(p.N)) {
            a = a + c[j] * u1[col];
            b = b + c[j ? j + 8 : 7] * u2[col];
        }
    }
    u[i] = c[8] * (a - b);
}

kernel void CoupledReverse(device const uint *ix [[buffer(0)]], device const Real *v [[buffer(1)]], device const Real *state [[buffer(2)]], device Real *interface [[buffer(3)]], device const Real *mask [[buffer(4)]], device Real *u1 [[buffer(5)]], constant InterfaceParams &params [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    constant PanelParams &s = params.Panel;
    constant AirParams &p = params.Air;
    if (i >= p.Slice) return;
    Real old = interface[p.Slice + i];
    Real star = -mask[i] * old + p.QRate * mv(s.J, i, ix, v, state, state + 2 * s.N);
    Real diff = interface[i] - p.Gamma2 * star + p.Gamma2 * mask[i] * old;
    interface[i] = diff;
    interface[3 * p.Slice + i] = star;
    if (mask[i].Hi > .5f) {
        u1[p.Lower * p.Slice + i] = Real(.5f) * (interface[2 * p.Slice + i] + diff);
        u1[(p.Lower + 1) * p.Slice + i] = Real(.5f) * (interface[2 * p.Slice + i] - diff);
    }
}
kernel void CoupledForward(device const Real *u [[buffer(0)]], device Real *interface [[buffer(1)]], constant AirParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.Slice) return;
    Real old = interface[p.Slice + i];
    Real current = interface[i];
    Real down = u[p.Lower * p.Slice + i], up = u[(p.Lower + 1) * p.Slice + i];
    Real next = down - up + p.Lambda2 * (interface[3 * p.Slice + i] - old);
    interface[p.Slice + i] = current;
    interface[i] = next;
    interface[2 * p.Slice + i] = down + up;
    interface[4 * p.Slice + i] = next + Real(2) * p.Gamma2 * current - old;
}
kernel void CoupledInterpolate(device const uint *ix [[buffer(0)]], device const Real *v [[buffer(1)]], device Real *state [[buffer(2)]], device const Real *input [[buffer(3)]], constant PanelParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < p.N) state[6 * p.N + i] = mv(p.I, i, ix, v, input);
}
kernel void CoupledSample(device const Real *u [[buffer(0)]], device const Real *u2 [[buffer(1)]], device const uint *receivers [[buffer(2)]], device Real *output [[buffer(3)]], constant AirParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < p.Receivers) output[p.Step * p.Channels + i] = p.Rate * (u[receivers[i]] - u2[receivers[i]]);
}
kernel void CoupledBoundary(device Real *u [[buffer(0)]], constant AirParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.N) return;
    uint y = i % p.Ny, x = (i % p.Slice) / p.Ny, z = i / p.Slice;
    if (x == 0 || x + 1 == p.Slice / p.Ny || y == 0 || y + 1 == p.Ny || z == 0 || z + 1 == p.N / p.Slice) u[i] = Real(0);
}
