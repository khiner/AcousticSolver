#include <metal_stdlib>
using namespace metal;
using R = StringReal;

inline R StringSum(R value, threadgroup R *shared, uint tid) {
    shared[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) shared[tid] = shared[tid] + shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    R result = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}
inline R StringNode(device const R *x, int i, uint n) {
    return i < 0 || uint(i) >= n ? R(0) : x[i];
}
inline R StringLap(device const R *x, int i, constant StringParams &p) {
    if (i < 0 || uint(i) >= p.Points) return R(0);
    return (StringNode(x, i - 1, p.Points) - R(2) * x[i] + StringNode(x, i + 1, p.Points)) * p.InvH2;
}
inline R StringBiharmonic(device const R *x, int i, constant StringParams &p) {
    return (StringLap(x, i - 1, p) - R(2) * StringLap(x, i, p) + StringLap(x, i + 1, p)) * p.InvH2;
}
inline R StringSlope(device const R *x, uint i, constant StringParams &p) {
    return (StringNode(x, int(i), p.Points) - StringNode(x, int(i) - 1, p.Points)) * p.InvH;
}

// Add one node's linear potential without changing the reduction order.
inline R StringAddLinearPotential(R value, device const R *x, uint i, constant StringParams &p) {
    R lap = StringLap(x + (i / p.Points) * p.Points, int(i % p.Points), p);
    value = value - R(.5) * p.H * (i < p.Points ? p.Tension : p.LongitudinalTension) * x[i] * lap;
    if (i < p.Points) value = value + R(.5) * p.H * p.Rigidity * lap * lap;
    return value;
}

// Shared geometric potential and its derivatives with respect to edge slopes.
struct StringGeometry {
    R U, V, Potential;
};
inline StringGeometry StringGeometric(R q, R longitudinal, constant StringParams &p) {
    R a = R(1) + longitudinal, radius = StringSqrt(a * a + q * q);
    // Rationalization retains small extensions near the unstretched state.
    R extension = (R(2) * longitudinal + longitudinal * longitudinal + q * q) / (radius + R(1));
    R coefficient = p.H * p.NonlinearModulus;
    if (p.Model == 0)
        return {coefficient * extension * q / radius, coefficient * extension * a / radius, R(.5) * coefficient * extension * extension};
    R radMinusA = (radius + a).Hi != 0 ? q * q / (radius + a) : radius - a;
    return {coefficient * q * extension / radius, coefficient * radMinusA / radius, coefficient * (R(.5) + R(.5) * q * q - radMinusA)};
}
// Evolve displacement increments directly. Keeping the unit inertial term
// exact avoids mass*inverseMass bias growing as dt becomes very small.
inline R StringIncrement(device const R *x, device const R *increment, uint i, bool full, constant StringParams &p) {
    R damping = p.Sigma0 * p.Dt, inverseDamping = R(1) / (R(1) + damping);
    R delta = increment[i];
    if (full) {
        uint n = p.Points, node = i % n;
        device const R *component = x + (i / n) * n;
        R tension = i < n ? p.Tension : p.LongitudinalTension;
        delta = delta - R(2) * damping * inverseDamping * delta;
        delta = delta + p.InverseMass * tension * p.Dt2 * StringLap(component, int(node), p);
        if (i < n) {
            delta = delta - p.InverseMass * p.Rigidity * p.Dt2 * StringBiharmonic(component, int(node), p);
            delta = delta + R(2) * p.Sigma1 * p.Dt * inverseDamping * StringLap(increment, int(node), p);
        }
    } else {
        // The upstream unsplit SAV recurrence omits explicit damping terms.
        delta = delta - damping * inverseDamping * (x[i] + delta);
    }
    return delta;
}

inline void StringReferenceStep(device const R *x, device const R *increment, device R *next, device R *nextIncrement, device R *scratch, threadgroup R *reduction, uint tid, constant StringParams &p) {
    uint n = p.Points, d = n * p.Components, edge = n + 1;
    device R *fluxU = scratch, *fluxV = fluxU + edge, *gradient = fluxV + edge;
    device R *csi = gradient + d, *chi = csi + d;
    if (p.Model <= 1) {
        for (uint i = tid; i < edge; i += 256) {
            auto geometry = StringGeometric(StringSlope(x, i, p), StringSlope(x + n, i, p), p);
            fluxU[i] = geometry.U;
            fluxV[i] = geometry.V;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < d; i += 256) {
            device const R *flux = i < n ? fluxU : fluxV;
            R force = (flux[i % n + 1] - flux[i % n]) * p.InvH;
            nextIncrement[i] = StringIncrement(x, increment, i, true, p) + p.InverseMass * p.Dt2 * p.InvH * force;
        }
    } else if (p.Model == 3) {
        R factor = p.Nonlinear ? R(.5) * p.Dt * StringSqrt(p.H * p.EA / p.Length) : R(0);
        R dotPrevious{};
        for (uint i = tid; i < n; i += 256) {
            gradient[i] = factor * StringLap(x, int(i), p);
            dotPrevious = dotPrevious + gradient[i] * (R(2) * x[i] - increment[i]);
        }
        R b1 = StringSum(dotPrevious, reduction, tid), dotCsi{}, dotChi{};
        for (uint i = tid; i < n; i += 256) {
            R q = gradient[i];
            csi[i] = StringIncrement(x, increment, i, true, p) - p.InverseMass * q * b1;
            chi[i] = p.InverseMass * q;
            dotCsi = dotCsi + q * csi[i];
            dotChi = dotChi + q * chi[i];
        }
        R correction = StringSum(dotCsi, reduction, tid) / (R(1) + StringSum(dotChi, reduction, tid));
        for (uint i = tid; i < n; i += 256) nextIncrement[i] = csi[i] - chi[i] * correction;
    } else {
        // Parallel cyclic reduction solves the SPD tridiagonal reference system.
        // Ping-pong rows preserve every neighbor read before the next elimination.
        device R *rows = chi + d, *other = rows + 4 * n;
        R factor = p.Nonlinear ? R(.25) * p.Dt2 * p.NonlinearModulus * p.InvH2 * p.InverseMass : R(0);
        for (uint i = tid; i < n; i += 256) {
            R ql = StringSlope(x, i, p), qr = StringSlope(x, i + 1, p);
            R left = factor * ql * ql, right = factor * qr * qr;
            rows[4 * i] = i ? -left : R(0);
            rows[4 * i + 1] = R(1) + left + right;
            rows[4 * i + 2] = i + 1 < n ? -right : R(0);
            R leftDifference = R(2) * (StringNode(x, int(i) - 1, n) - x[i]) - (StringNode(increment, int(i) - 1, n) - increment[i]);
            R rightDifference = R(2) * (StringNode(x, int(i) + 1, n) - x[i]) - (StringNode(increment, int(i) + 1, n) - increment[i]);
            rows[4 * i + 3] = StringIncrement(x, increment, i, true, p) + left * leftDifference + right * rightDifference;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint stride = 1; stride < n; stride <<= 1) {
            for (uint i = tid; i < n; i += 256) {
                R diagonal = rows[4 * i + 1], rhs = rows[4 * i + 3], lower{}, upper{};
                if (i >= stride) {
                    uint j = 4 * (i - stride);
                    R multiplier = -rows[4 * i] / rows[j + 1];
                    lower = multiplier * rows[j];
                    diagonal = diagonal + multiplier * rows[j + 2];
                    rhs = rhs + multiplier * rows[j + 3];
                }
                if (i + stride < n) {
                    uint j = 4 * (i + stride);
                    R multiplier = -rows[4 * i + 2] / rows[j + 1];
                    upper = multiplier * rows[j + 2];
                    diagonal = diagonal + multiplier * rows[j];
                    rhs = rhs + multiplier * rows[j + 3];
                }
                other[4 * i] = lower;
                other[4 * i + 1] = diagonal;
                other[4 * i + 2] = upper;
                other[4 * i + 3] = rhs;
            }
            threadgroup_barrier(mem_flags::mem_device);
            device R *swap = rows;
            rows = other;
            other = swap;
        }
        for (uint i = tid; i < n; i += 256) nextIncrement[i] = rows[4 * i + 3] / rows[4 * i + 1];
    }
    for (uint i = tid; i < d; i += 256) next[i] = x[i] + nextIncrement[i];
    threadgroup_barrier(mem_flags::mem_device);
}

// Reference integrators reconstruct psi from the midpoint potential and use
// their own discrete nonlinear energy; it is not the SAV quadratic surrogate.
inline R StringReferencePotential(device const R *x, device const R *next, device R *mid, threadgroup R *reduction, uint tid, constant StringParams &p, thread R &psi) {
    uint n = p.Points, d = n * p.Components;
    for (uint i = tid; i < d; i += 256) mid[i] = R(.5) * (x[i] + next[i]);
    threadgroup_barrier(mem_flags::mem_device);
    R potential{}, nonlinear{}, norm{}, cross{};
    for (uint i = tid; i <= n; i += 256) {
        R q = StringSlope(mid, i, p);
        if (p.Model <= 1) potential = potential + StringGeometric(q, StringSlope(mid + n, i, p), p).Potential;
        else if (p.Nonlinear) {
            R product = StringSlope(x, i, p) * StringSlope(next, i, p);
            if (p.Model == 2) {
                R factor = R(.125) * p.H * p.NonlinearModulus;
                potential = potential + factor * q * q * q * q;
                nonlinear = nonlinear + factor * product * product;
            } else {
                norm = norm + q * q;
                cross = cross + product;
            }
        }
    }
    potential = StringSum(potential, reduction, tid);
    if (p.Model <= 1) nonlinear = potential;
    else if (p.Model == 2) nonlinear = StringSum(nonlinear, reduction, tid);
    else {
        norm = StringSum(norm, reduction, tid);
        cross = StringSum(cross, reduction, tid);
        R factor = R(.125) * p.EA * p.H * p.H / p.Length;
        potential = factor * norm * norm;
        nonlinear = factor * cross * cross;
    }
    if (!p.Split) {
        R linear{};
        for (uint i = tid; i < d; i += 256) linear = StringAddLinearPotential(linear, mid, i, p);
        potential = potential + StringSum(linear, reduction, tid);
    }
    R shift = p.Model == 3 && p.Split ? R(0) : (p.Model <= 1 ? p.Shift : R(2) * p.Shift);
    psi = StringSqrt(R(2) * potential + shift);
    return nonlinear;
}

// One cooperative group owns a string. Bounded batches avoid dispatching every
// timestep, while device barriers keep neighbor reads and state rotations ordered.
kernel void StringAdvance(device R *state [[buffer(0)]], device R *scratch [[buffer(1)]], device R *scalars [[buffer(2)]], device R *samples [[buffer(3)]], device R *records [[buffer(4)]], device R *states [[buffer(5)]], constant StringParams &p [[buffer(6)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 reductionStorage[256];
    threadgroup R *reduction = reinterpret_cast<threadgroup R *>(reductionStorage);
    uint n = p.Points, d = n * p.Components, edge = n + 1;
    device R *x = state, *previous = state + d, *next = state + 2 * d;
    device R *increment = state + 3 * d, *nextIncrement = state + 4 * d;
    device R *fluxU = scratch, *fluxV = fluxU + edge, *gradient = fluxV + edge;
    device R *csi = gradient + d, *chi = csi + d;
    R psi = scalars[0], loss = scalars[1];
    R fac = p.Dt2 * p.InvH, quarter = R(.25) * fac;
    for (uint step = 0; step < p.Steps; ++step) {
        if (p.Reference) StringReferenceStep(x, increment, next, nextIncrement, scratch, reduction, tid, p);
        else {
            R normLocal{};
            if (p.Model == 3)
                for (uint i = tid; i < edge; i += 256) {
                    R q = StringSlope(x, i, p);
                    normLocal = normLocal + q * q;
                }
            R norm = StringSum(normLocal, reduction, tid);
            R potentialLocal{};
            for (uint i = tid; i < edge; i += 256) {
                R q = StringSlope(x, i, p), fu{}, fv{}, v{};
                if (p.Nonlinear) {
                    if (p.Model <= 1) {
                        auto geometry = StringGeometric(q, StringSlope(x + n, i, p), p);
                        fu = geometry.U;
                        fv = geometry.V;
                        v = geometry.Potential;
                    } else if (p.Model == 2) {
                        fu = R(.5) * p.H * p.NonlinearModulus * q * q * q;
                        v = R(.125) * p.H * p.NonlinearModulus * q * q * q * q;
                    } else {
                        fu = R(.5) * p.EA * p.H * p.H / p.Length * norm * q;
                        // The global KC energy is added once below.
                    }
                }
                fluxU[i] = fu;
                fluxV[i] = fv;
                potentialLocal = potentialLocal + v;
            }
            if (!p.Split)
                for (uint i = tid; i < d; i += 256) potentialLocal = StringAddLinearPotential(potentialLocal, x, i, p);
            R potential = StringSum(potentialLocal, reduction, tid);
            if (p.Model == 3 && p.Nonlinear)
                potential = potential + R(.125) * p.EA * p.H * p.H / p.Length * norm * norm;
            R divisor = StringSqrt(R(2) * potential + (p.Model <= 1 ? p.Shift : R(2) * p.Shift));
            threadgroup_barrier(mem_flags::mem_device);
            R dotPrevious{};
            for (uint i = tid; i < d; i += 256) {
                uint node = i % n;
                device R *flux = i < n ? fluxU : fluxV;
                R g = (flux[node] - flux[node + 1]) * p.InvH;
                if (!p.Split) {
                    device R *component = x + (i / n) * n;
                    R tension = i < n ? p.Tension : p.LongitudinalTension;
                    g = g - p.H * tension * StringLap(component, int(node), p);
                    if (i < n) g = g + p.H * p.Rigidity * StringBiharmonic(component, int(node), p);
                }
                if (p.Model == 3 && p.Split)
                    g = p.Nonlinear ? -StringSqrt(p.EA * p.H * p.H / p.Length) * StringLap(x, int(node), p) : R(0);
                else
                    g = divisor.Hi != 0 ? g / divisor : R(0);
                gradient[i] = g;
                dotPrevious = dotPrevious + g * increment[i];
            }
            R b1 = StringSum(dotPrevious, reduction, tid);
            R dotCsi{}, dotChi{};
            for (uint i = tid; i < d; i += 256) {
                R lin = StringIncrement(x, increment, i, p.Split, p);
                R g = gradient[i];
                csi[i] = lin - p.InverseMass * (quarter * g * b1 + fac * g * psi);
                chi[i] = p.InverseMass * quarter * g;
                dotCsi = dotCsi + g * csi[i];
                dotChi = dotChi + g * chi[i];
            }
            R c1 = StringSum(dotCsi, reduction, tid);
            R c2 = R(1) + StringSum(dotChi, reduction, tid);
            R psiDelta{};
            for (uint i = tid; i < d; i += 256) {
                nextIncrement[i] = csi[i] - chi[i] * (c1 / c2);
                next[i] = x[i] + nextIncrement[i];
                psiDelta = psiDelta + gradient[i] * (nextIncrement[i] + increment[i]);
            }
            psi = psi + R(.5) * StringSum(psiDelta, reduction, tid);
        }
        threadgroup_barrier(mem_flags::mem_device);
        R nonlinear = p.Reference ? StringReferencePotential(x, next, csi, reduction, tid, p, psi) : R(.5) * psi * psi;
        R kinetic{}, linear{}, dissipated{};
        for (uint i = tid; i < d; i += 256) {
            uint node = i % n;
            device R *component = x + (i / n) * n, *future = next + (i / n) * n;
            R velocity = nextIncrement[i] * p.InvDt;
            kinetic = kinetic + R(.5) * p.Mass * p.H * velocity * velocity;
            if (p.Split || p.Reference) {
                R tension = i < n ? p.Tension : p.LongitudinalTension;
                linear = linear - R(.5) * p.H * tension * next[i] * StringLap(component, int(node), p);
                if (i < n) linear = linear + R(.5) * p.H * p.Rigidity * StringLap(future, int(node), p) * StringLap(component, int(node), p);
            }
            R centered = R(.5) * (nextIncrement[i] + increment[i]) * p.InvDt;
            R work = -p.Sigma0 * centered * centered;
            if (i < n) work = work + p.Sigma1 * centered * StringLap(increment, int(node), p) * p.InvDt;
            dissipated = dissipated + R(2) * p.Mass * p.H * work;
        }
        kinetic = StringSum(kinetic, reduction, tid);
        linear = StringSum(linear, reduction, tid);
        loss = loss - p.Dt * StringSum(dissipated, reduction, tid);
        if (tid == 0) {
            ulong frame = ulong(p.Offset) + step;
            samples[frame * p.Components] = next[p.ReceiverU];
            if (p.Components == 2) samples[frame * 2 + 1] = next[n + p.ReceiverV];
            records[6 * frame] = kinetic;
            records[6 * frame + 1] = linear;
            records[6 * frame + 2] = nonlinear;
            records[6 * frame + 3] = loss;
            records[6 * frame + 4] = kinetic + linear + nonlinear + loss;
            records[6 * frame + 5] = psi;
        }
        if (p.RecordStates)
            for (uint i = tid; i < d; i += 256) states[(ulong(p.Offset) + step) * d + i] = next[i];
        threadgroup_barrier(mem_flags::mem_device);
        // Copying into canonical slots makes arbitrary Run() chunking deterministic.
        for (uint i = tid; i < d; i += 256) {
            previous[i] = x[i];
            x[i] = next[i];
            increment[i] = nextIncrement[i];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    if (tid == 0) {
        scalars[0] = psi;
        scalars[1] = loss;
    }
}
