#include <metal_stdlib>
using namespace metal;

inline int ImmersedCid(int x, int y, int z, constant ImmersedGridParams &g) {
    return x + g.Nx * (y + g.Ny * z);
}

inline bool ImmersedInPml(int x, int y, int z, constant ImmersedGridParams &g) {
    return x < g.PmlWidth || x >= g.Nx - g.PmlWidth || y < g.PmlWidth || y >= g.Ny - g.PmlWidth ||
        z < g.PmlWidth || z >= g.Nz - g.PmlWidth;
}

kernel void ImmersedVelocity(device const float *p [[buffer(0)]],
                             device float *vx [[buffer(1)]], device float *vy [[buffer(2)]],
                             device float *vz [[buffer(3)]],
                             device const float *pml_nv [[buffer(4)]], device const float *pml_dv [[buffer(5)]],
                             device const uchar *solid [[buffer(6)]],
                             constant ImmersedGridParams &g [[buffer(7)]],
                             uint3 tid [[thread_position_in_grid]]) {
    const int x = int(tid.x), y = int(tid.y), z = int(tid.z);
    if (x >= g.Nx || y >= g.Ny || z >= g.Nz) return;
    const int cid = ImmersedCid(x, y, z, g);
    const float center = p[cid];

    const int dx = min(x + 1, g.Nx - 1 - x);
    const int dy = min(y + 1, g.Ny - 1 - y);
    const int dz = min(z + 1, g.Nz - 1 - z);
    const float right = x + 1 < g.Nx ? p[cid + 1] : center;
    const float up = y + 1 < g.Ny ? p[cid + g.Nx] : center;
    const float back = z + 1 < g.Nz ? p[cid + g.Nx * g.Ny] : center;
    vx[cid] = x + 1 < g.Nx && (!g.HasSolid || (!solid[cid] && !solid[cid + 1])) ?
        (pml_nv[dx] * vx[cid] - g.InvRhoDtH * (right - center)) * pml_dv[dx] : 0.f;
    vy[cid] = y + 1 < g.Ny && (!g.HasSolid || (!solid[cid] && !solid[cid + g.Nx])) ?
        (pml_nv[dy] * vy[cid] - g.InvRhoDtH * (up - center)) * pml_dv[dy] : 0.f;
    vz[cid] = z + 1 < g.Nz && (!g.HasSolid || (!solid[cid] && !solid[cid + g.Nx * g.Ny])) ?
        (pml_nv[dz] * vz[cid] - g.InvRhoDtH * (back - center)) * pml_dv[dz] : 0.f;
}

kernel void ImmersedPressure(device float *p [[buffer(0)]],
                             device const float *vx [[buffer(1)]], device const float *vy [[buffer(2)]],
                             device const float *vz [[buffer(3)]],
                             device float *px [[buffer(4)]], device float *py [[buffer(5)]],
                             device float *pz [[buffer(6)]],
                             device const float *pml_np [[buffer(7)]], device const float *pml_dp [[buffer(8)]],
                             device const uchar *solid [[buffer(9)]],
                             constant ImmersedGridParams &g [[buffer(10)]],
                             uint3 tid [[thread_position_in_grid]]) {
    const int x = int(tid.x), y = int(tid.y), z = int(tid.z);
    if (x >= g.Nx || y >= g.Ny || z >= g.Nz) return;
    const int cid = ImmersedCid(x, y, z, g);
    if (g.HasSolid && solid[cid]) {
        p[cid] = px[cid] = py[cid] = pz[cid] = 0.f;
        return;
    }
    const float vx_left = x > 0 ? vx[cid - 1] : 0.f;
    const float vy_down = y > 0 ? vy[cid - g.Nx] : 0.f;
    const float vz_front = z > 0 ? vz[cid - g.Nx * g.Ny] : 0.f;
    const float div_x = vx[cid] - vx_left;
    const float div_y = vy[cid] - vy_down;
    const float div_z = vz[cid] - vz_front;
    if (ImmersedInPml(x, y, z, g)) {
        const int dx = min(x, g.Nx - 1 - x);
        const int dy = min(y, g.Ny - 1 - y);
        const int dz = min(z, g.Nz - 1 - z);
        const float next_x = (pml_np[dx] * px[cid] - g.RhoCcDtH * div_x) * pml_dp[dx];
        const float next_y = (pml_np[dy] * py[cid] - g.RhoCcDtH * div_y) * pml_dp[dy];
        const float next_z = (pml_np[dz] * pz[cid] - g.RhoCcDtH * div_z) * pml_dp[dz];
        px[cid] = next_x;
        py[cid] = next_y;
        pz[cid] = next_z;
        p[cid] = next_x + next_y + next_z;
    } else {
        p[cid] -= g.RhoCcDtH * (div_x + div_y + div_z);
    }
}

inline float ImmersedFilterForcing(device const ImmersedFilter *filters, device const ImmersedFilterState *states, int patch) {
    const ImmersedFilter f = filters[patch];
    const ImmersedFilterState s = states[patch];
    return f.B1 * s.X1 + f.B2 * s.X2 - f.A1 * s.Y1 - f.A2 * s.Y2 + s.Previous;
}

inline void ImmersedFilterAdvance(device const ImmersedFilter *filters,
                                  device ImmersedFilterState *states, int patch, float input) {
    const ImmersedFilter f = filters[patch];
    ImmersedFilterState s = states[patch];
    const float output = f.B0 * input + f.B1 * s.X1 + f.B2 * s.X2 - f.A1 * s.Y1 - f.A2 * s.Y2;
    s.X2 = s.X1;
    s.X1 = input;
    s.Y2 = s.Y1;
    s.Y1 = output;
    s.Previous = output;
    states[patch] = s;
}

inline float ImmersedVelocityValue(int stacked, int cells, device const float *vx,
                                   device const float *vy, device const float *vz) {
    const int component = stacked / cells;
    const int index = stacked - component * cells;
    return component == 0 ? vx[index] : (component == 1 ? vy[index] : vz[index]);
}

kernel void ImmersedVelocityHistory(device float *vx [[buffer(0)]], device float *vy [[buffer(1)]],
                                    device float *vz [[buffer(2)]],
                                    device const ImmersedFilter *filters [[buffer(3)]],
                                    device const ImmersedFilterState *states [[buffer(4)]],
                                    device const ImmersedIoCell *cells [[buffer(5)]],
                                    device const ImmersedWeightedIndex *refs [[buffer(6)]],
                                    constant ImmersedGridParams &g [[buffer(7)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(g.NumVelocityHistoryCells)) return;
    const ImmersedIoCell cell = cells[tid];
    float sum = 0.f;
    for (int i = cell.Begin; i < cell.End; ++i) {
        const int patch = refs[i].Index;
        sum += refs[i].Weight * ImmersedFilterForcing(filters, states, patch);
    }
    const int field_cells = g.Nx * g.Ny * g.Nz;
    const int component = cell.Ixyz / field_cells;
    const int index = cell.Ixyz - component * field_cells;
    if (component == 0) vx[index] -= .5f * g.Courant * sum;
    else if (component == 1) vy[index] -= .5f * g.Courant * sum;
    else vz[index] -= .5f * g.Courant * sum;
}

kernel void ImmersedVelocityGather(device const float *vx [[buffer(0)]], device const float *vy [[buffer(1)]],
                                   device const float *vz [[buffer(2)]], device const int *begin [[buffer(3)]],
                                   device const ImmersedWeightedIndex *refs [[buffer(4)]],
                                   device float *result [[buffer(5)]],
                                   constant ImmersedGridParams &g [[buffer(6)]],
                                   uint active [[thread_position_in_grid]]) {
    if (active >= uint(g.NumVelocityActive)) return;
    const int cells = g.Nx * g.Ny * g.Nz;
    float sum = 0.f;
    for (int i = begin[active]; i < begin[active + 1]; ++i)
        sum += refs[i].Weight * ImmersedVelocityValue(refs[i].Index, cells, vx, vy, vz);
    result[active] = sum;
}

kernel void ImmersedDense(device const float *input [[buffer(0)]],
                          device const float *inverse [[buffer(1)]], device float *output [[buffer(2)]],
                          constant ImmersedSolveParams &s [[buffer(3)]],
                          uint row [[thread_position_in_grid]]) {
    if (row >= uint(s.Count)) return;
    float sum = 0.f;
    for (int column = 0; column < s.Count; ++column) sum += inverse[column * s.Count + int(row)] * input[column];
    output[row] = sum;
}

kernel void ImmersedVelocityScatter(device float *vx [[buffer(0)]], device float *vy [[buffer(1)]],
                                    device float *vz [[buffer(2)]],
                                    device const ImmersedIoCell *cells [[buffer(3)]],
                                    device const ImmersedWeightedIndex *refs [[buffer(4)]],
                                    device const float *solution [[buffer(5)]],
                                    constant ImmersedGridParams &g [[buffer(6)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(g.NumVelocityCorrectionCells)) return;
    const ImmersedIoCell cell = cells[tid];
    float correction = 0.f;
    for (int i = cell.Begin; i < cell.End; ++i) correction += refs[i].Weight * solution[refs[i].Index];
    const int field_cells = g.Nx * g.Ny * g.Nz;
    const int component = cell.Ixyz / field_cells;
    const int index = cell.Ixyz - component * field_cells;
    if (component == 0) vx[index] -= correction;
    else if (component == 1) vy[index] -= correction;
    else vz[index] -= correction;
}

kernel void ImmersedVelocityAdvance(device const float *vx [[buffer(0)]], device const float *vy [[buffer(1)]],
                                    device const float *vz [[buffer(2)]], device const int *begin [[buffer(3)]],
                                    device const ImmersedWeightedIndex *refs [[buffer(4)]],
                                    device const ImmersedFilter *filters [[buffer(5)]],
                                    device ImmersedFilterState *states [[buffer(6)]],
                                    constant ImmersedGridParams &g [[buffer(7)]],
                                    uint patch [[thread_position_in_grid]]) {
    if (patch >= uint(g.NumPatches)) return;
    const ImmersedFilter f = filters[patch];
    if (f.B0 == 0.f && f.B1 == 0.f && f.B2 == 0.f && f.A1 == 0.f && f.A2 == 0.f) return;
    const int cells = g.Nx * g.Ny * g.Nz;
    float interpolated = 0.f;
    for (int i = begin[patch]; i < begin[patch + 1]; ++i)
        interpolated += refs[i].Weight * ImmersedVelocityValue(refs[i].Index, cells, vx, vy, vz);
    ImmersedFilterAdvance(filters, states, int(patch), interpolated);
}

kernel void ImmersedPressureHistory(device float *p [[buffer(0)]],
                                    device const ImmersedFilter *filters [[buffer(1)]],
                                    device const ImmersedFilterState *states [[buffer(2)]],
                                    device const ImmersedIoCell *cells [[buffer(3)]],
                                    device const ImmersedWeightedIndex *refs [[buffer(4)]],
                                    constant ImmersedGridParams &g [[buffer(5)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(g.NumPressureHistoryCells)) return;
    const ImmersedIoCell cell = cells[tid];
    float sum = 0.f;
    for (int i = cell.Begin; i < cell.End; ++i) {
        const int patch = refs[i].Index;
        sum += refs[i].Weight * ImmersedFilterForcing(filters, states, patch);
    }
    p[cell.Ixyz] -= .5f * g.Courant * sum;
}

kernel void ImmersedPressureGather(device const float *p [[buffer(0)]], device const int *begin [[buffer(1)]],
                                   device const ImmersedWeightedIndex *refs [[buffer(2)]],
                                   device float *result [[buffer(3)]],
                                   constant ImmersedGridParams &g [[buffer(4)]],
                                   uint active [[thread_position_in_grid]]) {
    if (active >= uint(g.NumPressureActive)) return;
    float sum = 0.f;
    for (int i = begin[active]; i < begin[active + 1]; ++i) sum += refs[i].Weight * p[refs[i].Index];
    result[active] = sum;
}

kernel void ImmersedPressureScatter(device float *p [[buffer(0)]],
                                    device const ImmersedIoCell *cells [[buffer(1)]],
                                    device const ImmersedWeightedIndex *refs [[buffer(2)]],
                                    device const float *solution [[buffer(3)]],
                                    constant ImmersedGridParams &g [[buffer(4)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(g.NumPressureCorrectionCells)) return;
    const ImmersedIoCell cell = cells[tid];
    float correction = 0.f;
    for (int i = cell.Begin; i < cell.End; ++i) correction += refs[i].Weight * solution[refs[i].Index];
    p[cell.Ixyz] -= correction;
}

kernel void ImmersedPressureAdvance(device const float *p [[buffer(0)]], device const int *begin [[buffer(1)]],
                                    device const ImmersedWeightedIndex *refs [[buffer(2)]],
                                    device const ImmersedFilter *filters [[buffer(3)]],
                                    device ImmersedFilterState *states [[buffer(4)]],
                                    constant ImmersedGridParams &g [[buffer(5)]],
                                    uint patch [[thread_position_in_grid]]) {
    if (patch >= uint(g.NumPatches)) return;
    const ImmersedFilter f = filters[patch];
    if (f.B0 == 0.f && f.B1 == 0.f && f.B2 == 0.f && f.A1 == 0.f && f.A2 == 0.f) return;
    float interpolated = 0.f;
    for (int i = begin[patch]; i < begin[patch + 1]; ++i) interpolated += refs[i].Weight * p[refs[i].Index];
    ImmersedFilterAdvance(filters, states, int(patch), interpolated);
}

kernel void ImmersedSource(device float *p [[buffer(0)]],
                           device const ImmersedIoCell *cells [[buffer(1)]],
                           device const ImmersedWeightedIndex *refs [[buffer(2)]],
                           device const float *signal [[buffer(3)]],
                           constant ImmersedGridParams &g [[buffer(4)]],
                           constant ImmersedStepParams &s [[buffer(5)]],
                           uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(g.NumSourceCells)) return;
    const ImmersedIoCell cell = cells[tid];
    float value = 0.f;
    for (int i = cell.Begin; i < cell.End; ++i) {
        const int source = refs[i].Index;
        const float average = .5f * (signal[s.Step * g.NumSources + source] +
                                     signal[(s.Step + 1) * g.NumSources + source]);
        value += refs[i].Weight * average;
    }
    p[cell.Ixyz] += g.SourceScale * value;
}

kernel void ImmersedSample(device const float *p [[buffer(0)]],
                           device const int *begin [[buffer(1)]],
                           device const ImmersedWeightedIndex *refs [[buffer(2)]],
                           device float *output [[buffer(3)]],
                           constant ImmersedGridParams &g [[buffer(4)]],
                           constant ImmersedStepParams &s [[buffer(5)]],
                           uint receiver [[thread_position_in_grid]]) {
    if (receiver >= uint(g.NumReceivers)) return;
    float value = 0.f;
    for (int i = begin[receiver]; i < begin[receiver + 1]; ++i) value += refs[i].Weight * p[refs[i].Index];
    output[s.Step * g.NumReceivers + receiver] = value;
}

kernel void ImmersedVelocityStream(device float *vx [[buffer(0)]], device float *vy [[buffer(1)]],
                                   device float *vz [[buffer(2)]], constant ImmersedGridParams &g [[buffer(3)]],
                                   uint3 tid [[thread_position_in_grid]]) {
    const int x = int(tid.x), y = int(tid.y), z = int(tid.z);
    if (x >= g.Nx || y >= g.Ny || z >= g.Nz) return;
    const int cid = ImmersedCid(x, y, z, g);
    vx[cid] *= g.InvRhoDtH;
    vy[cid] *= g.InvRhoDtH;
    vz[cid] *= g.InvRhoDtH;
}

kernel void ImmersedPressureStream(device float *p [[buffer(0)]], constant ImmersedGridParams &g [[buffer(1)]],
                                   uint3 tid [[thread_position_in_grid]]) {
    const int x = int(tid.x), y = int(tid.y), z = int(tid.z);
    if (x >= g.Nx || y >= g.Ny || z >= g.Nz) return;
    const int cid = ImmersedCid(x, y, z, g);
    p[cid] *= g.RhoCcDtH;
}
