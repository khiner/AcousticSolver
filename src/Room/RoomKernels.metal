// GPU kernels for the room-acoustics solver: the second-order scalar wave equation on a
// 7-point Cartesian or 13-point FCC grid with rigid boundary nodes and an Engquist-Majda
// absorbing shell at the outer grid box, after PFFDTD [Hamilton, MIT] and the scheme of
// Bilbao, Hamilton, Botts and Savioja 2016.
//
// Compiled at runtime with RoomParams.h prepended (see MetalContext::RoomPipeline).
//
// A step is a fixed sequence of dispatches, and the order is the scheme rather than an
// implementation detail:
//
//   RoomAir       u0 = A1*u1 - u0 + A2*sum(neighbours) over the interior box, restricted at a
//                 boundary node to the neighbours its adjacency mask keeps, and followed at
//                 the absorbing shell by the Engquist-Majda loss. RoomAirFcc is the same on
//                 the FCC stencil, where a twelve-bit mask is too wide to ride beside every
//                 node and a block rank reaches the packed list instead
//   RoomLossy     the boundary nodes that carry a material: the rigid value just written is
//                 corrected by the wall's frequency-dependent impedance, and each node's LRC
//                 branch state advances
//   RoomIo        receiver corners read u1, the *previous* level, and source corners add into
//                 u0, last, before the level swap
//
// Three passes the reference engine runs on their own are folded into the interior kernel
// here, each being a pure function of the node's position or of one byte beside it:
//
//   - the halo mirrors. The reference writes u1 reflected about the first interior layer into
//     the outer layers (0 <- 2, N-1 <- N-3). Reading is the same thing, and is what
//     RoomAxisSteps returns.
//   - saving the shell's previous level, and the shell update itself. The absorbing update
//     needs the value u0 held before the interior update overwrote it, which is the same value
//     the interior update subtracts, so one kernel keeps it in a register and no array of it
//     is needed. Which nodes are on the shell, and the Q that sets how hard they absorb, come
//     out of the position (see RoomShellQ).
//   - the rigid boundary pass, behind a grid-wide adjacency array that is all ones off the
//     boundary on the Cartesian grid, and behind a block rank into the packed list on the FCC
//     one, where twelve adjacency bits a node are too many to keep grid-wide. See RoomAirFcc.
//
// The FCC scheme steps the face-centred-cubic sublattice of the same Cartesian grid — the
// nodes whose index sum is even — where the twelve nearest neighbours are the face diagonals
// rather than the six axis steps. Half a Cartesian grid's points are idle under that, so the
// grid is stored folded in half along y: unfolded layer iy lands at min(iy, Nyf-1-iy), the
// mirror flips the parity, and the two halves interleave into a dense array with every node
// active. One spare y layer at the top of the folded array stands for the seam, and a step onto
// it lands back on the node itself. A folded y step therefore means +y for a node from the lower
// half and -y for one from the upper half, which is why the scene's adjacency bits are swapped
// in pairs on the way in.
//
// Everything is fp32, and A1 carries the reference's (1 + eps) diagonal shift.

#include <metal_stdlib>
using namespace metal;

// Each stencil's neighbour offsets as coordinate steps, in its adjacency mask's bit order.
constant int3 RoomCartSteps[RoomNumNeighbours] = {
    int3(1, 0, 0), int3(-1, 0, 0), int3(0, 1, 0), int3(0, -1, 0), int3(0, 0, 1), int3(0, 0, -1)};
constant int3 RoomFccSteps[RoomFccNeighbours] = {
    int3(1, 1, 0), int3(-1, -1, 0), int3(0, 1, 1), int3(0, -1, -1), int3(1, 0, 1), int3(-1, 0, -1),
    int3(1, -1, 0), int3(-1, 1, 0), int3(0, 1, -1), int3(0, -1, 1), int3(1, 0, -1), int3(-1, 0, 1)};

// The six axis steps from a node, as index offsets, with a step that would leave the grid
// replaced by its mirror image about the first interior layer. `fold_y` marks the FCC grid,
// whose far y layer is the fold seam and carries a copy of the layer below it, so a +y step
// from the last interior layer lands back on the node itself.
//
// Only nodes of the interior box (1 <= pos < N-1 on every axis) may ask: one axis step from
// there reaches at most the first halo layer, which is what these mirrors cover.
inline void RoomAxisSteps(int3 pos, int3 n, int nz, int nzny, bool fold_y, thread int *step) {
    step[0] = pos.x == n.x - 2 ? -nzny : nzny;
    step[1] = pos.x == 1 ? nzny : -nzny;
    step[2] = pos.y == n.y - 2 ? (fold_y ? 0 : -nz) : nz;
    step[3] = pos.y == 1 ? nz : -nz;
    step[4] = pos.z == n.z - 2 ? -1 : 1;
    step[5] = pos.z == 1 ? 1 : -1;
}

// The FCC stencil's twelve face diagonals as index offsets, in its adjacency mask's bit order.
// Each is a pair of axis steps, so mirroring the axes mirrors the diagonals, and a diagonal
// whose two components both leave the grid reflects in both.
inline void RoomFccOffsets(thread const int *step, thread int *off) {
    off[0] = step[0] + step[2];
    off[1] = step[1] + step[3];
    off[2] = step[2] + step[4];
    off[3] = step[3] + step[5];
    off[4] = step[0] + step[4];
    off[5] = step[1] + step[5];
    off[6] = step[0] + step[3];
    off[7] = step[1] + step[2];
    off[8] = step[2] + step[5];
    off[9] = step[3] + step[4];
    off[10] = step[0] + step[5];
    off[11] = step[1] + step[4];
}

// How many of the three axes a node is extreme on, which is the Q of the Engquist-Majda shell:
// 0 off the shell, 1 at a wall, 2 at an edge, 3 at a corner. The shell is the layer one node
// in from the outer grid box, so nothing about it is stored. On the FCC grid the far y layer
// is the fold seam, and a node is y-extreme exactly when its folded layer is 1 — both of that
// layer's unfolded preimages, 1 and Nyf-2, are extreme, and exactly one of them is on the
// sublattice.
inline int RoomShellQ(int3 pos, int3 n, bool fold_y) {
    return int(pos.x == 1 || pos.x == n.x - 2) + int(pos.y == 1 || (!fold_y && pos.y == n.y - 2)) +
        int(pos.z == 1 || pos.z == n.z - 2);
}

// The absorbing shell's second-order-accurate first-order Engquist-Majda condition, applied to
// the value the interior update just produced against the level it overwrote.
inline float RoomAbsorb(float updated, float previous, float l, int q) {
    if (q == 0) return updated;
    const float lq = l * float(q);
    return (updated + lq * previous) / (1.f + lq);
}

// The whole interior of a step: one pass over the interior box, one thread a node.
//
// A rigid boundary node's update is the interior one with the neighbours its adjacency mask
// drops removed from both the sum and the diagonal — the zero-flux condition for a wall lying
// half a cell outside the node — so both live here behind the grid-wide adjacency array.
//
// The two are not symmetric about the absorbing shell: the interior update runs first and the
// shell then mixes what it produced with the level it overwrote, while a boundary node has the
// shell applied to that level before its own stencil consumes it.
kernel void RoomAir(device float *u0, device const float *u1, device const uchar *adj_grid,
                    constant RoomGridParams &g, uint3 t [[thread_position_in_grid]]) {
    const int iz = int(t.x) + 1, iy = int(t.y) + 1, ix = int(t.z) + 1;
    if (ix >= g.Nx - 1 || iy >= g.Ny - 1 || iz >= g.Nz - 1) return;
    const int3 pos = int3(ix, iy, iz), n = int3(g.Nx, g.Ny, g.Nz);
    const int nzny = g.Nz * g.Ny;
    const int ii = ix * nzny + iy * g.Nz + iz;
    const uint adj = adj_grid[ii];
    const int q = RoomShellQ(pos, n, false);
    const float previous = u0[ii];

    int step[RoomNumNeighbours];
    RoomAxisSteps(pos, n, g.Nz, nzny, false, step);
    if (adj == RoomAirAdj) {
        float partial = g.A1 * u1[ii] - previous;
        for (int bit = 0; bit < RoomNumNeighbours; ++bit) partial += g.A2 * u1[ii + step[bit]];
        u0[ii] = RoomAbsorb(partial, previous, g.L, q);
        return;
    }
    const float b1 = 2.f - g.Sl2 * float(popcount(adj));
    float partial = b1 * u1[ii] - RoomAbsorb(previous, previous, g.L, q);
    for (int bit = 0; bit < RoomNumNeighbours; ++bit) {
        partial += g.A2 * float((adj >> bit) & 1) * u1[ii + step[bit]];
    }
    u0[ii] = partial;
}

// The same on the FCC stencil: twelve face diagonals instead of six axis steps and a diagonal
// of A1 = 2 - 12*Sl2 with A2 = l^2/4, summed in the reference's own order, which is the order
// the goldens carry.
//
// The rigid boundary is folded in here as it is on the Cartesian grid, but it cannot reach its
// adjacency the same way: twelve neighbours do not fit in a byte, so a grid-wide FCC array
// would cost two bytes a node against the Cartesian one's seven eighths of one, and on a grid
// whose levels stream at 658 GB/s that is more traffic than a second pass over the boundary
// nodes costs. Fusing it that way measured 1.069x, slower.
//
// What the fold needs instead is the packed list's row for a node, from the node's position
// alone. RoomBnBlockEntry is that: one occupancy word and one running count a block of 32
// nodes, so the row is the block's count plus the occupied nodes below this one in the block,
// and the whole lookup is a quarter of a byte a node rather than two. A node off the boundary
// pays one broadcast load a block for it and nothing else.
kernel void RoomAirFcc(device float *u0, device const float *u1, device const RoomBnBlockEntry *bn_blocks,
                       device const ushort *adj_bn, constant RoomGridParams &g, uint3 t [[thread_position_in_grid]]) {
    const int iz = int(t.x) + 1, iy = int(t.y) + 1, ix = int(t.z) + 1;
    if (ix >= g.Nx - 1 || iy >= g.Ny - 1 || iz >= g.Nz - 1) return;
    const int3 pos = int3(ix, iy, iz), n = int3(g.Nx, g.Ny, g.Nz);
    const int nzny = g.Nz * g.Ny;
    const int ii = ix * nzny + iy * g.Nz + iz;
    const int q = RoomShellQ(pos, n, true);
    const float previous = u0[ii];

    const RoomBnBlockEntry block = bn_blocks[ii / RoomBnBlock];
    const uint slot = uint(ii % RoomBnBlock);

    int step[RoomNumNeighbours], off[RoomFccNeighbours];
    RoomAxisSteps(pos, n, g.Nz, nzny, true, step);
    RoomFccOffsets(step, off);
    if (((block.Occupied >> slot) & 1u) == 0) {
        float partial = g.A1 * u1[ii] - previous;
        for (int bit = 0; bit < RoomFccNeighbours; ++bit) partial += g.A2 * u1[ii + off[bit]];
        u0[ii] = RoomAbsorb(partial, previous, g.L, q);
        return;
    }
    // A boundary node's update is the interior one with the neighbours its mask drops removed
    // from both the sum and the diagonal, over the level the shell has already absorbed.
    const uint adj = adj_bn[block.Rank + popcount(block.Occupied & ((1u << slot) - 1u))];
    const float b1 = 2.f - g.Sl2 * float(popcount(adj));
    float partial = b1 * u1[ii] - RoomAbsorb(previous, previous, g.L, q);
    for (int bit = 0; bit < RoomFccNeighbours; ++bit) {
        partial += g.A2 * float((adj >> bit) & 1) * u1[ii + off[bit]];
    }
    u0[ii] = partial;
}

// The frequency-dependent boundary's traffic with none of its work, and the reason it needs
// its own ceiling rather than being read against RoomStream's: this pass is a gather and a
// scatter over a scattered subset of nodes, across eight streams instead of two, so a dense
// two-level stream is not a roof it could ever reach. What it moves a lossy node is the node's
// own scalars, a read-modify-write of u0 at a grid index, and the LRC state of every branch
// its material carries — which is 176 of those bytes at the eleven branches every scene here
// uses. Same nodes, same buffers, same trip count, same threadgroup shape.
//
// Like RoomStream it destroys what it touches. See RoomGpu::LossyRoofline.
kernel void RoomLossyStream(device float *u0, device float *u0b, device const float *u2b,
                            device float *vh1, device float *gh1, device const int *bnl_ixyz,
                            device const float *ssaf_bnl, device const char *mat_bnl,
                            device const int *mat_mb, constant RoomGridParams &g,
                            uint nb [[thread_position_in_grid]]) {
    if (int(nb) >= g.Nbl) return;
    const int k = int(mat_bnl[nb]);
    const int ii = bnl_ixyz[nb];
    float acc = ssaf_bnl[nb] + u2b[nb] + u0[ii];
    const int mb = mat_mb[k];
    // Two loops over the branches, loading into registers and storing back, because that is
    // the shape the real pass has — one loop doing both would serialise each load against the
    // store before it and measure a dependency the pass does not carry.
    float vint[RoomMaxBranches], gint[RoomMaxBranches];
    for (int m = 0; m < mb; ++m) {
        const int nbm = m * g.Nbl + int(nb);
        vint[m] = vh1[nbm];
        gint[m] = gh1[nbm];
        acc += vint[m] + gint[m];
    }
    for (int m = 0; m < mb; ++m) {
        const int nbm = m * g.Nbl + int(nb);
        vh1[nbm] = acc + vint[m];
        gh1[nbm] = acc + gint[m];
    }
    u0[ii] = acc;
    u0b[nb] = acc;
}

// The interior update's traffic with none of its work: one read of the previous level and a
// read-modify-write of the level being computed, the twelve bytes a node-step no form of the
// scheme avoids. Over the same box with the same threadgroup shape, so it is this grid's own
// bandwidth ceiling. See RoomGpu::Roofline.
kernel void RoomStream(device float *u0, device const float *u1, constant RoomGridParams &g,
                       uint3 t [[thread_position_in_grid]]) {
    const int iz = int(t.x) + 1, iy = int(t.y) + 1, ix = int(t.z) + 1;
    if (ix >= g.Nx - 1 || iy >= g.Ny - 1 || iz >= g.Nz - 1) return;
    const int ii = ix * g.Nz * g.Ny + iy * g.Nz + iz;
    u0[ii] = g.A1 * u1[ii] - u0[ii];
}

// Frequency-dependent (lossy) boundary nodes. A lossy node is also a rigid node, so the rigid
// update has already left the zero-flux value in u0 — this pass replaces it with the value the
// wall's impedance admits, and advances the wall's own state.
//
// The wall is a parallel set of up to twelve LRC branches, each an admittance 1/(D s + E + F/s)
// discretised by the trapezoid rule. Branch m carries a velocity vh and its running integral
// gh at the node, and the node's update is
//
//   u^(n+1) = u_rigid - l * S * sum_m mu_t vh_m
//
// with S the staircase-corrected surface area and mu_t the two-point time average, and the
// branch step is implicit in u^(n+1) — which is why the sum is split: the part proportional to
// the new value is folded into the (1 + lo2*S*beta) division, and what is left of it depends
// only on the state from the previous step. beta is the material's summed b, so the division is
// one scalar however many branches the material has and the node solve stays node-local.
//
// u2b is the node's own value two steps back, held in a three-deep ring beside the two
// interior levels, because the grid at that level has already been overwritten. See
// RoomGpu::RunSteps for the rotation.
kernel void RoomLossy(device float *u0, device float *u0b, device const float *u2b,
                      device float *vh1, device float *gh1, device const int *bnl_ixyz,
                      device const float *ssaf_bnl, device const char *mat_bnl,
                      device const int *mat_mb, device const float *mat_beta,
                      device const RoomMatQuad *mat_quads, constant RoomGridParams &g,
                      uint nb [[thread_position_in_grid]]) {
    if (int(nb) >= g.Nbl) return;
    const int k = int(mat_bnl[nb]);
    const float ssaf = ssaf_bnl[nb];
    const float lo2_kbg = g.Lo2 * ssaf * mat_beta[k];
    const float fac = 2.f * g.Lo2 * ssaf / (1.f + lo2_kbg);

    const int ii = bnl_ixyz[nb];
    const float u2bint = u2b[nb];
    float u0bint = (u0[ii] + lo2_kbg * u2bint) / (1.f + lo2_kbg);

    const int mb = mat_mb[k];
    float vh1int[RoomMaxBranches], gh1int[RoomMaxBranches];
    for (int m = 0; m < mb; ++m) {
        const int nbm = m * g.Nbl + int(nb);
        const RoomMatQuad q = mat_quads[k * RoomMaxBranches + m];
        vh1int[m] = vh1[nbm];
        gh1int[m] = gh1[nbm];
        u0bint -= fac * (2.f * q.BDh * vh1int[m] - q.BFh * gh1int[m]);
    }

    const float du = u0bint - u2bint;
    for (int m = 0; m < mb; ++m) {
        const int nbm = m * g.Nbl + int(nb);
        const RoomMatQuad q = mat_quads[k * RoomMaxBranches + m];
        const float vh0m = q.B * du + q.Bd * vh1int[m] - 2.f * q.BFh * gh1int[m];
        gh1[nbm] = gh1int[m] + (vh0m + vh1int[m]) / 2.f;
        vh1[nbm] = vh0m;
    }
    u0[ii] = u0bint;
    u0b[nb] = u0bint; // the ring, so that two steps from now this is u2b
}

// Receivers read the previous level and sources add into the level just computed, last. Two
// unrelated scatters in one dispatch, because at this size a step's cost is dispatches and not
// work: the first Nr threads sample and the rest inject. Nothing this step writes u1, and
// source corners are distinct nodes, so neither scatter can race.
kernel void RoomIo(device float *u0, device float *out, device const float *u1, device const int *out_ixyz,
                   device const float *in_sigs, device const int *in_ixyz, constant RoomGridParams &g,
                   constant RoomStepParams &s, uint i [[thread_position_in_grid]]) {
    if (int(i) < g.Nr) {
        out[s.Step * g.Nr + int(i)] = u1[out_ixyz[i]];
        return;
    }
    const int n = int(i) - g.Nr;
    if (n >= g.Ns) return;
    u0[in_ixyz[n]] += in_sigs[n * g.Nt + s.Step];
}

// Discrete energy of the lossless scheme over a closed box of nodes, one partial a node for
// the host to sum in double.
//
// For u^(n+1) = 2u^n - u^(n-1) + M u^n with M = A2 Adj - Sl2 diag(K) symmetric, the conserved
// quantity is ||u^(n+1) - u^n||^2 - <u^(n+1), M u^n>, and the second term splits over edges as
// A2 sum_edges (du^(n+1))(du^n) plus (Sl2 - A2) on the diagonal. The edge form is used here
// because differences across a neighbouring pair stay small where the raw products do not,
// which is what keeps the fp32 partials from losing the drift being measured. The
// coefficients are the update's own, so the quantity is conserved by the arithmetic that
// actually runs and not only by the scheme it approximates.
//
// The box must be closed under the scheme's adjacency — every edge the update uses inside it
// stays inside it — or the sum is not conserved and the rung means nothing.
//
// The FCC stencil's twelve offsets come in plus/minus pairs like the Cartesian six, so the
// update operator is symmetric either way and the two stencils share the whole expression.
inline float RoomEnergyPartial(device const float *unew, device const float *uold, int3 pos,
                               constant RoomGridParams &g, constant RoomEnergyParams &e,
                               constant int3 *steps, int count) {
    const int nzny = g.Nz * g.Ny;
    const int ii = pos.x * nzny + pos.y * g.Nz + pos.z;
    const float a = unew[ii], b = uold[ii];
    const int3 box_lo = int3(e.X0, e.Y0, e.Z0), box_hi = int3(e.X1, e.Y1, e.Z1);
    float grad = 0.f;
    int k = 0;
    for (int bit = 0; bit < count; ++bit) {
        const int3 step = steps[bit], q = pos + step;
        if (any(q < box_lo) || any(q > box_hi)) continue;
        const int jj = ii + step.x * nzny + step.y * g.Nz + step.z;
        grad += (a - unew[jj]) * (b - uold[jj]);
        ++k;
    }
    return (a - b) * (a - b) + e.A2 * 0.5f * grad + e.Diag * float(k) * a * b;
}

kernel void RoomEnergy(device float *out, device const float *unew, device const float *uold,
                       constant RoomGridParams &g, constant RoomEnergyParams &e, uint i [[thread_position_in_grid]]) {
    const int ny = e.Y1 - e.Y0 + 1, nz = e.Z1 - e.Z0 + 1;
    if (int(i) >= (e.X1 - e.X0 + 1) * ny * nz) return;
    const int3 pos = int3(e.X0 + int(i) / (nz * ny), e.Y0 + (int(i) / nz) % ny, e.Z0 + int(i) % nz);
    out[i] = RoomEnergyPartial(unew, uold, pos, g, e, RoomCartSteps, RoomNumNeighbours);
}

kernel void RoomEnergyFcc(device float *out, device const float *unew, device const float *uold,
                          constant RoomGridParams &g, constant RoomEnergyParams &e, uint i [[thread_position_in_grid]]) {
    const int ny = e.Y1 - e.Y0 + 1, nz = e.Z1 - e.Z0 + 1;
    if (int(i) >= (e.X1 - e.X0 + 1) * ny * nz) return;
    const int3 pos = int3(e.X0 + int(i) / (nz * ny), e.Y0 + (int(i) / nz) % ny, e.Z0 + int(i) % nz);
    out[i] = RoomEnergyPartial(unew, uold, pos, g, e, RoomFccSteps, RoomFccNeighbours);
}

// --- The optimised implicit scheme of Smits & Bilbao 2025 -------------------------------
//
// An evaluation, not a stepper the solver uses: free space on a periodic box, no boundaries
// of any kind. See RoomImplicitParams for the scheme and for what the coefficients already
// have folded in.
//
// A step is one RoomImplicitRhs followed by a fixed number of RoomImplicitJacobi sweeps —
// the paper's own solver, chosen because the iteration count that reaches single precision
// is known in advance from the implicit operator's diagonal dominance. Both kernels move the
// same three streams a node, one read of each input grid and one write, so a step of P
// sweeps costs 3(P + 1) streams against an explicit step's 3.

// The 27-point neighbourhood of a node on a periodic box, as the three sums the scheme's
// Laplacians are built from: over the 6 face neighbours, the 12 edge neighbours and the 8
// corner neighbours. The periodic wrap is what keeps a plane wave an exact discrete
// eigenvector, which is what makes the dispersion measurable to machine precision.
static inline void RoomImplicitSums(device const float *u, int ix, int iy, int iz,
                                    constant RoomImplicitParams &p, thread float3 &s) {
    const int nzny = p.Nz * p.Ny;
    const int xm = (ix == 0 ? p.Nx - 1 : ix - 1) * nzny, x0 = ix * nzny, xp = (ix == p.Nx - 1 ? 0 : ix + 1) * nzny;
    const int ym = (iy == 0 ? p.Ny - 1 : iy - 1) * p.Nz, y0 = iy * p.Nz, yp = (iy == p.Ny - 1 ? 0 : iy + 1) * p.Nz;
    const int zm = (iz == 0 ? p.Nz - 1 : iz - 1), z0 = iz, zp = (iz == p.Nz - 1 ? 0 : iz + 1);

    s.x = u[xm + y0 + z0] + u[xp + y0 + z0] + u[x0 + ym + z0] + u[x0 + yp + z0] + u[x0 + y0 + zm] + u[x0 + y0 + zp];
    s.y = u[xm + ym + z0] + u[xm + yp + z0] + u[xp + ym + z0] + u[xp + yp + z0] +
        u[xm + y0 + zm] + u[xm + y0 + zp] + u[xp + y0 + zm] + u[xp + y0 + zp] +
        u[x0 + ym + zm] + u[x0 + ym + zp] + u[x0 + yp + zm] + u[x0 + yp + zp];
    s.z = u[xm + ym + zm] + u[xm + ym + zp] + u[xm + yp + zm] + u[xm + yp + zp] +
        u[xp + ym + zm] + u[xp + ym + zp] + u[xp + yp + zm] + u[xp + yp + zp];
}

// The Jacobi iteration's constant vector, from the two levels behind it.
kernel void RoomImplicitRhs(device float *bp, device const float *un, device const float *up,
                            constant RoomImplicitParams &p, uint3 t [[thread_position_in_grid]]) {
    const int iz = int(t.x), iy = int(t.y), ix = int(t.z);
    if (ix >= p.Nx || iy >= p.Ny || iz >= p.Nz) return;
    const int ii = ix * p.Nz * p.Ny + iy * p.Nz + iz;
    float3 sn, sp;
    RoomImplicitSums(un, ix, iy, iz, p, sn);
    RoomImplicitSums(up, ix, iy, iz, p, sp);
    bp[ii] = p.R1 * sn.x + p.R2 * sn.y + p.R3 * sn.z + p.Rc * un[ii] -
        (p.Q1 * sp.x + p.Q2 * sp.y + p.Q3 * sp.z + up[ii]);
}

// One Jacobi sweep, whose traffic is the same whatever the iteration index.
kernel void RoomImplicitJacobi(device float *xn, device const float *x, device const float *bp,
                               constant RoomImplicitParams &p, uint3 t [[thread_position_in_grid]]) {
    const int iz = int(t.x), iy = int(t.y), ix = int(t.z);
    if (ix >= p.Nx || iy >= p.Ny || iz >= p.Nz) return;
    const int ii = ix * p.Nz * p.Ny + iy * p.Nz + iz;
    float3 s;
    RoomImplicitSums(x, ix, iy, iz, p, s);
    xn[ii] = bp[ii] - (p.Q1 * s.x + p.Q2 * s.y + p.Q3 * s.z);
}

// The same three sums over a rigid box rather than a periodic one. A step that would leave
// the grid comes back to the node one step inside it, which is the Neumann image of a wall
// lying on the boundary node — exact for an axis-aligned box, and the same device the
// explicit schemes use for their halo. The box a grid of N nodes spans is therefore
// (N - 1) * h on that axis, and its discrete modes are cos(m*pi*i/(N-1)).
//
// This is not the paper's boundary condition. The paper carries a real admittance gamma and
// names frequency-dependent walls as future work; gamma = 0 is a rigid wall, and for an
// axis-aligned box the image above is what a rigid wall means. Nothing here reaches a wall
// that absorbs. See RoomImplicitBox in RoomImplicit.cpp for what that costs the scheme.
static inline void RoomImplicitSumsBox(device const float *u, int ix, int iy, int iz,
                                       constant RoomImplicitParams &p, thread float3 &s) {
    const int nzny = p.Nz * p.Ny;
    const int xm = (ix == 0 ? 1 : ix - 1) * nzny, x0 = ix * nzny, xp = (ix == p.Nx - 1 ? p.Nx - 2 : ix + 1) * nzny;
    const int ym = (iy == 0 ? 1 : iy - 1) * p.Nz, y0 = iy * p.Nz, yp = (iy == p.Ny - 1 ? p.Ny - 2 : iy + 1) * p.Nz;
    const int zm = (iz == 0 ? 1 : iz - 1), z0 = iz, zp = (iz == p.Nz - 1 ? p.Nz - 2 : iz + 1);

    s.x = u[xm + y0 + z0] + u[xp + y0 + z0] + u[x0 + ym + z0] + u[x0 + yp + z0] + u[x0 + y0 + zm] + u[x0 + y0 + zp];
    s.y = u[xm + ym + z0] + u[xm + yp + z0] + u[xp + ym + z0] + u[xp + yp + z0] +
        u[xm + y0 + zm] + u[xm + y0 + zp] + u[xp + y0 + zm] + u[xp + y0 + zp] +
        u[x0 + ym + zm] + u[x0 + ym + zp] + u[x0 + yp + zm] + u[x0 + yp + zp];
    s.z = u[xm + ym + zm] + u[xm + ym + zp] + u[xm + yp + zm] + u[xm + yp + zp] +
        u[xp + ym + zm] + u[xp + ym + zp] + u[xp + yp + zm] + u[xp + yp + zp];
}

kernel void RoomImplicitRhsBox(device float *bp, device const float *un, device const float *up,
                               constant RoomImplicitParams &p, uint3 t [[thread_position_in_grid]]) {
    const int iz = int(t.x), iy = int(t.y), ix = int(t.z);
    if (ix >= p.Nx || iy >= p.Ny || iz >= p.Nz) return;
    const int ii = ix * p.Nz * p.Ny + iy * p.Nz + iz;
    float3 sn, sp;
    RoomImplicitSumsBox(un, ix, iy, iz, p, sn);
    RoomImplicitSumsBox(up, ix, iy, iz, p, sp);
    bp[ii] = p.R1 * sn.x + p.R2 * sn.y + p.R3 * sn.z + p.Rc * un[ii] -
        (p.Q1 * sp.x + p.Q2 * sp.y + p.Q3 * sp.z + up[ii]);
}

kernel void RoomImplicitJacobiBox(device float *xn, device const float *x, device const float *bp,
                                  constant RoomImplicitParams &p, uint3 t [[thread_position_in_grid]]) {
    const int iz = int(t.x), iy = int(t.y), ix = int(t.z);
    if (ix >= p.Nx || iy >= p.Ny || iz >= p.Nz) return;
    const int ii = ix * p.Nz * p.Ny + iy * p.Nz + iz;
    float3 s;
    RoomImplicitSumsBox(x, ix, iy, iz, p, s);
    xn[ii] = bp[ii] - (p.Q1 * s.x + p.Q2 * s.y + p.Q3 * s.z);
}

// Source and receiver for the implicit stepper. Receivers read the level behind, the way
// RoomIo does, and the source adds into the level just solved before the rotation.
kernel void RoomImplicitIo(device float *un, device float *out, device const float *up,
                           device const int *out_ixyz, device const float *in_sigs,
                           device const int *in_ixyz, constant RoomImplicitParams &p,
                           constant RoomStepParams &st, uint i [[thread_position_in_grid]]) {
    if (int(i) < p.Nr) {
        out[int(i) * p.Nt + st.Step] = up[out_ixyz[i]];
        return;
    }
    const int k = int(i) - p.Nr;
    if (k >= p.Ns) return;
    un[in_ixyz[k]] += in_sigs[k * p.Nt + st.Step];
}

// The traffic a sweep cannot avoid, with none of its work: the twelve bytes of two reads and
// a write both kernels above move a node, over the same buffers and the same box.
kernel void RoomImplicitStream(device float *xn, device const float *x, device const float *bp,
                               constant RoomImplicitParams &p, uint3 t [[thread_position_in_grid]]) {
    const int iz = int(t.x), iy = int(t.y), ix = int(t.z);
    if (ix >= p.Nx || iy >= p.Ny || iz >= p.Nz) return;
    const int ii = ix * p.Nz * p.Ny + iy * p.Nz + iz;
    xn[ii] = bp[ii] - p.Q1 * x[ii];
}
