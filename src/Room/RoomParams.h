// Room solver: parameter blocks shared between host (C++) and device (MSL).
//
// This header is compiled both as C++ and, prepended to RoomKernels.metal, as MSL — keep it
// to plain structs of 4-byte int/float members so the layouts agree, and to fundamental
// types (it lands ahead of <metal_stdlib>, so nothing from the standard library exists yet).

#ifndef ROOM_PARAMS_H // Include guard rather than pragma once: textually prepended to RoomKernels.metal
#define ROOM_PARAMS_H

// Bit order of the Cartesian adjacency mask, which is the reference's adj_bn order:
// +x, -x, +y, -y, +z, -z.
enum : int { RoomNumNeighbours = 6 };

// Bit order of the FCC adjacency mask, the reference's again: (+x+y), (-x-y), (+y+z), (-y-z),
// (+x+z), (-x-z), (+x-y), (-x+y), (+y-z), (-y+z), (+x-z), (-x+z). Twelve bits, so an FCC mask
// needs a short where a Cartesian one fits in a byte.
enum : int { RoomFccNeighbours = 12 };

// What the Cartesian scheme's grid-wide adjacency array carries off the boundary. A boundary
// node sets at most six of the eight bits, so all-ones is free to mean ordinary air.
enum : int { RoomAirAdj = 0xFF };

// Nodes a block of the FCC scheme's boundary lookup covers. A power of two, so a node's block
// and its bit in that block are a shift and a mask.
enum : int { RoomBnBlock = 32 };

// One block of that lookup. Twelve adjacency bits do not fit in a byte, so the FCC scheme
// cannot afford the Cartesian one's grid-wide array — it keeps the masks packed in boundary
// order and reaches a node's row by rank: Occupied says which of the block's nodes are
// boundary nodes, Rank counts the ones ahead of the block, and the bits below a node in
// Occupied count the rest. Two words a block is 0.25 bytes a node against the two bytes a
// grid-wide array of shorts would cost. See RoomAirFcc.
struct RoomBnBlockEntry {
    unsigned int Occupied, Rank;
};

// Branches a material's impedance may carry (the reference's MMb). Per-node LRC state is
// allocated at this width whatever a scene uses.
enum : int { RoomMaxBranches = 12 };

// One branch of a material's discrete admittance. See RoomMaterialCoefficients.
struct RoomMatQuad {
    float B, Bd, BDh, BFh;
};

// The node at (ix, iy, iz) lives at ix*Nz*Ny + iy*Nz + iz, so z is the contiguous axis.
struct RoomGridParams {
    int Nx, Ny, Nz;
    int Nbl; // boundary nodes carrying a material: the frequency-dependent ones
    int Ns; // source corners
    int Nr; // receiver corners
    int Nt; // steps in the run: the row stride of the source and receiver arrays
    // A1 = 2 - Sl2*6, A2 = l^2, Sl2 = (1 + eps)*l^2 with eps the fp32 diagonal shift,
    // L = l (the Courant number, for the ABC shell), Lo2 = l/2 (the impedance boundary's
    // surface-to-volume factor). Computed in double on the host and narrowed here.
    float A1, A2, Sl2, L, Lo2;
};

// The index of the step being computed, which selects the source sample and the receiver row.
struct RoomStepParams {
    int Step;
};

// The optimised implicit scheme of Smits & Bilbao 2025, in the form its two kernels read it.
// The scheme is
//
//   (1 + Lim) u^(n+1) = (2 + 2 Lim + l^2 Lex) u^n - (1 + Lim) u^(n-1)
//
// with Lim and Lex each a weighted sum of the paper's three compact Laplacians — the 7-point
// face one, the 13-point edge one and the 9-point corner one — so both are 27-point stencils
// over the same neighbourhood. Writing S1, S2, S3 for the sums over a node's 6 face, 12 edge
// and 8 corner neighbours, those Laplacians are S1 - 6u, S2/4 - 3u and S3/4 - 2u.
//
// Everything here is already divided through by the diagonal of (1 + Lim), which is what both
// kernels want: RoomImplicitRhs writes the Jacobi iteration's constant vector
//
//   b' = R1 S1(u^n) + R2 S2(u^n) + R3 S3(u^n) + Rc u^n - Q1 S1(u^(n-1)) - Q2 S2(u^(n-1)) - Q3 S3(u^(n-1)) - u^(n-1)
//
// and RoomImplicitJacobi sweeps x <- b' - Q1 S1(x) - Q2 S2(x) - Q3 S3(x). The paper's optimum
// puts the face weight of Lim at zero, so Q1 is normally zero and the sweep touches 20
// neighbours, but nothing here assumes it.
struct RoomImplicitParams {
    int Nx, Ny, Nz;
    int Nr, Ns, Nt; // receiver corners, source corners, and the row stride of both arrays
    float R1, R2, R3, Rc;
    float Q1, Q2, Q3;
};

// The closed node box the discrete energy sums over (inclusive bounds), and the coefficients
// of the energy expression. See RoomEnergy in RoomKernels.metal.
struct RoomEnergyParams {
    int X0, X1, Y0, Y1, Z0, Z1;
    float A2; // the off-diagonal coefficient the update uses
    float Diag; // the per-neighbour diagonal shift A1 carries beyond the plain Laplacian's -6 A2
};

#endif // #ifndef ROOM_PARAMS_H
