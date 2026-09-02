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

// One block of the implicit scheme's boundary lookup, the same rank construction as
// RoomBnBlockEntry with one word more. Only 7-9% of a real room's nodes are wall, but the
// masked passes read six bytes a node grid-wide for what the wall alone needs — a uint of
// kept legs and a ushort into the term table. Here those ride in wall order and a node
// reaches its row by rank, and what stays grid-wide is three words a block of 32: which of
// the block's nodes are wall, which are outside the room, and how many wall rows precede the
// block. Outside is its own word rather than a term-table entry because it is what the other
// two classes are told apart against: an outside node is held at rest and reads nothing, an
// air node keeps all 26 legs and needs no mask at all. See RoomImplicitRhsWall.
struct RoomImplicitBlockEntry {
    unsigned int Wall, Outside, Rank;
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
    int Nw; // wall nodes carrying LRC branches, and the row stride of their branch state
    float R1, R2, R3, Rc;
    float Q1, Q2, Q3;
};

// What a boundary node needs beyond the uniform coefficients above, under the drop-ghost
// boundary of Smits & Bilbao 2025 Sec IV.
//
// Dropping a ghost neighbour costs the node both that neighbour's term and that neighbour's
// share of the diagonal, and the grid's zero halo already supplies the first: a term the sum
// should not have is a read of a node that holds zero. What it cannot supply is the second.
// Writing K_r for the neighbours of class r the node keeps, the implicit operator's diagonal
// at the node is d = 1 - (Q1 K_1 + Q2 K_2 + Q3 K_3)*d0 rather than the interior's d0, and
// every coefficient above was divided through by d0.
//
// So a boundary node's step is the uniform one plus three corrections. After the right-hand
// side: bp += Cn u^n + Cp u^(n-1), the two diagonal terms' error. After each Jacobi sweep:
// x *= Fd = d0/d, which is all a sweep's diagonal needs because the sweep's whole expression
// is one division by it.
//
// Every node's three numbers are a function of its kept counts and its G alone, and a room
// has far fewer distinct triples than nodes, so a wall node carries a ushort into a table of
// these — packed in wall order and reached by rank through RoomImplicitBlockEntry, so the
// corrections ride inside the uniform passes rather than in scattered passes of their own,
// which for a sweep that runs P times a step is what keeps them affordable. Ordinary air is
// {0, 0, 1} and a node outside the room is {0, 0, 0}, but neither reaches the table: both are
// classes of the block words, and the passes special-case them — air takes the uniform step
// and outside is held at the rest the halo's zeros rely on.
struct RoomImplicitTerm {
    float Cn, Cp, Fd;
};

// A wall node whose material is a set of parallel LRC branches rather than a real admittance.
//
// Bilbao & Hamilton 2017's appendix gives the recursion, and against the implicit scheme it
// comes to two things and no more. The branches' summed discrete admittance is one nonnegative
// scalar on the *diagonal*, which is where RoomImplicitTerm's Fd and Cp already put a real
// gamma — so nothing off-diagonal changes and the Jacobi sweeps never see the wall. And the
// branch history is a term on the right-hand side, which is all the memory the wall has.
//
// That split is what makes frequency-dependent walls affordable here: the sweeps run P times a
// step and stay uniform, while the two passes that know about the wall run once each, over the
// wall alone. It is the same shape the explicit path's RoomLossy already pays.
struct RoomImplicitWall {
    int Ixyz; // the node on the grid
    int Mat; // the material it carries
    float Psi; // lambda * beta / d0, what the branch history is scaled by on the right-hand side
};

// The closed node box the discrete energy sums over (inclusive bounds), and the coefficients
// of the energy expression. See RoomEnergy in RoomKernels.metal.
struct RoomEnergyParams {
    int X0, X1, Y0, Y1, Z0, Z1;
    float A2; // the off-diagonal coefficient the update uses
    float Diag; // the per-neighbour diagonal shift A1 carries beyond the plain Laplacian's -6 A2
};

#endif // #ifndef ROOM_PARAMS_H
