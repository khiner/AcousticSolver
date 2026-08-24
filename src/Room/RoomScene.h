#pragma once

// Room-acoustics scene: the discrete problem the solver steps, and the loader for the
// Scenes/Room<Name>/ layout that script/ConvertRoomScene writes.
//
// A scene is the PFFDTD voxelizer's own output, converted rather than re-derived, so this
// solver and the reference engines solve the identical discrete problem. Everything geometric
// is already baked in: which nodes are boundary nodes, which of their neighbours survive,
// where the source and receiver trilinear corners land and what each corner weighs.

#include "RoomParams.h"

#include <cstdint>
#include <string>
#include <vector>

struct RoomScene {
    // Which stencil steps this grid: the 7-point Cartesian one over the whole grid, or the
    // 13-point FCC one over the face-centred-cubic sublattice. An FCC grid is stored folded in
    // half along y, with Nyf = 2*(Ny - 1) the extent it stands for. See RoomKernels.metal.
    bool Fcc{false};
    int Neighbours() const { return Fcc ? int(RoomFccNeighbours) : int(RoomNumNeighbours); }
    int NyUnfolded() const { return Fcc ? 2 * (Ny - 1) : Ny; }

    // Node grid. The node at (ix, iy, iz) is at index ix*Nz*Ny + iy*Nz + iz.
    int Nx{0}, Ny{0}, Nz{0};
    double H{0}; // grid spacing, m
    double C{0}; // speed of sound, m/s
    double Ts{0}; // time step, s
    double Srate{0}; // 1 / Ts
    double L{0}, L2{0}; // Courant number and its square, as the reference recorded them

    int Nt{0}; // steps

    // Boundary nodes: node index, the packed adjacency mask (six bits Cartesian, twelve FCC —
    // see RoomParams.h), the material index (-1 rigid) and the staircase-corrected surface
    // area.
    std::vector<int> BnIxyz;
    std::vector<uint16_t> AdjBn;
    std::vector<int8_t> MatBn;
    std::vector<double> SafBn;
    int NumLossy{0};

    // Materials, indexed by MatBn. Each is a parallel set of LRC branches given as (D, E, F)
    // triples — the coefficients of the branch impedance D s + E + F/s, normalised to the
    // characteristic impedance of air. MatDef holds the triples of every material end to end,
    // so material k's rows start at three times the branch counts before it.
    std::vector<int> MatBranches;
    std::vector<double> MatDef;

    int NumMaterials() const { return int(MatBranches.size()); }

    // Source corners: node indices and one signal row each, Ns by Nt.
    std::vector<int> InIxyz;
    std::vector<double> InSigs;

    // Receiver corners: node indices and trilinear weights, eight corners a receiver in
    // receiver order.
    std::vector<int> OutIxyz;
    std::vector<double> OutAlpha;
    int NumReceivers{0};
    int CornersPerReceiver{8};

    std::string Output; // output stem, from the config

    int NumNodes() const { return Nx * Ny * Nz; }
    int NumSources() const { return int(InIxyz.size()); }
    int NumOutputs() const { return int(OutIxyz.size()); }
};

// Loads Scenes/Room<Name>/config.json and its flat sidecars.
RoomScene LoadRoomScene(const std::string &config_file);

// Steps `scene` for `n_steps` and returns one float64 row a receiver, the trilinear corners
// recombined and the source normalisation undone — the same signal the reference engines
// write. Normalising the source rewrites `scene`'s signal rows in place.
std::vector<double> RenderRoomScene(RoomScene &, int n_steps);

// `seconds`, when positive, renders only that much of the scene rather than its full step
// count.
void RunRoomScene(const std::string &config_file, double seconds = 0.);
