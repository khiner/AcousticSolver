#pragma once

// Grid geometry and the geometry-dependent tables of the SonicRadiation solver
// [Jin et al. 2025]: CQM weight rows for every band cell, the Laplacian set-difference
// bookkeeping (Eq. 12), and the per-element far-field interpolation stencils with their
// re-basing corrections (Eqs. 19-20).
//
// Building these is CPU work and happens off the step loop — once for static geometry,
// once per geometry epoch otherwise. The stepper consumes them read-only.

#include "Tdbem.h"

#include <chrono>
#include <functional>
#include <utility>

// Phase timing for one table build, printed when ACOUSTIC_RAD_BUILD_TIMES is set. An
// epoch's tables are built on a worker, so this is thread-local and reported by the thread
// that ran the build rather than accumulated into the shared phase profiler, whose map is
// not safe to share.
struct RadBuildTimes {
    bool On{false};
    double Last{0.};
    std::vector<std::pair<const char *, double>> Phases;

    void Restart();
    void Mark(const char *phase);
    void Report() const;
};
// This thread's timer. BuildRadiation restarts it and reports at the end.
RadBuildTimes &BuildTimes();

// Seconds on a monotonic clock, for the phase and step timings the solver prints.
inline double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct RadiationGrid {
    // ----- configured -----
    int Nx{46}, Ny{46}, Nz{46}; // total cells per axis, PML shell included
    double H{0.7 / 30.}; // cell size
    Eigen::Vector3d Origin{Eigen::Vector3d::Zero()}; // center of cell (0, 0, 0)
    double C{343.}, Rho{1.2};
    double CflFraction{1.}; // of the scalar scheme's 1/sqrt(3) limit
    // Chebyshev cell radii of the cell and element near sets. The paper's 3^3/"4^3"
    // neighborhoods are R1=1, R2=2, but that injection radius leaves the boundary-grid loop
    // unstable and floors grid accuracy near 26 dB. See VALIDATION.
    int R1{2}, R2{3};
    int PmlWidth{8};
    // Air damping, the reference solver's per-scene constant. Nonzero values also damp the
    // boundary-grid feedback loop, whose gain otherwise sits slightly above one.
    double Damping{0.};
    // Averages the interpolated far-field feedback over two steps ((1 + z^-1) / 2), nulling
    // the grid-Nyquist mode that loop amplifies. Only the far part of the Dirichlet update
    // sees the filter; the output paths are untouched.
    bool FilterFeedback{true};

    // ----- derived by Finalize() -----
    double Tau{0.};
    // Quadratic-ramp PML weights, indexed by distance to the nearest domain face.
    std::vector<double> PmlNv, PmlDv, PmlNp, PmlDp;

    void Finalize();

    // `res` interior cells spanning `domain` meters per axis, centered on the origin,
    // wrapped in the PML shell. Finalized.
    static RadiationGrid Cubic(int res, double domain, int pml_width = 8);

    size_t NumCells() const { return size_t(Nx) * Ny * Nz; }
    int Cid(int i, int j, int k) const { return i + Nx * (j + Ny * k); }
    int Cid(const Eigen::Vector3i &c) const { return Cid(c[0], c[1], c[2]); }
    Eigen::Vector3i CellCoords(int cid) const { return {cid % Nx, (cid / Nx) % Ny, cid / (Nx * Ny)}; }
    Eigen::Vector3d CellCenter(const Eigen::Vector3i &c) const { return Origin + H * c.cast<double>(); }
    Eigen::Vector3i CellOf(const Eigen::Vector3d &x) const;
    static int Cheb(const Eigen::Vector3i &a, const Eigen::Vector3i &b) {
        return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2])});
    }
};

// Unpacking a RadRef (see RadiationParams.h).
inline int RefEntry(RadRef ref) { return ref.Packed & ~RadRefNegative; }
inline double RefSign(RadRef ref) { return ref.Packed & RadRefNegative ? -1. : 1.; }

struct RadiationTables {
    std::vector<Eigen::Vector3i> ElemCell; // cell each element's centroid falls in
    std::vector<int> CellBand; // per cell: row index into CellElem, or -1

    // Cell-target weight table: rows are band cells (any element within Chebyshev R2+1),
    // evaluated at the cell center.
    PairTable CellElem;

    // Laplacian corrections (Eq. 12), CSR per interior cell with any differing neighbor set.
    std::vector<int> CorrCell, CorrBegin;
    std::vector<RadRef> CorrRefs;

    // Far-field interpolation per element: 8 corner cells with trilinear weights and
    // per-corner re-basing corrections from the corner's far set onto the element's. Flat,
    // element-major, like ListenerCorners below and like the buffer the GPU reads.
    std::vector<RadCorner> Interp;
    std::vector<RadRef> InterpRefs;

    // Listener stencils: the same trilinear corners as Interp, but carrying the corner
    // cell's near elements (the ones its far-field state leaves out) rather than a
    // re-basing difference. Built here so their pairs get weights like any other.
    std::vector<RadCorner> ListenerCorners;
    std::vector<RadRef> ListenerRefs;

    // Element near lists (Chebyshev <= R2 in cells), the source lists of the BEM's own
    // direct sums and the set the interpolation re-bases the far field onto. Tightening this
    // to the exact union of the element's own interpolation corners cuts the tables ~20% and
    // costs 0.6 dB of grid SNR, so the elements in the difference are earning their place.
    std::vector<std::vector<int>> ElemNear;

    // Assigns each element to its cell and fills ElemNear. Call before Tdbem::Init, whose
    // ElemElem rows are exactly these near lists.
    void AssignCells(const RadiationGrid &, const RadiationMesh &);
    // Builds every other table above, over the contour Tdbem::Init established. The cell
    // table carries weights only for the (cell, element) pairs some stencil references — the
    // set differences are thin shells, while the R2+1 neighbourhood they are drawn from is
    // not — so `listeners` must name every point that will be sampled. `staged` builds the
    // cell-element weights in the GPU's byte layout and skips the float ones, which only
    // CheckSetAlgebra reads.
    void Build(const RadiationGrid &, const CqmContour &, const RadiationMesh &, const QuadPolicy &, const std::vector<Eigen::Vector3d> &listeners, bool staged = false);

    // CellElem entry for a pair, -1 when absent. Rows are sorted by source element.
    int EntryOf(const RadiationGrid &, int cell, int elem) const;

    // Set-bookkeeping identity check (Eqs. 12/20): rebuilds a sample of cells' corrections
    // by direct membership sums over the table and diffs against the correction lists.
    // Returns the max relative mismatch. O(cells * M), so for validation runs only.
    double CheckSetAlgebra(const RadiationGrid &, const Tdbem &, int n_cells, int step) const;
};

// What a geometry epoch arguably owes the grid state. p(F_c) holds only the elements
// outside cell c's near set, so when an element crosses that boundary the stored value is no
// longer the quantity the scheme is integrating: an element that left the near set could
// have its retarded potential added back into p(F_c), and one that entered could have it
// removed. The paper does not address this. The correction applies at both time levels the
// second-order update reads, and is evaluated with the *old* pose's weights, since that is
// the geometry the history in the rings was accumulated under.
//
// It measures as a wash — the grid carries an FDTD-transported approximation of the far
// field, so an exact retarded potential for a migrating element trades one inconsistency for
// another of the same size — and ships opt-in as the reproducible measurement behind that.
// What actually governs the moving case is travel per epoch. See VALIDATION.
struct RebasePlan {
    std::vector<int> Cells; // cells whose near set changed
    std::vector<int> Begin; // CSR into Refs
    std::vector<RadRef> Refs; // entry in Table, signed by the direction of the crossing
    PairTable Table; // weights for exactly these pairs, at the old pose

    bool Empty() const { return Cells.empty(); }
};

// The plan for moving elements from `old_cells` to `new_cells`. `old_mesh` and `contour`
// must be the pose the history was accumulated under.
RebasePlan PlanRebase(const RadiationGrid &, const CqmContour &, const RadiationMesh &old_mesh, const std::vector<Eigen::Vector3i> &old_cells, const std::vector<Eigen::Vector3i> &new_cells, const QuadPolicy &);

// Builds `bem` (element near lists and their weight rows) and `tables` for `mesh` on
// `grid`, then sizes the history rings to cover every lag they reach. `previous` is the
// same element set's table at an earlier pose, whose retained pairs carry their weights
// over — see Tdbem::Init.
void BuildRadiation(const RadiationGrid &, const RadiationMesh &, const QuadPolicy &, const std::vector<Eigen::Vector3d> &listeners, Tdbem &, RadiationTables &, const PairTable *previous = nullptr, bool staged = false);

// Rest state at step 0: the Neumann samples for steps 0 and 1 (the stepper always needs
// one step of lookahead in the ring) and phi(0) from the initial diagonal solve.
void SeedRestState(Tdbem &, double tau, const std::function<double(int, double)> &neumann);
