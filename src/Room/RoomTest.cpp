// Validation harness for the room-acoustics solver. Each rung is an independent statement
// about the same stepper:
//
//   --modes    RoomShoebox, whose room is a whole number of cells on a side with the walls
//              half a cell outside the outermost nodes, so the discrete modes are exact
//              eigenvectors and the Morse & Ingard eigenfrequencies apply with no fitting.
//              Reports each mode's analytic frequency, the frequency the scheme's own
//              dispersion relation predicts, and the frequency measured off the receiver
//              signals. Under --fcc the predicted column carries the staircase wall's
//              first-order term as well — see BoxModes.
//   --energy   A sealed rigid box built here rather than voxelized, so the node set the
//              energy sums over is closed by construction. Seeded from rest and run with no
//              source at all, the lossless scheme's discrete energy is a conserved quantity
//              and the rung is how far it moves.
//   --tube     An impedance tube: a long duct, rigid but for the far end, which carries a
//              two-branch test wall. The reflection is read as the ratio of that run to the
//              same duct left rigid, so everything but the wall divides out, against the
//              coefficient the discrete wall's own admittance predicts.
//   --balance  A sealed box with one lossy wall. Room energy plus the energy stored in the
//              wall's branches plus everything the wall has dissipated is conserved, and the
//              rung is how far that moves.
//   --soak     Half a million single-precision steps on that box, driven throughout so the
//              wall never stops working, checking that nothing in it grows.
//   --golden   A scene against the reference engines' recorded receiver signals, scored the
//              way script/RunRoomReference --score scores. Both sides are single precision,
//              so the gate is sized from the precision floor the references themselves sit
//              at, not from byte identity. --scene and --gen pick the scene.
//   --fcc      Runs the rungs that have an FCC form on the 13-point grid instead. The
//              impedance tube and the energy balance stay Cartesian, since the boundary
//              update they exercise is the same kernel under either scheme.
//   --gate     Every rung at the settings its thresholds were recorded at, checked rather
//              than printed, exiting nonzero on a miss.
//   --roofline The interior update's bandwidth against a plain two-level stream over the same
//              box. --scene picks the grid, --repeat the dispatch count.
//   --implicit The Smits & Bilbao 2025 optimised implicit scheme in free space on a periodic
//              box: its dispersion against the relation its own coefficients give, the
//              oversampling ratio it needs for 1% worst-direction phase error, and what a
//              step costs against the traffic it cannot avoid. --count sets the box side and
//              --repeat the dispatch count of the cost measurement.
//   --projection A tiny runtime check of the implicit solver's zero-mean projection:
//              recursive reduction, interval scheduling, and preservation of outside zeros.
//   --marginal The R111 room at 8 cm: growth against Jacobi sweep count, then receiver and
//              mean behavior at three projection cadences. --steps may shorten every leg.
//   --projection-cost The isolated pair projection beside a whole implicit step, at the two
//              footprints used by the implicit cost rung.
//
// Set ACOUSTIC_ROOM_KERNEL_TIMES to have every dispatch isolated in its own command buffer
// and timed. That serializes the step, so it is a diagnostic run, not a mode to measure in.

#include "AabbTree.h"
#include "Parallel.h"
#include "RoomGpu.h"
#include "RoomScene.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using std::numbers::pi;

namespace {
int Failures = 0;

void Fail(const char *what, double got, const char *relation, double want) {
    std::printf("[gate] FAIL %s: %.6g, want %s %.6g\n", what, got, relation, want);
    ++Failures;
}
void AtLeast(const char *what, double got, double least) {
    if (!(got >= least)) Fail(what, got, ">=", least);
}
void AtMost(const char *what, double got, double most) {
    if (!(got <= most)) Fail(what, got, "<=", most);
}

double Now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template<typename F> double TimeGpu(int reps, F &&once) {
    auto &ctx = MetalContext::Get();
    for (int round = 0; round < 2; ++round) {
        for (int i = 0; i < reps; ++i) once();
        ctx.Sync();
        if (round == 0) ctx.TakeBatchGpuSeconds();
    }
    return ctx.TakeBatchGpuSeconds() / reps;
}

struct Box {
    int X0, X1, Y0, Y1, Z0, Z1;
    int Cells(int axis) const { return axis == 0 ? X1 - X0 + 1 : axis == 1 ? Y1 - Y0 + 1 :
                                                                             Z1 - Z0 + 1; }
};

size_t NodeIndex(const RoomScene &scene, int ix, int iy, int iz) {
    return (size_t(ix) * size_t(scene.Ny) + size_t(iy)) * size_t(scene.Nz) + size_t(iz);
}

// RoomKernels.metal's step tables written out per axis, because the geometry here needs the
// coordinate steps and not the index arithmetic.
constexpr int CartSteps[RoomNumNeighbours][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
constexpr int FccSteps[RoomFccNeighbours][3] = {{1, 1, 0}, {-1, -1, 0}, {0, 1, 1}, {0, -1, -1}, {1, 0, 1}, {-1, 0, -1}, {1, -1, 0}, {-1, 1, 0}, {0, 1, -1}, {0, -1, 1}, {1, 0, -1}, {-1, 0, 1}};

// A node's coordinates in the grid the scene stands for: the stored ones on a Cartesian grid,
// and on an FCC grid the *unfolded* ones, where its geometry lives. The fold is a mirror, so
// it flips the parity of the index sum, and an odd-parity stored node is therefore one the
// fold brought back from the far half of the original y extent.
struct Coord {
    int P[3];
    bool Upper; // came from the far half, so its stored y steps run backwards
};
Coord Unfold(const RoomScene &scene, int ii) {
    const int p[3] = {ii / (scene.Nz * scene.Ny), (ii / scene.Nz) % scene.Ny, ii % scene.Nz};
    if (!scene.Fcc || (p[0] + p[1] + p[2]) % 2 == 0) return {{p[0], p[1], p[2]}, false};
    return {{p[0], scene.NyUnfolded() - 1 - p[1], p[2]}, true};
}

// The node box a rigid room occupies, as the bounding box of its boundary nodes with the
// outer of the two boundary layers dropped. The voxelizer flags nodes on both sides of a
// surface, so the room's own shell is one node in from the bounding box.
Box RoomBox(const RoomScene &scene) {
    int lo[3] = {scene.Nx, scene.NyUnfolded(), scene.Nz}, hi[3] = {-1, -1, -1};
    for (int const ii : scene.BnIxyz) {
        const Coord c = Unfold(scene, ii);
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], c.P[a]);
            hi[a] = std::max(hi[a], c.P[a]);
        }
    }
    return {lo[0] + 1, hi[0] - 1, lo[1] + 1, hi[1] - 1, lo[2] + 1, hi[2] - 1};
}

// Whether the scheme's adjacency keeps every edge inside `box`: no interior node reaches out
// of it and no outside node reaches in. That is what makes the box's energy a conserved
// quantity, and what says the walls are where the mode arithmetic assumes.
//
// The box is in the coordinates Unfold reports, so on an FCC grid it is a box of the room and
// not of the array, and a stored y step means +y for a node from the near half of the fold and
// -y for one from the far half.
bool BoxIsClosed(const RoomScene &scene, const Box &box) {
    const int neighbours = scene.Neighbours();
    const int limit[3] = {scene.Nx, scene.NyUnfolded(), scene.Nz};
    std::vector<int> bn_of(size_t(scene.NumNodes()), -1);
    for (size_t i = 0; i < scene.BnIxyz.size(); ++i) bn_of[size_t(scene.BnIxyz[i])] = int(i);

    const auto inside = [&](const int p[3]) {
        return p[0] >= box.X0 && p[0] <= box.X1 && p[1] >= box.Y0 && p[1] <= box.Y1 && p[2] >= box.Z0 && p[2] <= box.Z1;
    };
    for (int ii = 0; ii < scene.NumNodes(); ++ii) {
        const int nb = bn_of[size_t(ii)];
        // Nodes the air pass never touches and that are not boundary nodes have no edges.
        const int stored[3] = {ii / (scene.Nz * scene.Ny), (ii / scene.Nz) % scene.Ny, ii % scene.Nz};
        const bool updated = stored[0] > 0 && stored[0] < scene.Nx - 1 && stored[1] > 0 && stored[1] < scene.Ny - 1 && stored[2] > 0 && stored[2] < scene.Nz - 1;
        if (nb < 0 && !updated) continue;
        const unsigned all = (1u << neighbours) - 1u;
        const unsigned adj = nb >= 0 ? scene.AdjBn[size_t(nb)] : all;
        const Coord c = Unfold(scene, ii);
        for (int bit = 0; bit < neighbours; ++bit) {
            if (!((adj >> bit) & 1)) continue;
            const int *step = scene.Fcc ? FccSteps[bit] : CartSteps[bit];
            int q[3];
            for (int a = 0; a < 3; ++a) q[a] = c.P[a] + (a == 1 && c.Upper ? -step[a] : step[a]);
            for (int a = 0; a < 3; ++a) {
                if (q[a] < 0 || q[a] >= limit[a]) return false;
            }
            if (inside(c.P) != inside(q)) return false;
        }
    }
    return true;
}

// The wall the impedance-tube and energy-balance rungs use, as the (D, E, F) triples PFFDTD's
// material fits arrive in. Two branches on purpose: one purely resistive, which fixes the
// reflection away from the resonance, and one series LRC resonant near 700 Hz with a
// half-power width of about an octave, which makes the reflection frequency-dependent and
// gives the wall stored energy as well as dissipation.
const std::vector<double> TestWall{0., 8., 0., 4.5e-4, 2., 8800.};

// X(f) of a windowed record at an arbitrary frequency. The rotation is re-anchored every block
// so its phase cannot drift over a record of hundreds of thousands of samples.
std::complex<double> Spectrum(const std::vector<double> &x, const std::vector<double> &win, size_t begin, size_t n, double srate, double f) {
    const double theta = -2. * pi * f / srate;
    const double dr = std::cos(theta), di = std::sin(theta);
    double re = 0., im = 0.;
    constexpr size_t BlockLen = 1024;
    for (size_t base = 0; base < n; base += BlockLen) {
        double wr = std::cos(theta * double(base)), wi = std::sin(theta * double(base));
        for (size_t k = base, end = std::min(base + BlockLen, n); k < end; ++k) {
            const double v = x[begin + k] * win[k];
            re += v * wr;
            im += v * wi;
            const double next = wr * dr - wi * di;
            wi = wr * di + wi * dr;
            wr = next;
        }
    }
    return {re, im};
}

double Power(const std::vector<double> &x, const std::vector<double> &win, size_t begin, size_t n, double srate, double f) {
    return std::norm(Spectrum(x, win, begin, n, srate, f));
}

// Golden-section search for the maximum of `fn` on [a, b].
template<typename F> double ArgMax(F &&fn, double a, double b) {
    constexpr double Gr = 0.6180339887498949;
    double c = b - Gr * (b - a), d = a + Gr * (b - a), fc = fn(c), fd = fn(d);
    for (int i = 0; i < 90 && b - a > 1e-11 * std::fabs(a); ++i) {
        if (fc > fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - Gr * (b - a);
            fc = fn(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + Gr * (b - a);
            fd = fn(d);
        }
    }
    return 0.5 * (a + b);
}

struct Mode {
    int N[3];
    double Analytic{0}; // Morse & Ingard, the continuous box
    double Discrete{0}; // the scheme's own dispersion relation at this mode's wavevector
    double Measured{0}; // off the receiver signals
    bool Bracketed{true};
};

// What the FCC shoebox's modes are held to, both recorded. The first is dominated by the
// staircase wall rather than by dispersion, and the second has that wall's first order
// predicted out and holds what its second order leaves — see BoxModes.
constexpr double FccModeLimit = 2.1; // per cent, vs Morse & Ingard
constexpr double FccDispersionLimit = 6e-4; // vs the FCC relation with its wall term

// The first `count` eigenfrequencies of a rigid box, and what the scheme puts each of them at.
//
// Substituting a plane wave into u(n+1) = A1 u(n) - u(n-1) + A2 sum_NN u(n) gives
//
//   sin^2(w Ts / 2) = A2 sum_v sin^2(beta.v h / 2) + (2 - A1 - NN A2) / 4
//
// with v running over the stencil's edge directions — the same expression for both schemes,
// since the FCC sum over twelve face diagonals collapses to six sin^2 terms by
// 1 - cos a cos b = sin^2((a+b)/2) + sin^2((a-b)/2). The update's diagonal shift is the
// constant term. Written with the ideal l^2 that term is 1.5 eps l^2 on the Cartesian grid,
// and written with the coefficients the stepper narrows to float it is 17% larger, worth 5e-5
// in frequency at the lowest mode of this box, so the coefficients in force are what goes in.
//
// On the Cartesian grid a rigid wall half a cell outside the outermost node makes
// cos(beta_d (x - x_wall)) with beta_d = n_d pi / L_d an exact eigenvector of the discrete
// operator, so no boundary correction enters anywhere. On the FCC grid it does not. A wall
// drops the neighbours pointing through it and adds their weight back to the node's own
// diagonal, which is the statement that each missing neighbour equals the node itself. On the
// Cartesian stencil that is exactly the half-cell mirror, because the missing neighbour's
// reflection *is* the node. An FCC face diagonal reflects onto a point that is not on the
// sublattice at all, and the substitution is right only where the mode is flat along the wall.
//
// What is left over is diagonal, so first-order perturbation theory gives it in closed form.
// A node on a wall normal to axis a is missing the four diagonals with a step through it, and
// their true values sum to 2(cos(beta_b h) + cos(beta_c h)) times the node's own — the wall's
// own axis drops out by the mirror. The scheme puts 4 Sl2/A2 there instead, so the residue is
//
//   E = 4 Sl2/A2 - 2 cos(beta_b h) - 2 cos(beta_c h)   per node of that wall,
//
// and the eigenvalue moves by the sum of E over the walls weighted by each one's share of the
// mode's energy — two faces of cos^2(beta_a h/2) over N_a cells, halved again when the mode
// varies along a.
std::vector<Mode> BoxModes(const double lengths[3], const int cells[3], double c, double h, double ts, const RoomCoefficients &k, int neighbours, int count) {
    const bool fcc = neighbours == int(RoomFccNeighbours);
    // One direction per plus/minus pair of neighbours. Both step tables list each pair
    // adjacent, so the even entries are exactly those directions.
    const int dirs = neighbours / 2;
    std::vector<Mode> modes;
    const int limit = 8;
    for (int n0 = 0; n0 <= limit; ++n0) {
        for (int n1 = 0; n1 <= limit; ++n1) {
            for (int n2 = 0; n2 <= limit; ++n2) {
                if (n0 + n1 + n2 == 0) continue;
                const int n[3] = {n0, n1, n2};
                double beta[3] = {0., 0., 0.}, k2 = 0.;
                for (int a = 0; a < 3; ++a) {
                    beta[a] = n[a] * pi / lengths[a];
                    k2 += beta[a] * beta[a];
                }
                double sines = 0.;
                for (int d = 0; d < dirs; ++d) {
                    const int *v = fcc ? FccSteps[2 * d] : CartSteps[2 * d];
                    const double s = std::sin(.5 * h * (beta[0] * v[0] + beta[1] * v[1] + beta[2] * v[2]));
                    sines += s * s;
                }
                double wall = 0.;
                for (int a = 0; a < 3 && fcc; ++a) {
                    const double share = 2. * std::pow(std::cos(.5 * beta[a] * h), 2.) / (cells[a] * (n[a] > 0 ? .5 : 1.));
                    double residue = 4. * double(k.Sl2) / double(k.A2);
                    for (int d = 0; d < 3; ++d) {
                        if (d != a) residue -= 2. * std::cos(beta[d] * h);
                    }
                    wall += share * residue;
                }
                const double arg = double(k.A2) * (sines - .25 * wall) + (2. - double(k.A1) - neighbours * double(k.A2)) / 4.;
                if (arg >= 1. || arg <= 0.) continue; // past the scheme's cutoff
                modes.push_back({{n0, n1, n2}, c * std::sqrt(k2) / (2. * pi), std::asin(std::sqrt(arg)) / (pi * ts), 0., true});
            }
        }
    }
    std::sort(modes.begin(), modes.end(), [](const Mode &a, const Mode &b) { return a.Analytic < b.Analytic; });
    modes.resize(std::min(size_t(count), modes.size()));
    return modes;
}

// `analytic_limit` (per cent) and `dispersion_limit` are the gate's thresholds. They belong to
// the scheme: the Cartesian box's modes are exact discrete eigenvectors and the FCC box's are
// not, so the two are not held to the same number. See BoxModes.
void ModeRung(const std::string &config_file, int repeat, int count, bool gate, double analytic_limit = 0., double dispersion_limit = 0.) {
    const double t0 = Now();
    RoomScene scene = LoadRoomScene(config_file);
    const Box box = RoomBox(scene);
    if (!BoxIsClosed(scene, box))
        throw std::runtime_error("The scene's rigid box is not closed under the scheme's adjacency, so its modes are not the analytic ones");

    // A longer record than the scene's own: the modes are undamped, so stepping past the
    // scene's duration costs only time and buys resolution. The closest pair among the first
    // sixteen is about 1.5 Hz apart, which one scene length does not separate.
    const int steps = scene.Nt * repeat;
    std::vector<double> padded(scene.InIxyz.size() * size_t(steps), 0.);
    for (size_t s = 0; s < scene.InIxyz.size(); ++s)
        std::copy_n(&scene.InSigs[s * size_t(scene.Nt)], scene.Nt, &padded[s * size_t(steps)]);
    scene.InSigs = std::move(padded);
    scene.Nt = steps;

    const auto rows = RenderRoomScene(scene, steps);
    const int cells[3] = {box.Cells(0), box.Cells(1), box.Cells(2)};
    const double lengths[3] = {cells[0] * scene.H, cells[1] * scene.H, cells[2] * scene.H};
    auto modes = BoxModes(lengths, cells, scene.C, scene.H, scene.Ts, RoomUpdateCoefficients(scene), scene.Neighbours(), count);

    std::vector<double> win(static_cast<size_t>(steps));
    for (size_t k = 0; k < win.size(); ++k) win[k] = 0.5 - 0.5 * std::cos(2. * pi * double(k) / double(win.size() - 1));

    ParallelFor(modes.size(), 1, [&](size_t m) {
        auto &mode = modes[m];
        // Pick the receiver that responds most to this mode, so a receiver near one of its
        // nodal surfaces cannot decide the measurement.
        int best = 0;
        double best_power = -1.;
        for (int r = 0; r < scene.NumReceivers; ++r) {
            const double p = Power(rows, win, size_t(r) * steps, size_t(steps), scene.Srate, mode.Discrete);
            if (p > best_power) best_power = p, best = r;
        }
        const auto power = [&](double f) { return Power(rows, win, size_t(best) * steps, size_t(steps), scene.Srate, f); };
        // Scan a window an order of magnitude wider than the deviation being measured, then
        // refine the bracket the peak falls in.
        constexpr int Scan = 201;
        const double span = 5e-3 * mode.Discrete;
        int peak = -1;
        double peak_power = -1.;
        for (int i = 0; i < Scan; ++i) {
            const double p = power(mode.Discrete - span + 2. * span * i / (Scan - 1));
            if (p > peak_power) peak_power = p, peak = i;
        }
        mode.Bracketed = peak > 0 && peak < Scan - 1;
        const double lo = mode.Discrete - span + 2. * span * (peak - 1) / (Scan - 1);
        const double hi = mode.Discrete - span + 2. * span * (peak + 1) / (Scan - 1);
        mode.Measured = ArgMax(power, lo, hi);
    });

    std::printf("[modes] %s %s box %d x %d x %d cells (%.4f x %.4f x %.4f m), %d steps at %.1f Hz\n", config_file.c_str(), scene.Fcc ? "FCC" : "cart", box.Cells(0), box.Cells(1), box.Cells(2), lengths[0], lengths[1], lengths[2], steps, scene.Srate);
    std::printf("  mode      analytic   %s     measured   meas/analytic   meas/predicted\n", scene.Fcc ? "disp+wall" : "dispersion");
    double worst_analytic = 0., worst_discrete = 0.;
    bool bracketed = true;
    for (const auto &m : modes) {
        const double e_analytic = m.Measured / m.Analytic - 1., e_discrete = m.Measured / m.Discrete - 1.;
        worst_analytic = std::max(worst_analytic, std::fabs(e_analytic));
        worst_discrete = std::max(worst_discrete, std::fabs(e_discrete));
        bracketed = bracketed && m.Bracketed;
        std::printf("  %d,%d,%d  %11.5f  %11.5f  %11.5f   %+.4f %%       %+.2e%s\n", m.N[0], m.N[1], m.N[2], m.Analytic, m.Discrete, m.Measured, 100. * e_analytic, e_discrete, m.Bracketed ? "" : "  UNBRACKETED");
    }
    const char *relation = scene.Fcc ? "the FCC relation with its wall term" : "the 7-point dispersion relation";
    std::printf("[modes] worst vs Morse & Ingard %.4f %%, worst vs %s %.2e | %.1fs\n", 100. * worst_analytic, relation, worst_discrete, Now() - t0);
    if (gate) {
        AtLeast("mode peaks bracketed", bracketed ? 1. : 0., 1.);
        // What separates the two columns is which errors are predicted. Deviation from Morse &
        // Ingard is the fp32 diagonal shift and dispersion, plus on the FCC grid a staircase
        // wall worth one and a half per cent. The right-hand column has all of that predicted
        // out and holds what is left over.
        AtMost("mode error vs Morse & Ingard", 100. * worst_analytic, analytic_limit);
        AtMost(scene.Fcc ? "mode error vs the FCC relation with its wall term" : "mode error vs the 7-point dispersion relation", worst_discrete, dispersion_limit);
    }
}

// A sealed rigid box, built here rather than voxelized, with the same two-layer wall the
// voxelizer produces: the box's own shell keeps only the neighbours inside it, and the layer
// of nodes just outside keeps everything except the neighbour across the wall. The box is then
// closed under the scheme's adjacency in both directions by construction, which is what makes
// its energy a conserved quantity.
//
// `lossy_face`, when it is not -1, is one of the six adjacency-bit directions: the face of the
// box's inner shell looking that way carries the test material instead of being rigid. Its
// nodes keep the same adjacency either way.
//
// `fcc` builds the same thing on the FCC stencil, where the box holds two interleaved
// sublattices that never exchange an edge — every one of the twelve face diagonals preserves
// the parity of the index sum. Seeding only the even one leaves a genuine FCC room with the
// odd nodes pinned at zero, and the fold seam sits outside the padding, so nothing about the
// folded storage enters.
RoomScene SealedBox(const int cells[3], int pad, int steps, Box &box, int lossy_face = -1, bool fcc = false) {
    RoomScene scene;
    scene.Fcc = fcc;
    scene.Nx = cells[0] + 2 * pad;
    scene.Ny = cells[1] + 2 * pad;
    scene.Nz = cells[2] + 2 * pad;
    scene.H = 0.0326857142857142;
    scene.C = 343.2;
    // The FCC stencil is stable to a Courant number of 1 where the Cartesian one stops at
    // 1/sqrt(3), and both are backed off by the reference's 0.999.
    scene.L = fcc ? 0.999 : 0.999 / std::numbers::sqrt3;
    scene.L2 = scene.L * scene.L;
    scene.Ts = scene.L * scene.H / scene.C;
    scene.Srate = 1. / scene.Ts;
    scene.Nt = steps;
    scene.Output = "SealedBox";
    box = {pad, pad + cells[0] - 1, pad, pad + cells[1] - 1, pad, pad + cells[2] - 1};

    const int neighbours = scene.Neighbours();
    const int lo[3] = {box.X0, box.Y0, box.Z0}, hi[3] = {box.X1, box.Y1, box.Z1};
    const auto in_box = [&](const int p[3]) {
        for (int a = 0; a < 3; ++a) {
            if (p[a] < lo[a] || p[a] > hi[a]) return false;
        }
        return true;
    };
    for (int ix = lo[0] - 1; ix <= hi[0] + 1; ++ix) {
        for (int iy = lo[1] - 1; iy <= hi[1] + 1; ++iy) {
            for (int iz = lo[2] - 1; iz <= hi[2] + 1; ++iz) {
                const int p[3] = {ix, iy, iz};
                uint16_t adj = 0;
                bool wall = false, lossy = false;
                for (int bit = 0; bit < neighbours; ++bit) {
                    const int *step = fcc ? FccSteps[bit] : CartSteps[bit];
                    const int q[3] = {ix + step[0], iy + step[1], iz + step[2]};
                    // The wall separates the box from everything else, so an edge crosses it
                    // exactly when its two ends disagree about being inside.
                    if (in_box(p) != in_box(q)) {
                        wall = true;
                        lossy = lossy || (bit == lossy_face && in_box(p));
                    } else {
                        adj |= uint16_t(1u << bit);
                    }
                }
                if (!wall) continue;
                scene.BnIxyz.push_back(int(NodeIndex(scene, ix, iy, iz)));
                scene.AdjBn.push_back(adj);
                scene.MatBn.push_back(lossy ? int8_t(0) : int8_t(-1));
                // A flat face of the staircase grid, so the node's share of the wall is one
                // cell of it: the surface-area correction the voxelizer would compute is 1.
                scene.SafBn.push_back(1.);
                scene.NumLossy += lossy;
            }
        }
    }
    if (lossy_face >= 0) {
        scene.MatBranches.assign(1, int(TestWall.size()) / 3);
        scene.MatDef = TestWall;
    }
    return scene;
}

// The Gaussian bump the energy and balance rungs start from, seeded into both levels so that
// nothing is injected after step zero. On the FCC stencil only the even sublattice is seeded,
// the odd one sharing no edge with it.
std::vector<float> GaussianSeed(const RoomScene &scene, const Box &box) {
    std::vector<float> seed(size_t(scene.NumNodes()), 0.f);
    const double centre[3] = {box.X0 + .41 * box.Cells(0), box.Y0 + .57 * box.Cells(1), box.Z0 + .33 * box.Cells(2)};
    constexpr double Sigma = 2.;
    for (int ix = box.X0; ix <= box.X1; ++ix) {
        for (int iy = box.Y0; iy <= box.Y1; ++iy) {
            for (int iz = box.Z0; iz <= box.Z1; ++iz) {
                if (scene.Fcc && (ix + iy + iz) % 2 != 0) continue;
                const double d[3] = {ix - centre[0], iy - centre[1], iz - centre[2]};
                const double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                seed[NodeIndex(scene, ix, iy, iz)] = float(std::exp(-r2 / (2. * Sigma * Sigma)));
            }
        }
    }
    return seed;
}

double PeakInBox(const RoomScene &scene, const float *u, const Box &box) {
    double peak = 0.;
    for (int ix = box.X0; ix <= box.X1; ++ix)
        for (int iy = box.Y0; iy <= box.Y1; ++iy)
            for (int iz = box.Z0; iz <= box.Z1; ++iz)
                peak = std::max(peak, std::fabs(double(u[NodeIndex(scene, ix, iy, iz)])));
    return peak;
}

// The (D, E, F) coefficients the branch update is carrying, recovered from the float quads
// rather than from the doubles they came from. Everything predicted analytically about this
// wall has to be predicted from the coefficients in force.
std::vector<std::array<double, 3>> BranchesInForce(const RoomScene &scene, int material) {
    const auto materials = RoomMaterialCoefficients(scene);
    std::vector<std::array<double, 3>> def;
    for (int m = 0; m < scene.MatBranches[size_t(material)]; ++m) {
        const auto &q = materials.Quads[size_t(material) * RoomMaxBranches + m];
        const double dh = double(q.BDh) / double(q.B), fh = double(q.BFh) / double(q.B);
        def.push_back({dh * scene.Ts, 1. / double(q.B) - 2. * dh - .5 * fh, fh / scene.Ts});
    }
    return def;
}

void EnergyRung(int steps, int reports, bool gate, bool fcc = false) {
    const double t0 = Now();
    const int cells[3] = {24, 20, 16};
    Box box{};
    RoomScene const scene = SealedBox(cells, 4, steps, box, -1, fcc);
    if (!BoxIsClosed(scene, box)) throw std::runtime_error("The sealed box is not closed under the scheme's adjacency");

    RoomGpu gpu;
    gpu.Init(scene);
    gpu.Seed(GaussianSeed(scene, box));
    const double e0 = gpu.Energy(box.X0, box.X1, box.Y0, box.Y1, box.Z0, box.Z1);

    std::printf("[energy] sealed %s %d x %d x %d cell box in a %d x %d x %d grid, %d steps, E(0) = %.12e\n", fcc ? "FCC" : "cart", cells[0], cells[1], cells[2], scene.Nx, scene.Ny, scene.Nz, steps, e0);
    double worst = 0., last = e0, peak = 0.;
    const int every = std::max(1, steps / reports);
    for (int done = 0; done < steps; done += every) {
        gpu.RunSteps(std::min(every, steps - done));
        last = gpu.Energy(box.X0, box.X1, box.Y0, box.Y1, box.Z0, box.Z1);
        worst = std::max(worst, std::fabs(last / e0 - 1.));
        peak = std::max(peak, PeakInBox(scene, gpu.Level(), box));
        std::printf("  step %7d  E = %.12e  drift %+.3e  max|u| %.6f\n", gpu.Step(), last, last / e0 - 1., peak);
    }
    std::printf("[energy] %d steps, worst relative drift %.3e (%.2e per step), max|u| %.6f | %.1fs\n", steps, worst, worst / steps, peak, Now() - t0);
    if (gate) {
        // Machine precision for fp32 state is 1.2e-7, and the drift is a few multiples of it
        // with no trend over the soak.
        AtMost("energy drift over the soak", worst, 2e-6);
        AtMost("sealed box peak amplitude", peak, 1.5);
    }
}

// The reflection coefficient the discrete wall presents to a normally incident plane wave at
// frequency `f`, referred to the wall plane half a cell outside its nodes.
//
// The wall condition, written with a ghost node so it reads as the interior stencil, is
// u(N+1) - u(N) = -(2 lo2 S / A2) sum_m mu_t vh_m. Substituting a time-harmonic plane wave into
// the branch update turns each branch into its continuous impedance evaluated not at i w but at
// the trapezoid rule's own frequency variable s = i (2/Ts) tan(w Ts / 2) — the branch update is
// the bilinear transform of D s + E + F/s, so that warping is the wall's, not an approximation
// — and gives sum_m mu_t vh_m = Ts s cos^2(w Ts / 2) Y(s) u(N), Y = sum_m 1/(D s + E + F/s).
// Eliminating u(N+1) against the incident and reflected waves leaves
//
//   R = (1 + k e^(i b h / 2)) / (1 - k e^(-i b h / 2)),  k = -(2 lo2 S) sin cos Y / (A2 sin(b h / 2))
//
// with b the discrete wavenumber. Rigid walls are k = 0 and R = 1, and a resistive wall at low
// frequency is (1 - Y)/(1 + Y), which is the continuous answer.
//
// Everything here is the scheme's own: A2 and the diagonal shift the interior carries set b
// through the 7-point dispersion relation, lo2 is the float the boundary update divides with,
// and the branch coefficients are the ones recovered from the float quads.
std::complex<double> WallReflection(double f, double ts, const RoomCoefficients &k, double lo2, double ssaf, const std::vector<std::array<double, 3>> &branches) {
    const double half = pi * f * ts, sh = std::sin(half), ch = std::cos(half);
    const std::complex<double> s{0., 2. / ts * sh / ch};
    std::complex<double> y{0., 0.};
    for (const auto &b : branches) y += 1. / (b[0] * s + b[1] + b[2] / s);

    const double shift = (2. - double(k.A1) - int(RoomNumNeighbours) * double(k.A2)) / 4.;
    const double sin_beta_half = std::sqrt((sh * sh - shift) / double(k.A2));
    const std::complex<double> kappa = -2. * lo2 * ssaf * sh * ch * y / (double(k.A2) * sin_beta_half);
    const double beta_half = std::asin(sin_beta_half);
    const std::complex<double> e{std::cos(beta_half), std::sin(beta_half)};
    return (1. + kappa * e) / (1. - kappa / e);
}

// Impedance tube: a duct one wavelength wide and hundreds long, sealed rigid everywhere except
// the far end, which carries the test wall. The tube is seeded with half a Gaussian centred on
// the rigid near-end wall plane, which by the mirror symmetry of that wall is the restriction
// of a symmetric free-field pulse, so what travels down the tube is a single wavefront and the
// incident and reflected transits at the receiver are hundreds of steps apart.
//
// The reflection is read as the ratio of two runs, this one and the same tube with the far end
// left rigid. Everything but the far wall is common to the pair — the source spectrum, the
// numerical dispersion of a long transit, the window — so the ratio is the wall's reflection
// with no propagation phase to model and no incident spectrum to divide out.
void TubeRung(int length, int steps, bool gate) {
    const double t0 = Now();
    const int cells[3] = {6, 6, length};
    constexpr int Pad = 4;

    // Where the transits land, in steps: a wavefront advances l cells a step, the receiver sits
    // half way down, and the reflection has the extra round trip past it.
    Box box{};
    const RoomScene wall = SealedBox(cells, Pad, steps, box, 4);
    const double per_cell = 1. / wall.L;
    const int receiver_cell = length / 2;
    const int direct = int(receiver_cell * per_cell), reflected = int((2 * length - receiver_cell) * per_cell);
    const int again = int((2 * length + receiver_cell) * per_cell);
    const int begin = (direct + reflected) / 2, end = std::min(steps, (reflected + again) / 2);
    if (end - begin < 64 || steps < reflected + 32) throw std::runtime_error("Impedance tube is too short to separate the incident and reflected transits");

    std::vector<std::vector<double>> records;
    for (int lossy = 1; lossy >= 0; --lossy) {
        // The lossy duct is `wall` itself — the two differ only in the far end's material.
        RoomScene scene = lossy ? wall : SealedBox(cells, Pad, steps, box, -1);
        scene.NumReceivers = 1;
        scene.CornersPerReceiver = 1;
        scene.OutIxyz = {int(NodeIndex(scene, box.X0 + cells[0] / 2, box.Y0 + cells[1] / 2, box.Z0 + receiver_cell))};
        scene.OutAlpha = {1.};

        // Half a Gaussian centred on the near-end wall plane, which lies half a cell outside
        // the outermost node. Seeded into both levels, so the tube starts at rest.
        constexpr double Sigma = 1.5;
        std::vector<float> seed(size_t(scene.NumNodes()), 0.f);
        for (int ix = box.X0; ix <= box.X1; ++ix) {
            for (int iy = box.Y0; iy <= box.Y1; ++iy) {
                for (int iz = box.Z0; iz <= box.Z1; ++iz) {
                    const double d = iz - (box.Z0 - .5);
                    seed[NodeIndex(scene, ix, iy, iz)] = float(std::exp(-d * d / (2. * Sigma * Sigma)));
                }
            }
        }

        RoomGpu gpu;
        gpu.Init(scene);
        gpu.Seed(seed);
        gpu.RunSteps(steps);
        const float *samples = gpu.Samples();
        std::vector<double> row(size_t(steps), 0.);
        for (int n = 0; n < steps; ++n) row[size_t(n)] = samples[n];
        records.push_back(std::move(row));
    }

    // A raised-cosine taper an eighth of the window wide at each end. Both records get the same
    // window, so it divides out of the ratio wherever the transit is inside the flat part,
    // which is what the leakage figure below checks.
    std::vector<double> win(size_t(steps), 0.);
    const int taper = (end - begin) / 8;
    for (int n = begin; n < end; ++n) {
        const int from_edge = std::min(n - begin, end - 1 - n);
        win[size_t(n)] = from_edge >= taper ? 1. : .5 - .5 * std::cos(pi * (from_edge + .5) / taper);
    }
    double peak = 0., edge = 0.;
    for (int n = begin; n < end; ++n) {
        const double v = std::fabs(records[0][size_t(n)]);
        if (n < begin + taper || n >= end - taper) edge = std::max(edge, v);
        else peak = std::max(peak, v);
    }

    const auto coeffs = RoomUpdateCoefficients(wall);
    const auto branches = BranchesInForce(wall, 0);
    const double lo2 = double(float(.5 * wall.L));

    std::printf("[tube] %d cell duct, receiver at %d, transits at steps %d and %d, window [%d, %d), taper %d\n", length, receiver_cell, direct, reflected, begin, end, taper);
    std::printf("  branches in force (D, E, F):");
    for (const auto &b : branches) std::printf("  (%.6g, %.6g, %.6g)", b[0], b[1], b[2]);
    std::printf("\n     f (Hz)   |R| measured   |R| analytic     arg meas     arg anal    |error|\n");

    constexpr int Points = 25;
    const double f_lo = 100., f_hi = 1800.;
    double worst = 0.;
    for (int i = 0; i < Points; ++i) {
        const double f = f_lo * std::pow(f_hi / f_lo, double(i) / (Points - 1));
        const auto measured = Spectrum(records[0], win, 0, size_t(steps), wall.Srate, f) / Spectrum(records[1], win, 0, size_t(steps), wall.Srate, f);
        const auto analytic = WallReflection(f, wall.Ts, coeffs, lo2, 1., branches);
        const double error = std::abs(measured - analytic);
        worst = std::max(worst, error);
        std::printf("  %9.2f   %12.6f   %12.6f   %+10.6f   %+10.6f   %.3e\n", f, std::abs(measured), std::abs(analytic), std::arg(measured), std::arg(analytic), error);
    }
    std::printf("[tube] worst |R measured - R analytic| over %.0f-%.0f Hz: %.3e, window-edge leakage %.2e of peak | %.1fs\n", f_lo, f_hi, worst, edge / peak, Now() - t0);
    if (gate) {
        // What is left is the reflected transit's own tail, which the window's flat part does
        // not quite hold — lengthening the duct pushes it down in proportion (a 900 cell duct
        // gives 1.2e-3, this one 8e-5), so the rung is measuring the window and not the wall.
        AtMost("impedance-tube reflection vs the discrete wall's analytic coefficient", worst, 2e-4);
        AtMost("impedance-tube window-edge leakage", edge / peak, .06);
    }
}

// A sealed box with one lossy wall. The room's discrete energy is no longer conserved on its
// own — the wall takes energy out of it, holds some in its branches and dissipates the rest —
// but the sum of the three is, exactly. The passivity statement of Bilbao & Hamilton 2019 read
// as an equality.
//
// Multiplying the branch update by mu_t vh gives, per branch and per step,
//
//   delta_t [ (D/2) vh^2 + (F/2) (Ts gh)^2 ] + Ts E (mu_t vh)^2 = Ts (delta_t. u)(mu_t vh)
//
// and the node update contributes exactly -l S sum_m mu_t vh_m times delta_t. u to the interior
// energy's own difference, so the two cancel term for term with the surface factor 2 l S / Ts.
// mu_t vh is read off the state as the step's change in gh, which is what gh integrates.
void BalanceRung(int steps, int reports, bool gate) {
    const double t0 = Now();
    const int cells[3] = {16, 14, 12};
    Box box{};
    RoomScene const scene = SealedBox(cells, 4, steps, box, 4);
    if (!BoxIsClosed(scene, box)) throw std::runtime_error("The sealed box is not closed under the scheme's adjacency");

    const auto lossy = RoomLossySubset(scene);
    const auto branches = BranchesInForce(scene, 0);
    const int nbl = int(lossy.Ixyz.size()), mb = int(branches.size());
    const double l = 2. * double(float(.5 * scene.L));

    RoomGpu gpu;
    gpu.Init(scene);
    gpu.Seed(GaussianSeed(scene, box));
    const double e0 = gpu.Energy(box.X0, box.X1, box.Y0, box.Y1, box.Z0, box.Z1);

    // The wall's stored energy, from the branch state as it stands.
    const auto stored = [&](const float *v, const float *g) {
        double total = 0.;
        for (int nb = 0; nb < nbl; ++nb) {
            const double factor = l * double(lossy.Ssaf[size_t(nb)]) / scene.Ts;
            for (int m = 0; m < mb; ++m) {
                const double vh = v[size_t(m) * nbl + nb], gh = g[size_t(m) * nbl + nb];
                total += factor * (branches[size_t(m)][0] * vh * vh + branches[size_t(m)][2] * scene.Ts * scene.Ts * gh * gh);
            }
        }
        return total;
    };

    std::printf("[balance] sealed %d x %d x %d cell box, %d lossy nodes x %d branches, %d steps, E(0) = %.12e\n", cells[0], cells[1], cells[2], nbl, mb, steps, e0);
    std::vector<float> previous(size_t(nbl) * mb, 0.f);
    double dissipated = 0., worst = 0., last_room = e0, last_wall = 0.;
    const int every = std::max(1, steps / reports);
    for (int done = 0; done < steps; ++done) {
        gpu.RunSteps(1);
        const float *g = gpu.BranchG();
        for (int nb = 0; nb < nbl; ++nb) {
            const double factor = 2. * l * double(lossy.Ssaf[size_t(nb)]);
            for (int m = 0; m < mb; ++m) {
                const size_t i = size_t(m) * nbl + nb;
                const double dg = double(g[i]) - double(previous[i]);
                dissipated += factor * branches[size_t(m)][1] * dg * dg;
                previous[i] = g[i];
            }
        }
        if ((done + 1) % every && done + 1 != steps) continue;
        last_room = gpu.Energy(box.X0, box.X1, box.Y0, box.Y1, box.Z0, box.Z1);
        last_wall = stored(gpu.BranchV(), gpu.BranchG());
        const double drift = (last_room + last_wall + dissipated) / e0 - 1.;
        worst = std::max(worst, std::fabs(drift));
        std::printf("  step %7d  room %.9e  wall %.6e  dissipated %.9e  drift %+.3e\n", gpu.Step(), last_room, last_wall, dissipated, drift);
    }
    std::printf("[balance] %d steps, %.3f%% of the energy taken by the wall, worst relative drift %.3e (%.2e per step) | %.1fs\n", steps, 100. * (1. - last_room / e0), worst, worst / steps, Now() - t0);
    if (gate) {
        // Nothing is being conserved unless the wall actually took most of the energy.
        AtLeast("fraction of the energy the lossy wall absorbed", 1. - last_room / e0, .9);
        AtMost("room plus wall plus dissipation drift", worst, 1e-5);
    }
}

// A long single-precision run on the same lossy box, driven the whole way by an impulse every
// thousand steps so the wall never stops working: nothing about the room or the boundary state
// may grow. Undriven, the room would decay to nothing in a fraction of the run and stop
// testing anything. The purely resistive branch's own recursion sits exactly on the unit
// circle, and its integrator gh accumulates every step of the soak.
void SoakRung(int steps, int reports, bool gate) {
    const double t0 = Now();
    const int cells[3] = {16, 14, 12};
    constexpr int Period = 1000;
    Box box{};
    RoomScene scene = SealedBox(cells, 4, steps, box, 4);
    scene.InIxyz = {int(NodeIndex(scene, box.X0 + 5, box.Y0 + 6, box.Z0 + 4))};
    scene.InSigs.assign(size_t(steps), 0.);
    for (int n = 0; n < steps; n += Period) scene.InSigs[size_t(n)] = 1.;

    RoomGpu gpu;
    gpu.Init(scene);
    const int nbl = gpu.LossyNodes();

    std::printf("[soak] sealed %d x %d x %d cell box with one lossy wall, driven every %d steps, %d steps\n", cells[0], cells[1], cells[2], Period, steps);
    double settled = 0., last = 0.;
    const int every = std::max(1, steps / reports);
    for (int done = 0, report = 0; done < steps; done += every, ++report) {
        gpu.RunSteps(std::min(every, steps - done));
        const double peak = PeakInBox(scene, gpu.Level(), box);
        double vpeak = 0., gpeak = 0.;
        const float *v = gpu.BranchV(), *g = gpu.BranchG();
        for (size_t i = 0; i < size_t(nbl) * RoomMaxBranches; ++i) {
            vpeak = std::max(vpeak, std::fabs(double(v[i])));
            gpeak = std::max(gpeak, std::fabs(double(g[i])));
        }
        // The room approaches the impulse train's steady response geometrically, so the level
        // half way through the soak is what the second half is judged against.
        if (report == reports / 2) settled = peak;
        last = peak;
        std::printf("  step %8d  max|u| %.6e  max|vh| %.6e  max|gh| %.6e%s\n", gpu.Step(), peak, vpeak, gpeak, std::isfinite(peak + vpeak + gpeak) ? "" : "  NOT FINITE");
        if (gate) AtMost("soak state stays finite", std::isfinite(peak + vpeak + gpeak) ? 0. : 1., 0.);
    }
    std::printf("[soak] %d steps, max|u| %.6e half way and %.6e at the end, ratio %.7f | %.1fs\n", steps, settled, last, last / settled, Now() - t0);
    if (gate) AtMost("soak peak amplitude at the end against its settled level", last / settled, 1.0001);
}

std::vector<double> ReadDoubles(const std::string &path) {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    if (!in) throw std::runtime_error("Failed to read reference: " + path);
    const auto bytes = size_t(in.tellg());
    std::vector<double> values(bytes / sizeof(double));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(values.data()), std::streamsize(bytes));
    return values;
}

// Scored the way script/RunRoomReference --score scores: per receiver, 20 log10 of the
// truth's norm over the difference's, plus the same over every receiver at once.
void GoldenRung(const std::string &config_file, const std::string &gen_dir, bool gate, double floor = 35.) {
    const double t0 = Now();
    RoomScene scene = LoadRoomScene(config_file);
    const auto rows = RenderRoomScene(scene, scene.Nt);
    const int nr = scene.NumReceivers, nt = scene.Nt;

    for (const char *stem : {"cpu_fp32", "cuda_fp32", "cpu_fp64"}) {
        const std::string path = gen_dir + "/" + stem + ".bin";
        if (!std::filesystem::exists(path)) {
            std::printf("[golden] %s: not present, skipped\n", path.c_str());
            continue;
        }
        const auto truth = ReadDoubles(path);
        if (truth.size() != size_t(nr) * nt) throw std::runtime_error("Reference shape disagrees with the scene: " + path);

        double num_all = 0., den_all = 0., worst = 1e300, peak = 0., max_diff = 0.;
        for (int r = 0; r < nr; ++r) {
            double num = 0., den = 0.;
            for (int n = 0; n < nt; ++n) {
                const double t = truth[size_t(r) * nt + n], o = rows[size_t(r) * nt + n];
                num += t * t;
                den += (t - o) * (t - o);
                peak = std::max(peak, std::fabs(t));
                max_diff = std::max(max_diff, std::fabs(t - o));
            }
            num_all += num;
            den_all += den;
            worst = std::min(worst, 10. * std::log10(num / den));
        }
        const double overall = 10. * std::log10(num_all / den_all);
        std::printf("[golden] %s vs %s: %.2f dB overall, %.2f dB worst receiver, max diff %.3g (peak %.3g) | %.1fs\n", scene.Output.c_str(), stem, overall, worst, max_diff, peak, Now() - t0);
        // The gate is against the single-precision CUDA golden, sized from the precision floor
        // the reference engines themselves sit at: 39.6 dB worst receiver on this scene,
        // against the fp64 truth. fp64 is reported for context only, since no fp32 solver can
        // beat its own precision floor against it.
        if (gate && std::string{stem} == "cuda_fp32") AtLeast("shoebox worst receiver vs the CUDA fp32 golden", worst, floor);
    }
}

#include "RoomImplicit.inc"

// Applies a masked Laplacian in double over the padded room layout.
template<typename T> std::vector<double> ImplicitApply(const ImplicitBox &box, const T *u, const double c[3]) {
    std::vector<double> out(box.Nodes, 0.);
    for (int ix = 0; ix < box.Nx; ++ix) {
        for (int iy = 0; iy < box.Ny; ++iy) {
            for (int iz = 0; iz < box.Nz; ++iz) {
                double s[3]{0., 0., 0.};
                int k[3]{0, 0, 0};
                for (const auto &leg : ImplicitLegs) {
                    if (!box.Shape.Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz)) {
                        s[leg.Class] += double(u[box.Index(ix + leg.Dx, iy + leg.Dy, iz + leg.Dz)]);
                        ++k[leg.Class];
                    }
                }
                double v = 0.;
                for (int r = 0; r < 3; ++r) v += c[r] * (s[r] - k[r] * double(u[box.Index(ix, iy, iz)]));
                out[box.Index(ix, iy, iz)] = v;
            }
        }
    }
    return out;
}

// Finds the walled energy limit 2/sqrt(rho), where rho is the largest eigenvalue of
// (-Lex, 1 + Lim), using double-precision power iteration and Jacobi inversion.
double ImplicitCriticalLambda(const ImplicitBox &box, int cap) {
    const double nim[3]{box.Nim(0), box.Nim(1), box.Nim(2)}, nex[3]{box.Nex(0), box.Nex(1), box.Nex(2)};
    std::vector<double> v(box.Nodes, 0.), w(box.Nodes, 0.);
    std::mt19937 rng{7};
    std::uniform_real_distribution<double> uniform{-1., 1.};
    for (int ix = 0; ix < box.Nx; ++ix)
        for (int iy = 0; iy < box.Ny; ++iy)
            for (int iz = 0; iz < box.Nz; ++iz)
                if (box.Shape.Inside(ix, iy, iz)) v[box.Index(ix, iy, iz)] = uniform(rng);

    auto dot = [&](const std::vector<double> &a, const std::vector<double> &b) {
        double q = 0.;
        for (int ix = 0; ix < box.Nx; ++ix)
            for (int iy = 0; iy < box.Ny; ++iy)
                for (int iz = 0; iz < box.Nz; ++iz) q += a[box.Index(ix, iy, iz)] * b[box.Index(ix, iy, iz)];
        return q;
    };
    auto const solve = [&](const std::vector<double> &rhs) {
        std::fill(w.begin(), w.end(), 0.);
        for (int sweep = 0; sweep < 12; ++sweep) {
            const auto off = ImplicitApply(box, w.data(), nim);
            for (int ix = 0; ix < box.Nx; ++ix)
                for (int iy = 0; iy < box.Ny; ++iy)
                    for (int iz = 0; iz < box.Nz; ++iz) {
                        const size_t i = box.Index(ix, iy, iz);
                        int k[3];
                        box.Shape.Counts(ix, iy, iz, k);
                        // Remove the diagonal already included in Lim*w.
                        const double lim = nim[0] * k[0] + nim[1] * k[1] + nim[2] * k[2];
                        w[i] = (rhs[i] - (off[i] + lim * w[i])) / (1. - lim);
                    }
        }
        return w;
    };
    auto const rayleigh = [&] {
        auto kv = ImplicitApply(box, v.data(), nex);
        for (auto &e : kv) e = -e;
        const auto mv = ImplicitApply(box, v.data(), nim);
        return dot(kv, v) / (dot(v, v) + dot(mv, v));
    };
    double rho = 0.;
    for (int i = 0; i < cap; ++i) {
        auto kv = ImplicitApply(box, v.data(), nex);
        for (auto &e : kv) e = -e;
        v = solve(kv);
        const double norm = std::sqrt(dot(v, v));
        for (auto &e : v) e /= norm;
        if ((i + 1) % 100) continue;
        const double next = rayleigh();
        if (next - rho < 1e-8 * next) break;
        rho = next;
    }
    return 2. / std::sqrt(rayleigh());
}

// Both Laplacians must annihilate constant pressure. This catches unmatched cut legs that a
// symmetry-only check cannot see.
double ImplicitNullResidual(const ImplicitBox &box) {
    const double nim[3]{box.Nim(0), box.Nim(1), box.Nim(2)}, nex[3]{box.Nex(0), box.Nex(1), box.Nex(2)};
    std::vector<double> one(box.Nodes, 0.);
    for (int ix = 0; ix < box.Nx; ++ix)
        for (int iy = 0; iy < box.Ny; ++iy)
            for (int iz = 0; iz < box.Nz; ++iz)
                if (box.Shape.Inside(ix, iy, iz)) one[box.Index(ix, iy, iz)] = 1.;
    const auto lim = ImplicitApply(box, one.data(), nim), lex = ImplicitApply(box, one.data(), nex);
    double worst = 0.;
    for (int ix = 0; ix < box.Nx; ++ix)
        for (int iy = 0; iy < box.Ny; ++iy)
            for (int iz = 0; iz < box.Nz; ++iz) {
                if (!box.Shape.Inside(ix, iy, iz)) continue;
                const size_t i = box.Index(ix, iy, iz);
                worst = std::max({worst, std::abs(lim[i]), std::abs(lex[i])});
            }
    return worst;
}

// Paper Sec. V energy: E = 1/2 <M d,d> + 1/2 <K u^(n+1),u^n>.
double ImplicitEnergy(const ImplicitBox &box, const float *unew, const float *uold) {
    const double n[3]{box.Nim(0), box.Nim(1), box.Nim(2)};
    const double e[3]{box.Nex(0), box.Nex(1), box.Nex(2)};
    std::vector<float> d(box.Nodes, 0.f);
    for (int ix = 0; ix < box.Nx; ++ix)
        for (int iy = 0; iy < box.Ny; ++iy)
            for (int iz = 0; iz < box.Nz; ++iz) {
                const size_t i = box.Index(ix, iy, iz);
                d[i] = unew[i] - uold[i];
            }
    const auto lim_d = ImplicitApply(box, d.data(), n);
    const auto lex_new = ImplicitApply(box, unew, e);
    const double l2 = box.Lambda * box.Lambda;
    double energy = 0.;
    for (int ix = 0; ix < box.Nx; ++ix)
        for (int iy = 0; iy < box.Ny; ++iy)
            for (int iz = 0; iz < box.Nz; ++iz) {
                const size_t i = box.Index(ix, iy, iz);
                energy += .5 * double(d[i]) * (double(d[i]) + lim_d[i]) - .5 * l2 * lex_new[i] * double(uold[i]);
            }
    return energy;
}

// Measures periodic plane-wave dispersion and long-run drift at the worst passing direction.
void ImplicitRung(const ImplicitScheme &scheme, int n, int steps) {
    const double t0 = Now();
    ImplicitBox box(scheme, n, n, n);
    const int dirs[3][3] = {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}};
    static const char *const Names[3]{"axis", "face diagonal", "cube diagonal"};
    std::vector<float> seed(box.Nodes);
    auto const seed_mode = [&](const double kv[3]) {
        for (int ix = 0; ix < n; ++ix)
            for (int iy = 0; iy < n; ++iy)
                for (int iz = 0; iz < n; ++iz)
                    seed[size_t(ix) * n * n + size_t(iy) * n + iz] = float(std::cos(kv[0] * ix + kv[1] * iy + kv[2] * iz));
        box.Prev.Upload(seed.data(), seed.size() * sizeof(float));
        box.Cur.Upload(seed.data(), seed.size() * sizeof(float));
    };

    std::printf("[implicit] %s free-space dispersion on a periodic %d^3 box, lambda %.3f, %d Jacobi sweeps | %.1fs\n", scheme.Name, n, box.Lambda, scheme.Sweeps, Now() - t0);
    double worst_predict = 0., worst_xi = 0., worst_kv[3] = {0., 0., 0.}, worst_wt = 0.;
    for (int d = 0; d < 3; ++d) {
        double last_ok_wt = 0., last_ok_kv[3] = {0., 0., 0.}, reported_err = 0.;
        int reported_m = 0;
        for (int m = 1;; ++m) {
            const double k = 2. * std::numbers::pi * m / n;
            const double kv[3] = {k * dirs[d][0], k * dirs[d][1], k * dirs[d][2]};
            const double kmag = std::sqrt(kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2]);
            if (kmag > std::numbers::pi * std::numbers::sqrt3) break;
            seed_mode(kv);
            box.Step();

            const double measured = double(box.Cur.As<float>()[0]) + 1.;
            const double predicted = ImplicitTwoCosOmegaT(box.P, kv[0], kv[1], kv[2]);
            if (!(std::abs(measured) < 2.) || !(std::abs(predicted) < 2.)) break;
            const double wt_m = std::acos(measured / 2.), wt_p = std::acos(predicted / 2.);
            const double v_m = wt_m / (box.Lambda * kmag), v_p = wt_p / (box.Lambda * kmag);
            worst_predict = std::max(worst_predict, std::abs(v_m - v_p));
            if (std::abs(1. - v_m) > scheme.Eps) break;
            last_ok_wt = wt_m;
            std::copy(std::begin(kv), std::end(kv), std::begin(last_ok_kv));
            reported_err = 1. - v_m;
            reported_m = m;
        }
        const double xi = std::numbers::pi / last_ok_wt;
        std::printf("  %-14s m %2d  wT/pi %.3f  phase error %+.3f%%  ->  xi %.2f, PPW %.2f\n", Names[d], reported_m, last_ok_wt / std::numbers::pi, 100. * reported_err, xi, 2. * xi * box.Lambda);
        if (xi > worst_xi) {
            worst_xi = xi;
            worst_wt = last_ok_wt;
            std::copy(std::begin(last_ok_kv), std::end(last_ok_kv), std::begin(worst_kv));
        }
    }
    std::printf("  worst direction: xi %.2f, PPW %.2f at %.2f%% phase error | vs the relation the floats in force give, worst %.2e\n", worst_xi, 2. * worst_xi * box.Lambda, 100. * scheme.Eps, worst_predict);

    seed_mode(worst_kv);
    double drift = 0., peak = 0.;
    const double c0 = std::cos(worst_wt / 2.);
    for (int i = 0; i < steps; ++i) {
        box.Step();
        const double a = double(box.Cur.As<float>()[0]);
        const double want = std::cos((i + 1.5) * worst_wt) / c0;
        drift = std::max(drift, std::abs(a - want));
        peak = std::max(peak, std::abs(a));
    }
    std::printf("  %d steps at that mode: peak amplitude %.6f against the exact %.6f, worst deviation %.2e | %.1fs\n", steps, peak, 1. / c0, drift, Now() - t0);
}

// Absorbed energy: 1/2 <G D,D>, D = u^(n+1) - u^(n-1).
double ImplicitLoss(const ImplicitBox &box, const float *unext, const float *uprev) {
    double loss = 0.;
    for (const auto &[i, g] : box.WallG) {
        const double d = double(unext[i]) - double(uprev[i]);
        loss += .5 * g * d * d;
    }
    return loss;
}

// Mean-free independent noise projects onto unknown unstable modes without exciting the
// physical constant-pressure null mode.
void ImplicitSeedNoise(ImplicitBox &box, uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> uniform{-1.f, 1.f};
    const int ny = box.Ny, nz = box.Nz;
    auto const row = [&](int ix, int iy, int iz) { return (size_t(ix) * size_t(ny) + size_t(iy)) * size_t(nz) + size_t(iz); };
    std::vector<float> level[2];
    for (auto &v : level) {
        v.assign(box.Room, 0.f);
        // Average only physical room nodes.
        double sum = 0.;
        size_t count = 0;
        for (int ix = 0; ix < box.Nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz) {
                    if (!box.Shape.Inside(ix, iy, iz)) continue;
                    const float e = uniform(rng);
                    v[row(ix, iy, iz)] = e;
                    sum += e;
                    ++count;
                }
        const float mean = float(sum / double(std::max<size_t>(count, 1)));
        for (int ix = 0; ix < box.Nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz)
                    if (box.Shape.Inside(ix, iy, iz)) v[row(ix, iy, iz)] -= mean;
    }
    box.Seed([&](int ix, int iy, int iz) { return level[0][(size_t(ix) * ny + size_t(iy)) * nz + size_t(iz)]; }, [&](int ix, int iy, int iz) { return level[1][(size_t(ix) * ny + size_t(iy)) * nz + size_t(iz)]; });
}

// Double-precision host step using the kernels' float coefficients and initial guess.
std::vector<double> ImplicitReferenceStep(const ImplicitBox &box, const float *un, const float *up) {
    const double R[3]{box.P.R1, box.P.R2, box.P.R3}, Q[3]{box.P.Q1, box.P.Q2, box.P.Q3};
    std::vector<double> cur(box.Nodes, 0.), prev(box.Nodes, 0.), bp(box.Nodes, 0.), x(box.Nodes, 0.), xn(box.Nodes, 0.);
    std::vector<RoomImplicitTerm> wall_terms(box.Nodes);
    for (size_t i = 0; i < box.Nodes; ++i) {
        cur[i] = double(un[i]);
        prev[i] = double(up[i]);
    }
    auto const sums = [&](const std::vector<double> &u, int ix, int iy, int iz, double s[3]) {
        s[0] = s[1] = s[2] = 0.;
        for (const auto &leg : ImplicitLegs)
            if (!box.Shape.Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz)) s[leg.Class] += u[box.Index(ix + leg.Dx, iy + leg.Dy, iz + leg.Dz)];
    };
    for (int ix = 0; ix < box.Nx; ++ix) {
        for (int iy = 0; iy < box.Ny; ++iy) {
            for (int iz = 0; iz < box.Nz; ++iz) {
                const size_t i = box.Index(ix, iy, iz);
                double sn[3], sp[3];
                sums(cur, ix, iy, iz, sn);
                sums(prev, ix, iy, iz, sp);
                double b = double(box.P.Rc) * cur[i] - prev[i];
                for (int r = 0; r < 3; ++r) b += R[r] * sn[r] - Q[r] * sp[r];
                const RoomImplicitTerm t = box.WallTerms(ix, iy, iz);
                wall_terms[i] = t;
                bp[i] = b + double(t.Cn) * cur[i] + double(t.Cp) * prev[i];
                x[i] = cur[i];
            }
        }
    }
    for (int m = 0; m < box.Scheme.Sweeps; ++m) {
        for (int ix = 0; ix < box.Nx; ++ix) {
            for (int iy = 0; iy < box.Ny; ++iy) {
                for (int iz = 0; iz < box.Nz; ++iz) {
                    const size_t i = box.Index(ix, iy, iz);
                    double sx[3];
                    sums(x, ix, iy, iz, sx);
                    double v = bp[i];
                    for (int r = 0; r < 3; ++r) v -= Q[r] * sx[r];
                    xn[i] = v * double(wall_terms[i].Fd);
                }
            }
        }
        x.swap(xn);
    }
    return x;
}

void ImplicitProjectionRung(const ImplicitScheme &scheme) {
    const double t0 = Now();
    ImplicitBox box(scheme, 16, 15, 14, true, 0., {.05}, ImplicitShape::Slanted, .4);
    box.SetMeanProjection(4);
    box.Seed([](int ix, int iy, int iz) { return 1. + .01 * ix - .02 * iy + .03 * iz; }, [](int ix, int iy, int iz) { return -.5 - .02 * ix + .01 * iy + .02 * iz; });

    const auto stats = [&](const float *u) {
        double sum = 0., outside = 0.;
        for (int ix = 0; ix < box.Nx; ++ix)
            for (int iy = 0; iy < box.Ny; ++iy)
                for (int iz = 0; iz < box.Nz; ++iz) {
                    const double v = u[box.Index(ix, iy, iz)];
                    if (box.Shape.Inside(ix, iy, iz)) sum += v;
                    else outside = std::max(outside, std::abs(v));
                }
        return std::pair{sum / double(box.InsideNodes), outside};
    };

    for (int n = 0; n < 3; ++n) box.Step();
    const auto before = stats(box.Cur.As<float>());
    box.Step(); // interval four: both stored levels are projected
    const auto cur = stats(box.Cur.As<float>()), prev = stats(box.Prev.As<float>());
    std::printf("[implicit] zero-mean projection on a 16 x 15 x 14 yawed room, every 4 steps\n");
    std::printf("  mean before interval %.3e; after: current %.3e, previous %.3e; outside max %.3e | %.1fs\n", before.first, cur.first, prev.first, std::max(cur.second, prev.second), Now() - t0);
    AtLeast("mean projection waits for its interval", std::abs(before.first), 1e-6);
    AtMost("mean projection current residual", std::abs(cur.first), 2e-7);
    AtMost("mean projection previous residual", std::abs(prev.first), 2e-7);
    AtMost("mean projection preserves outside zeros", std::max(cur.second, prev.second), 0.);
}

// Checks the Sec. IV drop-ghost wall against a host step and its exact energy balance.
void ImplicitWallRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int soak) {
    const double t0 = Now();
    ImplicitBox box(scheme, nx, ny, nz, true);
    ImplicitSeedNoise(box, 12345);

    std::printf("[implicit] %s drop-ghost wall on a %d x %d x %d room, lambda %.4f, %d Jacobi sweeps, %d wall nodes (%.1f%%)\n", scheme.Name, nx, ny, nz, box.Lambda, scheme.Sweeps, box.WallNodes, 100. * double(box.WallNodes) / double(box.Room));

    const std::vector<float> un(box.Cur.As<float>(), box.Cur.As<float>() + box.Nodes);
    const std::vector<float> up(box.Prev.As<float>(), box.Prev.As<float>() + box.Nodes);
    const auto reference = ImplicitReferenceStep(box, un.data(), up.data());
    box.Step();
    const float *got = box.Cur.As<float>();
    double worst = 0., scale = 0.;
    for (size_t i = 0; i < box.Nodes; ++i) {
        worst = std::max(worst, std::abs(double(got[i]) - reference[i]));
        scale = std::max(scale, std::abs(reference[i]));
    }
    std::printf("  one step against the same step in double on the host: worst %.2e, %.2e of the level\n", worst, worst / scale);
    std::printf("  the constant field through both Laplacians: worst %.2e, which is the null space the energy needs\n", ImplicitNullResidual(box));

    const double e0 = ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>());
    const int every = std::max(1, soak / 160);
    double drift = 0.;
    for (int i = 0; i < soak; ++i) {
        box.Step();
        if ((i + 1) % every) continue;
        drift = std::max(drift, std::abs(ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>()) - e0) / std::abs(e0));
    }
    std::printf("  %d steps: worst energy drift %.2e of E0 | %.1fs\n", soak, drift, Now() - t0);
}

// Fits the drop-ghost wall offset from rigid-box Rayleigh quotients against Morse & Ingard.
void ImplicitModeRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int count) {
    const double t0 = Now();
    const double h = 0.0326857142857142, c = 343.2;
    ImplicitBox const box(scheme, nx, ny, nz, true);
    const double ts = box.Lambda * h / c;
    const int n[3] = {nx, ny, nz};
    const double nim[3]{box.Nim(0), box.Nim(1), box.Nim(2)}, nex[3]{box.Nex(0), box.Nex(1), box.Nex(2)};

    struct Mode {
        int m[3];
        double Measured;
    };
    std::vector<Mode> modes;
    for (int a = 0; a <= 5; ++a)
        for (int b = 0; b <= 5; ++b)
            for (int d = 0; d <= 5; ++d)
                if (a + b + d) modes.push_back({{a, b, d}, 0.});
    auto nominal = [&](const Mode &p, double off) {
        double q = 0.;
        for (int a = 0; a < 3; ++a) q += std::pow(p.m[a] / ((n[a] + off) * h), 2.);
        return .5 * c * std::sqrt(q);
    };
    std::sort(modes.begin(), modes.end(), [&](const Mode &p, const Mode &q) { return nominal(p, 0.) < nominal(q, 0.); });
    if (int(modes.size()) > count) modes.resize(size_t(count));

    std::vector<double> v(box.Nodes, 0.);
    for (auto &mode : modes) {
        for (int ix = 0; ix < nx; ++ix) {
            const double cx = std::cos(std::numbers::pi * mode.m[0] * (ix + .5) / nx);
            for (int iy = 0; iy < ny; ++iy) {
                const double cy = std::cos(std::numbers::pi * mode.m[1] * (iy + .5) / ny);
                for (int iz = 0; iz < nz; ++iz) v[box.Index(ix, iy, iz)] = cx * cy * std::cos(std::numbers::pi * mode.m[2] * (iz + .5) / nz);
            }
        }
        const auto lim = ImplicitApply(box, v.data(), nim), lex = ImplicitApply(box, v.data(), nex);
        double num = 0., den = 0.;
        for (int ix = 0; ix < nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz) {
                    const size_t i = box.Index(ix, iy, iz);
                    num += -box.Lambda * box.Lambda * lex[i] * v[i];
                    den += (v[i] + lim[i]) * v[i];
                }
        mode.Measured = std::acos(1. - .5 * num / den) / (2. * std::numbers::pi * ts);
    }

    // Fit one isotropic wall offset across all modes.
    auto const residual = [&](double off) {
        double worst = 0.;
        for (const auto &mode : modes) worst = std::max(worst, std::abs(mode.Measured / nominal(mode, off) - 1.));
        return worst;
    };
    double best = 0., best_res = residual(0.);
    for (int i = -2000; i <= 4000; ++i) {
        const double off = i / 4000.;
        if (residual(off) < best_res) {
            best_res = residual(off);
            best = off;
        }
    }

    std::printf("[implicit] %s rigid %d x %d x %d room, %.3f x %.3f x %.3f m at the nominal wall, lambda %.4f\n", scheme.Name, nx, ny, nz, nx * h, ny * h, nz * h, box.Lambda);
    std::printf("  %-8s %12s %12s %10s %10s\n", "mode", "measured Hz", "wall at N h", "error", "fitted");
    for (const auto &mode : modes) {
        std::printf("  %d,%d,%d    %12.5f %12.5f %+9.4f %% %+9.4f %%\n", mode.m[0], mode.m[1], mode.m[2], mode.Measured, nominal(mode, 0.), 100. * (mode.Measured / nominal(mode, 0.) - 1.), 100. * (mode.Measured / nominal(mode, best) - 1.));
    }
    std::printf("  worst against a wall at N h %.3f %%, against the image's (N - 1) h %.3f %%\n", 100. * residual(0.), 100. * residual(-1.));
    std::printf("  the modes fit a wall at (N + %.3f) h, worst %.3f %% left | %.1fs\n", best, 100. * best_res, Now() - t0);
}

// Fits amplitude decay from the first 10 dB of energy after discarding the settling fifth.
template<typename F> double ImplicitDecay(ImplicitBox &box, F &&mode, int steps) {
    box.Seed(mode, mode);
    std::vector<double> logs, at;
    const int every = std::max(1, steps / 200);
    for (int i = 0; i < steps; ++i) {
        box.Step();
        if ((i + 1) % every) continue;
        const double e = ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>());
        if (!(e > 0.) || !std::isfinite(e)) break;
        logs.push_back(std::log(e));
        at.push_back(double(i + 1));
        if (logs.front() - logs.back() > std::numbers::ln10) break;
    }
    return -.5 * ImplicitSlope(at, logs, logs.size() / 5);
}

// Checks beta, energy balance, and axial decay for the paper's real-admittance wall. Analytic
// amplitude decay is c*gamma*(2/Lx + 1/Ly + 1/Lz).
void ImplicitAbsorbRung(const ImplicitScheme &scheme, int nx, int ny, int nz, double gamma, int steps) {
    const double t0 = Now();
    ImplicitBox box(scheme, nx, ny, nz, true, 0., {gamma});
    const ImplicitShape shape(ImplicitShape::Array, nx, ny, nz);
    const double nex[3]{box.Nex(0), box.Nex(1), box.Nex(2)};
    int k[3];
    std::printf("[implicit] %s admittance wall on a %d x %d x %d room, gamma %.3f, lambda %.4f, %d distinct terms\n", scheme.Name, nx, ny, nz, gamma, box.Lambda, box.Codes);
    std::printf("  beta on a flat face %.9f, which is 1 exactly when the wiring is right\n", ImplicitWallSample(shape, 0, ny / 2, nz / 2, nex, k, true));

    ImplicitSeedNoise(box, 12345);
    const int balance = std::min(steps, 2000);
    std::vector<float> behind(box.Prev.As<float>(), box.Prev.As<float>() + box.Nodes);
    const double e0 = ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>());
    double loss = 0., drift = 0.;
    for (int i = 0; i < balance; ++i) {
        box.Step();
        const float *next = box.Cur.As<float>();
        loss += ImplicitLoss(box, next, behind.data());
        drift = std::max(drift, std::abs(ImplicitEnergy(box, next, box.Prev.As<float>()) + loss - e0) / std::abs(e0));
        const float *now = box.Prev.As<float>();
        behind.assign(now, now + box.Nodes);
    }
    std::printf("  %d steps from noise: worst |E + loss - E0| %.2e of E0, and the wall took %.1f%% of E0 out\n", balance, drift, 100. * loss / e0);

    const double side[3]{nx + ImplicitWallOffset, ny + ImplicitWallOffset, nz + ImplicitWallOffset};
    const double exact = gamma * box.Lambda * (2. / side[0] + 1. / side[1] + 1. / side[2]);
    std::printf("  %-6s %16s %16s %8s   (the wall at (N + %.2f) h)\n", "mode", "nepers a step", "analytic", "ratio", ImplicitWallOffset);
    for (int m = 1; m <= 4; ++m) {
        const double decay = ImplicitDecay(box, [&](int ix, int, int) { return std::cos(std::numbers::pi * m * (ix + .5) / nx); }, steps);
        std::printf("  %-6d %16.6e %16.6e %8.4f\n", m, decay, exact, decay / exact);
    }
    std::printf("  | %.1fs\n", Now() - t0);
}

// Compares fitted modes and decay for a yawed box with and without staircase compensation.
void ImplicitYawedRung(const ImplicitScheme &scheme, int n, double angle, double gamma, int count, int steps) {
    const double t0 = Now();
    const double h = 0.0326857142857142, c = 343.2;
    ImplicitBox box(scheme, n, n, n, true, 0., {gamma}, ImplicitShape::Slanted, angle);
    const ImplicitShape &shape = box.Shape;
    const double ts = box.Lambda * h / c;
    const double cells[3]{2. * shape.Ext[0], 2. * shape.Ext[1], 2. * shape.Ext[2]};
    const double nim[3]{box.Nim(0), box.Nim(1), box.Nim(2)}, nex[3]{box.Nex(0), box.Nex(1), box.Nex(2)};

    std::printf("[implicit] %s a box yawed %.1f degrees inside a %d^3 array, %.3f x %.3f x %.3f m, gamma %.3f\n", scheme.Name, angle * 180. / std::numbers::pi, n, cells[0] * h, cells[1] * h, cells[2] * h, gamma);
    std::printf("  %zu nodes of the array are the room (%.1f%%), %d of them wall, over %d distinct terms\n", box.InsideNodes, 100. * double(box.InsideNodes) / double(box.Room), box.WallNodes, box.Codes);

    struct Mode {
        int M[3];
        double F;
    };
    std::vector<Mode> modes;
    for (int a = 0; a <= 3; ++a)
        for (int b = 0; b <= 3; ++b)
            for (int d = 0; d <= 3; ++d) {
                if (a + b + d == 0) continue;
                double q = 0.;
                const int m[3]{a, b, d};
                for (int e = 0; e < 3; ++e) q += std::pow(m[e] / (cells[e] * h), 2.);
                modes.push_back({{a, b, d}, .5 * c * std::sqrt(q)});
            }
    std::sort(modes.begin(), modes.end(), [](const Mode &p, const Mode &q) { return p.F < q.F; });
    if (int(modes.size()) > count) modes.resize(size_t(count));

    std::vector<double> v(box.Nodes, 0.), measured;
    for (const auto &mode : modes) {
        std::fill(v.begin(), v.end(), 0.);
        for (int ix = 0; ix < n; ++ix) {
            for (int iy = 0; iy < n; ++iy) {
                for (int iz = 0; iz < n; ++iz) {
                    if (!shape.Inside(ix, iy, iz)) continue;
                    const double p[3]{ix - shape.Cen[0], iy - shape.Cen[1], iz - shape.Cen[2]};
                    double q[3], value = 1.;
                    shape.Local(p, q);
                    for (int a = 0; a < 3; ++a) value *= std::cos(std::numbers::pi * mode.M[a] * (q[a] + shape.Ext[a]) / cells[a]);
                    v[box.Index(ix, iy, iz)] = value;
                }
            }
        }
        const auto lim = ImplicitApply(box, v.data(), nim), lex = ImplicitApply(box, v.data(), nex);
        double num = 0., den = 0.;
        for (size_t i = 0; i < box.Nodes; ++i) {
            num += -box.Lambda * box.Lambda * lex[i] * v[i];
            den += (v[i] + lim[i]) * v[i];
        }
        measured.push_back(std::acos(1. - .5 * num / den) / (2. * std::numbers::pi * ts));
    }

    // Fit the wall offset before attributing the remaining error to staircasing.
    auto analytic = [&](const Mode &mode, double off) {
        double q = 0.;
        for (int a = 0; a < 3; ++a) q += std::pow(mode.M[a] / ((cells[a] + off) * h), 2.);
        return .5 * c * std::sqrt(q);
    };
    auto const residual = [&](double off) {
        double worst = 0.;
        for (size_t i = 0; i < modes.size(); ++i) worst = std::max(worst, std::abs(measured[i] / analytic(modes[i], off) - 1.));
        return worst;
    };
    double best = 0., best_res = residual(0.);
    for (int i = -4000; i <= 4000; ++i) {
        const double off = i / 1000.;
        if (residual(off) < best_res) {
            best_res = residual(off);
            best = off;
        }
    }

    std::printf("  %-8s %12s %12s %10s %10s\n", "mode", "analytic Hz", "measured Hz", "nominal", "fitted");
    for (size_t i = 0; i < modes.size(); ++i) {
        std::printf("  %d,%d,%d    %12.5f %12.5f %+9.4f %% %+9.4f %%\n", modes[i].M[0], modes[i].M[1], modes[i].M[2], analytic(modes[i], best), measured[i], 100. * (measured[i] / analytic(modes[i], 0.) - 1.), 100. * (measured[i] / analytic(modes[i], best) - 1.));
    }
    std::printf("  the modes fit a room %+.3f cells on the shape's extents, worst %.3f %% left against %.3f %% nominal\n", best, 100. * best_res, 100. * residual(0.));

    ImplicitBox raw(scheme, n, n, n, true, 0., {gamma}, ImplicitShape::Slanted, angle, false);
    const double fitted[3]{cells[0] + best, cells[1] + best, cells[2] + best};
    const double exact = gamma * box.Lambda * (2. / fitted[2] + 1. / fitted[0] + 1. / fitted[1]);
    std::printf("  %-6s %14s %8s %14s %8s   (analytic %.6e)\n", "mode", "compensated", "ratio", "staircased", "ratio", exact);
    for (int m = 1; m <= 4; ++m) {
        auto const mode = [&](int ix, int iy, int iz) {
            const double p[3]{ix - shape.Cen[0], iy - shape.Cen[1], iz - shape.Cen[2]};
            double q[3];
            shape.Local(p, q);
            return std::cos(std::numbers::pi * m * (q[2] + shape.Ext[2]) / cells[2]);
        };
        const double good = ImplicitDecay(box, mode, steps), bad = ImplicitDecay(raw, mode, steps);
        std::printf("  %-6d %14.6e %8.4f %14.6e %8.4f\n", m, good, good / exact, bad, bad / exact);
    }
    std::printf("  | %.1fs\n", Now() - t0);
}

// Trapezoidal-rule admittance at s = i(2/Ts)tan(w Ts/2).
std::complex<double> ImplicitAdmittance(const ImplicitWall &wall, double f) {
    const std::complex<double> s{0., 2. / wall.Ts * std::tan(std::numbers::pi * f * wall.Ts)};
    std::complex<double> y{0., 0.};
    for (int m = 0; m < wall.Branches(); ++m) {
        y += 1. / (wall.Def[3 * size_t(m)] * s + wall.Def[3 * size_t(m) + 1] + wall.Def[3 * size_t(m) + 2] / s);
    }
    return y;
}

// LRC stored and dissipated energy. Recover beta from the uploaded Psi so accounting uses the
// coefficients in force.
struct ImplicitWallState {
    double Stored, Loss;
};
ImplicitWallState ImplicitWallEnergy(const ImplicitBox &box, std::vector<float> &before) {
    const int mb = box.Wall.Branches(), nw = box.P.Nw;
    const RoomImplicitWall *list = box.WallList.As<RoomImplicitWall>();
    const float *vh = box.Vh.As<float>(), *gh = box.Gh.As<float>();
    const double d0 = box.Diagonal();
    ImplicitWallState out{0., 0.};
    for (int i = 0; i < nw; ++i) {
        const double beta = double(list[i].Psi) * d0 / box.Lambda;
        for (int m = 0; m < mb; ++m) {
            const double dh = box.Wall.Def[3 * size_t(m)] / box.Wall.Ts, eh = box.Wall.Def[3 * size_t(m) + 1];
            const double fh = box.Wall.Def[3 * size_t(m) + 2] * box.Wall.Ts;
            const size_t j = size_t(m) * size_t(nw) + size_t(i);
            const double v = double(vh[j]), g = double(gh[j]), mu = .5 * (v + double(before[j]));
            out.Stored += .5 * box.Lambda * beta * (dh * v * v + fh * g * g);
            out.Loss += box.Lambda * beta * eh * mu * mu;
        }
    }
    before.assign(vh, vh + size_t(mb) * size_t(nw));
    return out;
}

// Checks pure-resistance equivalence, LRC energy balance, and mode-dependent resonant decay.
void ImplicitLrcRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int steps) {
    const double t0 = Now();
    const double h = 0.0326857142857142, c = 343.2;
    const double lambda = std::min(scheme.Lambda, ImplicitCourantLimit(scheme)), ts = lambda * h / c;
    const double side[3]{nx + ImplicitWallOffset, ny + ImplicitWallOffset, nz + ImplicitWallOffset};
    const double geometry = lambda * (2. / side[0] + 1. / side[1] + 1. / side[2]);

    std::printf("[implicit] %s LRC walls on a %d x %d x %d room, lambda %.4f, %.1f kHz step rate\n", scheme.Name, nx, ny, nz, lambda, 1e-3 / ts);

    const double gamma = .05;
    ImplicitBox real(scheme, nx, ny, nz, true, 0., {gamma});
    ImplicitBox same(scheme, nx, ny, nz, true, 0., ImplicitWall{0., ts, {0., 1. / gamma, 0.}});
    ImplicitSeedNoise(real, 12345);
    ImplicitSeedNoise(same, 12345);
    for (int i = 0; i < 64; ++i) {
        real.Step();
        same.Step();
    }
    double worst = 0., scale = 0.;
    const float *a = real.Cur.As<float>(), *b = same.Cur.As<float>();
    for (size_t i = 0; i < real.Nodes; ++i) {
        worst = std::max(worst, std::abs(double(a[i]) - double(b[i])));
        scale = std::max(scale, std::abs(double(a[i])));
    }
    std::printf("  a branch of pure resistance against the gamma = %.2f it equals, 64 steps: worst %.2e of the level\n", gamma, worst / scale);

    const double f0 = 200., d = .02, e = d * 2. * std::numbers::pi * f0, f = d * std::pow(2. * std::numbers::pi * f0, 2.);
    const ImplicitWall material{0., ts, {d, e, f, 0., 100., 0.}};
    ImplicitBox box(scheme, nx, ny, nz, true, 0., material);
    std::printf("  %d branches, a resonance at %.0f Hz beside a resistive floor, %d wall nodes\n", material.Branches(), f0, box.P.Nw);

    ImplicitSeedNoise(box, 12345);
    std::vector<float> before(size_t(material.Branches()) * size_t(box.P.Nw), 0.f);
    const double e0 = ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>());
    double loss = 0., drift = 0., held = 0.;
    const int balance = std::min(steps, 2000);
    for (int i = 0; i < balance; ++i) {
        box.Step();
        const ImplicitWallState w = ImplicitWallEnergy(box, before);
        loss += w.Loss;
        held = std::max(held, w.Stored);
        drift = std::max(drift, std::abs(ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>()) + w.Stored + loss - e0) / std::abs(e0));
    }
    std::printf("  %d steps from noise: worst |E + wall + loss - E0| %.2e of E0, the branches holding at most %.2e of E0 and having dissipated %.1f%%\n", balance, drift, held / e0, 100. * loss / e0);

    std::printf("  %-6s %10s %10s %16s %16s %8s\n", "mode", "Hz", "Re{Y}", "nepers a step", "analytic", "ratio");
    for (int m = 1; m <= 4; ++m) {
        const double hz = .5 * c * m / (side[0] * h);
        const double y = ImplicitAdmittance(material, hz).real();
        const double decay = ImplicitDecay(box, [&](int ix, int, int) { return std::cos(std::numbers::pi * m * (ix + .5) / nx); }, steps);
        std::printf("  %-6d %10.3f %10.5f %16.6e %16.6e %8.4f\n", m, hz, y, decay, y * geometry, decay / (y * geometry));
    }
    std::printf("  | %.1fs\n", Now() - t0);
}

struct ImplicitVoxelisedScene {
    ImplicitModel Model;
    ImplicitVoxels Voxels;
    double Seconds;
};

ImplicitVoxelisedScene LoadImplicitVoxelisedScene(const std::string &path, double h) {
    const double t0 = Now();
    ImplicitModel model = ImplicitLoadModel(path, h, 2);
    const ImplicitMesh mesh(model.V, model.F);
    ImplicitVoxels voxels = ImplicitVoxelise(model, mesh);
    return {std::move(model), std::move(voxels), Now() - t0};
}

void ImplicitSceneRung(const ImplicitScheme &scheme, const std::string &path, double h, double gamma, const ImplicitVoxelisedScene &scene) {
    const double t0 = Now();
    const ImplicitModel &model = scene.Model;
    const ImplicitVoxels &vox = scene.Voxels;
    const int nx = vox.N[0], ny = vox.N[1], nz = vox.N[2];
    const size_t nodes = size_t(nx) * size_t(ny) * size_t(nz);
    const ImplicitShape shape(vox);

    const double nex[3]{ImplicitNex(scheme, 0), ImplicitNex(scheme, 1), ImplicitNex(scheme, 2)};
    const double nim[3]{ImplicitNim(scheme, 0), ImplicitNim(scheme, 1), ImplicitNim(scheme, 2)};
    const double lambda = std::min(scheme.Lambda, ImplicitCourantLimit(scheme));
    const RoomImplicitParams p = ImplicitCoefficients(scheme, lambda, nx, ny, nz);
    const double d0 = 1. - (6. * nim[0] + 12. * nim[1] + 8. * nim[2]), l2 = lambda * lambda;

    struct Sample {
        int K[3], Mat;
        double Beta;
    };
    std::vector<Sample> wall;
    size_t room = 0, axis_wall = 0, negative = 0, interior_cut = 0;
    double lo = 1e30, hi = -1e30, mean = 0., total = 0., clamped = 0.;
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int iz = 0; iz < nz; ++iz) {
                const size_t i = vox.At(ix, iy, iz);
                room += size_t(vox.Filled[i] != 0);
                if (!vox.Filled[i] || !vox.Blocked[i]) continue;
                int k[3]{0, 0, 0};
                double n[3], beta = 0.;
                shape.Normal(ix, iy, iz, n);
                for (const auto &leg : ImplicitLegs) {
                    if (!shape.Blocked(ix, iy, iz, leg.Dx, leg.Dy, leg.Dz)) {
                        ++k[leg.Class];
                        continue;
                    }
                    beta += nex[leg.Class] * (n[0] * leg.Dx + n[1] * leg.Dy + n[2] * leg.Dz);
                    // Count cut legs whose opposite endpoint is also live.
                    const int jx = ix + leg.Dx, jy = iy + leg.Dy, jz = iz + leg.Dz;
                    if (jx >= 0 && jy >= 0 && jz >= 0 && jx < nx && jy < ny && jz < nz && vox.Filled[vox.At(jx, jy, jz)]) ++interior_cut;
                }
                if (k[0] < 6) ++axis_wall;
                lo = std::min(lo, beta);
                hi = std::max(hi, beta);
                if (beta < 0.) {
                    ++negative;
                    clamped -= beta;
                    beta = 0.;
                }
                mean += beta;
                total += beta;
                wall.push_back({{k[0], k[1], k[2]}, vox.Mat[i], beta});
            }
        }
    }
    mean /= double(std::max<size_t>(wall.size(), 1));

    // Material participates in the term key because it changes the wall diagonal.
    auto const distinct = [&](int bits) {
        const double step = bits && hi > 0. ? hi / double((1 << bits) - 1) : 0.;
        std::vector<std::array<float, 4>> t;
        t.reserve(wall.size());
        for (const auto &w : wall) {
            const double beta = step > 0. ? std::round(w.Beta / step) * step : w.Beta;
            const double d = 1. - (nim[0] * w.K[0] + nim[1] * w.K[1] + nim[2] * w.K[2]);
            const double ex = nex[0] * w.K[0] + nex[1] * w.K[1] + nex[2] * w.K[2];
            const double g = .5 * lambda * beta * gamma;
            t.push_back({float(w.Mat), float((2. * d - l2 * ex) / d0 - double(p.Rc)), float(1. - (d - g) / d0), float(d0 / (d + g))});
        }
        std::sort(t.begin(), t.end());
        return size_t(std::unique(t.begin(), t.end()) - t.begin());
    };

    std::printf("[implicit] %s voxelising %s at %.0f mm: %d x %d x %d, %.3fM nodes, %d triangles over %zu materials | %.1fs\n", scheme.Name, std::filesystem::path{path}.parent_path().filename().string().c_str(), 1e3 * h, nx, ny, nz, double(nodes) / 1e6, int(model.F.rows()), model.Materials.size(), scene.Seconds);
    std::printf("  the fill reaches %.3fM nodes, %.1f%% of the box — the other %.1f%% is outside, ended by the block's word\n", double(room) / 1e6, 100. * double(room) / double(nodes), 100. * (1. - double(room) / double(nodes)));
    std::printf("  wall nodes %zu over 26 directions, %zu over the 6 a reference voxeliser marks (%.1f%% of them)\n", wall.size(), axis_wall, 100. * double(axis_wall) / double(std::max<size_t>(wall.size(), 1)));
    std::printf("  beta %.4f to %.4f, mean %.4f (a flat wall is 1) — %zu came out negative and went rigid, %.4f%% of the wall's absorption\n", lo, hi, mean, negative, 100. * clamped / (total + clamped));
    std::printf("  distinct terms %zu exact, %zu with beta to 8 bits (the production code keeps exact uint indices)\n", distinct(0), distinct(8));
    std::printf("  %zu cut legs join two nodes that are both in the room, which no zero across the wall can drop | %.1fs\n", interior_cut, scene.Seconds + Now() - t0);
}

// End-to-end mesh-room operator, stability, energy, and timing check.
void ImplicitRoomRung(const ImplicitScheme &scheme, const std::string &path, double h, int steps, int reps, const ImplicitVoxels &vox) {
    const double t0 = Now();
    ImplicitBox box(scheme, vox.N[0], vox.N[1], vox.N[2], true, 0., {}, ImplicitShape::Voxels, 0., true, &vox);
    std::printf("[implicit] %s stepping %s at %.0f mm: %.3fM nodes, %zu of them room, %d wall, %d distinct terms\n", scheme.Name, std::filesystem::path{path}.parent_path().filename().string().c_str(), 1e3 * h, double(box.Room) / 1e6, box.InsideNodes, box.WallNodes, box.Codes);

    ImplicitSeedNoise(box, 12345);
    const std::vector<float> un(box.Cur.As<float>(), box.Cur.As<float>() + box.Nodes);
    const std::vector<float> up(box.Prev.As<float>(), box.Prev.As<float>() + box.Nodes);
    const auto reference = ImplicitReferenceStep(box, un.data(), up.data());
    box.Step();
    const float *got = box.Cur.As<float>();
    double worst = 0., scale = 0.;
    for (size_t i = 0; i < box.Nodes; ++i) {
        worst = std::max(worst, std::abs(double(got[i]) - reference[i]));
        scale = std::max(scale, std::abs(reference[i]));
    }
    std::printf("  one step against the same step in double on the host: worst %.2e, %.2e of the level\n", worst, worst / scale);
    std::printf("  the constant field through both Laplacians: worst %.2e, which is the null space the energy needs\n", ImplicitNullResidual(box));

    size_t interior_cut = 0;
    for (int ix = 0; ix < vox.N[0]; ++ix)
        for (int iy = 0; iy < vox.N[1]; ++iy)
            for (int iz = 0; iz < vox.N[2]; ++iz) {
                const size_t i = vox.At(ix, iy, iz);
                if (!vox.Filled[i] || !vox.Blocked[i]) continue;
                for (const auto &leg : ImplicitLegs) {
                    const int jx = ix + leg.Dx, jy = iy + leg.Dy, jz = iz + leg.Dz;
                    if (jx < 0 || jy < 0 || jz < 0 || jx >= vox.N[0] || jy >= vox.N[1] || jz >= vox.N[2]) continue;
                    if ((vox.Blocked[i] & (1u << leg.Bit)) && vox.Filled[vox.At(jx, jy, jz)]) ++interior_cut;
                }
            }

    const double free_limit = std::min(scheme.Lambda, ImplicitCourantLimit(scheme));
    std::printf("  %zu cut legs join two nodes both in the room, which only the mask can drop\n", interior_cut);
    std::printf("  %-10s %14s %22s\n", "lambda", "energy drift", "growth, dB per 1000");
    double holds = 0.;
    for (const double scale : {1.0, 0.95, 0.9}) {
        ImplicitBox leg(scheme, vox.N[0], vox.N[1], vox.N[2], true, free_limit * scale, {}, ImplicitShape::Voxels, 0., true, &vox);
        ImplicitSeedNoise(leg, 12345);
        const double e0 = ImplicitEnergy(leg, leg.Cur.As<float>(), leg.Prev.As<float>());
        const int every = std::max(1, steps / 40);
        std::vector<double> logs, at;
        double drift = 0., m = 0.;
        for (int i = 0; i < steps; ++i) {
            leg.Step();
            if ((i + 1) % every) continue;
            const float *u = leg.Cur.As<float>();
            for (size_t j = 0; j < leg.Nodes; j += 97) m = std::max(m, std::abs(double(u[j])));
            if (!(m > 0.) || !std::isfinite(m)) break;
            logs.push_back(std::log(m));
            at.push_back(double(i + 1));
            m = 0.;
            drift = std::max(drift, std::abs(ImplicitEnergy(leg, u, leg.Prev.As<float>()) - e0) / std::abs(e0));
        }
        const double db = 1000. * ImplicitSlope(at, logs) * 20. / std::numbers::ln10;
        const bool held = db < 0.02;
        if (held && holds == 0.) holds = free_limit * scale;
        std::printf("  %-10.4f %14.2e %+21.4f  %s\n", free_limit * scale, drift, db, held ? "holds" : "GROWS");
    }
    if (holds > 0.) std::printf("  largest holding: lambda %.4f, %.0f%% of the free-space limit\n", holds, 100. * holds / free_limit);

    auto const measure = [&](bool wall) { return TimeGpu(reps, [&] { box.Step(wall); }); };
    double walled = 1e30, bare = 1e30;
    for (int round = 0; round < 3; ++round) {
        walled = std::min(walled, measure(true));
        bare = std::min(bare, measure(false));
    }
    std::printf("  a step %.4f ms against %.4f without the geometry (+%.1f%%), %.1f ps a node-step | %.1fs\n", 1e3 * walled, 1e3 * bare, 100. * (walled / bare - 1.), 1e12 * walled / double(box.Room), Now() - t0);
}

// Direct-DFT band magnitudes using one rotation recurrence per bin.
std::vector<double> RenderBandDb(const std::vector<double> &x, double srate, double f0, double df, int bins) {
    std::vector<double> db(size_t(bins), 0.);
    for (int k = 0; k < bins; ++k) {
        const double w = 2. * std::numbers::pi * (f0 + df * k) / srate;
        const double cr = std::cos(w), ci = -std::sin(w);
        double pr = 1., pj = 0., re = 0., im = 0.;
        for (const double v : x) {
            re += v * pr;
            im += v * pj;
            const double nr = pr * cr - pj * ci;
            pj = pr * ci + pj * cr;
            pr = nr;
        }
        db[size_t(k)] = 10. * std::log10(re * re + im * im + 1e-300);
    }
    return db;
}

// Third-octave smoothing reduces sensitivity to sub-cell source and receiver shifts.
std::vector<double> RenderSmoothDb(const std::vector<double> &db, double f0, double df) {
    const double half = std::pow(2., 1. / 6.);
    const int bins = int(db.size());
    std::vector<double> out(db.size());
    for (int k = 0; k < bins; ++k) {
        const double f = f0 + df * k;
        const int klo = std::max(0, int(std::ceil((f / half - f0) / df)));
        const int khi = std::min(bins - 1, int(std::floor((f * half - f0) / df)));
        double power = 0.;
        for (int j = klo; j <= khi; ++j) power += std::pow(10., db[size_t(j)] / 10.);
        out[size_t(k)] = 10. * std::log10(power / double(khi - klo + 1));
    }
    return out;
}

// Band-limited Schroeder T30 with raised-cosine edges to avoid brick-wall ringing.
double RenderBandT30(const std::vector<double> &x, double srate) {
    constexpr double Df = .5;
    const int bins = int((450. - 30.) / Df) + 1;
    std::vector<double> y(x.size(), 0.);
    for (int k = 0; k < bins; ++k) {
        const double f = 30. + Df * k;
        double taper = 1.;
        if (f < 60.) taper = .5 - .5 * std::cos(std::numbers::pi * (f - 30.) / 30.);
        else if (f > 350.) taper = .5 + .5 * std::cos(std::numbers::pi * (f - 350.) / 100.);
        const double w = 2. * std::numbers::pi * f / srate;
        const double cr = std::cos(w), ci = -std::sin(w);
        double pr = 1., pj = 0., re = 0., im = 0.;
        for (const double v : x) {
            re += v * pr;
            im += v * pj;
            const double nr = pr * cr - pj * ci;
            pj = pr * ci + pj * cr;
            pr = nr;
        }
        pr = 1.;
        pj = 0.;
        for (double &v : y) {
            v += taper * (re * pr + im * pj);
            const double nr = pr * cr - pj * ci;
            pj = pr * ci + pj * cr;
            pr = nr;
        }
    }
    std::vector<double> tail(y.size() + 1, 0.);
    for (size_t n = y.size(); n > 0; --n) tail[n - 1] = tail[n] + y[n - 1] * y[n - 1];
    double t5 = -1., t35 = -1.;
    for (size_t n = 0; n < y.size(); ++n) {
        const double db = 10. * std::log10(tail[n] / tail[0] + 1e-300);
        if (t5 < 0. && db < -5.) t5 = double(n) / srate;
        if (db < -35.) {
            t35 = double(n) / srate;
            break;
        }
    }
    return t5 >= 0. && t35 >= 0. ? 2. * (t35 - t5) : 0.;
}

// Builds an outward-wound rotated box and maps one local point into node coordinates.
ImplicitModel ImplicitRotatedBox(const double dims[3], const double rot[3][3], double h, int margin, const double local[3], double node[3]) {
    ImplicitModel model;
    model.Materials = {"wall"};
    model.Mat.assign(12, 0);
    Eigen::MatrixXd v(8, 3);
    for (int corner = 0; corner < 8; ++corner)
        for (int a = 0; a < 3; ++a) {
            double x = 0.;
            for (int b = 0; b < 3; ++b) x += rot[a][b] * (corner >> b & 1 ? .5 : -.5) * dims[b];
            v(corner, a) = x;
        }
    for (int a = 0; a < 3; ++a) {
        double lo = 1e30, hi = -1e30, world = 0.;
        for (int corner = 0; corner < 8; ++corner) {
            lo = std::min(lo, v(corner, a));
            hi = std::max(hi, v(corner, a));
        }
        for (int corner = 0; corner < 8; ++corner) v(corner, a) = (v(corner, a) - lo) / h + margin;
        model.N[a] = int(std::ceil((hi - lo) / h)) + 2 * margin + 1;
        model.Source[a] = -lo / h + margin;
        for (int b = 0; b < 3; ++b) world += rot[a][b] * local[b];
        node[a] = (world - lo) / h + margin;
    }
    model.V = std::move(v);
    model.F = ImplicitBoxFaces();
    return model;
}

// Measures the R111 render's marginal line after resetting both means at the end of the pulse.
void ImplicitMarginalRung(const ImplicitScheme &base, int cap) {
    const double t0 = Now(), h = .08, c0 = 343.2, gamma = .05;
    const double dims[3]{4. * std::sqrt(5.), 4. * std::numbers::sqrt3, 4.};
    const double lambda = .95 * std::min(base.Lambda, ImplicitCourantLimit(base));
    const double ts = lambda * h / c0;
    int steps = int(std::lround(2. / ts));
    if (cap > 0) steps = std::min(steps, cap);
    if (steps < 512) throw std::runtime_error("--marginal needs at least 512 steps");
    const int reset = 256, sample_every = 32;

    const double s2 = 1. / std::numbers::sqrt2, s3 = 1. / std::numbers::sqrt3, q = std::numbers::sqrt2 / std::numbers::sqrt3;
    const double r110[3][3]{{s2, -s2, 0.}, {s2, s2, 0.}, {0., 0., 1.}};
    const double ry[3][3]{{s3, 0., q}, {0., 1., 0.}, {-q, 0., s3}};
    double r111[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            r111[i][j] = 0.;
            for (int k = 0; k < 3; ++k) r111[i][j] += r110[i][k] * ry[k][j];
        }
    const double rcv_local[3]{.5 * dims[0] - .5, .5 * dims[1] - .5, .5 * dims[2] - .5};
    double rcv[3];
    const ImplicitModel model = ImplicitRotatedBox(dims, r111, h, 2, rcv_local, rcv);
    const ImplicitMesh mesh(model.V, model.F);
    const ImplicitVoxels vox = ImplicitVoxelise(model, mesh);

    const double sigma = 1.06e-3, pulse_t0 = 6. * sigma;
    std::vector<float> sig(size_t(steps), 0.f);
    for (int n = 0; n < steps; ++n) {
        const double t = n * ts;
        sig[size_t(n)] = float(-(t - pulse_t0) / sigma * std::exp(-(t - pulse_t0) * (t - pulse_t0) / (2. * sigma * sigma)));
    }

    struct Result {
        int Sweeps, Every;
        double Growth, MaxMean, FinalMean, T30;
        std::vector<double> Ir, Db;
    };
    constexpr double F0 = 30., F1 = 400., Df = .5;
    constexpr int Bins = int((F1 - F0) / Df) + 1;
    const auto run = [&](int sweeps, int every) {
        const double leg_t0 = Now();
        ImplicitScheme scheme = base;
        scheme.Sweeps = sweeps;
        ImplicitBox box(scheme, vox.N[0], vox.N[1], vox.N[2], true, lambda, {gamma}, ImplicitShape::Voxels, 0., true, &vox);
        const int sn[3]{int(std::lround(model.Source[0])), int(std::lround(model.Source[1])), int(std::lround(model.Source[2]))};
        const int rn[3]{int(std::lround(rcv[0])), int(std::lround(rcv[1])), int(std::lround(rcv[2]))};
        box.SetIo({int(box.Index(sn[0], sn[1], sn[2]))}, sig, {int(box.Index(rn[0], rn[1], rn[2]))}, steps);
        std::vector<double> at, magnitude;
        for (int n = 0; n < steps; ++n) {
            box.StepIo(n);
            if (n + 1 == reset) box.SetMeanProjection(every ? every : steps + 1);
            if (n + 1 < reset || (n + 1 - reset) % sample_every) continue;
            const double m = std::max(std::abs(box.MeanLevel(box.Cur)), std::abs(box.MeanLevel(box.Prev)));
            if (m > 0. && std::isfinite(m)) {
                at.push_back((n + 1 - reset) * ts);
                magnitude.push_back(m);
            }
        }

        // Four-readback maxima suppress beating between the two marginal lines.
        std::vector<double> envelope_t, envelope_log;
        constexpr size_t Window = 4;
        for (size_t first = 0; first < magnitude.size(); first += Window) {
            const size_t last = std::min(first + Window, magnitude.size());
            double peak = 0.;
            for (size_t i = first; i < last; ++i) peak = std::max(peak, magnitude[i]);
            if (peak > 0.) {
                envelope_t.push_back(at[last - 1]);
                envelope_log.push_back(std::log(peak));
            }
        }
        const double growth = every ? 0. : 20. * ImplicitSlope(envelope_t, envelope_log, envelope_log.size() / 3) / std::numbers::ln10;
        const float *out = box.Out.As<float>();
        Result result{sweeps, every, growth, magnitude.empty() ? 0. : *std::max_element(magnitude.begin(), magnitude.end()), magnitude.empty() ? 0. : magnitude.back(), 0., std::vector<double>(out, out + steps), {}};
        result.T30 = RenderBandT30(result.Ir, 1. / ts);
        result.Db = RenderSmoothDb(RenderBandDb(result.Ir, 1. / ts, F0, Df, Bins), F0, Df);
        const std::string projection = every ? std::to_string(every) : "off";
        std::printf("  P %-2d project %-4s  growth %+8.3f dB/s  max mean %.3e  final %.3e  T30 %.3f s | %.1fs\n", sweeps, projection.c_str(), growth, result.MaxMean, result.FinalMean, result.T30, Now() - leg_t0);
        std::fflush(stdout);
        return result;
    };

    std::printf("[implicit marginal] R111 %.1fM-node room at %.0f mm, %d steps (%.3f s), source cleared and means reset at step %d\n", double(size_t(vox.N[0]) * vox.N[1] * vox.N[2]) / 1e6, 1e3 * h, steps, steps * ts, reset);
    std::vector<Result> sweeps;
    for (const int p : {7, 6, 8, 9, 10}) sweeps.push_back(run(p, 0));
    const Result &baseline = sweeps.front();
    std::printf("  projection cadence against P 7 unprojected receiver:\n");
    for (const int every : {1024, 256, 64}) {
        const Result projected = run(base.Sweeps, every);
        double gain = 0.;
        for (int k = 0; k < Bins; ++k) gain += projected.Db[size_t(k)] - baseline.Db[size_t(k)];
        gain /= double(Bins);
        double mean = 0., worst = 0.;
        for (int k = 0; k < Bins; ++k) {
            const double d = std::abs(projected.Db[size_t(k)] - baseline.Db[size_t(k)] - gain);
            mean += d;
            worst = std::max(worst, d);
        }
        std::printf("    every %-4d  T30 %+7.3f%%  spectrum mean %.4f dB, worst %.4f dB\n", every, 100. * (projected.T30 / baseline.T30 - 1.), mean / double(Bins), worst);
    }
    std::printf("  | %.1fs\n", Now() - t0);
}

// Reproduces Smits & Bilbao Fig. 5 at R100, R110, and R111 orientations, comparing compensated
// and raw staircase decay against the validated grid-aligned explicit solver.
void ImplicitRenderRung(const ImplicitScheme &scheme, double h, double seconds, double gamma, int cap, int project_every) {
    const double t0 = Now();
    const double dims[3]{4. * std::sqrt(5.), 4. * std::numbers::sqrt3, 4.};
    const double lambda = .95 * std::min(scheme.Lambda, ImplicitCourantLimit(scheme));
    const double c0 = 343.2;
    const double ts = lambda * h / c0;
    int steps = int(std::lround(seconds / ts));
    if (cap > 0) steps = std::min(steps, cap);
    const double length = steps * ts;

    constexpr double F0 = 30., F1 = 400., Df = .5;
    constexpr int Bins = int((F1 - F0) / Df) + 1;
    const double sigma = 1.06e-3, t0s = 6. * sigma;
    const auto pulse = [&](double t) { return -(t - t0s) / sigma * std::exp(-(t - t0s) * (t - t0s) / (2. * sigma * sigma)); };

    const std::string project_desc = project_every ? "every " + std::to_string(project_every) + " steps" : "off";
    std::printf("[implicit render] %s the rotated room of Smits & Bilbao 2025 Fig. 5: %.3f x %.3f x %.3f m at %.0f mm, gamma %.3f, %.2f s to a corner receiver, mean projection %s\n", scheme.Name, dims[0], dims[1], dims[2], 1e3 * h, gamma, length, project_desc.c_str());
    std::fflush(stdout);
    std::filesystem::create_directories("room");
    nlohmann::json meta = nlohmann::json::array();
    const std::string project_tag = project_every ? "-p" + std::to_string(project_every) : "";
    const auto write_ir = [&](const std::string &name, const std::vector<double> &ir, double srate) {
        const std::string file = "room/implicit-render" + project_tag + "-" + name + ".bin";
        std::ofstream{file, std::ofstream::binary}.write(reinterpret_cast<const char *>(ir.data()), std::streamsize(ir.size() * sizeof(double)));
        meta.push_back({{"name", name}, {"file", file}, {"srate", srate}, {"samples", ir.size()}});
    };

    int cells[3];
    for (int a = 0; a < 3; ++a) cells[a] = int(std::lround(dims[a] / h));
    const double ref_ts = (0.999 / std::numbers::sqrt3) * h / c0;
    const int ref_steps = int(std::lround(length / ref_ts));
    Box box;
    RoomScene scene = SealedBox(cells, 2, ref_steps, box);
    scene.H = h;
    scene.Ts = scene.L * h / scene.C;
    scene.Srate = 1. / scene.Ts;
    scene.MatBranches.assign(1, 1);
    scene.MatDef = {0., 1. / gamma, 0.};
    for (size_t i = 0; i < scene.BnIxyz.size(); ++i) {
        const int idx = scene.BnIxyz[i];
        const int iz = idx % scene.Nz, iy = (idx / scene.Nz) % scene.Ny, ix = idx / (scene.Nz * scene.Ny);
        if (ix < box.X0 || ix > box.X1 || iy < box.Y0 || iy > box.Y1 || iz < box.Z0 || iz > box.Z1) continue;
        scene.MatBn[i] = 0;
        scene.SafBn[i] = 6. - double(std::popcount(uint32_t(scene.AdjBn[i])));
        ++scene.NumLossy;
    }
    const int off = int(std::lround(.5 / h));
    scene.InIxyz = {int(NodeIndex(scene, box.X0 + (cells[0] - 1) / 2, box.Y0 + (cells[1] - 1) / 2, box.Z0 + (cells[2] - 1) / 2))};
    scene.InSigs.assign(size_t(ref_steps), 0.);
    for (int n = 0; n < ref_steps; ++n) scene.InSigs[size_t(n)] = pulse(n * scene.Ts);
    scene.NumReceivers = 1;
    scene.CornersPerReceiver = 1;
    scene.OutIxyz = {int(NodeIndex(scene, box.X1 - off, box.Y1 - off, box.Z1 - off))};
    scene.OutAlpha = {1.};
    auto const ref = RenderRoomScene(scene, ref_steps);
    const double ref_t30 = RenderBandT30(ref, scene.Srate);
    const auto ref_db = RenderSmoothDb(RenderBandDb(ref, scene.Srate, F0, Df, Bins), F0, Df);
    write_ir("slf-r100", ref, scene.Srate);
    std::printf("  SLF reference %d x %d x %d, %d steps at %.0f Hz: T30 %.3f s over %.0f-%.0f Hz | %.1fs\n", scene.Nx, scene.Ny, scene.Nz, ref_steps, scene.Srate, ref_t30, F0, F1, Now() - t0);
    std::fflush(stdout);

    const double s2 = 1. / std::numbers::sqrt2, s3 = 1. / std::numbers::sqrt3, q = std::numbers::sqrt2 / std::numbers::sqrt3;
    const double eye[3][3]{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
    const double r110[3][3]{{s2, -s2, 0.}, {s2, s2, 0.}, {0., 0., 1.}};
    const double ry[3][3]{{s3, 0., q}, {0., 1., 0.}, {-q, 0., s3}};
    double r111[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            r111[i][j] = 0.;
            for (int k = 0; k < 3; ++k) r111[i][j] += r110[i][k] * ry[k][j];
        }

    struct Leg {
        const char *Name;
        const double (*Rot)[3];
        bool Comp;
    };
    const Leg legs[]{{"r100", eye, true}, {"r110", r110, true}, {"r111", r111, true}, {"r110-raw", r110, false}, {"r111-raw", r111, false}};
    std::printf("  %-9s %9s %9s %8s %9s %10s %10s\n", "leg", "nodes", "wall", "T30 s", "vs ref", "mean dB", "worst dB");
    for (const Leg &leg : legs) {
        const double t1 = Now();
        const double rcv_local[3]{.5 * dims[0] - .5, .5 * dims[1] - .5, .5 * dims[2] - .5};
        double rcv[3];
        const ImplicitModel model = ImplicitRotatedBox(dims, leg.Rot, h, 2, rcv_local, rcv);
        const ImplicitMesh mesh(model.V, model.F);
        const ImplicitVoxels vox = ImplicitVoxelise(model, mesh);
        ImplicitBox ib(scheme, vox.N[0], vox.N[1], vox.N[2], true, lambda, {gamma}, ImplicitShape::Voxels, 0., leg.Comp, &vox);
        const int sn[3]{int(std::lround(model.Source[0])), int(std::lround(model.Source[1])), int(std::lround(model.Source[2]))};
        const int rn[3]{int(std::lround(rcv[0])), int(std::lround(rcv[1])), int(std::lround(rcv[2]))};
        if (!ib.Shape.Inside(sn[0], sn[1], sn[2]) || !ib.Shape.Inside(rn[0], rn[1], rn[2])) throw std::runtime_error("render source or receiver landed outside the room");
        std::vector<float> sig(size_t(steps), 0.f);
        for (int n = 0; n < steps; ++n) sig[size_t(n)] = float(pulse(n * ts));
        ib.SetIo({int(ib.Index(sn[0], sn[1], sn[2]))}, sig, {int(ib.Index(rn[0], rn[1], rn[2]))}, steps);
        if (project_every) ib.SetMeanProjection(project_every);
        for (int n = 0; n < steps; ++n) ib.StepIo(n);
        const float *out = ib.Out.As<float>();
        const std::vector<double> ir(out, out + steps);
        const double t30 = RenderBandT30(ir, 1. / ts);
        const auto db = RenderSmoothDb(RenderBandDb(ir, 1. / ts, F0, Df, Bins), F0, Df);
        double gain = 0.;
        for (int k = 0; k < Bins; ++k) gain += db[size_t(k)] - ref_db[size_t(k)];
        gain /= double(Bins);
        double mean = 0., worst = 0.;
        for (int k = 0; k < Bins; ++k) {
            const double d = std::abs(db[size_t(k)] - ref_db[size_t(k)] - gain);
            mean += d;
            worst = std::max(worst, d);
        }
        mean /= double(Bins);
        write_ir(leg.Name, ir, 1. / ts);
        std::printf("  %-9s %8.1fM %9d %8.3f %+8.1f%% %10.2f %10.2f | %.1fs\n", leg.Name, double(ib.Room) / 1e6, ib.WallNodes, t30, 100. * (t30 / ref_t30 - 1.), mean, worst, Now() - t1);
        std::fflush(stdout);
    }
    const std::string meta_file = "room/implicit-render" + project_tag + ".json";
    std::ofstream{meta_file} << meta.dump() << '\n';
    std::printf("  wrote room/implicit-render%s-*.bin (float64) and %s | %.1fs\n", project_tag.c_str(), meta_file.c_str(), Now() - t0);
}

// Surveys beta, exact term counts, and outside volume on closed-form geometries.
void ImplicitGeometryRung(const ImplicitScheme &scheme, const ImplicitShape &shape, double gamma) {
    const double t0 = Now();
    const double nex[3]{ImplicitNex(scheme, 0), ImplicitNex(scheme, 1), ImplicitNex(scheme, 2)};
    const double nim[3]{ImplicitNim(scheme, 0), ImplicitNim(scheme, 1), ImplicitNim(scheme, 2)};
    const double lambda = std::min(scheme.Lambda, ImplicitCourantLimit(scheme));
    const RoomImplicitParams p = ImplicitCoefficients(scheme, lambda, shape.N[0], shape.N[1], shape.N[2]);
    const double d0 = 1. - (6. * nim[0] + 12. * nim[1] + 8. * nim[2]), l2 = lambda * lambda;

    struct Sample {
        int K[3];
        double Beta;
    };
    std::vector<Sample> wall;
    size_t inside = 0, axis_wall = 0, negative = 0;
    double lo = 1e30, hi = -1e30, mean = 0.;
    for (int ix = 0; ix < shape.N[0]; ++ix) {
        for (int iy = 0; iy < shape.N[1]; ++iy) {
            for (int iz = 0; iz < shape.N[2]; ++iz) {
                if (!shape.Inside(ix, iy, iz)) continue;
                ++inside;
                int k[3];
                const double beta = ImplicitWallSample(shape, ix, iy, iz, nex, k, true);
                if (k[0] == 6 && k[1] == 12 && k[2] == 8) continue;
                if (k[0] < 6) ++axis_wall;
                lo = std::min(lo, beta);
                hi = std::max(hi, beta);
                mean += beta;
                negative += size_t(beta < 0.);
                wall.push_back({{k[0], k[1], k[2]}, beta});
            }
        }
    }
    const size_t nodes = size_t(shape.N[0]) * shape.N[1] * shape.N[2];
    mean /= double(std::max<size_t>(wall.size(), 1));

    auto const distinct = [&](int bits) {
        const double step = bits && hi > 0. ? hi / double((1 << bits) - 1) : 0.;
        std::vector<std::array<float, 3>> t;
        t.reserve(wall.size());
        for (const auto &w : wall) {
            const double beta = step > 0. ? std::round(w.Beta / step) * step : w.Beta;
            const double d = 1. - (nim[0] * w.K[0] + nim[1] * w.K[1] + nim[2] * w.K[2]);
            const double ex = nex[0] * w.K[0] + nex[1] * w.K[1] + nex[2] * w.K[2];
            const double g = .5 * lambda * beta * gamma;
            t.push_back({float((2. * d - l2 * ex) / d0 - double(p.Rc)), float(1. - (d - g) / d0), float(d0 / (d + g))});
        }
        std::sort(t.begin(), t.end());
        return size_t(std::unique(t.begin(), t.end()) - t.begin());
    };
    std::vector<std::array<int, 3>> classes;
    classes.reserve(wall.size());
    for (const auto &w : wall) classes.push_back({w.K[0], w.K[1], w.K[2]});
    std::sort(classes.begin(), classes.end());
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());

    std::printf("[implicit] %s %s in a %d x %d x %d array, gamma %.3f\n", scheme.Name, shape.Name(), shape.N[0], shape.N[1], shape.N[2], gamma);
    std::printf("  inside %.3fM nodes, %.1f%% of the array — the other %.1f%% is outside, ended by the block's word\n", double(inside) / 1e6, 100. * double(inside) / double(nodes), 100. * (1. - double(inside) / double(nodes)));
    std::printf("  wall nodes %zu over 26 directions, %zu over the 6 a reference voxeliser marks (%.1f%% of them)\n", wall.size(), axis_wall, 100. * double(axis_wall) / double(std::max<size_t>(wall.size(), 1)));
    std::printf("  beta %.4f to %.4f, mean %.4f, %zu negative (a flat wall is 1)\n", lo, hi, mean, negative);
    std::printf("  distinct terms %zu exact, %zu with beta to 8 bits, over %zu kept-count classes\n", distinct(0), distinct(8), classes.size());
    std::printf("  | %.1fs\n", Now() - t0);
}

// Measures the walled Courant limit from both the energy pencil and noise growth.
void ImplicitBoxCourantRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int soak) {
    const double t0 = Now();
    const double free_limit = std::min(scheme.Lambda, ImplicitCourantLimit(scheme));
    std::printf("[implicit] %s %d x %d x %d room: what Courant number the drop-ghost wall holds, %d steps a leg\n", scheme.Name, nx, ny, nz, soak);
    const double critical = ImplicitCriticalLambda(ImplicitBox(scheme, nx, ny, nz, true), 20000);
    std::printf("  free-space limit %.4f, the wall's own %.4f; seeded with noise so every eigenvalue is excited\n", free_limit, critical);

    double best = 0.;
    for (const double scale : {1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5}) {
        const double lambda = free_limit * scale;
        ImplicitBox box(scheme, nx, ny, nz, true, lambda);
        ImplicitSeedNoise(box, 12345);

        const double e0 = ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>());
        std::vector<double> logs, at;
        const int every = std::max(1, soak / 160);
        double m = 0.;
        for (int i = 0; i < soak; ++i) {
            box.Step();
            if ((i + 1) % every) continue;
            const float *u = box.Cur.As<float>();
            for (size_t j = 0; j < box.Nodes; j += 97) m = std::max(m, std::abs(double(u[j])));
            if (!(m > 0.) || !std::isfinite(m)) break;
            logs.push_back(std::log(m));
            at.push_back(double(i + 1));
            m = 0.;
        }
        const double rate = ImplicitSlope(at, logs);
        const double db_per_k = 1000. * rate * 20. / std::numbers::ln10;
        const bool held = db_per_k < 0.02;
        if (held && lambda > best) best = lambda;
        const double drift = std::abs(ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>()) - e0) / std::abs(e0);
        std::printf("  lambda %.4f (%.0f%% of free space)  growth %+8.4f dB per 1000 steps, energy drift %.1e  %s\n", lambda, 100. * scale, db_per_k, drift, held ? "holds" : "GROWS");
    }
    if (best > 0.) std::printf("  largest holding: lambda %.4f, %.0f%% of the free-space limit | %.1fs\n", best, 100. * best / free_limit, Now() - t0);
    else std::printf("  nothing held: the wall is unstable at every Courant number tried | %.1fs\n", Now() - t0);
}

// Compares implicit passes and a whole step with matching traffic ceilings.
void ImplicitCostRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int reps, bool rigid = false) {
    ImplicitBox box(scheme, nx, ny, nz, rigid);
    auto &ctx = MetalContext::Get();
    auto const time = [&](MTL::ComputePipelineState *pso, std::initializer_list<GpuSlice> buffers) {
        return TimeGpu(reps, [&] { ctx.Dispatch(pso, box.Tiles, box.Threads, buffers, &box.P, sizeof box.P); });
    };
    const double stream = time(box.Stream, {&box.Tmp, &box.Cur, &box.Bp});
    const double sweep = time(box.Jacobi, {&box.Tmp, &box.Cur, &box.Bp});
    const double rhs = time(box.Rhs, {&box.Bp, &box.Cur, &box.Prev});
    const double gb = 12. * double(box.Room) / 1e9;

    auto const whole = [&](bool wall) { return TimeGpu(reps, [&] { box.Step(wall); }); };
    double step = 1e30, bare = 1e30;
    for (int round = 0; round < (rigid ? 3 : 1); ++round) {
        step = std::min(step, whole(true));
        if (rigid) bare = std::min(bare, whole(false));
    }
    if (!rigid) bare = step;

    std::printf("[implicit cost] %s %s %d x %d x %d, %.3fM nodes, %.1f MB a pass, %.1f MB of state\n", scheme.Name, rigid ? "walled" : "periodic", nx, ny, nz, double(box.Room) / 1e6, 1e3 * gb, 4. * 4. * double(box.Nodes) / 1e6);
    std::printf("  three-stream pass %8.4f ms  %6.0f GB/s\n", 1e3 * stream, gb / stream);
    std::printf("  Jacobi sweep      %8.4f ms  %6.0f GB/s  %.0f%% of the stream\n", 1e3 * sweep, gb / sweep, 100. * stream / sweep);
    std::printf("  right-hand side   %8.4f ms  %6.0f GB/s  %.0f%% of the stream\n", 1e3 * rhs, gb / rhs, 100. * stream / rhs);
    if (rigid) {
        const double walled_sweep = TimeGpu(reps, [&] { ctx.Dispatch(box.JacobiFused, box.Tiles, box.Threads, {&box.Tmp, &box.Cur, &box.Bp, &box.Blocks, &box.Code, &box.Terms, &box.Keep}, &box.P, sizeof box.P); });
        const size_t nblocks = (box.Nodes + RoomBnBlock - 1) / RoomBnBlock;
        const double geom_mb = (12. * double(nblocks) + 6. * double(box.WallNodes)) / 1e6;
        std::printf("  Jacobi sweep, walled %8.4f ms  %6.0f GB/s  against three words a block of 32, %.1f%% more traffic\n", 1e3 * walled_sweep, gb / walled_sweep, 100. * (12. / double(RoomBnBlock)) / 12.);
        std::printf("  wall: %d nodes (%.2f%% of the room), %d distinct terms, %.1f MB of blocks and rows\n", box.WallNodes, 100. * double(box.WallNodes) / double(box.Room), box.Codes, geom_mb);
        std::printf("  the wall costs %.1f%% of a step (%.4f ms against %.4f without it)\n", 100. * (step / bare - 1.), 1e3 * step, 1e3 * bare);
    }
    std::printf("  a whole step      %8.4f ms  %6.0f GB/s over %d passes  |  %.1f ps/node-step\n", 1e3 * step, double(scheme.Sweeps + 1) * gb / step, scheme.Sweeps + 1, 1e12 * step / double(box.Room));
}

// Measures the pair projection and its cadence-amortized step overhead.
void ImplicitProjectionCostRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int reps) {
    ImplicitBox box(scheme, nx, ny, nz, true, 0., {.05});
    box.SetMeanProjection(1 << 28);
    const double step = TimeGpu(reps, [&] { box.Step(); });
    const double pair = TimeGpu(reps, [&] { box.ProjectPair(); });
    std::printf("[implicit projection cost] %s %d x %d x %d, %.3fM nodes\n", scheme.Name, nx, ny, nz, double(box.Room) / 1e6);
    std::printf("  whole step %.4f ms; project both levels %.4f ms (%.2f steps)\n", 1e3 * step, 1e3 * pair, pair / step);
    for (const int every : {64, 256, 1024}) std::printf("  every %-4d  amortized step overhead %.3f%%\n", every, 100. * pair / (step * every));
}

// Boundary-free grid for large-footprint streaming measurements.
RoomScene AirGrid(int nx, int ny, int nz, bool fcc) {
    RoomScene scene;
    scene.Fcc = fcc;
    scene.Nx = nx;
    scene.Ny = ny;
    scene.Nz = nz;
    scene.H = 0.0326857142857142;
    scene.C = 343.2;
    scene.L = fcc ? 0.999 : 0.999 / std::numbers::sqrt3;
    scene.L2 = scene.L * scene.L;
    scene.Ts = scene.L * scene.H / scene.C;
    scene.Srate = 1. / scene.Ts;
    scene.Nt = 1;
    scene.Output = "AirGrid";
    return scene;
}

void RooflineRung(const RoomScene &scene, const std::string &label, int reps) {
    const double t0 = Now();
    RoomGpu gpu;
    gpu.Init(scene);
    const auto band = gpu.Roofline(reps);
    const double gb = band.Bytes / 1e9;
    std::printf("[roofline] %s %s %d x %d x %d, %.2fM nodes, %.0f MB a pass\n", label.c_str(), scene.Fcc ? "FCC" : "cart", scene.Nx, scene.Ny, scene.Nz, double(scene.NumNodes()) / 1e6, 1e3 * gb);
    std::printf("  two-level stream  %8.4f ms  %6.0f GB/s\n", 1e3 * band.Stream, gb / band.Stream);
    std::printf("  interior update   %8.4f ms  %6.0f GB/s  %.0f%% of the stream\n", 1e3 * band.Update, gb / band.Update, 100. * band.Stream / band.Update);

    const auto lossy = gpu.LossyRoofline(reps);
    if (lossy.Bytes > 0.) {
        const double lgb = lossy.Bytes / 1e9;
        std::printf("  lossy nodes       %d of %d boundary, %.0f MB a pass\n", int(std::count_if(scene.MatBn.begin(), scene.MatBn.end(), [](int8_t m) { return m >= 0; })), int(scene.BnIxyz.size()), 1e3 * lgb);
        std::printf("  boundary stream   %8.4f ms  %6.0f GB/s\n", 1e3 * lossy.Stream, lgb / lossy.Stream);
        std::printf("  boundary update   %8.4f ms  %6.0f GB/s  %.0f%% of the stream\n", 1e3 * lossy.Update, lgb / lossy.Update, 100. * lossy.Stream / lossy.Update);
    }
    std::printf("  %.1f ps a node-step, %.2f GB of state | %.1fs\n", 1e12 * band.Update / double(int64_t(scene.Nx - 2) * (scene.Ny - 2) * (scene.Nz - 2)), double(scene.NumNodes()) * 8. / 1e9, Now() - t0);
}
} // namespace

int main(int argc, char *const *argv) {
    bool modes = false, energy = false, tube = false, balance = false, soak = false, golden = false, gate = false, any = false, fcc = false, roofline = false, implicit = false, projection = false, marginal = false, projection_cost = false, render = false;
    int repeat = 4, count = 16, steps = 0, reports = 10, length = 2000, project_every = 1024;
    int cells[3] = {0, 0, 0};
    std::string config_file, gen_dir;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--fcc") fcc = true;
        else if (arg == "--modes") modes = any = true;
        else if (arg == "--energy") energy = any = true;
        else if (arg == "--tube") tube = any = true;
        else if (arg == "--balance") balance = any = true;
        else if (arg == "--soak") soak = any = true;
        else if (arg == "--golden") golden = any = true;
        else if (arg == "--gate") gate = any = true;
        else if (arg == "--roofline") roofline = any = true;
        else if (arg == "--implicit") implicit = any = true;
        else if (arg == "--projection") projection = any = true;
        else if (arg == "--marginal") marginal = any = true;
        else if (arg == "--projection-cost") projection_cost = any = true;
        else if (arg == "--render") render = any = true;
        else if (arg == "--no-project") project_every = 0;
        else if (arg == "--repeat" && a + 1 < argc) repeat = std::atoi(argv[++a]);
        else if (arg == "--count" && a + 1 < argc) count = std::atoi(argv[++a]);
        else if (arg == "--steps" && a + 1 < argc) steps = std::atoi(argv[++a]);
        else if (arg == "--reports" && a + 1 < argc) reports = std::atoi(argv[++a]);
        else if (arg == "--length" && a + 1 < argc) length = std::atoi(argv[++a]);
        else if (arg == "--project-every" && a + 1 < argc) {
            project_every = std::atoi(argv[++a]);
            if (project_every < 1) {
                std::printf("--project-every needs a positive step interval\n");
                return 1;
            }
        } else if (arg == "--scene" && a + 1 < argc) config_file = argv[++a];
        else if (arg == "--gen" && a + 1 < argc) gen_dir = argv[++a];
        else if (arg == "--cells" && a + 1 < argc) {
            // One extent, or three separated by commas: --cells 1024 or --cells 1200,700,700
            if (std::sscanf(argv[++a], "%d,%d,%d", &cells[0], &cells[1], &cells[2]) != 3) cells[1] = cells[2] = cells[0];
            if (cells[0] < 6 || cells[1] < 6 || cells[2] < 6) {
                std::printf("--cells needs at least 6 on every axis\n");
                return 1;
            }
        } else {
            std::printf("unknown arg %s\n", arg.c_str());
            return 1;
        }
    }
    if (!any) modes = energy = tube = balance = golden = true; // default ladder, soak on request
    const std::string shoebox = fcc ? "RoomShoeboxFcc" : "RoomShoebox";
    if (config_file.empty()) config_file = "../Scenes/" + shoebox + "/config.json";
    if (gen_dir.empty()) gen_dir = "../gen/room/" + shoebox;

    if (modes) ModeRung(config_file, repeat, count, false);
    if (energy) EnergyRung(steps ? steps : 100000, reports, false, fcc);
    if (tube) TubeRung(length, steps ? steps : 7200, false);
    if (balance) BalanceRung(steps ? steps : 20000, reports, false);
    if (soak) SoakRung(steps ? steps : 500000, reports, false);
    if (golden) GoldenRung(config_file, gen_dir, false);
    if (roofline) {
        const int reps = repeat == 4 ? 200 : repeat;
        if (cells[0] > 0) RooflineRung(AirGrid(cells[0], cells[1], cells[2], fcc), "air grid", reps);
        else RooflineRung(LoadRoomScene(config_file), config_file, reps);
    }
    if (implicit) {
        for (const auto &s : {Implicit2Pct, Implicit1Pct, ImplicitHalfPct}) ImplicitRung(s, count == 16 ? 96 : count, steps ? steps : 2000);
        ImplicitWallRung(Implicit1Pct, 40, 33, 28, steps ? steps : 20000);
        ImplicitModeRung(Implicit1Pct, 40, 33, 28, count == 16 ? 16 : count);
        ImplicitAbsorbRung(Implicit1Pct, 40, 33, 28, 0.05, steps ? steps : 2000);
        ImplicitYawedRung(Implicit1Pct, 96, 0.4, 0.05, count == 16 ? 12 : count, steps ? steps : 2000);
        ImplicitLrcRung(Implicit1Pct, 40, 33, 28, steps ? steps : 2000);
        for (const double angle : {0., 0.4}) ImplicitVoxelRung(Implicit1Pct, ImplicitShape(ImplicitShape::Slanted, 48, 48, 48, angle));
        for (const char *model : {"CTK_Church", "Musikverein_ConcertHall"}) {
            const std::string root = std::getenv("PFFDTD_DIR") ? std::getenv("PFFDTD_DIR") : "../../pffdtd";
            const std::string path = root + "/data/models/" + model + "/model_export.json";
            if (std::filesystem::exists(path)) {
                const auto scene = LoadImplicitVoxelisedScene(path, .183);
                ImplicitSceneRung(Implicit1Pct, path, .183, .05, scene);
                ImplicitRoomRung(Implicit1Pct, path, .183, steps ? steps : 4000, repeat == 4 ? 100 : repeat, scene.Voxels);
            } else std::printf("[implicit] no model at %s, skipping the voxeliser survey\n", path.c_str());
        }
        for (const auto &shape : {ImplicitShape(ImplicitShape::Array, 96, 96, 96), ImplicitShape(ImplicitShape::Slanted, 96, 96, 96, 0.4), ImplicitShape(ImplicitShape::Sphere, 96, 96, 96), ImplicitShape(ImplicitShape::Obstacle, 96, 96, 96)})
            ImplicitGeometryRung(Implicit1Pct, shape, 0.05);
        ImplicitBoxCourantRung(Implicit1Pct, 40, 33, 28, steps ? steps : 40000);
        ImplicitCostRung(Implicit1Pct, 116, 76, 41, repeat == 4 ? 200 : repeat);
        ImplicitCostRung(Implicit1Pct, 384, 384, 384, repeat == 4 ? 20 : repeat);
        ImplicitCostRung(Implicit1Pct, 114, 74, 39, repeat == 4 ? 200 : repeat, true);
        ImplicitCostRung(Implicit1Pct, 382, 382, 382, repeat == 4 ? 20 : repeat, true);
    }
    if (projection) ImplicitProjectionRung(Implicit1Pct);
    if (marginal) ImplicitMarginalRung(Implicit1Pct, steps);
    if (projection_cost) {
        ImplicitProjectionCostRung(Implicit1Pct, 114, 74, 39, repeat == 4 ? 200 : repeat);
        ImplicitProjectionCostRung(Implicit1Pct, 382, 382, 382, repeat == 4 ? 20 : repeat);
    }
    if (render) ImplicitRenderRung(Implicit1Pct, .02, 2., .05, steps, project_every);
    if (gate) {
        ModeRung("../Scenes/RoomShoebox/config.json", 4, 16, true, .035, 3e-5);
        EnergyRung(100000, 10, true, false);
        TubeRung(2000, 7200, true);
        BalanceRung(20000, 10, true);
        SoakRung(500000, 10, true);
        GoldenRung("../Scenes/RoomShoebox/config.json", "../gen/room/RoomShoebox", true, 35.);
        ModeRung("../Scenes/RoomShoeboxFcc/config.json", 4, 16, true, FccModeLimit, FccDispersionLimit);
        EnergyRung(100000, 10, true, true);
        GoldenRung("../Scenes/RoomShoeboxFcc/config.json", "../gen/room/RoomShoeboxFcc", true, 34.);
        std::printf("[gate] %s (%d failures)\n", Failures ? "FAIL" : "PASS", Failures);
    }
    return Failures ? 1 : 0;
}
