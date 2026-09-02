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
//
// Set ACOUSTIC_ROOM_KERNEL_TIMES to have every dispatch isolated in its own command buffer
// and timed. That serializes the step, so it is a diagnostic run, not a mode to measure in.

#include "Parallel.h"
#include "RoomGpu.h"
#include "RoomScene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
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

// --- The optimised implicit scheme of Smits & Bilbao 2025 -------------------------------
//
// Free space on a periodic box, no boundaries: the question is whether the interior economics
// beat the explicit FCC scheme's, and the paper's own boundary conditions are
// frequency-independent, so they could not carry these scenes' walls anyway.

// A scheme in the paper's parameters: the weights of the three compact Laplacians in the
// explicit and implicit operators, the Courant number it is stable to, the dispersion-error
// threshold it was optimised against, and the Jacobi iteration count Table II records for it.
struct ImplicitScheme {
    const char *Name;
    double Lex[3], Lim[3], Lambda, Eps;
    int Sweeps;
};

// Table II, row IM at each of the three dispersion-error thresholds. Each is measured against
// its own threshold, since that is what its parameters were optimised for. The 1% row is the
// threshold both explicit schemes are calibrated at.
constexpr ImplicitScheme Implicit2Pct{"IM 2%", {0.390, 0.610, 0.000}, {0., 2.09e-3, 41.17e-3}, 0.845, .02, 7};
constexpr ImplicitScheme Implicit1Pct{"IM 1%", {0.404, 0.553, 0.043}, {0., 9.75e-3, 28.37e-3}, 0.840, .01, 7};
constexpr ImplicitScheme ImplicitHalfPct{"IM 0.5%", {0.400, 0.580, 0.020}, {0., 0.91e-3, 29.95e-3}, 0.850, .005, 6};

// Where the drop-ghost wall effectively sits, in cells beyond the outermost node, as
// ImplicitModeRung fits it. The full cell puts it at 0.5 and the 27-point stencil's own
// boundary layer pulls it in; the fit drifts with grid size (0.48 at N ~ 11, 0.44 at N ~ 33),
// so this is the value for room-sized grids and the rung that measures it prints its own.
constexpr double ImplicitWallOffset = 0.44;

// The Courant limit the scheme's own parameters give, from the paper's Eq. 16: with
// lbar_ex(b) = -sum_r l_r^ex L_r(b) and lbar_im(b) = 1 + sum_r l_r^im L_r(b) evaluated at the
// three corner cases of Table I, stability needs lambda^2 <= 4 lbar_im / lbar_ex at each.
// Recomputed rather than read off Table II's lambda_max, which is rounded to three places:
// IM 0.5%'s 0.850 is above its own limit of 0.8494, and grows a free-space mode by four orders
// of magnitude in 2000 steps.
double ImplicitCourantLimit(const ImplicitScheme &s) {
    static constexpr double L[3][3] = {{-4., -4., -4.}, {-8., -4., 0.}, {-12., 0., -4.}};
    double limit = 1e30;
    for (const auto &lb : L) {
        double ex = 0., im = 1.;
        for (int r = 0; r < 3; ++r) {
            ex -= s.Lex[r] * lb[r];
            im += s.Lim[r] * lb[r];
        }
        if (ex > 0.) limit = std::min(limit, std::sqrt(4. * im / ex));
    }
    return limit;
}

// The scheme's coefficients as the kernels hold them, narrowed to float once here so that
// everything predicted about the scheme is predicted from the numbers in force. The three
// Laplacians contribute S1 - 6u, S2/4 - 3u and S3/4 - 2u, so a weight vector l gives neighbour
// coefficients (l0, l1/4, l2/4) and a diagonal of -(6 l0 + 3 l1 + 2 l2), and the whole system
// is divided through by the implicit operator's diagonal.
RoomImplicitParams ImplicitCoefficients(const ImplicitScheme &s, double lambda, int nx, int ny, int nz) {
    const double e[3] = {s.Lex[0], s.Lex[1] / 4., s.Lex[2] / 4.};
    const double n[3] = {s.Lim[0], s.Lim[1] / 4., s.Lim[2] / 4.};
    const double ediag = 6. * s.Lex[0] + 3. * s.Lex[1] + 2. * s.Lex[2];
    const double jd = 1. - (6. * s.Lim[0] + 3. * s.Lim[1] + 2. * s.Lim[2]);
    const double l2 = lambda * lambda;
    return {nx, ny, nz, 0, 0, 0, 0, float((2. * n[0] + l2 * e[0]) / jd), float((2. * n[1] + l2 * e[1]) / jd), float((2. * n[2] + l2 * e[2]) / jd), float((2. * jd - l2 * ediag) / jd), float(n[0] / jd), float(n[1] / jd), float(n[2] / jd)};
}

// What the scheme in force does to a plane wave, from those same floats. On a periodic box a
// plane wave is an exact discrete eigenvector, and the neighbour sums have symbols
// S1 = 2(cx+cy+cz), S2 = 4(cx cy + cx cz + cy cz), S3 = 8 cx cy cz, so the modal amplitude
// obeys a^(n+1) = 2 cos(wT) a^n - a^(n-1) with the ratio below. Started from a^0 = a^1 = 1,
// a^2 + 1 is 2 cos(wT) exactly, which is what the rung reads off the grid in one step.
double ImplicitTwoCosOmegaT(const RoomImplicitParams &p, double kx, double ky, double kz) {
    const double cx = std::cos(kx), cy = std::cos(ky), cz = std::cos(kz);
    const double s1 = 2. * (cx + cy + cz), s2 = 4. * (cx * cy + cx * cz + cy * cz), s3 = 8. * cx * cy * cz;
    const double off = p.Q1 * s1 + p.Q2 * s2 + p.Q3 * s3;
    return 1. + ((p.R1 * s1 + p.R2 * s2 + p.R3 * s3 + p.Rc) - (off + 1.)) / (1. + off);
}

// The scheme's neighbour coefficients, undivided: the implicit operator's off-diagonals per
// offset class, and the explicit one's. The three compact Laplacians contribute S1 - 6u,
// S2/4 - 3u and S3/4 - 2u, so a weight vector l gives (l0, l1/4, l2/4).
double ImplicitNim(const ImplicitScheme &s, int r) { return r == 0 ? s.Lim[0] : s.Lim[r] / 4.; }
double ImplicitNex(const ImplicitScheme &s, int r) { return r == 0 ? s.Lex[0] : s.Lex[r] / 4.; }

// The room a grid holds, for the questions the boundary asks of geometry: is this node inside,
// and what is the true outward normal at it. Three cases, all in closed form — the array
// itself, a box yawed off the grid so its walls staircase, and a sphere, which carries every
// orientation and a curvature at once. The last two are what the staircase-compensation paper
// validates against, and between them there is no mesh and no voxeliser standing between the
// geometry and the numbers.
//
// Node i sits at i, and the wall lies half a cell beyond the outermost node the room keeps —
// the convention the box's own modes measure at (N + 0.44) h. So `inside` means at least half
// a cell in from the surface, and the extents below are where the wall is, not where the last
// node is.
struct ImplicitShape {
    enum Kind { Array,
                Slanted,
                Sphere,
                Obstacle };

    ImplicitShape(Kind k, int nx, int ny, int nz, double angle = 0.) : K(k), N{nx, ny, nz}, Cos(std::cos(angle)), Sin(std::sin(angle)) {
        // Off the grid's centre by a different fraction of a cell on each axis. A shape
        // centred on the grid is symmetric under the octahedral group, and its distinct
        // boundary terms then come out forty-eight times too few — which would read as the
        // code byte having room it does not have.
        static constexpr double Break[3]{0.37, 0.19, 0.23};
        double fit = 1e30;
        for (int a = 0; a < 3; ++a) {
            Cen[a] = .5 * (N[a] - 1) + (k == Array ? 0. : Break[a]);
            fit = std::min(fit, std::min(Cen[a], N[a] - 1 - Cen[a]) + .5);
        }
        // A sphere is the largest the array holds; an obstacle is a column standing in it,
        // small enough that its own wall is nowhere near the array's.
        Rad = k == Obstacle ? .4 * fit : fit;
        Ext[2] = std::min(Cen[2], N[2] - 1 - Cen[2]) + .5;
        // The largest yawed box the array holds: a yaw of theta grows a half-extent's
        // footprint on both in-plane axes by |cos| + |sin|.
        Ext[0] = Ext[1] = std::min(fit, Ext[2]) / (std::abs(Cos) + std::abs(Sin));
    }

    const char *Name() const {
        static const char *const Names[4]{"the array", "a yawed box", "a sphere", "a column in the array"};
        return Names[K];
    }

    void Local(const double p[3], double q[3]) const {
        q[0] = Cos * p[0] + Sin * p[1];
        q[1] = Cos * p[1] - Sin * p[0];
        q[2] = p[2];
    }

    double FromCentre(int ix, int iy, int iz, double p[3]) const {
        p[0] = ix - Cen[0];
        p[1] = iy - Cen[1];
        p[2] = iz - Cen[2];
        return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    }

    bool Inside(int ix, int iy, int iz) const {
        if (ix < 0 || iy < 0 || iz < 0 || ix >= N[0] || iy >= N[1] || iz >= N[2]) return false;
        if (K == Array) return true;
        double p[3];
        const double radius = FromCentre(ix, iy, iz, p);
        if (K == Sphere) return radius <= Rad - .5;
        if (K == Obstacle) return radius >= Rad + .5; // and inside the array, tested above
        double q[3];
        Local(p, q);
        return std::abs(q[0]) <= Ext[0] - .5 && std::abs(q[1]) <= Ext[1] - .5 && std::abs(q[2]) <= Ext[2] - .5;
    }

    // The true outward unit normal of the surface nearest an inside node — the mesh's normal,
    // not the staircase's, which is the whole point of the compensation. Zero where there is
    // no surface to be near, which is also where there are no ghosts.
    void Normal(int ix, int iy, int iz, double n[3]) const {
        n[0] = n[1] = n[2] = 0.;
        if (K == Array) {
            n[0] = double(ix == N[0] - 1) - double(ix == 0);
            n[1] = double(iy == N[1] - 1) - double(iy == 0);
            n[2] = double(iz == N[2] - 1) - double(iz == 0);
        } else {
            double p[3];
            const double radius = FromCentre(ix, iy, iz, p);
            if (K == Sphere) {
                for (int a = 0; a < 3; ++a) n[a] = p[a];
            } else if (K == Obstacle) {
                // Two surfaces here, and the node belongs to whichever is nearer — which is
                // what a mesh's nearest-triangle lookup would decide. Out of the room means
                // *into* the column, so its normal points inward.
                int face = 0;
                for (int a = 1; a < 3; ++a)
                    if (Wall(a) - std::abs(p[a]) < Wall(face) - std::abs(p[face])) face = a;
                if (radius - Rad < Wall(face) - std::abs(p[face])) {
                    for (int a = 0; a < 3; ++a) n[a] = -p[a];
                } else {
                    n[face] = p[face] < 0. ? -1. : 1.;
                }
            } else {
                double q[3];
                Local(p, q);
                int face = 0; // the wall the node is nearest, which is the triangle it belongs to
                for (int a = 1; a < 3; ++a)
                    if (Ext[a] - std::abs(q[a]) < Ext[face] - std::abs(q[face])) face = a;
                double m[3]{0., 0., 0.};
                m[face] = q[face] < 0. ? -1. : 1.;
                n[0] = Cos * m[0] - Sin * m[1];
                n[1] = Sin * m[0] + Cos * m[1];
                n[2] = m[2];
            }
        }
        const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (len > 0.)
            for (int a = 0; a < 3; ++a) n[a] /= len;
    }

    // Where the array's own wall is on an axis, as a distance from the centre.
    double Wall(int a) const { return std::min(Cen[a], N[a] - 1 - Cen[a]) + .5; }

    // How many of the 6 face, 12 edge and 8 corner neighbours a node keeps under the
    // drop-ghost boundary. A face neighbour steps on one axis, an edge neighbour on two and a
    // corner neighbour on three, so when the room is the array the counts are the elementary
    // symmetric polynomials in how many steps each axis has left, and the interior's
    // (6, 12, 8) is the case every axis has both.
    void Counts(int ix, int iy, int iz, int k[3]) const {
        if (K == Array) {
            const int a = int(ix > 0) + int(ix < N[0] - 1), b = int(iy > 0) + int(iy < N[1] - 1), c = int(iz > 0) + int(iz < N[2] - 1);
            k[0] = a + b + c;
            k[1] = a * b + a * c + b * c;
            k[2] = a * b * c;
            return;
        }
        k[0] = k[1] = k[2] = 0;
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    const int r = dx * dx + dy * dy + dz * dz;
                    if (r && Inside(ix + dx, iy + dy, iz + dz)) ++k[r - 1];
                }
    }

    Kind K;
    int N[3]; // NOLINT(modernize-use-default-member-init) the constructor's arguments are where it comes from
    double Cos, Sin;
    double Cen[3], Ext[3], Rad;
};

// What a node's ghost set gives it: the counts it keeps in each offset class, and the paper's
// `beta_i = sum_r nex_r * sum_j (n_i . j)` over the ghost offsets j, with n_i the true outward
// normal.
//
// On a grid-aligned flat wall every ghost offset has n.j = 1 in its own class, so beta comes
// out at Lex0 + Lex1 + Lex2 = 1 exactly — that is the check that the paper's two readings of
// alpha_r have not been crossed. A slanted or curved wall gets less, and that is the staircase
// compensation doing its work: the aggregate absorbing weight follows the true wall area
// rather than the staircased one, which is what keeps absorption from being over-represented.
double ImplicitWallSample(const ImplicitShape &shape, int ix, int iy, int iz, const double nex[3], int k[3], bool compensated) {
    double n[3];
    shape.Normal(ix, iy, iz, n);
    k[0] = k[1] = k[2] = 0;
    double beta = 0., faces = 0., projected = 0.;
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                const int r = dx * dx + dy * dy + dz * dz;
                if (!r) continue;
                if (shape.Inside(ix + dx, iy + dy, iz + dz)) {
                    ++k[r - 1];
                    continue;
                }
                const double projection = n[0] * dx + n[1] * dy + n[2] * dz;
                beta += nex[r - 1] * projection;
                if (r == 1) { // the only ghosts that stand for a face of the node's cell
                    ++faces;
                    projected += projection;
                }
            }
    // Uncompensated: the node's wall keeps the *staircased* area rather than the true area it
    // stands for, which is the error the 2024 compensation paper measures at -18% to -42% in
    // T60 on rotated boxes and -34% on a sphere. Per node that ratio is the exposed faces over
    // their projection onto the true surface, which is 1 on a grid-aligned wall — so this and
    // the compensated reading agree exactly where staircasing costs nothing.
    if (!compensated && projected > 0.) beta *= faces / projected;
    return beta;
}

// Least-squares slope of `y` against `x` from sample `from` on, which is how both soaks read a
// rate: a growth or a decay per step means the same thing at any run length, where "did it
// move" mostly measures how long the run was. Infinite when there is too little to fit.
double ImplicitSlope(const std::vector<double> &x, const std::vector<double> &y, size_t from = 0) {
    if (x.size() < from + 4) return std::numeric_limits<double>::infinity();
    const double count = double(x.size() - from);
    double sx = 0., sy = 0., sxx = 0., sxy = 0.;
    for (size_t i = from; i < x.size(); ++i) {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        sxy += x[i] * y[i];
    }
    return (count * sxy - sx * sy) / (count * sxx - sx * sx);
}

// What a wall is: a real admittance, the paper's own case, or a set of parallel LRC branches,
// which is what every scene here carries. Both reach the scheme the same way — one nonnegative
// scalar on the diagonal — and the branches also carry the memory that makes them
// frequency-dependent. See RoomImplicitWall.
struct ImplicitWall {
    double Gamma{0.}; // a real admittance, when the wall has no branches
    double Ts{0.}; // the time step the branches are discretised at
    std::vector<double> Def; // (D, E, F) triples, the format the scenes' materials come in
    int Branches() const { return int(Def.size()) / 3; }
};

struct ImplicitBox {
    // `rigid` swaps the periodic wrap for the drop-ghost wall of Smits & Bilbao 2025 Sec IV,
    // which turns the dispersion instrument into a room: every neighbour outside the room
    // leaves both the sum and the diagonal. The grid then carries a one-node halo that no
    // pass ever writes, so a neighbour outside the room reads the zero it was allocated with
    // and the sums need no mask, and RoomImplicitTerm carries the diagonal the dropped ghosts
    // took with them. The wall lies half a cell outside the outermost node, so a room of N
    // nodes is N*h on that axis rather than the image's (N-1)*h.
    //
    // `gamma` is the paper's real admittance, uniform over the wall. Nothing about the step
    // changes with it: G = (lambda/2) beta gamma is diagonal, so it lands on A's diagonal and
    // on the u^(n-1) coefficient, which is where RoomImplicitTerm's Fd and Cp already are. It
    // does not move the Courant limit either — E >= 0 is a condition on Lim and Lex alone, and
    // a nonnegative G only makes A more diagonally dominant, which the Jacobi sweep likes.
    //
    // `kind` is the room the array holds. Array is the array itself, and the two others are a
    // staircase cut out of it — their outside nodes are stepped like any other and held at zero
    // by their own terms, so nothing about the passes knows the difference. `compensated` off
    // gives beta the staircase's normal instead of the surface's, which is the absorption bias
    // the compensation exists to remove.
    ImplicitBox(const ImplicitScheme &s, int nx, int ny, int nz, bool rigid = false, double lambda = 0., ImplicitWall wall = {}, ImplicitShape::Kind kind = ImplicitShape::Array, double angle = 0., bool compensated = true)
        : Scheme(s), Lambda(lambda > 0. ? lambda : std::min(s.Lambda, ImplicitCourantLimit(s))), Wall(std::move(wall)), Compensated(compensated),
          Shape(kind, nx, ny, nz, angle), Pad(rigid ? 1 : 0),
          Nx(nx), Ny(ny), Nz(nz), Gx(nx + 2 * Pad), Gy(ny + 2 * Pad), Gz(nz + 2 * Pad),
          P(ImplicitCoefficients(s, Lambda, Gx, Gy, Gz)), Nodes(size_t(Gx) * Gy * Gz), Room(size_t(nx) * ny * nz) {
        for (auto *b : {&Prev, &Cur, &Bp, &Tmp}) b->ResizeZeroed(Nodes * sizeof(float));
        auto &ctx = MetalContext::Get();
        Rhs = ctx.RoomPipeline(rigid ? "RoomImplicitRhsNoWall" : "RoomImplicitRhs");
        Jacobi = ctx.RoomPipeline(rigid ? "RoomImplicitJacobiNoWall" : "RoomImplicitJacobi");
        Stream = ctx.RoomPipeline(rigid ? "RoomImplicitStreamWall" : "RoomImplicitStream");
        Threads = {32, 4, 4};
        Tiles = {(uint32_t(nz) + Threads.x - 1) / Threads.x, (uint32_t(ny) + Threads.y - 1) / Threads.y, (uint32_t(nx) + Threads.z - 1) / Threads.z};
        if (!rigid) return;

        RhsFused = ctx.RoomPipeline("RoomImplicitRhsWall");
        JacobiFused = ctx.RoomPipeline("RoomImplicitJacobiWall");

        // The branch coefficients, through the scenes' own derivation so that the arithmetic
        // the wall runs is the arithmetic the explicit path runs.
        if (Wall.Branches()) {
            RoomScene scene;
            scene.Ts = Wall.Ts;
            scene.MatBranches = {Wall.Branches()};
            scene.MatDef = Wall.Def;
            Mats = RoomMaterialCoefficients(scene);
            WallRhsPso = ctx.RoomPipeline("RoomImplicitWallRhs");
            WallStepPso = ctx.RoomPipeline("RoomImplicitWallStep");
        }
        std::vector<RoomImplicitWall> lossy;

        // Two bytes a node into a table of the distinct terms. A rigid box needs three of them
        // and a yawed one twenty-one, so a byte would do for both; a sphere needs forty
        // thousand, because beta is a continuous function of the surface's normal and very
        // nearly every wall node is then its own term. That is what sets the width.
        //
        // The table is indexed directly here, which holds to 65,535 terms and so to the grids
        // this validates on. A production room has more wall nodes than that, and the way out
        // is the packing the terms already invite: Cn depends on the kept counts alone, of
        // which there are at most a few dozen, so six bits of class and ten of quantised beta
        // fit the same two bytes. That is a change to this table and not to the kernels.
        std::vector<RoomImplicitTerm> terms{{0.f, 0.f, 1.f}, {0.f, 0.f, 0.f}}; // air, and outside
        std::vector<uint16_t> code(Nodes, 0);
        for (int ix = 0; ix < nx; ++ix) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int iz = 0; iz < nz; ++iz) {
                    if (!Shape.Inside(ix, iy, iz)) {
                        code[Index(ix, iy, iz)] = 1; // Fd of zero, which holds it at rest
                        continue;
                    }
                    ++InsideNodes;
                    int k[3];
                    Shape.Counts(ix, iy, iz, k);
                    if (k[0] == 6 && k[1] == 12 && k[2] == 8) continue; // ordinary air
                    if (WallScalar() != 0.) WallG.emplace_back(int(Index(ix, iy, iz)), NodeG(ix, iy, iz));
                    if (Wall.Branches()) lossy.push_back({int(Index(ix, iy, iz)), 0, float(Lambda * Beta(ix, iy, iz) / Diagonal())});
                    const RoomImplicitTerm t = WallTerms(ix, iy, iz);
                    size_t c = 2;
                    while (c < terms.size() && !(terms[c].Cn == t.Cn && terms[c].Cp == t.Cp && terms[c].Fd == t.Fd)) ++c;
                    if (c == terms.size()) terms.push_back(t);
                    if (c > 65535) throw std::runtime_error("more than 65535 distinct boundary terms");
                    code[Index(ix, iy, iz)] = uint16_t(c);
                    ++WallNodes;
                }
            }
        }
        Codes = int(terms.size());
        Code.ResizeZeroed(code.size() * sizeof(uint16_t));
        Code.Upload(code.data(), code.size() * sizeof(uint16_t));
        Terms.ResizeZeroed(terms.size() * sizeof(RoomImplicitTerm));
        Terms.Upload(terms.data(), terms.size() * sizeof(RoomImplicitTerm));
        if (lossy.empty()) return;

        P.Nw = int(lossy.size());
        WallList.ResizeZeroed(lossy.size() * sizeof(RoomImplicitWall));
        WallList.Upload(lossy.data(), lossy.size() * sizeof(RoomImplicitWall));
        const int mb = Wall.Branches();
        MatMb.ResizeZeroed(sizeof(int));
        MatMb.Upload(&mb, sizeof(int));
        MatQuads.ResizeZeroed(Mats.Quads.size() * sizeof(RoomMatQuad));
        MatQuads.Upload(Mats.Quads.data(), Mats.Quads.size() * sizeof(RoomMatQuad));
        for (auto *b : {&Vh, &Gh}) b->ResizeZeroed(size_t(mb) * lossy.size() * sizeof(float));
        Ub.ResizeZeroed(lossy.size() * sizeof(float));
        WallThreads = {64, 1, 1};
        WallTiles = {(uint32_t(P.Nw) + WallThreads.x - 1) / WallThreads.x, 1, 1};
    }

    // What rides on the diagonal: a real admittance, or the branches' summed discrete one.
    double WallScalar() const { return Wall.Branches() ? double(Mats.Beta[0]) : Wall.Gamma; }

    // The scheme's neighbour coefficients, undivided: the implicit operator's off-diagonals
    // and the explicit one's, per offset class.
    double Nim(int r) const { return ImplicitNim(Scheme, r); }
    double Nex(int r) const { return ImplicitNex(Scheme, r); }
    // The implicit operator's diagonal at a node keeping k neighbours a class, and the
    // explicit one's, both as the interior case when no counts are given.
    double Diagonal(const int k[3] = nullptr) const {
        static constexpr int Full[3]{6, 12, 8};
        const int *c = k ? k : Full;
        return 1. - (Nim(0) * c[0] + Nim(1) * c[1] + Nim(2) * c[2]);
    }
    double ExDiagonal(const int k[3] = nullptr) const {
        static constexpr int Full[3]{6, 12, 8};
        const int *c = k ? k : Full;
        return Nex(0) * c[0] + Nex(1) * c[1] + Nex(2) * c[2];
    }

    size_t Index(int ix, int iy, int iz) const { return (size_t(ix + Pad) * Gy + size_t(iy + Pad)) * Gz + size_t(iz + Pad); }

    // What a node's step needs beyond the uniform coefficients, from the neighbours it keeps.
    // Rc and the u^(n-1) coefficient are what the uniform pass writes with the interior
    // diagonal d0 in them, so what a wall node needs added back is the difference against its
    // own diagonal d, and the whole sweep's diagonal is one factor d0/d because the sweep is
    // one division by it. An interior node's terms are {0, 0, 1}, which is code 0.
    RoomImplicitTerm WallTerms(int ix, int iy, int iz) const {
        if (!Shape.Inside(ix, iy, iz)) return {0.f, 0.f, 0.f};
        int k[3];
        Shape.Counts(ix, iy, iz, k);
        const double d = Diagonal(k), ex = ExDiagonal(k), d0 = Diagonal(), g = NodeG(ix, iy, iz);
        return {float((2. * d - Lambda * Lambda * ex) / d0 - double(P.Rc)), float(1. - (d - g) / d0), float(d0 / (d + g))};
    }

    // The wall's admittance term at a node, G = (lambda/2) beta gamma. Zero off the wall, and
    // zero everywhere when the wall is rigid, which is what leaves WallTerms the rigid ones.
    double Beta(int ix, int iy, int iz) const {
        const double nex[3]{Nex(0), Nex(1), Nex(2)};
        int k[3];
        return ImplicitWallSample(Shape, ix, iy, iz, nex, k, Compensated);
    }
    double NodeG(int ix, int iy, int iz) const {
        const double scalar = WallScalar();
        return scalar == 0. ? 0. : .5 * Lambda * Beta(ix, iy, iz) * scalar;
    }

    // Seeds the two levels from functions of the room coordinates, leaving the halo at rest.
    // They are separate because seeding both alike starts the scheme from rest, and a mode
    // started from rest carries an amplitude of 1/cos(wT/2) — which at the top of the band is
    // large enough to be mistaken for growth.
    template<typename F, typename G> void Seed(F &&cur, G &&prev) {
        std::vector<float> a(Nodes, 0.f), b(Nodes, 0.f);
        for (int ix = 0; ix < Nx; ++ix)
            for (int iy = 0; iy < Ny; ++iy)
                for (int iz = 0; iz < Nz; ++iz) {
                    if (!Shape.Inside(ix, iy, iz)) continue; // outside stays at the rest the sums need
                    a[Index(ix, iy, iz)] = float(cur(ix, iy, iz));
                    b[Index(ix, iy, iz)] = float(prev(ix, iy, iz));
                }
        Cur.Upload(a.data(), a.size() * sizeof(float));
        Prev.Upload(b.data(), b.size() * sizeof(float));
        if (!P.Nw) return;
        // Seeding restarts the run, so the wall goes back to rest — and its record of the
        // level two steps back becomes the level the seed just put behind.
        const size_t branch_bytes = size_t(Wall.Branches()) * size_t(P.Nw) * sizeof(float);
        Vh.Zero(branch_bytes);
        Gh.Zero(branch_bytes);
        std::vector<float> behind(size_t(P.Nw));
        const RoomImplicitWall *list = WallList.As<RoomImplicitWall>();
        for (int i = 0; i < P.Nw; ++i) behind[size_t(i)] = b[size_t(list[i].Ixyz)];
        Ub.Upload(behind.data(), behind.size() * sizeof(float));
    }

    // Four grids of state, which is the paper's S = 4: the two levels behind, the Jacobi
    // constant vector, and one scratch grid the sweeps ping-pong through. With an odd sweep
    // count the answer lands in the grid the oldest level occupied, so nothing is copied and
    // the rotation is a swap of handles.
    //
    // `wall` off encodes the same passes without the byte that carries the wall's terms. It
    // steps nothing meaningful — the diagonal it leaves at the interior's value is wrong at
    // every node the wall touches — and exists so that what the wall costs can be read against
    // the same room rather than against a periodic box of a different shape.
    void Step(bool wall = true) {
        auto &ctx = MetalContext::Get();
        const bool fused = WallNodes > 0 && wall;
        if (fused) ctx.Dispatch(RhsFused, Tiles, Threads, {&Bp, &Cur, &Prev, &Code, &Terms}, &P, sizeof P);
        else ctx.Dispatch(Rhs, Tiles, Threads, {&Bp, &Cur, &Prev}, &P, sizeof P);
        // The wall's memory, once a step and over the wall alone. The sweeps below it never
        // see the branches: what they carry rides on the diagonal, inside Fd.
        if (fused && P.Nw) ctx.Dispatch(WallRhsPso, WallTiles, WallThreads, {&Bp, &Vh, &Gh, &WallList, &MatMb, &MatQuads}, &P, sizeof P);
        const GpuBuffer *src = &Cur, *dst = &Prev;
        for (int m = 0; m < Scheme.Sweeps; ++m) {
            if (fused) ctx.Dispatch(JacobiFused, Tiles, Threads, {dst, src, &Bp, &Code, &Terms}, &P, sizeof P);
            else ctx.Dispatch(Jacobi, Tiles, Threads, {dst, src, &Bp}, &P, sizeof P);
            src = dst;
            dst = (dst == &Prev) ? &Tmp : &Prev;
        }
        if (src != &Prev) { // an even sweep count leaves it in the scratch grid
            Prev.Swap(Tmp);
        }
        Prev.Swap(Cur); // Cur holds the new level, Prev the one it replaced
        // And the branch step, node-local, now that the level it needs exists.
        if (fused && P.Nw) ctx.Dispatch(WallStepPso, WallTiles, WallThreads, {&Cur, &Prev, &Ub, &Vh, &Gh, &WallList, &MatMb, &MatQuads}, &P, sizeof P);
    }

    const ImplicitScheme &Scheme;
    double Lambda; // the scheme's own Courant limit where Table II's rounded value exceeds it
    ImplicitWall Wall; // rigid, a real admittance, or LRC branches
    RoomMaterials Mats; // the branches' discrete coefficients, empty without them
    bool Compensated; // whether beta reads the surface's normal or the staircase's
    ImplicitShape Shape; // the room the array holds
    int Pad; // one node of halo with a wall, none on a periodic box
    int Nx, Ny, Nz; // the room
    int Gx, Gy, Gz; // the array the room sits in
    RoomImplicitParams P;
    size_t Nodes, Room;
    size_t InsideNodes{0}; // nodes of the room, which is all of them when it is the array
    int WallNodes{0}; // nodes the wall takes a neighbour from, 0 on a periodic box
    int Codes{1}; // distinct boundary terms, the width of the table the code indexes
    // G at the nodes that have one, for the host's energy accounting. Empty when gamma is
    // zero, which is every rung but the absorbing one — a grid-wide copy of it would be as
    // large as a level.
    std::vector<std::pair<int, double>> WallG;
    GpuBuffer Prev, Cur, Bp, Tmp, Code, Terms;
    // The LRC wall: its node list and material, a branch-major (vh, gh), and the node's own
    // level two steps back, which the grid no longer holds when the branches want it.
    GpuBuffer WallList, MatMb, MatQuads, Vh, Gh, Ub;
    MTL::ComputePipelineState *Rhs, *Jacobi, *Stream, *RhsFused{}, *JacobiFused{}, *WallRhsPso{}, *WallStepPso{};
    Dim3 Tiles, Threads, WallTiles{}, WallThreads{};
};

// One of the scheme's two Laplacians applied over the room, in double, laid out on the same
// padded grid its argument is. `c` is the operator's three neighbour coefficients; the halo's
// zeros mask the sums, and the diagonal follows the counts the node keeps, which is the whole
// boundary condition.
template<typename T> std::vector<double> ImplicitApply(const ImplicitBox &box, const T *u, const double c[3]) {
    std::vector<double> out(box.Nodes, 0.);
    for (int ix = 0; ix < box.Nx; ++ix) {
        for (int iy = 0; iy < box.Ny; ++iy) {
            for (int iz = 0; iz < box.Nz; ++iz) {
                double s[3]{0., 0., 0.};
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dz = -1; dz <= 1; ++dz) {
                            const int r = dx * dx + dy * dy + dz * dz;
                            if (r) s[r - 1] += double(u[box.Index(ix + dx, iy + dy, iz + dz)]);
                        }
                int k[3];
                box.Shape.Counts(ix, iy, iz, k);
                double v = 0.;
                for (int r = 0; r < 3; ++r) v += c[r] * (s[r] - k[r] * double(u[box.Index(ix, iy, iz)]));
                out[box.Index(ix, iy, iz)] = v;
            }
        }
    }
    return out;
}

// The largest Courant number at which the scheme's energy is positive, which is the wall's own
// stability limit and a different number from the free-space one the scheme is published with.
// E >= 0 iff 1 + Lim + (lambda^2/4) Lex > 0, so the limit is 2/sqrt(rho) with rho the largest
// eigenvalue of the pencil (-Lex, 1 + Lim).
//
// Power iteration in double, run until the Rayleigh quotient settles, with 1 + Lim inverted by
// the same Jacobi sweeps the stepper uses — its diagonal dominance is what the whole scheme
// rests on, so a handful reach double precision. The constant vector is in Lex's null space
// and the iteration maps it to zero, so the scheme's physical null mode needs no removing.
// The quotient approaches rho from below, so a run cut short reports the limit too high.
double ImplicitCriticalLambda(const ImplicitBox &box, int cap) {
    const double nim[3]{box.Nim(0), box.Nim(1), box.Nim(2)}, nex[3]{box.Nex(0), box.Nex(1), box.Nex(2)};
    std::vector<double> v(box.Nodes, 0.), w(box.Nodes, 0.);
    std::mt19937 rng{7};
    std::uniform_real_distribution<double> uniform{-1., 1.};
    for (int ix = 0; ix < box.Nx; ++ix)
        for (int iy = 0; iy < box.Ny; ++iy)
            for (int iz = 0; iz < box.Nz; ++iz)
                if (box.Shape.Inside(ix, iy, iz)) v[box.Index(ix, iy, iz)] = uniform(rng);

    // The inner product over the room, and M w = rhs solved by sweeps of the scheme's own.
    auto dot = [&](const std::vector<double> &a, const std::vector<double> &b) {
        double q = 0.;
        for (int ix = 0; ix < box.Nx; ++ix)
            for (int iy = 0; iy < box.Ny; ++iy)
                for (int iz = 0; iz < box.Nz; ++iz) q += a[box.Index(ix, iy, iz)] * b[box.Index(ix, iy, iz)];
        return q;
    };
    auto solve = [&](const std::vector<double> &rhs) {
        std::fill(w.begin(), w.end(), 0.);
        for (int sweep = 0; sweep < 12; ++sweep) {
            const auto off = ImplicitApply(box, w.data(), nim);
            for (int ix = 0; ix < box.Nx; ++ix)
                for (int iy = 0; iy < box.Ny; ++iy)
                    for (int iz = 0; iz < box.Nz; ++iz) {
                        const size_t i = box.Index(ix, iy, iz);
                        int k[3];
                        box.Shape.Counts(ix, iy, iz, k);
                        // `off` is Lim w, which already carries -lim*w at the node, so adding
                        // that back leaves the neighbour sum alone and M's diagonal is 1 - lim.
                        const double lim = nim[0] * k[0] + nim[1] * k[1] + nim[2] * k[2];
                        w[i] = (rhs[i] - (off[i] + lim * w[i])) / (1. - lim);
                    }
        }
        return w;
    };
    auto rayleigh = [&] {
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

// The scheme's discrete energy at the half step between two levels, from Sec V of the paper:
// E = 1/2 <M d, d> + 1/2 <K u^(n+1), u^n> with d = u^(n+1) - u^n, M = 1 + Lim and
// K = -lambda^2 Lex. Both operators masked as above, so this is the boundary's own energy
// balance and not the interior's. With a rigid wall it is conserved exactly, which is the
// machine-precision check the drop-ghost boundary has in place of an exact discrete mode.
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

// Free-space dispersion, measured. The box is periodic and cubic, so k = 2 pi m / N along
// whichever direction the mode vector picks, the mode is an exact discrete eigenvector, and
// seeding both levels with it makes the next level read back 2 cos(wT) - 1 at every node.
// Sweeping m walks a direction out in frequency, and the last mode inside the scheme's error
// threshold gives the oversampling ratio it needs (the paper's Eq. 38). The mode grid
// quantises that ratio, so what is reported is the conservative side of the crossing.
//
// Then the worst direction's last passing mode is run out for `steps` steps. With both levels
// seeded alike the modal amplitude is a^n = cos((n - 1/2) wT) / cos(wT / 2) exactly, so the
// deviation from it is everything the scheme's realisation adds: the Jacobi truncation at a
// fixed sweep count, and single precision.
void ImplicitRung(const ImplicitScheme &scheme, int n, int steps) {
    const double t0 = Now();
    ImplicitBox box(scheme, n, n, n);
    const int dirs[3][3] = {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}};
    static const char *const Names[3]{"axis", "face diagonal", "cube diagonal"};
    std::vector<float> seed(box.Nodes);
    auto seed_mode = [&](const double kv[3]) {
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

            // Read where the seed is largest, so the ratio is not a small difference of small
            // numbers: the node at the origin, where the cosine is 1.
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

// What the absorbing wall takes out in one step. The scheme fixes it exactly: E falls by
// 1/2 <G D, D> with D = u^(n+1) - u^(n-1) and G = (lambda/2) beta gamma, so E plus the running
// loss is what is conserved. G is diagonal, so this is a sum over the wall alone.
double ImplicitLoss(const ImplicitBox &box, const float *unext, const float *uprev) {
    double loss = 0.;
    for (const auto &[i, g] : box.WallG) {
        const double d = double(unext[i]) - double(uprev[i]);
        loss += .5 * g * d * d;
    }
    return loss;
}

// Independent noise on the two levels, each with its mean removed.
//
// Noise, not a mode: a wall's instability is one eigenvalue outside the unit circle and which
// one is not known in advance, so the seed has to project onto all of them — a mode soak can
// read clean for twenty thousand steps and diverge by eighty. The mean comes off because the
// constant vector is in both Laplacians' null space, a physical null mode whose round-off
// wander is not the wall moving.
void ImplicitSeedNoise(ImplicitBox &box, uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_real_distribution<float> uniform{-1.f, 1.f};
    std::vector<float> level[2];
    for (auto &v : level) {
        v.resize(box.Room);
        for (auto &e : v) e = uniform(rng);
        const double mean = std::accumulate(v.begin(), v.end(), 0.) / double(box.Room);
        for (auto &e : v) e -= float(mean);
    }
    const int ny = box.Ny, nz = box.Nz;
    box.Seed([&](int ix, int iy, int iz) { return level[0][(size_t(ix) * ny + size_t(iy)) * nz + size_t(iz)]; }, [&](int ix, int iy, int iz) { return level[1][(size_t(ix) * ny + size_t(iy)) * nz + size_t(iz)]; });
}

// One step of the scheme on the host in double, in the kernels' own float coefficients and
// from the same initial guess: the right-hand side, then the scheme's fixed number of Jacobi
// sweeps. This is what says the masked operator the kernels run is the operator on paper —
// the drop-ghost wall has no exact discrete mode to read a closed form against, so the
// alternative would be believing the kernel because it does not diverge.
std::vector<double> ImplicitReferenceStep(const ImplicitBox &box, const float *un, const float *up) {
    const double R[3]{box.P.R1, box.P.R2, box.P.R3}, Q[3]{box.P.Q1, box.P.Q2, box.P.Q3};
    std::vector<double> cur(box.Nodes, 0.), prev(box.Nodes, 0.), bp(box.Nodes, 0.), x(box.Nodes, 0.), xn(box.Nodes, 0.);
    for (size_t i = 0; i < box.Nodes; ++i) {
        cur[i] = double(un[i]);
        prev[i] = double(up[i]);
    }
    // The halo is zero, so the sums need no mask: an offset that leaves the room reads it.
    auto sums = [&](const std::vector<double> &u, int ix, int iy, int iz, double s[3]) {
        s[0] = s[1] = s[2] = 0.;
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dz = -1; dz <= 1; ++dz) {
                    const int r = dx * dx + dy * dy + dz * dz;
                    if (r) s[r - 1] += u[box.Index(ix + dx, iy + dy, iz + dz)];
                }
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
                bp[i] = b + double(t.Cn) * cur[i] + double(t.Cp) * prev[i];
                x[i] = cur[i]; // the sweeps start from u^n, as the GPU's ping-pong does
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
                    xn[i] = v * double(box.WallTerms(ix, iy, iz).Fd);
                }
            }
        }
        x.swap(xn);
    }
    return x;
}

// The drop-ghost wall of Smits & Bilbao 2025 Sec IV, in the two ways it can be checked to
// machine precision.
//
// The first is the operator: one step of the GPU's kernels against the same step taken on the
// host in double. The second is the wall's own energy balance, which is what replaces the
// exact discrete mode the Neumann image had. With a rigid wall the paper's
// E = 1/2 <M d, d> + 1/2 <K u^(n+1), u^n> is conserved exactly, and it is conserved by the
// boundary as well as the interior — an operator that is not exactly symmetric and negative
// semidefinite cannot conserve it, whatever a mode soak says.
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

    // The energy the scheme conserves, sampled rather than accumulated: it is a function of
    // the two levels alone, so nothing about it depends on how often it is read.
    const double e0 = ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>());
    const int every = std::max(1, soak / 160); // the readback synchronises, so sample it
    double drift = 0.;
    for (int i = 0; i < soak; ++i) {
        box.Step();
        if ((i + 1) % every) continue;
        drift = std::max(drift, std::abs(ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>()) - e0) / std::abs(e0));
    }
    std::printf("  %d steps: worst energy drift %.2e of E0 | %.1fs\n", soak, drift, Now() - t0);
}

// The implicit scheme in a rigid box, against Morse & Ingard. This is the rung that says
// whether the scheme is a room solver rather than a dispersion instrument, and it is available
// without any reference implementation: a rigid box has analytic eigenfrequencies at any grid
// spacing, so nothing about it needs a golden.
//
// Drop-ghost is a full-cell finite-volume termination (Hamilton, DAFx 2014): the wall lies on
// the cell face, half a cell beyond the outermost node, so a room of N nodes is N*h wide and
// the node at index i sits at (i + 1/2)h from the wall. The product cosine sampled there is
// therefore the mode's trial vector, but — unlike under the image — it is not an exact
// discrete eigenvector: the two differ at wall-crossing diagonal offsets. So the frequency is
// read as the Rayleigh quotient <K v, v> / <M v, v> of that trial vector, which is what the
// generalised eigenvalue is stationary about, rather than as one step of the recurrence.
//
// What that measures is where the wall effectively sits. Fitting one isotropic offset across
// the modes leaves a residual that is the 27-point stencil's own boundary layer, and shrinks
// with the grid rather than with the mode.
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
    // Ordered by the frequency the nominal wall gives, which is the order the fit reports in.
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

    // The offset the modes agree on, and what is left after it. A scan rather than a solve:
    // the residual is not quadratic in the offset and the grid it is scanned on is finer than
    // anything the number is quoted to.
    auto residual = [&](double off) {
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

// The amplitude decay a seeded mode shows, in nepers a step, read off the energy rather than
// off a node: E goes as amplitude squared, so the slope of log E is -2 alpha T, and there is no
// sinusoid to fit.
//
// The window is bounded by decay rather than by step count, and this is the whole difficulty
// of the measurement. The seed's own error excites modes it is not, and with a
// frequency-dependent wall those absorb at their own rate rather than at this one's — so the
// longer the run, the more of the survivors and the less of the mode being measured is in the
// slope. Ten decibels of it, which is the early-decay window a room's T10 is read over and for
// the same reason. The first fifth goes because a mode started from rest settles first.
template<typename F> double ImplicitDecay(ImplicitBox &box, F &&mode, int steps) {
    box.Seed(mode, mode);
    std::vector<double> logs, at;
    const int every = std::max(1, steps / 200); // each sample reads back and re-sums the energy
    for (int i = 0; i < steps; ++i) {
        box.Step();
        if ((i + 1) % every) continue;
        const double e = ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>());
        if (!(e > 0.) || !std::isfinite(e)) break;
        logs.push_back(std::log(e));
        at.push_back(double(i + 1));
        if (logs.front() - logs.back() > std::numbers::ln10) break; // ten decibels of energy
    }
    return -.5 * ImplicitSlope(at, logs, logs.size() / 5);
}

// The paper's admittance wall: a real gamma, reaching the node through the beta its own ghost
// set gives it. Nothing in the kernels changes for it — see ImplicitBox's constructor — so what
// this rung checks is the two numbers the host now computes, beta and G.
//
// Three readings, in the order they would fail. beta on a grid-aligned flat wall is 1 exactly
// or the paper's two alpha_r readings have been crossed. The energy balance is then exact:
// with a wall that absorbs, what is conserved is E plus the running loss, and that is a
// machine-precision statement the way conservation was for the rigid wall. And the decay is
// the physics — the discrete room against the analytic one.
//
// The analytic side is a box with the same gamma on all six walls. For an axial mode the
// pressure is antinodal at the two walls it runs into and uniform across the four it does not,
// so the modal amplitude decay is alpha = c gamma (2/Lx + 1/Ly + 1/Lz), the slab's 2 gamma c/L
// with the side walls added. It does not depend on the mode index, which is what makes it
// readable off the energy: the modes the seed's own error also excites decay at the same rate,
// so contamination moves the amplitude and leaves the slope alone. E goes as amplitude
// squared, so the slope of log E is -2 alpha T.
void ImplicitAbsorbRung(const ImplicitScheme &scheme, int nx, int ny, int nz, double gamma, int steps) {
    const double t0 = Now();
    ImplicitBox box(scheme, nx, ny, nz, true, 0., {gamma});
    const ImplicitShape shape(ImplicitShape::Array, nx, ny, nz);
    const double nex[3]{box.Nex(0), box.Nex(1), box.Nex(2)};
    int k[3];
    std::printf("[implicit] %s admittance wall on a %d x %d x %d room, gamma %.3f, lambda %.4f, %d distinct terms\n", scheme.Name, nx, ny, nz, gamma, box.Lambda, box.Codes);
    std::printf("  beta on a flat face %.9f, which is 1 exactly when the wiring is right\n", ImplicitWallSample(shape, 0, ny / 2, nz / 2, nex, k, true));

    // The balance leg is the short one: D spans two steps, so every step has to be counted and
    // both levels read back.
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

// The same room, stepped, with its walls off the grid — which is the experiment the whole
// staircase compensation exists for.
//
// A yawed box is a box: its modes are still (c/2) sqrt(sum (m_a/L_a)^2) at its own side
// lengths, and an axial mode still decays at c gamma (2/Lx + 1/Ly + 1/Lz). Nothing analytic
// about it changes when it is turned off the grid. What changes is that every wall is now a
// staircase, so the two readings separate:
//
//  - **the frequencies**, which staircasing should leave alone. Bilbao & Hamilton 2017 is
//    explicit that mode frequencies converge with h and decay times do not, and that is the
//    reason the compensation is load-bearing rather than a refinement.
//  - **the decay**, which staircasing biases because the staircased surface is larger than the
//    one it stands for. The rung runs it twice: once with beta reading the surface's normal,
//    and once with it reading the staircase's own, which is what an implementation that never
//    looked at the mesh would do.
//
// Frequencies are the Rayleigh quotient of the true room's continuous mode sampled at the
// nodes it keeps, the same variational reading the grid-aligned room uses. Decay is read off
// the energy.
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

    // The room's own modes, in the order the true box gives them.
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

    // Which room the modes say this is. The shape's own extents are not it: the inside test
    // keeps a node when it is half a cell in from the surface, and against a centre placed a
    // fraction of a cell off the grid that leaves the outermost node somewhere between half a
    // cell and a cell and a half from the wall — an ambiguity the size of the errors being
    // read. So the same fit the grid-aligned room uses, one offset in cells across all three
    // sides, and what is left after it is the staircase's own contribution.
    auto analytic = [&](const Mode &mode, double off) {
        double q = 0.;
        for (int a = 0; a < 3; ++a) q += std::pow(mode.M[a] / ((cells[a] + off) * h), 2.);
        return .5 * c * std::sqrt(q);
    };
    auto residual = [&](double off) {
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

    // And the decay, compensated against not, at the room the modes just fitted — so what the
    // ratio carries is absorption and not where the wall is. Modes along the room's own long
    // axis, seeded in its own coordinates.
    ImplicitBox raw(scheme, n, n, n, true, 0., {gamma}, ImplicitShape::Slanted, angle, false);
    const double fitted[3]{cells[0] + best, cells[1] + best, cells[2] + best};
    const double exact = gamma * box.Lambda * (2. / fitted[2] + 1. / fitted[0] + 1. / fitted[1]);
    std::printf("  %-6s %14s %8s %14s %8s   (analytic %.6e)\n", "mode", "compensated", "ratio", "staircased", "ratio", exact);
    for (int m = 1; m <= 4; ++m) {
        auto mode = [&](int ix, int iy, int iz) {
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

// The wall's discrete admittance at a frequency. The trapezoid rule maps s to
// i(2/Ts)tan(w Ts/2), so this is the continuous admittance at the warped frequency — which at a
// room's modes against a 12 kHz step rate is the continuous one to a part in ten thousand, but
// the warp is what the wall actually realises and it costs nothing to say so.
std::complex<double> ImplicitAdmittance(const ImplicitWall &wall, double f) {
    const std::complex<double> s{0., 2. / wall.Ts * std::tan(std::numbers::pi * f * wall.Ts)};
    std::complex<double> y{0., 0.};
    for (int m = 0; m < wall.Branches(); ++m) {
        y += 1. / (wall.Def[3 * size_t(m)] * s + wall.Def[3 * size_t(m) + 1] + wall.Def[3 * size_t(m) + 2] / s);
    }
    return y;
}

// What the LRC wall holds, and what left it over the step just taken.
//
// Multiplying the branch relation by mu(vh) telescopes it: Dh vh^2 + Fh gh^2 is stored and
// 2 Eh mu(vh)^2 leaves a step. So what a room with frequency-dependent walls conserves is the
// interior energy plus (lambda/2) beta sum_m [Dh vh^2 + Fh gh^2] at every wall node, plus
// everything dissipated so far. That identity — a boundary energy with branch storage over a
// 27-point operator — is the one the 2017 analysis leaves at nearest-neighbour cells and the
// 2025 analysis leaves out, and it is what this rung reads.
//
// beta comes back out of the node's own Psi rather than being recomputed: Psi is lambda beta
// over the interior diagonal, and those are the numbers actually in force.
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

// Frequency-dependent walls: the scheme's walls as every scene here actually carries them, a
// set of parallel LRC branches rather than a real admittance.
//
// Three readings, in the order they would fail.
//
//  - A branch that is pure resistance *is* a real admittance: D = F = 0 leaves b = gamma, and
//    the history term identically zero. So the two paths have to agree to the last bit, and
//    that is the check that the branch algebra is wired to the same place gamma was.
//  - The energy identity above, which is the machine-precision statement for a wall that both
//    stores and dissipates.
//  - And the physics: with a resonant branch the wall's Re{Y} varies across the room's own
//    modes, and each mode has to decay at c Re{Y(f)} (2/Lx + 1/Ly + 1/Lz). A wall that only
//    happened to absorb the right *average* amount would fail this.
void ImplicitLrcRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int steps) {
    const double t0 = Now();
    const double h = 0.0326857142857142, c = 343.2;
    const double lambda = std::min(scheme.Lambda, ImplicitCourantLimit(scheme)), ts = lambda * h / c;
    const double side[3]{nx + ImplicitWallOffset, ny + ImplicitWallOffset, nz + ImplicitWallOffset};
    // alpha = Re{Y} * this, in nepers a step: the same room the real-admittance rung measures.
    const double geometry = lambda * (2. / side[0] + 1. / side[1] + 1. / side[2]);

    std::printf("[implicit] %s LRC walls on a %d x %d x %d room, lambda %.4f, %.1f kHz step rate\n", scheme.Name, nx, ny, nz, lambda, 1e-3 / ts);

    // One branch of pure resistance, against the real admittance it is.
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

    // A resonance and a resistive floor, so Re{Y} varies across the room's own modes rather
    // than sitting at one number the decay could match by accident.
    const double f0 = 200., d = .02, e = d * 2. * std::numbers::pi * f0, f = d * std::pow(2. * std::numbers::pi * f0, 2.);
    const ImplicitWall material{0., ts, {d, e, f, 0., 100., 0.}};
    ImplicitBox box(scheme, nx, ny, nz, true, 0., material);
    std::printf("  %d branches, a resonance at %.0f Hz beside a resistive floor, %d wall nodes\n", material.Branches(), f0, box.P.Nw);

    // The energy the wall stores and dissipates, beside the interior's.
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

// What the wall looks like on geometry that is not the array, which is the one thing a box
// cannot show. Nothing here steps: all of it is a property of the voxelisation, and the shapes
// answer the inside test and the true normal in closed form.
//
// Three numbers the next build item rests on.
//
//  - **beta where the surface is slanted or curved.** It is provably nonnegative for a planar
//    boundary, because the ghost set is then exactly the outward half-space. Sharp convex
//    edges and thin fins are the cases that could break it, and a staircased sphere carries
//    every orientation at once. A negative beta is not cosmetic: G rides on A's diagonal, so a
//    negative one takes the wall out of the energy framework that makes it stable.
//  - **how many distinct boundary terms a staircase produces.** RoomImplicitTerm is reached by
//    a byte a node, so 255 is the ceiling, and a rectangular room's four says nothing about a
//    real one. The terms split as Cn from the kept counts alone and Cp, Fd from those and
//    G = (lambda/2) beta gamma, so what is reported is both: the classes the counts give, which
//    is what a rigid wall needs, and the triples once a continuous beta is in them.
//  - **how much of the bounding box is outside the room.** Those nodes are masked by the same
//    byte, but they are not skipped — every pass carries them at full cost, and an implicit
//    step is eight passes where an explicit one is a single pass.
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
                if (k[0] == 6 && k[1] == 12 && k[2] == 8) continue; // ordinary air
                if (k[0] < 6) ++axis_wall; // and in the reference voxeliser's boundary set too
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

    // The distinct terms, exactly and with beta rounded to `bits`. A byte a node can index 255
    // of them, so the second number is what says whether quantising beta is a way to keep it.
    auto distinct = [&](int bits) {
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
    std::printf("  inside %.3fM nodes, %.1f%% of the array — the other %.1f%% is outside and every pass carries it\n", double(inside) / 1e6, 100. * double(inside) / double(nodes), 100. * (1. - double(inside) / double(nodes)));
    std::printf("  wall nodes %zu over 26 directions, %zu over the 6 a reference voxeliser marks (%.1f%% of them)\n", wall.size(), axis_wall, 100. * double(axis_wall) / double(std::max<size_t>(wall.size(), 1)));
    std::printf("  beta %.4f to %.4f, mean %.4f, %zu negative (a flat wall is 1)\n", lo, hi, mean, negative);
    std::printf("  distinct terms %zu exact, %zu with beta to 8 bits, over %zu kept-count classes\n", distinct(0), distinct(8), classes.size());
    std::printf("  | %.1fs\n", Now() - t0);
}

// What Courant number the wall actually holds at, beside the free-space one the scheme is
// published with.
//
// The paper's stability analysis covers the periodic operator, which is symmetric. So is the
// drop-ghost wall: an offset that crosses the wall is dropped from both of the nodes it would
// have joined, and the diagonal each loses is the count it dropped. (The Neumann image is not
// — a wall-crossing diagonal offset maps onto a node that does not map back — and it grows at
// every Courant number this sweep tries.) The energy bound the wall inherits from that
// symmetry is E >= 0 iff 1 + Lim + (lambda^2/4) Lex > 0, which is a matrix condition on the
// walled operator rather than the free-space limit, so what it comes to is measured here.
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

        // A growth *rate*, not a verdict on one soak length. An unstable wall here grows slowly
        // enough that a short run reads clean — at the published Courant number mode (0,1,1) is
        // still within 4% of exact after 20,000 steps and reaches 1e6 by 80,000 — so asking
        // "did the envelope move" mostly measures how long the run was. Fitting log(envelope)
        // against step index gives a number that means the same thing at any soak length, and
        // that a short run reports as small and positive rather than as stable.
        const double e0 = ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>());
        std::vector<double> logs, at;
        const int every = std::max(1, soak / 160); // the readback synchronises, so sample it
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
        const double rate = ImplicitSlope(at, logs); // infinite if the run left the floats behind
        // dB per thousand steps, which is the scale a room render is judged at: over the
        // full-bandwidth scene's 77,078 steps +0.01 becomes +0.8 dB and +0.5 becomes 385.
        // Against 0.02 rather than zero: the legs below the limit scatter by about +-0.012 at
        // this soak length, which is the fit's own resolution and not motion in the envelope.
        // Over the full-bandwidth scene's 77,078 steps 0.02 dB per thousand is 1.5 dB.
        const double db_per_k = 1000. * rate * 20. / std::numbers::ln10;
        const bool held = db_per_k < 0.02;
        if (held && lambda > best) best = lambda;
        // The energy beside the envelope, because the two separate the two ways this can fail.
        // The scheme conserves E whatever lambda is; what a lambda past the limit costs is E's
        // positivity, and an indefinite E is conserved while the amplitude it no longer bounds
        // grows. So conserved-and-growing is the Courant failure and drifting is a broken
        // operator, and neither reads as the other.
        const double drift = std::abs(ImplicitEnergy(box, box.Cur.As<float>(), box.Prev.As<float>()) - e0) / std::abs(e0);
        std::printf("  lambda %.4f (%.0f%% of free space)  growth %+8.4f dB per 1000 steps, energy drift %.1e  %s\n", lambda, 100. * scale, db_per_k, drift, held ? "holds" : "GROWS");
    }
    if (best > 0.) std::printf("  largest holding: lambda %.4f, %.0f%% of the free-space limit | %.1fs\n", best, 100. * best / free_limit, Now() - t0);
    else std::printf("  nothing held: the wall is unstable at every Courant number tried | %.1fs\n", Now() - t0);
}

// What a step of the implicit scheme costs, against the traffic it cannot avoid. A sweep and
// the plain stream move the same twelve bytes a node, so the stream is the box's ceiling and
// the sweep's share of it says whether the scheme reaches the memory system the way the
// explicit ones do.
void ImplicitCostRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int reps, bool rigid = false) {
    ImplicitBox box(scheme, nx, ny, nz, rigid);
    auto &ctx = MetalContext::Get();
    // GPU seconds `once` takes, after a discarded warm-up round of the same length — the GPU's
    // clock ramp reads the first rounds of any microbenchmark several times too slow.
    auto measure = [&](auto &&once) {
        for (int round = 0; round < 2; ++round) {
            for (int i = 0; i < reps; ++i) once();
            ctx.Sync();
            if (round == 0) ctx.TakeBatchGpuSeconds();
        }
        return ctx.TakeBatchGpuSeconds() / reps;
    };
    auto time = [&](MTL::ComputePipelineState *pso, std::initializer_list<GpuSlice> buffers) {
        return measure([&] { ctx.Dispatch(pso, box.Tiles, box.Threads, buffers, &box.P, sizeof box.P); });
    };
    const double stream = time(box.Stream, {&box.Tmp, &box.Cur, &box.Bp});
    const double sweep = time(box.Jacobi, {&box.Tmp, &box.Cur, &box.Bp});
    const double rhs = time(box.Rhs, {&box.Bp, &box.Cur, &box.Prev});
    const double gb = 12. * double(box.Room) / 1e9;

    // A whole step, encoded the way it runs, so the dispatch cost of a short pass is in it,
    // beside the same step with the wall left out. Alternated and taken at their best, because
    // the two are read against one another and the GPU's clock drifts over a rung by more than
    // the difference between them.
    auto whole = [&](bool wall) { return measure([&] { box.Step(wall); }); };
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
        const double walled_sweep = measure([&] { ctx.Dispatch(box.JacobiFused, box.Tiles, box.Threads, {&box.Tmp, &box.Cur, &box.Bp, &box.Code, &box.Terms}, &box.P, sizeof box.P); });
        std::printf("  Jacobi sweep, walled %8.4f ms  %6.0f GB/s  against two bytes a node, %.0f%% more traffic\n", 1e3 * walled_sweep, gb / walled_sweep, 200. / 12.);
        std::printf("  wall: %d nodes (%.2f%% of the room), %d distinct terms, %.1f MB of codes\n", box.WallNodes, 100. * double(box.WallNodes) / double(box.Room), box.Codes, 2. * double(box.Nodes) / 1e6);
        std::printf("  the wall costs %.1f%% of a step (%.4f ms against %.4f without it)\n", 100. * (step / bare - 1.), 1e3 * step, 1e3 * bare);
    }
    // The step's own rate, not its share of the isolated stream. At a few megabytes a pass
    // that stream is latency-bound and swings by a factor of three run to run, while the step
    // — eight passes alternating between buffers rather than chaining on one — is steady.
    std::printf("  a whole step      %8.4f ms  %6.0f GB/s over %d passes  |  %.1f ps/node-step\n", 1e3 * step, double(scheme.Sweeps + 1) * gb / step, scheme.Sweeps + 1, 1e12 * step / double(box.Room));
}

// How much of the memory system a step reaches. The interior update and a plain two-level
// stream over the same box move the same twelve bytes a node-step, so the stream is the grid's
// own ceiling under the conditions of the moment and the update's share of it is the roofline
// statement.
// A grid of nothing but air, for asking how a scheme streams at a size no scene here reaches.
// Every wall is the absorbing shell, so there are no boundary nodes to build and none of the
// O(volume) surface search SealedBox does — which is what makes a 10^9-node grid constructible
// in the time it takes to allocate it. The spacing is a real one so the coefficients are, but
// nothing about this grid is a room: it is the interior update's streaming behaviour alone.
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

    // The boundary pass against its own ceiling, not the grid pass's: it gathers and scatters
    // over a subset of the nodes across eight streams, and a dense stream is not its roof.
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

int main(int argc, char **argv) {
    bool modes = false, energy = false, tube = false, balance = false, soak = false, golden = false, gate = false, any = false, fcc = false, roofline = false, implicit = false;
    int repeat = 4, count = 16, steps = 0, reports = 10, length = 2000;
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
        else if (arg == "--repeat" && a + 1 < argc) repeat = std::atoi(argv[++a]);
        else if (arg == "--count" && a + 1 < argc) count = std::atoi(argv[++a]);
        else if (arg == "--steps" && a + 1 < argc) steps = std::atoi(argv[++a]);
        else if (arg == "--reports" && a + 1 < argc) reports = std::atoi(argv[++a]);
        else if (arg == "--length" && a + 1 < argc) length = std::atoi(argv[++a]);
        else if (arg == "--scene" && a + 1 < argc) config_file = argv[++a];
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
        // Two footprints, because the ceiling is a property of the footprint. The first is
        // RoomChurch's own 21.0 x 13.7 x 7.4 m room at the 1% scheme's grid step of 183 mm,
        // from the 2.68 points per wavelength measured above at 700 Hz. The second is large
        // enough to be served by DRAM rather than cache, where a full-bandwidth room sits.
        ImplicitWallRung(Implicit1Pct, 40, 33, 28, steps ? steps : 20000);
        ImplicitModeRung(Implicit1Pct, 40, 33, 28, count == 16 ? 16 : count);
        ImplicitAbsorbRung(Implicit1Pct, 40, 33, 28, 0.05, steps ? steps : 2000);
        ImplicitYawedRung(Implicit1Pct, 96, 0.4, 0.05, count == 16 ? 12 : count, steps ? steps : 2000);
        ImplicitLrcRung(Implicit1Pct, 40, 33, 28, steps ? steps : 2000);
        // What a wall that is not grid-aligned does to beta and to the term count, on the two
        // shapes the staircase-compensation paper validates against. The yaw is deliberately
        // not a nice fraction of a right angle, so no wall lands on the grid.
        // The column is the one of the four whose wall is convex into the room, which is where
        // beta could go negative — a box and a sphere are concave from the inside everywhere.
        for (const auto &shape : {ImplicitShape(ImplicitShape::Array, 96, 96, 96), ImplicitShape(ImplicitShape::Slanted, 96, 96, 96, 0.4), ImplicitShape(ImplicitShape::Sphere, 96, 96, 96), ImplicitShape(ImplicitShape::Obstacle, 96, 96, 96)})
            ImplicitGeometryRung(Implicit1Pct, shape, 0.05);
        ImplicitBoxCourantRung(Implicit1Pct, 40, 33, 28, steps ? steps : 40000);
        ImplicitCostRung(Implicit1Pct, 116, 76, 41, repeat == 4 ? 200 : repeat);
        ImplicitCostRung(Implicit1Pct, 384, 384, 384, repeat == 4 ? 20 : repeat);
        // The same two footprints walled, with the room two nodes smaller on each axis so the
        // array it sits in is the periodic box above: a halo moves the strides, and a stride
        // is not what a wall costs.
        ImplicitCostRung(Implicit1Pct, 114, 74, 39, repeat == 4 ? 200 : repeat, true);
        ImplicitCostRung(Implicit1Pct, 382, 382, 382, repeat == 4 ? 20 : repeat, true);
    }
    // The gate ignores every sweep flag above: a threshold belongs to the configuration it
    // was recorded at, and the sweeps exist to leave that configuration.
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
