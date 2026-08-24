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
#include <numbers>
#include <stdexcept>
#include <string>
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
    return {nx, ny, nz, float((2. * n[0] + l2 * e[0]) / jd), float((2. * n[1] + l2 * e[1]) / jd), float((2. * n[2] + l2 * e[2]) / jd), float((2. * jd - l2 * ediag) / jd), float(n[0] / jd), float(n[1] / jd), float(n[2] / jd)};
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

struct ImplicitBox {
    ImplicitBox(const ImplicitScheme &s, int nx, int ny, int nz)
        : Scheme(s), Lambda(std::min(s.Lambda, ImplicitCourantLimit(s))), P(ImplicitCoefficients(s, Lambda, nx, ny, nz)), Nodes(size_t(nx) * ny * nz) {
        for (auto *b : {&Prev, &Cur, &Bp, &Tmp}) b->ResizeZeroed(Nodes * sizeof(float));
        auto &ctx = MetalContext::Get();
        Rhs = ctx.RoomPipeline("RoomImplicitRhs");
        Jacobi = ctx.RoomPipeline("RoomImplicitJacobi");
        Stream = ctx.RoomPipeline("RoomImplicitStream");
        Threads = {32, 4, 4};
        Tiles = {(uint32_t(nz) + Threads.x - 1) / Threads.x, (uint32_t(ny) + Threads.y - 1) / Threads.y, (uint32_t(nx) + Threads.z - 1) / Threads.z};
    }

    // Four grids of state, which is the paper's S = 4: the two levels behind, the Jacobi
    // constant vector, and one scratch grid the sweeps ping-pong through. With an odd sweep
    // count the answer lands in the grid the oldest level occupied, so nothing is copied and
    // the rotation is a swap of handles.
    void Step() {
        auto &ctx = MetalContext::Get();
        ctx.Dispatch(Rhs, Tiles, Threads, {&Bp, &Cur, &Prev}, &P, sizeof P);
        const GpuBuffer *src = &Cur, *dst = &Prev;
        for (int m = 0; m < Scheme.Sweeps; ++m) {
            ctx.Dispatch(Jacobi, Tiles, Threads, {dst, src, &Bp}, &P, sizeof P);
            src = dst;
            dst = (dst == &Prev) ? &Tmp : &Prev;
        }
        if (src != &Prev) { // an even sweep count leaves it in the scratch grid
            Prev.Swap(Tmp);
        }
        Prev.Swap(Cur); // Cur holds the new level, Prev the one it replaced
    }

    const ImplicitScheme &Scheme;
    double Lambda; // the scheme's own Courant limit where Table II's rounded value exceeds it
    RoomImplicitParams P;
    size_t Nodes;
    GpuBuffer Prev, Cur, Bp, Tmp;
    MTL::ComputePipelineState *Rhs, *Jacobi, *Stream;
    Dim3 Tiles, Threads;
};

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

// What a step of the implicit scheme costs, against the traffic it cannot avoid. A sweep and
// the plain stream move the same twelve bytes a node, so the stream is the box's ceiling and
// the sweep's share of it says whether the scheme reaches the memory system the way the
// explicit ones do.
void ImplicitCostRung(const ImplicitScheme &scheme, int nx, int ny, int nz, int reps) {
    ImplicitBox box(scheme, nx, ny, nz);
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
    const double gb = 12. * double(box.Nodes) / 1e9;

    // A whole step, encoded the way it runs, so the dispatch cost of a short pass is in it.
    const double step = measure([&] { box.Step(); });

    std::printf("[implicit cost] %s %d x %d x %d, %.3fM nodes, %.1f MB a pass, %.1f MB of state\n", scheme.Name, nx, ny, nz, double(box.Nodes) / 1e6, 1e3 * gb, 4. * 4. * double(box.Nodes) / 1e6);
    std::printf("  three-stream pass %8.4f ms  %6.0f GB/s\n", 1e3 * stream, gb / stream);
    std::printf("  Jacobi sweep      %8.4f ms  %6.0f GB/s  %.0f%% of the stream\n", 1e3 * sweep, gb / sweep, 100. * stream / sweep);
    std::printf("  right-hand side   %8.4f ms  %6.0f GB/s  %.0f%% of the stream\n", 1e3 * rhs, gb / rhs, 100. * stream / rhs);
    // The step's own rate, not its share of the isolated stream. At a few megabytes a pass
    // that stream is latency-bound and swings by a factor of three run to run, while the step
    // — eight passes alternating between buffers rather than chaining on one — is steady.
    std::printf("  a whole step      %8.4f ms  %6.0f GB/s over %d passes  |  %.1f ps/node-step\n", 1e3 * step, double(scheme.Sweeps + 1) * gb / step, scheme.Sweeps + 1, 1e12 * step / double(box.Nodes));
}

// How much of the memory system a step reaches. The interior update and a plain two-level
// stream over the same box move the same twelve bytes a node-step, so the stream is the grid's
// own ceiling under the conditions of the moment and the update's share of it is the roofline
// statement.
void RooflineRung(const std::string &config_file, int reps) {
    const double t0 = Now();
    RoomScene const scene = LoadRoomScene(config_file);
    RoomGpu gpu;
    gpu.Init(scene);
    const auto band = gpu.Roofline(reps);
    const double gb = band.Bytes / 1e9;
    std::printf("[roofline] %s %s %d x %d x %d, %.2fM nodes, %.0f MB a pass\n", config_file.c_str(), scene.Fcc ? "FCC" : "cart", scene.Nx, scene.Ny, scene.Nz, double(scene.NumNodes()) / 1e6, 1e3 * gb);
    std::printf("  two-level stream  %8.4f ms  %6.0f GB/s\n", 1e3 * band.Stream, gb / band.Stream);
    std::printf("  interior update   %8.4f ms  %6.0f GB/s  %.0f%% of the stream | %.1fs\n", 1e3 * band.Update, gb / band.Update, 100. * band.Stream / band.Update, Now() - t0);
}
} // namespace

int main(int argc, char **argv) {
    bool modes = false, energy = false, tube = false, balance = false, soak = false, golden = false, gate = false, any = false, fcc = false, roofline = false, implicit = false;
    int repeat = 4, count = 16, steps = 0, reports = 10, length = 2000;
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
        else {
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
    if (roofline) RooflineRung(config_file, repeat == 4 ? 200 : repeat);
    if (implicit) {
        for (const auto &s : {Implicit2Pct, Implicit1Pct, ImplicitHalfPct}) ImplicitRung(s, count == 16 ? 96 : count, steps ? steps : 2000);
        // Two footprints, because the ceiling is a property of the footprint. The first is
        // RoomChurch's own 21.0 x 13.7 x 7.4 m room at the 1% scheme's grid step of 183 mm,
        // from the 2.68 points per wavelength measured above at 700 Hz. The second is large
        // enough to be served by DRAM rather than cache, where a full-bandwidth room sits.
        ImplicitCostRung(Implicit1Pct, 116, 76, 41, repeat == 4 ? 200 : repeat);
        ImplicitCostRung(Implicit1Pct, 384, 384, 384, repeat == 4 ? 20 : repeat);
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
