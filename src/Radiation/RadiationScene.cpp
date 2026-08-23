// Scene runner for the SonicRadiation solver. See RadiationScene.h.
//
// Per scene: the config's grid (grown until the body clears the PML shell and its own near
// sets), the object mesh retargeted to that grid by RadiationMesh::Remesh, one CQM table
// build, and then a pure GPU step loop. The boundary condition is the modal body's normal
// acceleration, g = dp/dn = -rho a_n, at element centroids: the modal solver runs at the
// grid's timestep and one GEMM per batch turns its accelerations into per-element ones.
//
// Listeners are written on both output paths — the grid sample and the Eq. 5 retarded
// potential — because they have opposite error profiles (grid dispersion versus CQM
// frequency warping), and which one wins depends on the listener's distance.

#include "RadiationScene.h"

#include "AabbTree.h"
#include "Mesh.h"
#include "ModalSound.h"
#include "Parallel.h"
#include "Profile.h"
#include "RadiationGpu.h"
#include "json.hpp"

#define ACCELERATE_NEW_LAPACK // the modern interface, as everywhere else this links BLAS
#include <Accelerate/Accelerate.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;
using Eigen::Vector3d, Eigen::Vector3i;

namespace {
// Where the solver's listener output goes, relative to the working directory — kept apart
// from the WaveBlender path's <output>.bin.
constexpr const char *OutputDir = "radiation";

// A rigid pose from the scene's animation track. The convention is the WaveBlender path's:
// the keyframe rotates and translates the mesh as the file gave it, and the scene's offset
// places the result, so a world position is rotation * (rest - offset) + translation +
// offset. Identity keyframes leave the rest mesh exactly where it was.
struct Pose {
    Vector3d Translation{Vector3d::Zero()};
    Eigen::Quaterniond Rotation{Eigen::Quaterniond::Identity()};

    Vector3d Apply(const Vector3d &rest, const Vector3d &offset) const { return Rotation * (rest - offset) + Translation + offset; }
};

struct PoseTrack {
    std::vector<double> Times;
    std::vector<Pose> Keys;
    bool Moves{false}; // any keyframe differs from the first, so epochs are needed at all

    static PoseTrack Read(const std::string &path) {
        PoseTrack track;
        std::ifstream file{path};
        if (!file.good()) throw std::runtime_error("Failed to read animation: " + path);
        std::string line;
        std::getline(file, line); // header
        while (std::getline(file, line)) {
            std::istringstream is{line};
            double t;
            Vector3d translation;
            Eigen::Vector4d q; // w x y z, as the file writes it
            if (!(is >> t >> translation[0] >> translation[1] >> translation[2] >> q[0] >> q[1] >> q[2] >> q[3])) continue;
            track.Times.push_back(t);
            track.Keys.push_back({translation, Eigen::Quaterniond{q[0], q[1], q[2], q[3]}.normalized()});
        }
        if (track.Keys.empty()) throw std::runtime_error("Animation has no keyframes: " + path);
        for (const auto &key : track.Keys) {
            track.Moves = track.Moves || (key.Translation - track.Keys.front().Translation).norm() > 1e-9 ||
                key.Rotation.angularDistance(track.Keys.front().Rotation) > 1e-9;
        }
        return track;
    }

    Pose At(double t) const {
        if (t <= Times.front()) return Keys.front();
        if (t >= Times.back()) return Keys.back();
        const size_t k = size_t(std::ranges::upper_bound(Times, t) - Times.begin()) - 1;
        const double span = Times[k + 1] - Times[k];
        const double alpha = span > 1e-12 ? (t - Times[k]) / span : 0.;
        return {(1. - alpha) * Keys[k].Translation + alpha * Keys[k + 1].Translation, Keys[k].Rotation.slerp(alpha, Keys[k + 1].Rotation)};
    }
};

// The rest mesh at `pose`, finalized so centroids, normals and areas match the new pose.
RadiationMesh Posed(const RadiationMesh &rest, const Vector3d &offset, const Pose &pose) {
    RadiationMesh out = rest;
    for (auto &v : out.Vertices) v = pose.Apply(v, offset);
    out.Finalize();
    return out;
}

// How far any point of the body moves between two poses, measured on the rest bounding
// box's corners — the extremes of a rigid motion sit there, so this bounds the travel that
// governs how stale a table set is (see RebasePlan for what that staleness costs).
double Travel(const std::array<Vector3d, 8> &corners, const Vector3d &offset, const Pose &a, const Pose &b) {
    double worst = 0.;
    for (const auto &corner : corners) worst = std::max(worst, (a.Apply(corner, offset) - b.Apply(corner, offset)).norm());
    return worst;
}

// Growing axis-aligned bounds, over the rest mesh and then over the poses it sweeps through.
struct Bounds {
    Vector3d Lo{Vector3d::Constant(1e30)}, Hi{Vector3d::Constant(-1e30)};

    void Add(const Vector3d &p) {
        Lo = Lo.cwiseMin(p);
        Hi = Hi.cwiseMax(p);
    }
    Vector3d Centre() const { return 0.5 * (Lo + Hi); }
    std::array<Vector3d, 8> Corners() const {
        std::array<Vector3d, 8> corners;
        for (int c = 0; c < 8; ++c) corners[c] = {c & 1 ? Hi[0] : Lo[0], c & 2 ? Hi[1] : Lo[1], c & 4 ? Hi[2] : Lo[2]};
        return corners;
    }
};

// The modal body as a Neumann source: normal acceleration at element centroids. The modal
// solver steps at the grid's timestep, so nothing resamples between it and the boundary
// condition. Accelerations are a central difference of the modal velocities plus the
// rigid-body acceleration noise of the active contact impulses (ModalSound::AccelPulse).
struct ModalNeumann {
    ModalNeumann(const json &object, const RadiationMesh &mesh, double tau, double ts, double rho)
        : Solver(object["dataPrefix"], object["meshFile"], object["material"], tau), Tau(tau), Ts(ts), Rho(rho),
          NumElems(mesh.NumElems()), NumModes(int(Solver.EigenVectorsNormal.cols())) {
        const std::vector<double> offset_values = object["offset"];
        const Vector3d offset{offset_values[0], offset_values[1], offset_values[2]};

        Eigen::MatrixXd v0;
        Eigen::MatrixXi f;
        if (!ReadObj(std::string{object["meshFile"]}, v0, f)) throw std::runtime_error("Failed to read mesh");
        AabbTree<double> tree;
        tree.Init(v0, f);

        // Mode-to-element transfer: each element's normal picks up the eigenvector's normal
        // component at the closest point of the original surface, barycentrically blended.
        // Body-frame, so it is built once — both the element normal and the eigenvector
        // rotate with the body. Only the world normals the rigid-body term projects turn.
        ModeToElem.resize(NumElems, NumModes);
        RestNormals.resize(NumElems, 3);
        ParallelFor(NumElems, 32, [&](size_t i) {
            const Vector3d n = mesh.Normals[i];
            RestNormals.row(i) = n.transpose().cast<float>();
            const Eigen::RowVector3d rest = (mesh.Centroids[i] - offset).transpose();
            int tri = 0;
            Eigen::RowVector3d closest;
            tree.ClosestPoint(rest, tri, closest);
            const Eigen::RowVector3d w = BarycentricWeights(closest, v0.row(f(tri, 0)), v0.row(f(tri, 1)), v0.row(f(tri, 2)));
            Eigen::RowVectorXd acc = Eigen::RowVectorXd::Zero(NumModes);
            for (int c = 0; c < 3; ++c) {
                const int vertex = f(tri, c);
                acc += w[c] * n.dot(Vector3d{Solver.Normals.row(vertex).transpose()}) * Solver.EigenVectorsNormal.row(vertex);
            }
            ModeToElem.row(i) = acc.cast<float>();
        });
        Normals = RestNormals;
    }

    // Turns the world normals to `rotation`, at a geometry epoch.
    void SetRotation(const Eigen::Quaterniond &rotation) {
        Normals = RestNormals * rotation.toRotationMatrix().transpose().cast<float>();
    }

    // g = dp/dn for steps [first, first + n), row-major over elements.
    void Fill(int first, int n, std::vector<float> &out) {
        const profile::Scope scope{"radiation/neumann"};
        Qddot.resize(NumModes, n);
        AccelN.resize(NumElems, n);
        for (int k = 0; k < n; ++k) {
            const int step = first + k;
            Qddot.col(k) = (QdotAt(step + 1) - QdotAt(step - 1)) / (2. * Tau);
            const Vector3d a = AccelPulse(Ts + step * Tau);
            AccelN.col(k) = (Normals * a.cast<float>()).array();
        }
        const Eigen::MatrixXf qddot = Qddot.cast<float>();
        out.resize(size_t(NumElems) * n);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NumElems, n, NumModes, 1.f, ModeToElem.data(), NumElems, qddot.data(), NumModes, 0.f, out.data(), NumElems);
        for (int k = 0; k < n; ++k) {
            float *col = out.data() + size_t(k) * NumElems;
            for (int i = 0; i < NumElems; ++i) col[i] = float(-Rho) * (col[i] + AccelN(i, k));
        }
    }

    ModalSound::Solver Solver;

private:
    double Tau, Ts, Rho;
    int NumElems, NumModes;
    Eigen::MatrixXf ModeToElem; // (elements x modes), column-major for the batch GEMM
    Eigen::Matrix<float, Eigen::Dynamic, 3> RestNormals, Normals;
    Eigen::MatrixXd Qddot;
    Eigen::MatrixXf AccelN;

    // Modal velocities at the last few steps. Access is monotone with a one-step lookahead
    // and lookbehind, so three slots suffice.
    Eigen::VectorXd Ring[3];
    int Produced{0};

    const Eigen::VectorXd &QdotAt(int step) {
        if (step < 0) {
            if (Rest.size() != NumModes) Rest = Eigen::VectorXd::Zero(NumModes);
            return Rest;
        }
        while (Produced <= step) {
            Ring[Produced % 3] = Solver.QDotCPlus;
            Solver.Step(Ts + Produced * Tau);
            ++Produced;
        }
        return Ring[step % 3];
    }
    Eigen::VectorXd Rest;

    // Rigid-body acceleration from the contact impulses active at `time`.
    Vector3d AccelPulse(double time) const {
        Vector3d accel = Vector3d::Zero();
        for (const auto &impulse : Solver.Impulses.Records) accel += ModalSound::AccelPulse(Solver, impulse, time);
        return accel;
    }
};

// Grows the grid until the body clears the PML shell and the reach of its own near sets at
// every pose it takes, keeping the configured cell size and the physical positions of
// everything in it. A moving body is fitted over its whole trajectory, because the grid is
// fixed for the run while the tables are rebuilt against it.
void FitGridToBox(RadiationGrid &grid, const Vector3d &box_lo, const Vector3d &box_hi) {
    const int margin = grid.PmlWidth + grid.R2 + 2;
    const Vector3i lo = grid.CellOf(box_lo), hi = grid.CellOf(box_hi);
    int n[3]{grid.Nx, grid.Ny, grid.Nz};
    for (int d = 0; d < 3; ++d) {
        const int pad_lo = std::max(0, margin - lo[d]), pad_hi = std::max(0, margin - (n[d] - 1 - hi[d]));
        n[d] += pad_lo + pad_hi;
        grid.Origin[d] -= pad_lo * grid.H;
    }
    grid.Nx = n[0];
    grid.Ny = n[1];
    grid.Nz = n[2];
}
} // namespace

void RunRadiationScene(const std::string &config_file, double seconds) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0); // progress lines are the only view of a long run
    const double t_start = Now();
    std::ifstream config_stream{config_file};
    if (!config_stream.good()) throw std::runtime_error("Failed to read config: " + config_file);
    json config;
    config_stream >> config;

    RadiationGrid grid;
    grid.Nx = config["Nx"];
    grid.Ny = config["Ny"];
    grid.Nz = config["Nz"];
    grid.H = config["cellsize"];
    grid.C = 343.2; // the WaveBlender path's SimParams defaults, so the two are comparable
    grid.Rho = 1.204;
    grid.Damping = config["damping"];
    grid.Origin = {-0.5 * (grid.Nx - 1) * grid.H, -0.5 * (grid.Ny - 1) * grid.H, -0.5 * (grid.Nz - 1) * grid.H};
    grid.Finalize();

    const double ts = config["ts"];
    // `seconds` shortens the render without touching the config: a scene's first few
    // hundred milliseconds are usually enough to compare against another solver.
    const double config_tf = config["tf"];
    const double tf = seconds > 0. ? std::min(config_tf, ts + seconds) : config_tf;
    const int blend_rate = config["blendrate"];

    // One rigid modal body: the scope the method covers.
    const auto &objects = config["objects"];
    if (objects.size() != 1) throw std::runtime_error("The radiation solver takes exactly one Modal object");
    const auto &object = *objects.begin();
    if (object["shader"] != "Modal") throw std::runtime_error("The radiation solver takes a Modal object");

    const std::vector<double> offset_values = object["offset"];
    const Vector3d offset{offset_values[0], offset_values[1], offset_values[2]};
    auto mesh = RadiationMesh::FromObj(object["meshFile"], offset);
    const int faces_in = mesh.NumElems();
    // The exterior representation assumes a closed boundary, and remeshing preserves
    // topology rather than repairing it, so an open input is a scene the method cannot take.
    if (!mesh.IsClosed()) throw std::runtime_error("Object boundary is not a closed manifold");
    {
        // The paper's constraint: no element larger than a cell.
        const profile::Scope scope{"radiation/remesh"};
        const double max_edge = 1.05 * grid.H;
        mesh.Remesh(RadiationMesh::RemeshTarget * max_edge, max_edge);
    }
    if (!mesh.IsClosed()) throw std::runtime_error("Remeshing did not preserve a closed manifold");
    std::printf("[radiation] mesh %d -> %d elements, max edge %.2f cells\n", faces_in, mesh.NumElems(), mesh.MaxEdge / grid.H);

    const int steps_per_batch = std::max(1, int(std::lround(1. / (blend_rate * grid.Tau))));
    const int total_steps = int(std::ceil((tf - ts) / grid.Tau));

    // The animation track, which is what decides whether this is a static-geometry scene.
    const PoseTrack track = object.contains("animFile") && !std::string{object["animFile"]}.empty() ? PoseTrack::Read(object["animFile"]) : PoseTrack{{ts}, {Pose{}}, false};

    // Poses at batch boundaries, and the box the body sweeps through them: the grid is fixed
    // for the run while the tables are rebuilt against it, so it has to clear the body at
    // every pose, not just the first.
    Bounds rest;
    for (const auto &centroid : mesh.Centroids) rest.Add(centroid);
    const auto rest_corners = rest.Corners();
    const auto pose_at = [&](int step) { return track.At(ts + step * grid.Tau); };
    std::vector<Pose> batch_poses;
    for (int step = 0; step <= total_steps; step += steps_per_batch) batch_poses.push_back(pose_at(step));
    batch_poses.push_back(pose_at(total_steps)); // the last batch is short unless it divides
    Bounds swept;
    for (const auto &pose : batch_poses)
        for (const auto &corner : rest_corners) swept.Add(pose.Apply(corner, offset));
    FitGridToBox(grid, swept.Lo, swept.Hi);
    std::printf("[radiation] grid %dx%dx%d h=%.4f tau=%.3fus (%.0f Hz) steps=%d batch=%d\n", grid.Nx, grid.Ny, grid.Nz, grid.H, grid.Tau * 1e6, 1. / grid.Tau, total_steps, steps_per_batch);

    // Geometry epochs, paced by travel rather than by a step count, since travel between
    // table builds is what governs accuracy (see RebasePlan). Tables are built for the
    // *midpoint* of the interval they serve, which halves the worst staleness for nothing,
    // and epochs land on batch boundaries so a batch's Neumann samples belong to one pose.
    // Far-field re-basing is left out: it measures as a wash, and it would have to run
    // between the batches it separates.
    constexpr double EpochTravel = 0.25; // cells of travel per table build
    std::vector<int> epoch_start{0};
    if (track.Moves) {
        for (int begin = 0; begin < total_steps;) {
            int end = std::min(begin + steps_per_batch, total_steps);
            while (end < total_steps) {
                const Pose mid = pose_at((begin + end) / 2);
                const double stale = std::max(Travel(rest_corners, offset, mid, pose_at(begin)), Travel(rest_corners, offset, mid, pose_at(end)));
                if (stale >= EpochTravel * grid.H) break;
                end = std::min(end + steps_per_batch, total_steps);
            }
            if (end < total_steps) epoch_start.push_back(end);
            begin = end;
        }
    }
    const auto epoch_end = [&](size_t k) { return k + 1 < epoch_start.size() ? epoch_start[k + 1] : total_steps; };
    const auto epoch_pose = [&](size_t k) { return pose_at((epoch_start[k] + epoch_end(k)) / 2); };

    // Built from the rest mesh, whose closest-point lookups are in the source mesh's own
    // frame; only the world normals the rigid-body term projects onto turn with the body.
    ModalNeumann source{object, mesh, grid.Tau, ts, grid.Rho};

    // Listener positions, in the same world frame the WaveBlender path uses.
    std::vector<Vector3d> listeners;
    std::vector<std::string> names;
    for (const auto &listener : config["listeners"]) {
        if (listener["format"] != "Mono") throw std::runtime_error("Only Mono listeners are supported");
        const std::vector<double> position = listener["position"];
        listeners.emplace_back(position[0], position[1], position[2]);
        names.push_back(listener["output"]);
    }

    // One epoch's geometry: the posed mesh and every table built over it. Nothing here is
    // read once the GPU has bound it, so the previous epoch's set is freed at the swap.
    struct Epoch {
        std::unique_ptr<Tdbem> Bem;
        std::unique_ptr<RadiationTables> Tables;
        RadPackedTables Packed;
        PairTable Points;
        Eigen::Quaterniond Rotation;
        double MeanWindow{0.}; // lags per cell-table entry, before the padding the GPU wants
    };
    const RadiationMesh rest_mesh = mesh;
    // `hist_len` zero sizes the rings from these tables, as the first build does; later
    // epochs are held to the ring the run was sized for, which they carry their history in.
    const auto build_epoch = [&](const Pose &pose, int hist_len, const PairTable *previous) {
        Epoch out;
        out.Rotation = pose.Rotation;
        out.Bem = std::make_unique<Tdbem>();
        out.Tables = std::make_unique<RadiationTables>();
        // The first epoch keeps its float weights: the rest state is seeded by a CPU
        // convolution over them. Every epoch after it is read only by the GPU, so its weights
        // are quantized as they are built and the float ones never exist.
        BuildRadiation(grid, Posed(rest_mesh, offset, pose), {}, listeners, *out.Bem, *out.Tables, previous, previous != nullptr);
        out.Points = out.Bem->BuildPointTable(listeners);
        out.Bem->EnsureHistLen(hist_len > 0 ? hist_len - 1 : out.Points.MaxLagEnd(), 0);
        if (hist_len > 0 && out.Bem->HistLen != hist_len) throw std::runtime_error("A geometry epoch outgrew the history ring");
        // Packed here rather than at the swap: it is most of what applying an epoch costs and
        // none of it needs the main thread. The cell weights are dropped with it, since
        // nothing reads them again and two epochs' worth is gigabytes; the element weights
        // stay, because the next epoch carries most of them over.
        out.MeanWindow = double(out.Tables->CellElem.Weights.size()) / double(2 * out.Tables->CellElem.Entries.size());
        out.Packed = PackRadiation(*out.Bem, *out.Tables);
        out.Tables->CellElem.Weights = {};
        return out;
    };

    // Scoped only here: later epochs build on a worker, and the phase profiler's map is not
    // safe to share between threads.
    Epoch current = [&] {
        const profile::Scope scope{"radiation/build_tables"};
        return build_epoch(epoch_pose(0), 0, nullptr);
    }();
    if (track.Moves) {
        // The rings carry across epochs, so they have to cover every pose's lags. Receding
        // from a listener by d pushes its lags out by d / (c tau) and leaves the window shape
        // alone, so the run's reach is this epoch's plus the furthest the body ever recedes.
        // A later epoch that outgrows the result fails loudly in build_epoch.
        const Vector3d centre = rest.Centre();
        double receding = 0.;
        for (const auto &pose : batch_poses)
            for (const auto &listener : listeners)
                receding = std::max(receding, (listener - pose.Apply(centre, offset)).norm() - (listener - epoch_pose(0).Apply(centre, offset)).norm());
        const int reach = std::max({current.Bem->ElemElem.MaxLagEnd(), current.Tables->CellElem.MaxLagEnd(), current.Points.MaxLagEnd()});
        current.Bem->EnsureHistLen(reach + Tdbem::RecedingLags(receding, grid.C, grid.Tau), 0);
    }
    const int hist_len = current.Bem->HistLen;
    source.SetRotation(current.Rotation);

    // The rest state carries the Neumann samples for steps 0 and 1, which the first step's
    // convolutions read. Taking them from the source rather than assuming zero is what lets a
    // scene already excited at `ts` start correctly.
    std::vector<float> seed;
    source.Fill(0, 2, seed);
    SeedRestState(*current.Bem, grid.Tau, [&](int e, double t) {
        return double(seed[(t > 0. ? size_t(rest_mesh.NumElems()) : 0) + size_t(e)]);
    });

    RadiationGpu gpu;
    gpu.Init(grid, *current.Bem, *current.Tables, current.Packed);
    gpu.SetListeners(int(listeners.size()), current.Points);
    std::printf("[radiation] tables %.0f MB | band rows %zu entries %zu | corr cells %zu refs %zu | interp refs %zu | "
                "elem entries %zu | hist %d | mean lag window %.1f | epochs %zu | build %.1fs\n",
                gpu.Bytes() / 1048576., current.Tables->CellElem.Begin.size() - 1, current.Tables->CellElem.Entries.size(), current.Tables->CorrCell.size(), current.Tables->CorrRefs.size(), current.Tables->InterpRefs.size(), current.Bem->ElemElem.Entries.size(), hist_len, current.MeanWindow, epoch_start.size(), Now() - t_start);

    // Output lands in its own directory: a different solver from the one writing
    // <output>.bin beside it, at a different sample rate.
    std::filesystem::create_directories(OutputDir);
    std::vector<std::ofstream> grid_out, eq5_out;
    for (const auto &name : names) {
        const auto stem = std::filesystem::path{OutputDir} / name;
        grid_out.emplace_back(stem.string() + "_grid.bin", std::ofstream::binary);
        eq5_out.emplace_back(stem.string() + "_eq5.bin", std::ofstream::binary);
        std::ofstream{stem.string() + ".json"} << json{{"srate", 1. / grid.Tau}, {"paths", {"grid", "eq5"}}}.dump();
    }

    const double t_steps = Now();
    std::vector<float> g_rows;
    const int n_listeners = int(listeners.size());
    std::vector<float> deinterleaved;
    // The next epoch's tables are built on a worker while the GPU steps the current one, so a
    // rebuild costs wall time only when it outruns the epoch it hides behind. `stall` is that
    // overrun, and is the number to watch when a scene moves fast enough to matter.
    std::future<Epoch> pending;
    size_t epoch = 0;
    double stall = 0., applying = 0.;
    if (epoch_start.size() > 1) pending = std::async(std::launch::async, build_epoch, epoch_pose(1), hist_len, &current.Bem->ElemElem);
    for (int step = 0; step < total_steps; step += steps_per_batch) {
        const int take = std::min(steps_per_batch, total_steps - step);
        if (epoch + 1 < epoch_start.size() && step >= epoch_start[epoch + 1]) {
            const double t_wait = Now();
            Epoch next = pending.get();
            stall += Now() - t_wait;
            const double t_apply = Now();
            gpu.ApplyEpoch({}, *next.Bem, *next.Tables, next.Packed);
            gpu.SetListeners(n_listeners, next.Points);
            applying += Now() - t_apply;
            source.SetRotation(next.Rotation);
            current = std::move(next);
            ++epoch;
            if (epoch + 1 < epoch_start.size()) pending = std::async(std::launch::async, build_epoch, epoch_pose(epoch + 1), hist_len, &current.Bem->ElemElem);
        }
        // Each step appends its successor's Neumann samples, so a batch starting at step q
        // supplies rows for q + 2 through q + take + 1.
        source.Fill(step + 2, take, g_rows);
        gpu.RunSteps(take, g_rows.data());

        const float *grid_samples = gpu.GridSamples();
        const float *eq5_samples = gpu.PointSamples();
        deinterleaved.resize(take);
        for (int l = 0; l < n_listeners; ++l) {
            for (const auto &[samples, out] : {std::pair{grid_samples, &grid_out}, std::pair{eq5_samples, &eq5_out}}) {
                for (int k = 0; k < take; ++k) deinterleaved[k] = samples[size_t(k) * n_listeners + l];
                (*out)[l].write(reinterpret_cast<const char *>(deinterleaved.data()), take * sizeof(float));
            }
        }
        if ((step / steps_per_batch) % 100 == 0)
            std::printf("[radiation] t=%.3fs (%d/%d steps, %.1fs)\n", ts + step * grid.Tau, step, total_steps, Now() - t_start);
    }
    std::printf("[radiation] done: %d steps in %.1fs (%.2f ms/step), total %.1fs\n", total_steps, Now() - t_steps, 1e3 * (Now() - t_steps) / std::max(1, total_steps), Now() - t_start);
    if (epoch_start.size() > 1)
        std::printf("[radiation] %zu geometry epochs, %.1f steps each, %.2fs binding them, %.2fs stalled on table builds\n", epoch_start.size(), double(total_steps) / double(epoch_start.size()), applying, stall);
    for (const auto &kernel : gpu.Timings()) {
        if (kernel.Count) std::printf("[radiation]   %-12s %6.3f ms/step\n", kernel.Name, 1e3 * kernel.Seconds / double(kernel.Count));
    }
    profile::Report();
}
