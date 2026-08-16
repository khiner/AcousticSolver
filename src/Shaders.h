#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (Shaders.h) — Metal port.
//
// Declares all acoustic shader types (sound sources / boundary conditions) as plain
// structs over a shared ObjectBase, dispatched through std::variant — no virtual
// classes. See the .cpp files associated with each sound source for implementation
// details. GPU kernels live in FDTD/Kernels.metal.

#include "AabbTree.h"
#include "BubbleFactorPipeline.h"
#include "FDTDCommon.h"
#include "FluidSound.h"
#include "Mesh.h"
#include "MetalContext.h"
#include "ModalSound.h"

#include <fstream>
#include <string>
#include <variant>
#include <vector>

enum class ShaderClass {
    Monopole,
    Speaker,
    Occluder,
    Bubbles,
    Modal,
    ThinShell,
    Point,
    Density,
};

// Data shared by every object: identity, shader timing, surface geometry, rigid
// animation state, and the per-batch boundary sample points the shader kernels consume.
struct ObjectBase {
    ObjectBase(ShaderClass type, bool has_shader, int blend_rate, int shader_srate, double ts)
        : Type(type), HasShader(has_shader), Srate(shader_srate), Dt(1. / shader_srate), Ts(ts), NSamples(shader_srate / blend_rate + 1) {}

    ShaderClass Type;
    bool HasShader;
    bool Changed{true}; // whether the object has moved / its geometry has changed since the last batch
    int NPoints{0}; // number of boundary points

    int Srate; // shader sample rate
    double Dt; // shader timestep size
    int Step{0}; // current shader timestep
    double Ts; // simulation start time
    int NSamples; // number of shader samples in time (i.e., the batchsize)

    Eigen::MatrixX<real> V0; // original, non-animated surface mesh vertex positions
    Eigen::MatrixXi F; // surface mesh faces

    // Current and next surface mesh vertex positions.
    // (Due to how we rasterize, we need to "look ahead" to the next triangle mesh.)
    Eigen::MatrixX<real> V1, V2;

    // B and BN are the set of all boundary positions and corresponding boundary normals
    // to compute shader samples at. See SetSamplePoints().
    Eigen::Matrix<real, Eigen::Dynamic, 3, Eigen::RowMajor> B, BN;
    GpuBuffer GpuB, GpuBN;

    // ----- Animation -----
    double T1{0}, T2{0}; // animation keyframe endpoint times
    Eigen::Vector3d Translation1, Translation2;
    Eigen::Vector4d Rotation1, Rotation2; // rotation quaternions
    std::ifstream AnimFile;

    void ReadAnimation();

    // Sets this batch's (NPoints, 3) boundary sample points and their normals.
    void SetSamplePoints(const Eigen::MatrixX<real> &b, const Eigen::MatrixX<real> &bn);

    // ----- Closest point query (must be called after Tree.Init()) -----
    AabbTree<real> Tree;
    // For each query position in `b`, the index of the closest triangle of mesh `v`, F.
    void ClosestPoint(Eigen::VectorXi &i_out, const Eigen::MatrixX<real> &b, const Eigen::MatrixX<real> &v) const;
    // As above, plus barycentric weights into `w_out`.
    void ClosestPoint(Eigen::VectorXi &i_out, Eigen::MatrixX<real> &w_out, const Eigen::MatrixX<real> &b, const Eigen::MatrixX<real> &v) const;
};

// Basic mono-frequency, monopole source shader used for testing.
struct Monopole {
    Monopole(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, real freq_hz, real speed, real c = 343.2)
        : Obj(ShaderClass::Monopole, true, blend_rate, shader_srate, ts), FreqHz(freq_hz), Speed(speed), C(c) {
        ReadObj(mesh_file, Obj.V2, Obj.F);
    }
    void Compute(GpuBuffer &vb, int global_bid);

    ObjectBase Obj;

private:
    real FreqHz;
    real Speed; // x-direction
    real C; // speed of sound
};

// Speaker shader for pre-recorded input audio.
struct Speaker {
    Speaker(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, const std::string &wav_file, int direction, const std::string &anim_file)
        : Obj(ShaderClass::Speaker, true, blend_rate, shader_srate, ts), Direction(direction) {
        ReadWav(wav_file);
        ReadObj(mesh_file, Obj.V0, Obj.F);
        Obj.V2 = Obj.V0;
        if (!anim_file.empty()) {
            Obj.AnimFile.open(anim_file);
            Obj.ReadAnimation();
        }
    }
    void Compute(GpuBuffer &vb, int global_bid);

    ObjectBase Obj;

private:
    int Direction; // 0: Left, 1: Right, 2: Down, 3: Up, 4: Front, 5: Back (non axis-aligned directions unsupported, as in the reference)

    void ReadWav(const std::string &wav_file);
    GpuBuffer Audio;
};

// Rigid, non-vibrating shader for animated occluders.
struct Occluder {
    Occluder(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, const std::string &anim_file)
        : Obj(ShaderClass::Occluder, !anim_file.empty(), blend_rate, shader_srate, ts) {
        ReadObj(mesh_file, Obj.V0, Obj.F);
        Obj.V2 = Obj.V0;
        if (!anim_file.empty()) {
            Obj.AnimFile.open(anim_file);
            Obj.ReadAnimation();
        }
    }
    void Compute(GpuBuffer &, int) {
        if (Obj.AnimFile.is_open()) Obj.ReadAnimation();
        else if (Obj.Step > 0) Obj.Changed = false;
        Obj.Step += Obj.NSamples - 1; // needed for ReadAnimation()
    }

    ObjectBase Obj;
};

// Bubble-based water sound shader.
struct Bubbles {
    Bubbles(int blend_rate, int shader_srate, double ts, const std::string &bub_file, std::string fluid_mesh_dir, real dx = 0.)
        : Obj(ShaderClass::Bubbles, true, blend_rate, shader_srate, ts), Solver(bub_file, 1. / shader_srate, 1, ts),
          FactorPipeline(std::make_unique<BubbleFactorPipeline>(Solver, 1. / shader_srate, ts)), Dx(dx), FluidMeshDir(std::move(fluid_mesh_dir)) {
        ReadFluidMesh();
    }
    void Compute(GpuBuffer &vb, int global_bid);

    ObjectBase Obj;

private:
    FluidSound::Solver<double> Solver;
    std::unique_ptr<BubbleFactorPipeline> FactorPipeline; // precomputes mass-matrix factorizations (declared after Solver: destroyed first)
    real Dx; // FDTD cell size (for flux normalization)

    std::string FluidMeshDir;
    void ReadFluidMesh();

    std::vector<int> ActiveOscIds; // Active during the current batch

    // --- Host (CPU) ---
    std::vector<real> BubData;
    Eigen::MatrixX<real> BubVels;

    // --- Device (GPU) ---
    // GpuBubData and GpuBubVels hold one slot per minibatch (uploads must not overwrite
    // data that already-encoded, not-yet-executed minibatch kernels read).
    GpuBuffer GpuBubData;
    GpuBuffer BubToBoundary;
    GpuBuffer GpuBubVels;
    GpuBuffer Flux;
};

// Modal sound (plus acceleration noise) shader for vibrating rigid bodies.
struct Modal {
    Modal(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, const std::string &anim_file, const std::string &data_prefix, const std::string &material)
        : Obj(ShaderClass::Modal, true, blend_rate, shader_srate, ts), Solver(data_prefix, mesh_file, material, 1. / shader_srate) {
        ReadObj(mesh_file, Obj.V0, Obj.F);
        Obj.V2 = Obj.V0;
        Obj.Tree.Init(Obj.V0, Obj.F);

        if (!anim_file.empty()) {
            Obj.AnimFile.open(anim_file);
            Obj.ReadAnimation();
        }
    }
    void Compute(GpuBuffer &vb, int global_bid);

    ObjectBase Obj;

private:
    ModalSound::Solver Solver;
    Eigen::Vector3d AccelSum{Eigen::Vector3d::Zero()};

    void SetModeToBoundary(Eigen::MatrixX<real> &mode_to_boundary) const;

    // --- Host (CPU) ---
    Eigen::MatrixX<real> ModeToBoundary1, ModeToBoundary2;
    Eigen::MatrixX<real> ModeVels;
    Eigen::MatrixX<real> AccelNoise;

    // --- Device (GPU) ---
    GpuBuffer GpuModeToBoundary1, GpuModeToBoundary2;
    GpuBuffer GpuModeVels;
    GpuBuffer GpuAccelNoise;
};

// Thin shell shader (right now, it's just loading precomputed simulation data from disk).
struct Shell {
    Shell(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, std::string anim_dir, std::string acc_dir, const std::string &map_file)
        : Obj(ShaderClass::ThinShell, true, blend_rate, shader_srate, ts), ShellAnimDir(std::move(anim_dir)), ShellAccelDir(std::move(acc_dir)),
          MegaBatchSize(100 * (Obj.NSamples - 1) + 1) { // load 100 batches of accel data at once
        ReadObj(mesh_file, Obj.V0, Obj.F);
        Obj.V2 = Obj.V0;
        ReadVertexMap(map_file);

        ReadShellAnimation();
        VertAccel0 = Eigen::RowVectorXd::Zero(Obj.V2.rows() * 3);
        VertAccel1 = Eigen::RowVectorXd::Zero(Obj.V2.rows() * 3);

        VertVels = Eigen::MatrixXd::Zero(1, Obj.V2.rows() * 3);
    }
    void Compute(GpuBuffer &vb, int global_bid);

    ObjectBase Obj;

private:
    std::string ShellAnimDir;
    std::string ShellAccelDir; // Per-vertex acceleration data
    void ReadShellAnimation();

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> VertDisplace;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> VertVels;
    Eigen::RowVectorXd VertAccel0, VertAccel1; // current and next vertex accelerations (read from disk)

    void ReadVertexMap(const std::string &map_file);
    std::vector<int> VertMap; // mapping from original (mesh) vertex indices to internal re-ordered indices

    // --- Host (CPU) ---
    Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> BatchVels;
    int MegaBatchSize{0};

    // (No device buffers because we copy BatchVels directly to the shader data buffer.)
};

// Point force shader. (Named PointSource rather than Point to avoid colliding with the
// legacy ::Point that Apple's MacTypes.h defines at global scope.)
struct PointSource {
    PointSource(int blend_rate, int shader_srate, double ts, const std::string &impulse_file, real dx)
        : Obj(ShaderClass::Point, true, blend_rate, shader_srate, ts), Dx(dx) {
        ReadImpulses(impulse_file);
        GetActiveImpulseList();
    }
    void Compute(GpuBuffer &force, int global_bid);

    ObjectBase Obj;

private:
    real Dx; // FDTD cell size

    void ReadImpulses(const std::string &filename);
    void GetActiveImpulseList();

    // --- All impulse data ---
    std::vector<double> Times; // impact times
    std::vector<double> Tau; // Hertz contact timescale
    std::vector<Eigen::Vector3<real>> DV; // body's velocity change
    std::vector<Eigen::Vector3<real>> Positions;

    // Impulse indices sorted by start time, for windowed active-impulse queries.
    std::vector<int> IdsByTime;
    double MaxTau{0};

    // --- Host (CPU) ---
    std::vector<int> ActiveImpulseIds; // ascending index order (fixes row order, and thus rounding)
    std::vector<real> ImpulseData;
    Eigen::MatrixX<real> ImpulseVels;

    // --- Device (GPU) ---
    GpuBuffer GpuImpulseData;
    GpuBuffer ImpulseToBoundary;
    GpuBuffer GpuImpulseVels;
};

// User-defined density ("auxiliary beta").
struct Density {
    Density(int blend_rate, int shader_srate, double ts, std::string beta_dir)
        : Obj(ShaderClass::Density, true, blend_rate, shader_srate, ts), BetaDir(std::move(beta_dir)) {
        ReadDensity();
    }
    void Compute(GpuBuffer &, int) {
        Obj.Step += Obj.NSamples - 1;
        ReadDensity();
    }

    ObjectBase Obj;

private:
    std::string BetaDir;

    static constexpr int Fps{60}; // hard-coded frame rate, as in the reference

    void ReadDensity();
};

using ObjectVariant = std::variant<Monopole, Speaker, Occluder, Bubbles, Modal, Shell, PointSource, Density>;

inline ObjectBase &Base(ObjectVariant &object) {
    return std::visit([](auto &o) -> ObjectBase & { return o.Obj; }, object);
}
inline const ObjectBase &Base(const ObjectVariant &object) {
    return std::visit([](const auto &o) -> const ObjectBase & { return o.Obj; }, object);
}
// Computes shader values at all sampled positions and times for the current batch,
// writing (NSamples * NPoints) values into `vb` at row offset `global_bid`.
inline void Compute(ObjectVariant &object, GpuBuffer &vb, int global_bid) {
    std::visit([&](auto &o) { o.Compute(vb, global_bid); }, object);
}
