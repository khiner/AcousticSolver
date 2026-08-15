#pragma once

// Ported from WaveBlender (c) 2024 Kangrui Xue (Shaders.h) — Metal port.
//
// Declares the Shader class and all sound sources based on it. See the .cpp files
// associated with each sound source for implementation details. GPU kernels live in
// FDTD/Kernels.metal, raw device pointers became GpuBuffers, and the cuBLAS handles
// are gone (the one Sgemv call is now the bub_flux kernel).

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>

#include "igl/AABB.h"
#include "igl/barycentric_coordinates.h"
#include "igl/read_triangle_mesh.h"

#include "FluidSound.h"
#include "ModalSound.h"

#include "KernelParams.h"
#include "MetalContext.h"

using REAL = float;

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

// Acoustic shader virtual class for representing vibration boundary conditions or forcing terms.
class Shader {
public:
    Shader(int blend_rate, int shader_srate, double ts) : Srate(shader_srate), Dt(1. / Srate), Ts(ts), NSamples(shader_srate / blend_rate + 1) {}
    virtual ~Shader() = default;

    // Given a device buffer and global index offset, populates the buffer with boundary velocity data.
    // A total of (NSamples * NPoints) values will be written, at row offset `global_bid`.
    virtual void Compute(GpuBuffer &vb, int global_bid) = 0;

    // Sets the shader sample points for the current batch:
    // (NPoints, 3) boundary points to compute shader samples at, and their normals.
    void SetSamplePoints(const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &bn) {
        NPoints = b.rows();
        B = b;
        BN = bn;

        GpuB.Resize(B.size() * sizeof(REAL));
        GpuBN.Resize(BN.size() * sizeof(REAL));
        GpuB.Upload(B.data(), B.size() * sizeof(REAL));
        GpuBN.Upload(BN.data(), BN.size() * sizeof(REAL));
    }

    ShaderClass Type;
    int NPoints{0}; // number of boundary points

protected:
    int Srate{48'000}; // shader sample rate
    double Dt{1. / 48'000}; // shader timestep size
    int Step{0}; // current shader timestep
    double Ts{0}; // simulation start time

    int NSamples{0}; // number of shader samples in time (i.e., the batchsize)

    // B and BN are the set of all boundary positions and corresponding boundary normals
    // to compute shader samples at. \see SetSamplePoints()
    Eigen::Matrix<REAL, Eigen::Dynamic, 3, Eigen::RowMajor> B, BN;
    GpuBuffer GpuB, GpuBN;
};

// Extends Shader class with object geometry and animation handling.
class Object : public Shader {
public:
    Object(int blend_rate, int shader_srate, double ts) : Shader(blend_rate, shader_srate, ts) {}

    bool HasShader{false};
    bool Changed{true}; // whether the object has moved / its geometry has changed since the last batch

    Eigen::MatrixX<REAL> V0; // original, non-animated surface mesh vertex positions
    Eigen::MatrixXi F; // surface mesh faces

    // Current and next surface mesh vertex positions.
    // (Due to how we rasterize, we need to "look ahead" to the next triangle mesh.)
    Eigen::MatrixX<REAL> V1, V2;

protected:
    // ----- Closest point query -----
    igl::AABB<Eigen::MatrixX<REAL>, 3> Tree;
    void ClosestPoint(Eigen::VectorXi &i_out, const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &v);
    void ClosestPoint(Eigen::VectorXi &i_out, Eigen::MatrixX<REAL> &w_out, const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &v);

    // ----- Animation and I/O -----
    double T1{0}, T2{0}; // animation keyframe endpoint times
    Eigen::Vector3d Translation1, Translation2; // translation vectors
    Eigen::Vector4d Rotation1, Rotation2; // rotation quaternions

    std::ifstream AnimFile; // animation file stream
    void ReadAnimation(); // reads animation data for current Step

    // Reads .obj file and saves vertices and faces to V and F.
    static bool ReadObj(const std::string &filename, Eigen::MatrixX<REAL> &v, Eigen::MatrixXi &f) { return igl::read_triangle_mesh(filename, v, f); }
};

// Basic mono-frequency, monopole source shader used for testing.
class Monopole : public Object {
public:
    Monopole(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, REAL freq_hz, REAL speed, REAL c = 343.2) : Object(blend_rate, shader_srate, ts), FreqHz(freq_hz), Speed(speed), C(c) {
        HasShader = true;
        Type = ShaderClass::Monopole;
        ReadObj(mesh_file, V2, F);
    }
    void Compute(GpuBuffer &vb, int global_bid) override;

private:
    REAL FreqHz{2000}; // source frequency
    REAL Speed{0}; // source speed (x-direction)
    REAL C{343.2}; // speed of sound
};

// Speaker shader for pre-recorded input audio.
class Speaker : public Object {
public:
    Speaker(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, const std::string &wav_file, int direction, const std::string &anim_file) : Object(blend_rate, shader_srate, ts), Direction(direction) {
        HasShader = true;
        Type = ShaderClass::Speaker;
        ReadWav(wav_file);
        ReadObj(mesh_file, V0, F);
        V2 = V0;
        if (anim_file != "") {
            AnimFile.open(anim_file);
            ReadAnimation();
        }
    }
    void Compute(GpuBuffer &vb, int global_bid) override;

private:
    int Direction{0}; // 0: Left, 1: Right, 2: Down, 3: Up, 4: Front, 5: Back (non axis-aligned directions unsupported, as in the reference)

    void ReadWav(const std::string &wav_file);
    GpuBuffer Audio;
};

// Rigid, non-vibrating shader for animated occluders.
class Occluder : public Object {
public:
    Occluder(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, const std::string &anim_file) : Object(blend_rate, shader_srate, ts) {
        HasShader = anim_file != "";
        Type = ShaderClass::Occluder;
        ReadObj(mesh_file, V0, F);
        V2 = V0;
        if (anim_file != "") {
            AnimFile.open(anim_file);
            ReadAnimation();
        }
    }
    void Compute(GpuBuffer &, int) override {
        if (AnimFile.is_open()) ReadAnimation();
        else if (Step > 0) Changed = false;
        Step += NSamples - 1; // needed for ReadAnimation()
    }
};

// Bubble-based water sound shader.
class Bubbles : public Object {
public:
    Bubbles(int blend_rate, int shader_srate, double ts, const std::string &bub_file, std::string fluid_mesh_dir, REAL dx = 0.) : Object(blend_rate, shader_srate, ts), Solver(bub_file, 1. / shader_srate, 1, ts), Dx(dx), FluidMeshDir(std::move(fluid_mesh_dir)) {
        HasShader = true;
        Type = ShaderClass::Bubbles;
        ReadFluidMesh();
    }
    void Compute(GpuBuffer &vb, int global_bid) override;

private:
    FluidSound::Solver<double> Solver; // solver for timestepping bubble oscillations
    REAL Dx{0}; // FDTD cell size (for flux normalization)

    std::string FluidMeshDir; // path to directory containing fluid meshes
    void ReadFluidMesh();

    std::vector<int> ActiveOscIds; // IDs of active oscillators during current batch

    // --- Host (CPU) ---
    std::vector<REAL> BubData;
    Eigen::MatrixX<REAL> BubVels;

    // --- Device (GPU) ---
    GpuBuffer GpuBubData;
    GpuBuffer BubToBoundary;
    GpuBuffer GpuBubVels;
    GpuBuffer Flux; // for flux normalization
};

// Modal sound (plus acceleration noise) shader for vibrating rigid bodies.
class Modal : public Object {
public:
    Modal(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, const std::string &anim_file, const std::string &data_prefix, const std::string &material) : Object(blend_rate, shader_srate, ts), Solver(data_prefix, mesh_file, material, 1. / shader_srate) {
        HasShader = true;
        Type = ShaderClass::Modal;
        ReadObj(mesh_file, V0, F);
        V2 = V0;
        Tree.init(V0, F);

        if (anim_file != "") {
            AnimFile.open(anim_file);
            ReadAnimation();
        }
    }
    void Compute(GpuBuffer &vb, int global_bid) override;

private:
    ModalSound::Solver Solver; // solver for timestepping modal vibrations
    Eigen::Vector3d AccelSum{Eigen::Vector3d::Zero()};

    void SetModeToBoundary(Eigen::MatrixX<REAL> &mode_to_boundary);

    // --- Host (CPU) ---
    Eigen::MatrixX<REAL> ModeToBoundary1, ModeToBoundary2;
    Eigen::MatrixX<REAL> ModeVels;
    Eigen::MatrixX<REAL> AccelNoise;

    // --- Device (GPU) ---
    GpuBuffer GpuModeToBoundary1, GpuModeToBoundary2;
    GpuBuffer GpuModeVels;
    GpuBuffer GpuAccelNoise;
};

// Thin shell shader (right now, it's just loading precomputed simulation data from disk).
class Shell : public Object {
public:
    Shell(int blend_rate, int shader_srate, double ts, const std::string &mesh_file, std::string anim_dir, std::string acc_dir, const std::string &map_file) : Object(blend_rate, shader_srate, ts), ShellAnimDir(std::move(anim_dir)), ShellAccelDir(std::move(acc_dir)) {
        HasShader = true;
        Type = ShaderClass::ThinShell;
        ReadObj(mesh_file, V0, F);
        V2 = V0;
        ReadVertexMap(map_file);

        ReadShellAnimation();
        VertAccel0 = Eigen::RowVectorXd::Zero(V2.rows() * 3);
        VertAccel1 = Eigen::RowVectorXd::Zero(V2.rows() * 3);

        MegaBatchSize = 100 * (NSamples - 1) + 1; // load 100 batches of accel data at once
        VertVels = Eigen::MatrixXd::Zero(1, V2.rows() * 3);
    }
    void Compute(GpuBuffer &vb, int global_bid) override;

private:
    std::string ShellAnimDir; // directory containing shell animation data
    std::string ShellAccelDir; // directory containing shell vertex acceleration data
    void ReadShellAnimation();

    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> VertDisplace;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> VertVels;
    Eigen::RowVectorXd VertAccel0, VertAccel1; // current and next vertex accelerations (read from disk)

    void ReadVertexMap(const std::string &map_file);
    std::map<int, int> VertMap; // mapping from original (mesh) vertex indices to internal re-ordered indices

    // --- Host (CPU) ---
    Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> BatchVels;
    int MegaBatchSize{0};

    // (No device buffers because we copy BatchVels directly to the shader data buffer.)
};

// Point force shader.
class Point : public Object {
public:
    Point(int blend_rate, int shader_srate, double ts, const std::string &impulse_file, REAL dx) : Object(blend_rate, shader_srate, ts), Dx(dx) {
        HasShader = true;
        Type = ShaderClass::Point;
        ReadImpulses(impulse_file);
        GetActiveImpulseList();
    }
    void Compute(GpuBuffer &force, int global_bid) override;

private:
    REAL Dx{0}; // FDTD cell size

    void ReadImpulses(const std::string &filename);
    void GetActiveImpulseList();

    // --- All impulse data ---
    std::vector<double> Times; // impact times
    std::vector<double> Tau; // Hertz contact timescale
    std::vector<Eigen::Vector3<REAL>> DV; // body's velocity change
    std::vector<Eigen::Vector3<REAL>> Positions; // impact positions

    // --- Host (CPU) ---
    std::set<int> ActiveImpulseIds;
    std::vector<REAL> ImpulseData;
    Eigen::MatrixX<REAL> ImpulseVels;

    // --- Device (GPU) ---
    GpuBuffer GpuImpulseData;
    GpuBuffer ImpulseToBoundary;
    GpuBuffer GpuImpulseVels;
};

// User-defined density ("auxiliary beta").
class Density : public Object {
public:
    Density(int blend_rate, int shader_srate, double ts, std::string beta_dir) : Object(blend_rate, shader_srate, ts), BetaDir(std::move(beta_dir)) {
        HasShader = true;
        Type = ShaderClass::Density;
        ReadDensity();
    }
    void Compute(GpuBuffer &, int) override {
        Step += NSamples - 1;
        ReadDensity();
    }

private:
    std::string BetaDir;

    static constexpr int Fps{60}; // hard-coded frame rate, as in the reference

    void ReadDensity() {
        std::vector<Eigen::Vector3<REAL>> pos;
        std::vector<int> betas;

        const int frame = (Step * Dt + Ts) * Fps + 1;
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << frame;
        const auto filename = BetaDir + "betaField_6x40x6_" + ss.str() + ".txt";

        std::ifstream in_file{filename};
        std::string line;

        std::getline(in_file, line); // Skip first line
        while (!line.empty() && in_file.good()) {
            std::getline(in_file, line);
            std::istringstream is{line};

            REAL density, px, py, pz;
            is >> density >> px >> py >> pz;

            pos.emplace_back(px, py, pz);
            betas.push_back(int(100 * density));
        }
        V2 = Eigen::MatrixX<REAL>::Zero(pos.size(), 3);
        F = Eigen::MatrixXi::Zero(betas.size(), 3);
        for (int r = 0; r < pos.size(); ++r) {
            V2.row(r) = pos[r];
            F(r, 0) = betas[r]; // Hack to encode density in cell face
        }
        Changed = in_file.is_open();
    }
};
