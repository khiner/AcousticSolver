#pragma once

#include <Eigen/Core>
#include <complex>
#include <cstdint>
#include <vector>

namespace bem {
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Triangles = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
using Vector = Eigen::VectorXcd;
using Matrix = Eigen::MatrixXcd;

struct Problem {
    Points Vertices;
    Triangles Faces;
    Points Listeners;
    Eigen::Vector3d Source{0., 100., 0.};
    double Frequency{1000.};
    double SoundSpeed{344.};
    std::vector<int> HeightVertices;
};

struct Operators {
    Matrix Boundary, Listener;
    Vector Incident;
};

struct CompressionStats {
    uint64_t DenseEntries{}, StoredEntries{};
    int Blocks{}, LowRankBlocks{}, MaxRank{};
    double RelativeError{}, MaxBlockError{};
};

struct AssemblyProfile {
    double QuadratureSeconds{}, CompressionSeconds{}, PackingSeconds{}, PreconditionerSeconds{}, OtherSeconds{};
};

enum class Preconditioner { None,
                            BlockJacobi,
                            TwoLevel };
struct PreconditionerStats {
    Preconditioner Kind{Preconditioner::None};
    double Seconds{};
    uint64_t StoredEntries{};
    int LocalBlocks{}, CoarseSize{};
};
struct SolverOptions {
    double CompressionTolerance{2e-6}, SolveTolerance{1e-6};
    int LeafSize{128}, MaxBlockSize{1024}, Restart{120}, MaxIterations{1200};
    Preconditioner Precondition{Preconditioner::TwoLevel};
};

struct Result {
    Vector SurfacePressure, ScatteredPressure;
    Eigen::VectorXd HeightGradient;
    double Diffusion{};
    double ForwardResidual{}, AdjointResidual{};
    double AssemblySeconds{}, SolveSeconds{}, DerivativeSeconds{};
    int ForwardIterations{}, AdjointIterations{}, IndependentRows{};
    double IndependentForwardResidual{}, IndependentAdjointResidual{};
    CompressionStats BoundaryCompression, ListenerCompression;
    PreconditionerStats Preconditioning;
    AssemblyProfile Assembly;
};

// Constant-element exterior rigid-boundary collocation, exp(-ikr) convention.
// Boundary pressure includes the incident field; listener pressure is scattered only.
// Dense oracle calls are limited to 2048 triangles.
void Validate(const Problem &);
void ValidateDense(const Problem &);
Matrix ReferenceRows(const Problem &, const std::vector<int> &rows, bool boundary, bool transpose = false);
Operators AssembleReference(const Problem &);
Result SolveReference(const Problem &);
Result SolveMetal(const Problem &, const SolverOptions & = {});

// Plain transpose adjoint: the pressure sensitivity already contains conjugation.
Eigen::VectorXd ReferenceGradient(const Problem &, const Vector &surface, const Vector &sensitivity, const Vector &adjoint);
Eigen::VectorXd MetalGradient(const Problem &, const Vector &surface, const Vector &sensitivity, const Vector &adjoint);
Vector PressureSensitivity(const Vector &);
} // namespace bem
