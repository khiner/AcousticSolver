#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <complex>
#include <json.hpp>
#include <memory>
#include <string>
#include <vector>

namespace thermoviscous {
using Complex = std::complex<double>;
using Vector = Eigen::VectorXcd;
using Matrix = Eigen::MatrixXcd;
using Sparse = Eigen::SparseMatrix<Complex>;
using Point = Eigen::Vector3d;
struct Material {
    double Density{1.199238418502307}, SoundSpeed{343.98620840428265};
    double Viscosity{1.8126668012737552e-5}, BulkViscosity{1.0876000807642531e-5};
    double HeatCapacityP{1012.3677822977633}, HeatCapacityV{722.5520495555156}, Conductivity{.02517780857486048};
};
struct Constants {
    Complex Ka, Kh, Kv, TauA, TauH, PhiA, PhiH;
};
struct Settings {
    int Quadrature{6}, Restart{120}, MaxIterations{1000};
    double Tolerance{1e-7}, InnerTolerance{1e-10}, QuadratureTolerance{1e-6}, LossDecay{25};
    bool Exterior{true}, IncidentLosses{false};
    std::string AcousticBackend{"dense"}, Method{"iterative"};
    double DirectMemoryGiB{2};
    double CompressionTolerance{1e-10};
    int FastQuadrature{12}, LeafSize{64};
};
struct Element {
    std::vector<int> Geometry, Dofs;
    double Inset{};
};
struct Problem {
    std::vector<Point> Vertices, Sources, Receivers;
    std::vector<Element> Elements;
    Vector Velocity;
    Material Fluid;
    Settings Options;
    double Frequency{100};
};
struct AcousticOperator {
    virtual ~AcousticOperator() = default;
    virtual Vector Apply(const Vector &x, bool doubleLayer) const = 0;
    virtual size_t Bytes() const = 0;
};
struct Operators {
    Matrix Ha, Ga;
    std::unique_ptr<AcousticOperator> Acoustic;
    Vector ApplyH(const Vector &x) const { return Acoustic ? Acoustic->Apply(x, true) : Vector(Ha * x); }
    Vector ApplyG(const Vector &x) const { return Acoustic ? Acoustic->Apply(x, false) : Vector(Ga * x); }
    Sparse Hh, Gh, Hv, Gv, Dc, Dr, Nd;
    Constants Physics;
    double AssemblySeconds{};
};
struct Solution {
    Vector Pa, Dpa, Ph, Dph, Vv, Dvv, Pressure;
    int Iterations{}, InnerIterations{};
    double RelativeResidual{}, BoundaryResidual{}, SolveSeconds{}, PressureSeconds{}, RecoverySeconds{};
};
Problem Read(const nlohmann::json &config);
Constants WaveConstants(const Material &fluid, double frequency);
Operators Assemble(const Problem &problem);
Vector DirectPressure(const Problem &problem, const Operators &operators);
Solution Solve(const Problem &problem, const Operators &operators);
Vector Evaluate(const Problem &problem, const Operators &operators, const Solution &solution);
} // namespace thermoviscous
