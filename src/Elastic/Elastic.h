#pragma once
#include "Dg/DgBasis.h"
#include <Eigen/SparseCore>
#include <functional>
#include <json.hpp>
#include <map>

namespace elastic {
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using V3 = Eigen::Vector3d;
using Json = nlohmann::json;
struct Material {
    double Density{}, Bulk{}, Speed{}, Impedance{};
    Mat Stiffness;
    bool Fluid() const { return Stiffness.size() == 0; }
    int Fields() const { return Fluid() ? 4 : 9; }
};
Material EvaluateMaterial(const Json &, const V3 &);
Mat NormalMatrix(const V3 &);
Vec Flux(const Vec &, const Vec &, const Material &, const Material &, const V3 &, double penalty);
struct Element {
    int Offset{}, Tag{};
    bool Fluid{};
    Mat Geometry;
    Mat InverseMass, EnergyMass, PhysicalMass, Points;
    std::array<Mat, 3> Weak;
    std::vector<Mat> QuadratureInverse;
    int Fields() const { return Fluid ? 4 : 9; }
};
struct Face {
    int Left{}, Right{-1};
    std::string Boundary;
    Mat L, R, Points, Normals;
    std::array<std::vector<int>, 2> TraceNodes;
    // Rows and columns use canonical face coefficients, with left fields followed by right fields.
    Mat Load;
    Vec Weights;
    std::vector<Material> LeftMaterial, RightMaterial;
};
struct Receiver {
    int ElementIndex{}, Field{};
    double Sign{1};
    Vec Weights;
};
struct Source {
    int ElementIndex{};
    Vec Load;
    double Amplitude{}, Frequency{}, Time{};
    double Drive(double time) const;
};
struct RunOptions {
    bool Cpu{};
    double Dt{};
};
struct Trace {
    Mat Samples, Energy;
    Vec State;
    double Seconds{}, Dt{}, SetupSeconds{}, GpuSeconds{};
    uint64_t Steps{};
};
class Model {
public:
    Model(const Json &config, const Json &mesh, bool assembleReference = true);
    int Degree{}, Nodes{}, Size{};
    bool WeightAdjusted{};
    Mat MassInterpolation;
    Vec Weights;
    double DtLimit{}, Penalty{}, CflRealBound{}, CflImaginaryBound{};
    std::vector<Element> Elements;
    std::vector<Face> Faces;
    std::vector<Receiver> Receivers;
    std::vector<Source> Sources;
    Vec Initial;
    std::function<Vec(const V3 &, int, double)> BoundaryState;
    std::function<Vec(double)> PrescribedLoad;
    Eigen::SparseMatrix<double, Eigen::RowMajor> System, BoundaryPower;
    Vec Rhs(const Vec &, double time = 0) const;
    void AddForcing(Vec &, double time) const;
    Eigen::Matrix<double, 5, 1> Diagnostics(const Vec &, double time) const;
    Vec Apply(const Vec &) const;
    Vec ApplyMatrixFree(const Vec &) const;
    double Energy(const Vec &, bool physical = false) const;
    double Power(const Vec &, const Vec &) const;
    Eigen::Vector3d Powers(const Vec &, double time) const;
    Vec Project(const std::function<Vec(const V3 &, int)> &) const;
    Trace RunReference(double seconds, double sampleRate, double dt = 0) const;
    Trace Run(double seconds, double sampleRate, const RunOptions & = {}) const;

private:
    struct Geometry {
        Mat X;
        std::vector<Eigen::Matrix3d> J;
    };
    int GeometryDegree{};
    Mat V, ReferenceInverse, GeometryInverse;
    Geometry Map(const Element &, const Mat &points, bool positive = true) const;
    std::vector<long> RowStarts;
    void Assemble(bool assembleReference);
    void SetTimestep(std::vector<std::map<int, Mat>> &weakBlocks);
    std::pair<int, Vec> Locate(const V3 &, int region) const;
};
} // namespace elastic
