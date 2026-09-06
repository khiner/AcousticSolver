#pragma once
#include "json.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace dg::reference {
namespace fs = std::filesystem;
using Json = nlohmann::json;
template<class T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Mat = Matrix<double>;
using Vec = Eigen::VectorXd;
using State = Mat;
inline void Require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> std::vector<T> Read(const fs::path &path, size_t count = SIZE_MAX) {
    const auto bytes = fs::file_size(path);
    Require(bytes % sizeof(T) == 0 && (count == SIZE_MAX || bytes / sizeof(T) == count), "Wrong array size: " + path.string());
    std::vector<T> result(bytes / sizeof(T));
    Require(bool(std::ifstream(path, std::ios::binary).read(reinterpret_cast<char *>(result.data()), bytes)), "Cannot read: " + path.string());
    return result;
}
template<class T> void Write(const fs::path &path, std::span<const T> data) {
    Require(bool(std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(data.data()), data.size_bytes())), "Cannot write: " + path.string());
}
template<class T> void Write(const fs::path &path, const std::vector<T> &data) { Write<T>(path, std::span<const T>(data)); }
Json Load(const fs::path &);
void Save(const fs::path &, const Json &);
std::string Hash(const fs::path &);
void NewDirectory(const fs::path &);
Mat ReadMatrix(const fs::path &, int rows, int cols);
Mat ReadNodes(const fs::path &, int count);
struct Basis {
    Mat V;
    std::array<Mat, 3> D;
};
Basis Bernstein(const Mat &, int degree);
struct Quadrature {
    Mat Points;
    Vec Weights;
    std::array<Mat, 4> Faces;
    Vec FaceWeights;
};
Quadrature Integrate(int order);
struct Mesh {
    Json Metadata;
    fs::path Directory;
    int E, N, Nfp, F, Degree, Receivers;
    double C, Z;
    Mat Nodes, XYZ, Initial, ReceiverWeights, ReceiverXYZ;
    std::vector<int32_t> Vm, Vp, Boundary, ReceiverElements;
    Mesh(const fs::path &directory, const fs::path &nodes);
};
struct Element {
    Mat Physical, H, A, G, L, WeakG, WeakL;
    Vec Moments, Jacobian;
    std::array<Mat, 4> Normals;
    std::array<Vec, 4> Measures;
};
struct Reference {
    Mesh Source;
    Mat InverseBasis, V, T, W, InverseMass;
    std::array<Mat, 4> FaceBasis;
    std::vector<Element> Elements;
    Json MassChecks;
    double MinimumJacobian = INFINITY;
    explicit Reference(const fs::path &mesh, const fs::path &nodes, bool wadg = true);
    State Rhs(const State &) const;
    double Energy(const State &) const;
    double Mean(const State &) const;
    double Dissipation(const State &) const;
    Json Check() const;
    Json Run(const fs::path &, int steps = 1024, double dt = 0x1p-18) const;
    void Export(const fs::path &, double offset = 2, int reduced_order = 0) const;
};
Json Compare(const fs::path &reference, const fs::path &candidate, const fs::path &tables, int stride = 1);
Json Score(const fs::path &mesh, const fs::path &record, const fs::path &nodes, int order = 0);
void CheckAnalytic(const fs::path &);
struct Fixture {
    const char *Name;
    const char *MeshName;
    int Degree, Order;
};
inline constexpr std::array<Fixture, 2> Fixtures{{{"degree6", "r1_l0_p6", 6, 12}, {"radius_half", "r0.5_l1_p4", 4, 9}}};
inline const fs::path Project = ACOUSTIC_PROJECT_ROOT;
inline const fs::path FixtureRoot = Project / "tests/fixtures/dg";
void VerifyFixtures();
void Prepare(const fs::path &output);
void ImportUpstream(const fs::path &captures, const fs::path &checkout, const fs::path &output);
void Validate(const fs::path &output, const fs::path &inputs, const fs::path &executable);
} // namespace dg::reference
