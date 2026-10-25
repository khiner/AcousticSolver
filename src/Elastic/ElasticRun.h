#pragma once
#include "Elastic.h"
#include <memory>

namespace elastic {
using Sparse = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Entries = std::vector<Eigen::Triplet<double>>;
void AppendBlock(Entries &, const Mat &, int row, int col);
Sparse MakeSparse(int rows, int cols, const Entries &);
inline constexpr std::array<double, 5> RkA{0, -.41789047449985195, -1.192151694642677, -1.6977846924715279, -1.5141834442571558};
inline constexpr std::array<double, 5> RkB{.14965902199922912, .37921031299962726, .8229550293869817, .6994504559491221, .15305724796815198};
inline constexpr std::array<double, 5> RkC{0, .14965902199922912, .37040095736420475, .6222557631344432, .9582821306746903};
// The left half-disk |z| <= 3 lies inside this RK method's stability region; see README.md.
inline constexpr double RkRadius = 3;
inline constexpr int CommandSteps = 128;
double SpectralNorm(Mat);
double StateScale(const Model &);
struct Stepper {
    virtual ~Stepper() = default;
    virtual void Advance(int frame, int frames, int substeps, double rate) = 0;
    virtual void Finish(Trace &) = 0;
};
std::unique_ptr<Stepper> GpuStepper(const Model &, Trace const &);
} // namespace elastic
