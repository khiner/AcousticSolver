// Ported from WaveBlender (c) 2024 Kangrui Xue (Object.cpp) — Metal port.
// ObjectBase geometry, animation, and closest-point implementation.

#include "Shaders.h"

#include "Mesh.h"
#include "Parallel.h"
#include "Profile.h"

#include <sstream>

void ObjectBase::SetSamplePoints(const Eigen::MatrixX<real> &b, const Eigen::MatrixX<real> &bn, std::vector<int> &&keys) {
    NPoints = b.rows();
    B = b;
    BN = bn;
    SampleKeys = std::move(keys);

    GpuB.Resize(B.size() * sizeof(real));
    GpuBN.Resize(BN.size() * sizeof(real));
    GpuB.Upload(B.data(), B.size() * sizeof(real));
    GpuBN.Upload(BN.data(), BN.size() * sizeof(real));
}

void ObjectBase::ReadAnimation() {
    const int lookahead = NSamples - 1;
    std::string line;

    if (Step == 0) std::getline(AnimFile, line); // First line is metadata

    while (AnimFile.good() && !AnimFile.eof() && T2 <= (Step + lookahead) * Dt + Ts) {
        std::getline(AnimFile, line);
        T1 = T2;
        Translation1 = Translation2;
        Rotation1 = Rotation2;

        std::istringstream is{line};
        is >> T2 >> Translation2[0] >> Translation2[1] >> Translation2[2] >> Rotation2[0] >> Rotation2[1] >> Rotation2[2] >> Rotation2[3];
    }
    if (Step == 0) {
        Translation1 = Translation2;
        Rotation1 = Rotation2;
    }

    // Same fixed threshold as the reference (see its TODO)
    if (Step > 0 && (Translation2 - Translation1).norm() < 1e-6 && (Rotation2 - Rotation1).norm() < 1e-6) {
        Changed = false;
        return;
    }
    const Eigen::Quaterniond quat1{Rotation1[0], Rotation1[1], Rotation1[2], Rotation1[3]};
    const Eigen::Quaterniond quat2{Rotation2[0], Rotation2[1], Rotation2[2], Rotation2[3]};

    const double alpha = ((Step + lookahead) * Dt + Ts - T1) / (T2 - T1);
    const Eigen::Quaternion<real> quat = quat1.slerp(alpha, quat2).cast<real>();
    const Eigen::Vector3<real> translation = ((1. - alpha) * Translation1 + alpha * Translation2).cast<real>();

    V1 = V2;
    for (int r = 0; r < V2.rows(); ++r) {
        V2.row(r) = quat * V0.row(r);
        V2.row(r) += translation;
    }
    Changed = true;
}

void ObjectBase::ClosestPoint(Eigen::VectorXi &i_out, const Eigen::MatrixX<real> &b, const Eigen::MatrixX<real> &v) const {
    i_out.resize(b.rows());
    ParallelChunks(b.rows(), 32, [&](size_t begin, size_t end) {
        Eigen::RowVector3<real> closest;
        for (size_t r = begin; r < end; ++r) Tree.ClosestPoint(b.row(r), i_out[r], closest);
    });
}

void ObjectBase::ClosestPoint(Eigen::VectorXi &i_out, Eigen::MatrixX<real> &w_out, const Eigen::MatrixX<real> &b, const Eigen::MatrixX<real> &v) const {
    const profile::Scope scope{"cpu/closest_point"};
    i_out.resize(b.rows());
    w_out.resize(b.rows(), 3);
    ParallelChunks(b.rows(), 32, [&](size_t begin, size_t end) {
        Eigen::RowVector3<real> closest;
        for (size_t r = begin; r < end; ++r) {
            int &index = i_out[r];
            Tree.ClosestPoint(b.row(r), index, closest);
            w_out.row(r) = BarycentricWeights(closest, v.row(F(index, 0)), v.row(F(index, 1)), v.row(F(index, 2)));
        }
    });
}
