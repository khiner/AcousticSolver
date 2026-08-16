// Ported from WaveBlender (c) 2024 Kangrui Xue (Object.cpp) — Metal port.
// ObjectBase geometry, animation, and closest-point implementation.

#include "Parallel.h"
#include "Shaders.h"

#include <sstream>

namespace {
// Barycentric weights of `p` within triangle (a, b, c).
Eigen::RowVector3<REAL> BarycentricWeights(const Eigen::RowVector3<REAL> &p, const Eigen::RowVector3<REAL> &a, const Eigen::RowVector3<REAL> &b, const Eigen::RowVector3<REAL> &c) {
    const Eigen::RowVector3<REAL> v0 = b - a, v1 = c - a, v2 = p - a;
    const REAL d00 = v0.dot(v0), d01 = v0.dot(v1), d11 = v1.dot(v1), d20 = v2.dot(v0), d21 = v2.dot(v1);
    const REAL denom = d00 * d11 - d01 * d01;
    const REAL w1 = (d11 * d20 - d01 * d21) / denom, w2 = (d00 * d21 - d01 * d20) / denom;
    return Eigen::RowVector3<REAL>{REAL(1) - (w1 + w2), w1, w2};
}
} // namespace

void ObjectBase::SetSamplePoints(const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &bn) {
    NPoints = b.rows();
    B = b;
    BN = bn;

    GpuB.Resize(B.size() * sizeof(REAL));
    GpuBN.Resize(BN.size() * sizeof(REAL));
    GpuB.Upload(B.data(), B.size() * sizeof(REAL));
    GpuBN.Upload(BN.data(), BN.size() * sizeof(REAL));
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
    // Scale Rot Trans
    const Eigen::Quaterniond quat1{Rotation1[0], Rotation1[1], Rotation1[2], Rotation1[3]};
    const Eigen::Quaterniond quat2{Rotation2[0], Rotation2[1], Rotation2[2], Rotation2[3]};

    const double alpha = ((Step + lookahead) * Dt + Ts - T1) / (T2 - T1);
    const Eigen::Quaternion<REAL> quat = quat1.slerp(alpha, quat2).cast<REAL>();
    const Eigen::Vector3<REAL> translation = ((1. - alpha) * Translation1 + alpha * Translation2).cast<REAL>();

    V1 = V2;
    for (int r = 0; r < V2.rows(); ++r) {
        V2.row(r) = quat * V0.row(r);
        V2.row(r) += translation;
    }
    Changed = true;
}

void ObjectBase::ClosestPoint(Eigen::VectorXi &i_out, const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &v) const {
    i_out.resize(b.rows());
    ParallelChunks(b.rows(), 256, [&](size_t begin, size_t end) {
        Eigen::RowVector3<REAL> closest;
        for (size_t r = begin; r < end; ++r) Tree.ClosestPoint(v, F, b.row(r), i_out[r], closest);
    });
}

void ObjectBase::ClosestPoint(Eigen::VectorXi &i_out, Eigen::MatrixX<REAL> &w_out, const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &v) const {
    i_out.resize(b.rows());
    w_out.resize(b.rows(), 3);
    ParallelChunks(b.rows(), 256, [&](size_t begin, size_t end) {
        Eigen::RowVector3<REAL> closest;
        for (size_t r = begin; r < end; ++r) {
            int &index = i_out[r];
            Tree.ClosestPoint(v, F, b.row(r), index, closest);
            w_out.row(r) = BarycentricWeights(closest, v.row(F(index, 0)), v.row(F(index, 1)), v.row(F(index, 2)));
        }
    });
}
