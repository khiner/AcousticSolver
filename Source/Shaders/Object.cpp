// Ported from WaveBlender (c) 2024 Kangrui Xue (Object.cpp) — Metal port.

#include "Shaders.h"

void Object::ReadAnimation() {
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

    std::cout << T1 << ", " << T2 << std::endl;

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

// For each query position, computes the closest point on this object's surface mesh (must be called after Tree.init()).
// `i_out` receives the triangle index for each closest point.
void Object::ClosestPoint(Eigen::VectorXi &i_out, const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &v) {
    Eigen::VectorX<REAL> sqr_d;
    Eigen::MatrixX<REAL> p;
    Tree.squared_distance(v, F, b, sqr_d, i_out, p);
}

// Closest point query with additional barycentric weight computation into `w_out`.
void Object::ClosestPoint(Eigen::VectorXi &i_out, Eigen::MatrixX<REAL> &w_out, const Eigen::MatrixX<REAL> &b, const Eigen::MatrixX<REAL> &v) {
    Eigen::VectorX<REAL> sqr_d;
    Eigen::MatrixX<REAL> p;
    Tree.squared_distance(v, F, b, sqr_d, i_out, p);

    Eigen::MatrixX<REAL> tri1(NPoints, 3);
    Eigen::MatrixX<REAL> tri2(NPoints, 3);
    Eigen::MatrixX<REAL> tri3(NPoints, 3);
    for (int bid = 0; bid < NPoints; ++bid) {
        tri1.row(bid) = v.row(F(i_out[bid], 0));
        tri2.row(bid) = v.row(F(i_out[bid], 1));
        tri3.row(bid) = v.row(F(i_out[bid], 2));
    }
    igl::barycentric_coordinates(p, tri1, tri2, tri3, w_out);
}
