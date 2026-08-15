// Ported from WaveBlender (c) 2024 Kangrui Xue (ModalShader.cu) — Metal port.
// Implements the Modal acoustic shader (kernel: mode_matmul in Kernels.metal).

#include <numbers>

#include "Shaders.h"

void Modal::SetModeToBoundary(Eigen::MatrixX<REAL> &mode_to_boundary) {
    const Eigen::Quaterniond quat1{Rotation1[0], Rotation1[1], Rotation1[2], Rotation1[3]};
    const Eigen::Quaterniond quat2{Rotation2[0], Rotation2[1], Rotation2[2], Rotation2[3]};

    const auto &eigen_vectors_normal = Solver._eigenVectorsNormal;
    const auto &normals = Solver._normals;

    // Compute interpolated rotation + translation
    const double alpha = ((Step * Dt + Ts) - T1) / (T2 - T1);
    const Eigen::Quaternion<REAL> quat = quat1.slerp(alpha, quat2).cast<REAL>();
    const Eigen::RowVector3<REAL> translation = ((1. - alpha) * Translation1 + alpha * Translation2).cast<REAL>();

    Eigen::VectorXi indices; // Closest triangle indices + barycentric weights
    Eigen::MatrixX<REAL> weights;

    // For closest point query, first transform boundary face positions into rest frame
    const Eigen::Matrix3<REAL> inv_rot = quat.inverse().toRotationMatrix();
    Eigen::Matrix<REAL, Eigen::Dynamic, 3, Eigen::RowMajor> b = B.rowwise() - translation;
    b = b.eval() * inv_rot.transpose();

    ClosestPoint(indices, weights, b, V0);

    // Set entries of mode-to-boundary matrix (assumes already allocated)
    for (int bid = 0; bid < NPoints; ++bid) {
        const Eigen::Vector3<REAL> bn = BN.row(bid);
        for (int vert = 0; vert < 3; ++vert) {
            REAL weight = weights(bid, vert); // barycentric weights
            const int vertex_id = F(indices[bid], vert);

            const Eigen::Vector3<REAL> normal = normals.row(vertex_id).cast<REAL>();
            weight *= bn.dot(quat * normal);

            mode_to_boundary.row(bid) += weight * eigen_vectors_normal.row(vertex_id).cast<REAL>();
        }
    }
}

void Modal::Compute(GpuBuffer &vb, int global_bid) {
    const int n_modes = Solver._qDot_c_plus.size();
    const Eigen::Quaterniond quat1{Rotation1[0], Rotation1[1], Rotation1[2], Rotation1[3]};
    const Eigen::Quaterniond quat2{Rotation2[0], Rotation2[1], Rotation2[2], Rotation2[3]};

    // Build mode-to-boundary matrix at batch start (we will eventually interpolate in between batch start and end)
    ModeToBoundary1 = Eigen::MatrixX<REAL>::Zero(NPoints, n_modes);
    SetModeToBoundary(ModeToBoundary1);

    GpuModeToBoundary1.Resize(ModeToBoundary1.size() * sizeof(REAL));
    GpuModeToBoundary1.Upload(ModeToBoundary1.data(), ModeToBoundary1.size() * sizeof(REAL));

    // Initialize surface velocity buffers (we will set the data as we go)
    ModeVels = Eigen::MatrixX<REAL>::Zero(n_modes, NSamples);
    AccelNoise = Eigen::MatrixX<REAL>::Zero(NPoints, NSamples);

    Eigen::Vector3<double> accel_pulse1; // current acceleration
    Eigen::Vector3<double> accel_pulse2 = Eigen::Vector3<double>::Zero(); // next acceleration

    // Timestep modal vibrations
    for (int k = 0; k < NSamples; ++k) {
        ModeVels.col(k) = Solver._qDot_c_plus.cast<REAL>();

        // -------------------- Acceleration noise -------------------- //
        const double time = Step * Dt + Ts;

        const double alpha = (time - T1) / (T2 - T1);
        const Eigen::Quaterniond quat = quat1.slerp(alpha, quat2);

        std::vector<ModalSound::ImpactRecord> impact_records;
        auto &impulse_series = Solver._impulseSeries;

        for (int idx = 0; idx < impulse_series._impulses.size(); ++idx) {
            const auto &record = impulse_series._impulses[idx];
            if (time >= record.timestamp && time < record.timestamp + record.supportLength) impact_records.push_back(record);
        }

        accel_pulse1 = accel_pulse2;
        accel_pulse2 = Eigen::Vector3<double>::Zero();

        for (const auto &impulse : impact_records) {
            if (impulse.supportLength < 1e-12) continue;

            const Eigen::Vector3d &j_vec = impulse.impactVector;
            const Eigen::Vector3d r = impulse.impactPosition - Solver._centerOfMass;

            double s = 0.;
            if (time <= impulse.timestamp + impulse.supportLength && time >= impulse.timestamp) s = std::sin(std::numbers::pi * (time - impulse.timestamp) / impulse.supportLength);

            // Translational acceleration
            accel_pulse2 += j_vec * (std::numbers::pi * s / (2. * impulse.supportLength * Solver._mass));

            // Rotational acceleration (Eq. 13) from [Chadwick et al. 2012]
            const Eigen::Matrix3d &i_inv = Solver._I_Inv;
            const Eigen::Vector3d rot_alpha = (i_inv * r.cross(j_vec)) * (std::numbers::pi * s) / (2. * impulse.supportLength);
            accel_pulse2 += rot_alpha.cross(r);
        }

        accel_pulse2 = quat * accel_pulse2.eval(); // rotate vector from rest frame to animation frame

        for (int bid = 0; bid < NPoints; ++bid) { // set normal velocities
            if (BN(bid, 0) != 0.f) AccelNoise(bid, k) = AccelSum[0];
            else if (BN(bid, 1) != 0.f) AccelNoise(bid, k) = AccelSum[1];
            else if (BN(bid, 2) != 0.f) AccelNoise(bid, k) = AccelSum[2];
        }
        // ------------------------------------------------------------------------------- //

        Solver.step(Step * Dt + Ts);
        if (k >= NSamples - 1) break;

        AccelSum += (accel_pulse1 + accel_pulse2) / 2.f * Dt; // trapezoidal rule integrator
        Step += 1;
    }

    GpuModeVels.Resize(ModeVels.size() * sizeof(REAL));
    GpuModeVels.Upload(ModeVels.data(), ModeVels.size() * sizeof(REAL));

    GpuAccelNoise.Resize(AccelNoise.size() * sizeof(REAL));
    GpuAccelNoise.Upload(AccelNoise.data(), AccelNoise.size() * sizeof(REAL));

    // Read next animation data in order to build mode-to-boundary matrix at batch end
    if (AnimFile.is_open()) ReadAnimation();

    ModeToBoundary2 = Eigen::MatrixX<REAL>::Zero(NPoints, n_modes);
    SetModeToBoundary(ModeToBoundary2);

    GpuModeToBoundary2.Resize(ModeToBoundary2.size() * sizeof(REAL));
    GpuModeToBoundary2.Upload(ModeToBoundary2.data(), ModeToBoundary2.size() * sizeof(REAL));

    // Multiply mode-to-boundary matrix with modal velocities on GPU
    const Dim3 threads{16, 16, 1};
    const Dim3 matmul_blocks{uint32_t(NPoints + 15) / 16, uint32_t(NSamples + 15) / 16, 1};

    const ModeMatmulParams params{global_bid, NPoints, n_modes, NSamples};
    MetalContext::Get().Dispatch("mode_matmul", matmul_blocks, threads, {&vb, &GpuModeToBoundary1, &GpuModeToBoundary2, &GpuModeVels, &GpuAccelNoise}, &params, sizeof(params));
}
