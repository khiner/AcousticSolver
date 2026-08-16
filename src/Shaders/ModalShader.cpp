// Ported from WaveBlender (c) 2024 Kangrui Xue (ModalShader.cu) — Metal port.
// Implements the Modal acoustic shader (kernel: mode_matmul in Kernels.metal).
//
// The impact-record scan is windowed per batch: candidates are collected once, then the
// exact per-sample check runs on that subset in record order (order affects the
// acceleration sum's rounding). Mode-to-boundary rows accumulate in parallel.

#include <numbers>

#include "Shaders.h"

#include "KernelParams.h"
#include "Parallel.h"

void Modal::SetModeToBoundary(Eigen::MatrixX<REAL> &mode_to_boundary) const {
    const auto &base = Obj;
    const Eigen::Quaterniond quat1{base.Rotation1[0], base.Rotation1[1], base.Rotation1[2], base.Rotation1[3]};
    const Eigen::Quaterniond quat2{base.Rotation2[0], base.Rotation2[1], base.Rotation2[2], base.Rotation2[3]};

    const auto &eigen_vectors_normal = Solver._eigenVectorsNormal;
    const auto &normals = Solver._normals;

    // Compute interpolated rotation + translation
    const double alpha = ((base.Step * base.Dt + base.Ts) - base.T1) / (base.T2 - base.T1);
    const Eigen::Quaternion<REAL> quat = quat1.slerp(alpha, quat2).cast<REAL>();
    const Eigen::RowVector3<REAL> translation = ((1. - alpha) * base.Translation1 + alpha * base.Translation2).cast<REAL>();

    Eigen::VectorXi indices; // Closest triangle indices + barycentric weights
    Eigen::MatrixX<REAL> weights;

    // For closest point query, first transform boundary face positions into rest frame
    const Eigen::Matrix3<REAL> inv_rot = quat.inverse().toRotationMatrix();
    Eigen::Matrix<REAL, Eigen::Dynamic, 3, Eigen::RowMajor> b = base.B.rowwise() - translation;
    b = b.eval() * inv_rot.transpose();

    base.ClosestPoint(indices, weights, b, base.V0);

    // Set entries of mode-to-boundary matrix (assumes already allocated)
    ParallelFor(base.NPoints, 64, [&](size_t bid) {
        const Eigen::Vector3<REAL> bn = base.BN.row(bid);
        for (int vert = 0; vert < 3; ++vert) {
            REAL weight = weights(bid, vert); // barycentric weights
            const int vertex_id = base.F(indices[bid], vert);

            const Eigen::Vector3<REAL> normal = normals.row(vertex_id).cast<REAL>();
            weight *= bn.dot(quat * normal);

            mode_to_boundary.row(bid) += weight * eigen_vectors_normal.row(vertex_id).cast<REAL>();
        }
    });
}

void Modal::Compute(GpuBuffer &vb, int global_bid) {
    auto &base = Obj;
    const int n_modes = Solver._qDot_c_plus.size();
    const Eigen::Quaterniond quat1{base.Rotation1[0], base.Rotation1[1], base.Rotation1[2], base.Rotation1[3]};
    const Eigen::Quaterniond quat2{base.Rotation2[0], base.Rotation2[1], base.Rotation2[2], base.Rotation2[3]};

    // Build mode-to-boundary matrix at batch start (we will eventually interpolate in between batch start and end)
    ModeToBoundary1 = Eigen::MatrixX<REAL>::Zero(base.NPoints, n_modes);
    SetModeToBoundary(ModeToBoundary1);

    GpuModeToBoundary1.Resize(ModeToBoundary1.size() * sizeof(REAL));
    GpuModeToBoundary1.Upload(ModeToBoundary1.data(), ModeToBoundary1.size() * sizeof(REAL));

    // Initialize surface velocity buffers (we will set the data as we go)
    ModeVels = Eigen::MatrixX<REAL>::Zero(n_modes, base.NSamples);
    AccelNoise = Eigen::MatrixX<REAL>::Zero(base.NPoints, base.NSamples);

    // Records whose support can overlap this batch (the exact per-sample check below
    // selects from this subset, preserving record order).
    const auto &impulses = Solver._impulseSeries._impulses;
    const double batch_t1 = base.Step * base.Dt + base.Ts;
    const double batch_t2 = (base.Step + base.NSamples - 1) * base.Dt + base.Ts;
    std::vector<const ModalSound::ImpactRecord *> batch_records;
    for (const auto &record : impulses) {
        if (record.timestamp + record.supportLength > batch_t1 && record.timestamp <= batch_t2) batch_records.push_back(&record);
    }

    Eigen::Vector3<double> accel_pulse1; // current acceleration
    Eigen::Vector3<double> accel_pulse2 = Eigen::Vector3<double>::Zero(); // next acceleration

    // Timestep modal vibrations
    for (int k = 0; k < base.NSamples; ++k) {
        ModeVels.col(k) = Solver._qDot_c_plus.cast<REAL>();

        // -------------------- Acceleration noise -------------------- //
        const double time = base.Step * base.Dt + base.Ts;

        const double alpha = (time - base.T1) / (base.T2 - base.T1);
        const Eigen::Quaterniond quat = quat1.slerp(alpha, quat2);

        accel_pulse1 = accel_pulse2;
        accel_pulse2 = Eigen::Vector3<double>::Zero();

        for (const auto *record : batch_records) {
            const auto &impulse = *record;
            if (!(time >= impulse.timestamp && time < impulse.timestamp + impulse.supportLength)) continue;
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

        for (int bid = 0; bid < base.NPoints; ++bid) { // set normal velocities
            if (base.BN(bid, 0) != 0.f) AccelNoise(bid, k) = AccelSum[0];
            else if (base.BN(bid, 1) != 0.f) AccelNoise(bid, k) = AccelSum[1];
            else if (base.BN(bid, 2) != 0.f) AccelNoise(bid, k) = AccelSum[2];
        }
        // ------------------------------------------------------------------------------- //

        Solver.step(base.Step * base.Dt + base.Ts);
        if (k >= base.NSamples - 1) break;

        AccelSum += (accel_pulse1 + accel_pulse2) / 2.f * base.Dt; // trapezoidal rule integrator
        base.Step += 1;
    }

    GpuModeVels.Resize(ModeVels.size() * sizeof(REAL));
    GpuModeVels.Upload(ModeVels.data(), ModeVels.size() * sizeof(REAL));

    GpuAccelNoise.Resize(AccelNoise.size() * sizeof(REAL));
    GpuAccelNoise.Upload(AccelNoise.data(), AccelNoise.size() * sizeof(REAL));

    // Read next animation data in order to build mode-to-boundary matrix at batch end
    if (base.AnimFile.is_open()) base.ReadAnimation();

    ModeToBoundary2 = Eigen::MatrixX<REAL>::Zero(base.NPoints, n_modes);
    SetModeToBoundary(ModeToBoundary2);

    GpuModeToBoundary2.Resize(ModeToBoundary2.size() * sizeof(REAL));
    GpuModeToBoundary2.Upload(ModeToBoundary2.data(), ModeToBoundary2.size() * sizeof(REAL));

    // Multiply mode-to-boundary matrix with modal velocities on GPU
    const Dim3 threads{16, 16, 1};
    const Dim3 matmul_blocks{uint32_t(base.NPoints + 15) / 16, uint32_t(base.NSamples + 15) / 16, 1};

    const ModeMatmulParams params{global_bid, base.NPoints, n_modes, base.NSamples};
    MetalContext::Get().Dispatch("mode_matmul", matmul_blocks, threads, {&vb, &GpuModeToBoundary1, &GpuModeToBoundary2, &GpuModeVels, &GpuAccelNoise}, &params, sizeof(params));
}
